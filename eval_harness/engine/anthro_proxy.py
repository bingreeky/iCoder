#!/usr/bin/env python3
"""Anthropic-native -> OpenAI translation proxy. Drop-in for proxy_rr.py.

Why this exists: some gateways speak Anthropic Messages format ONLY (e.g.
Anthropic-native gateways for Claude Code). They return
{'type':'message','content':[{'type':'text','text':...}]} with NO OpenAI
`choices` key. The eval harness uses langchain ChatOpenAI / OpenAI clients,
which parse `choices[0].message.content` -> content=None -> TypeError ->
empty .sv / no kernel. proxy_rr.py is a dumb passthrough (injects auth,
rewrites model, retries) — it does NOT translate. So an Anthropic-native
upstream poisons every bench row with empty output.

This proxy sits on :PROXY_PORT (same slot proxy_rr.py would) and:
  1. accepts OpenAI POST /v1/chat/completions (what the harness sends)
  2. translates messages -> Anthropic /v1/messages (system split out, content
     blocks passed through; max_tokens defaulted to 49152 if absent — Anthropic
     REQUIRES it, OpenAI clients always send it for these benches)
  3. forwards to the upstream Anthropic gateway with x-api-key +
     anthropic-version headers (NOT Bearer — that's OpenAI-only and gets 401)
  4. translates the Anthropic response back to OpenAI choices format
  5. on transient upstream faults, retries with the same backoff ladder as
     proxy_rr.py / expand/llm.py; on terminal failure passes the REAL
     upstream status + body through (so preflight sees a 503, not a synthetic
     500).

Non-streaming only — every eval_harness bench uses langchain invoke() /
openai create() (0 stream=True calls anywhere in the tree). A streaming
client would need SSE chunk translation, which is out of scope; if one
appears the proxy returns a normal JSON completion and the client errors
loudly (better than silent empty poisoning).

Env (set by serve_vllm.sh api mode, same names as proxy_rr.py):
  PROXY_API_KEY      -> sent as x-api-key (Anthropic auth header)
  PROXY_SERVED_NAME  -> rewrites request `model` field (KB hardcodes "default")
  PROXY_PORT         -> listen port (default 8000)
  PROXY_MAX_RETRIES, PROXY_RETRY_BACKOFF_CAP, PROXY_UPSTREAM_TIMEOUT
  THINK=0 strips <think> blocks from the mapped content (default THINK=1
       keeps reasoning; Anthropic reasoning is NOT in content for these
       non-thinking-served models, so this is a near-no-op safety net).
"""
import itertools
import json
import os
import re
import ssl
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError

# PROXY_INSECURE_TLS=1: disable upstream TLS verification. ONLY for dev boxes
# behind a self-signed MITM proxy (curl trusts it via system CA, Python
# certifi/urllib does not) so the proxy can still reach the gateway for a local
# test. Pods run on clean network — leave unset there.
if os.environ.get("PROXY_INSECURE_TLS", "") in ("1", "true", "True"):
    ssl._create_default_https_context = ssl._create_unverified_context

# Positional: PORT [UPSTREAM_ORIGIN ...]. Mirrors proxy_rr.py's argv shape so
# serve_vllm.sh can launch either interchangeably. Only the FIRST upstream is
# used (these gateways are single-endpoint); extras tolerated for compat.
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
UPSTREAMS = [u.rstrip("/") for u in sys.argv[2:]] if len(sys.argv) > 2 else []
UPSTREAM = UPSTREAMS[0] if UPSTREAMS else os.environ.get("PROXY_UPSTREAM_URL", "")

PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "").strip()
PROXY_SERVED_NAME = os.environ.get("PROXY_SERVED_NAME", "").strip()
THINK = os.environ.get("THINK", "1") not in ("0", "false", "False", "")
# THINK_FORCE=1: inject Anthropic extended thinking into EVERY request
# (budget_tokens from THINK_BUDGET, default 24000). Anthropic thinking
# REQUIRES temperature=1 and forbids top_p/top_k, so when forced we pin
# temperature=1 and drop top_p regardless of what the OpenAI client sent.
THINK_FORCE = os.environ.get("THINK_FORCE", "0") in ("1", "true", "True")
THINK_BUDGET = int(os.environ.get("THINK_BUDGET", "24000"))

PROXY_MAX_RETRIES = int(os.environ.get("PROXY_MAX_RETRIES", "5"))
PROXY_RETRY_BACKOFF_CAP = int(os.environ.get("PROXY_RETRY_BACKOFF_CAP", "20"))
# When THINK_FORCE is on, opus intermittently returns a response whose content
# blocks are thinking+tool_use with NO text block (it "decides" to call a tool
# that doesn't exist) -> anth_to_oai emits empty content -> the bench row gets
# an empty .v/.sv -> silent failure. This is ~40% of opus thinking responses on
# heavy prompts (ArchX), non-deterministic at temp=1. Retry the upstream request
# when the 200 response carries no text block; opus usually emits text on retry.
# Bound to avoid runaway load under a sustained tool_use storm.
PROXY_RETRY_EMPTY = int(os.environ.get("PROXY_RETRY_EMPTY", "3"))
PROXY_RETRY_429_MAX = int(os.environ.get("PROXY_RETRY_429_MAX",
                                          str(PROXY_MAX_RETRIES)))
_upstream_to = os.environ.get("PROXY_UPSTREAM_TIMEOUT", "86400")
PROXY_UPSTREAM_TIMEOUT = None if _upstream_to == "None" else int(_upstream_to)

ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")


def strip_think(text):
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\A.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.lstrip("\n")


# ---- retry classification (mirrors proxy_rr.py / expand/llm.py:184-203) ----
def _classify(err):
    code = getattr(err, "code", None)
    low = str(err).lower()
    is_quota = code == 429 or "quota" in low or "rate limit" in low
    is_membership = code == 402 or "membership" in low
    is_transient = (
        is_quota or is_membership
        or code in (500, 502, 503, 504)
        or "bad gateway" in low or "service unavailable" in low
        or "overloaded" in low  # Anthropic 529 overloaded
        or "timeout" in low or "timed out" in low
        or isinstance(err, (URLError, TimeoutError, ConnectionError, OSError))
    )
    if not is_transient:
        return False, "client" if code else "other"
    if is_quota:
        return True, "quota"
    if is_membership:
        return True, "membership"
    return True, "transient"


def _backoff_delay(kind, attempt):
    if kind == "quota":
        return 30.0 * (attempt + 1)
    return float(min(3 * (2 ** attempt), PROXY_RETRY_BACKOFF_CAP))


# ---- OpenAI -> Anthropic request translation ----
def oai_to_anth(body):
    """Translate an OpenAI chat.completions request body to Anthropic
    /v1/messages body. Returns the Anthropic dict, or None if the body is
    not a parseable chat request (caller then passes it through untouched)."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return None
    system_parts = []
    anth_msgs = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        # OpenAI content can be a str or a list of {type:"text", text:...}.
        # Anthropic accepts the SAME block shape ({type:"text", text:...}) and
        # a bare str, so pass both through unchanged.
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        system_parts.append(blk.get("text", ""))
            continue
        # OpenAI "tool"/"function" roles are out of scope for the eval benches
        # (no tool-use in gen). Skip anything we can't map rather than crash.
        if role not in ("user", "assistant"):
            continue
        anth_msgs.append({"role": role, "content": content if content is not None else ""})
    anth = {
        "model": body.get("model") or PROXY_SERVED_NAME or "",
        "messages": anth_msgs,
        # Anthropic REQUIRES max_tokens (no default). OpenAI clients always send
        # it for these benches; default only as a safety net.
        "max_tokens": int(body.get("max_tokens") or 49152),
    }
    if system_parts:
        anth["system"] = "\n\n".join(system_parts)
    # THINK_FORCE: inject extended thinking. Anthropic thinking REQUIRES
    # temperature=1 and forbids top_p/top_k, so pin temp=1 and drop top_p
    # regardless of what the OpenAI client sent (VE sends temp=0.8, top_p=0.95).
    # Without this normalization the upstream rejects with 400
    # "temperature: expected 1.0 for extended thinking".
    # ALSO: Anthropic requires thinking.budget_tokens < max_tokens (else 400
    # "budget_tokens must be less than max_tokens"). VE sends max_tokens=49152
    # so THINK_BUDGET=32000 is fine, but other clients (CVDP/ArchX-edge) may send
    # a small max_tokens -> clamp budget to max_tokens-1024 (leave room for the
    # answer). If max_tokens is too small to think meaningfully (<2048 budget),
    # SKIP thinking on that request (respect client temp/top_p) rather than 400.
    if THINK_FORCE:
        budget = min(THINK_BUDGET, anth["max_tokens"] - 1024)
    else:
        budget = None
    if budget is not None and budget >= 1024:
        anth["temperature"] = 1
        anth.pop("top_p", None)
        anth["thinking"] = {"type": "enabled", "budget_tokens": budget}
    else:
        # no thinking (THINK_FORCE off, or max_tokens too small to budget) ->
        # respect the client's temperature/top_p (Anthropic supports both).
        if "temperature" in body and body["temperature"] is not None:
            anth["temperature"] = body["temperature"]
        if "top_p" in body and body["top_p"] is not None:
            anth["top_p"] = body["top_p"]
    if body.get("stop"):
        stop = body["stop"]
        anth["stop_sequences"] = stop if isinstance(stop, list) else [stop]
    return anth


# ---- Anthropic -> OpenAI response translation ----
def anth_to_oai(anth, model):
    """Translate an Anthropic /v1/messages response to OpenAI chat.completion."""
    blocks = anth.get("content") or []
    if isinstance(blocks, str):
        text = blocks
    else:
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
    if not THINK:
        text = strip_think(text)
    finish = {"end_turn": "stop", "max_tokens": "length",
              "stop_sequence": "stop", "tool_use": "tool_calls"}.get(
        anth.get("stop_reason"), "stop")
    usage = anth.get("usage") or {}
    return {
        "id": anth.get("id", "chatcmpl-anthro"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": (usage.get("input_tokens") or 0)
                            + (usage.get("output_tokens") or 0),
        },
    }


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Health check only — do NOT echo the upstream URL (info disclosure).
        self._send(200, json.dumps({"status": "ok"}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        is_chat = self.path.endswith("/chat/completions")
        is_completions = self.path.endswith("/completions")
        if not (is_chat or is_completions):
            # Unknown POST path — pass through best-effort (Anthropic gateways
            # don't serve other /v1/* endpoints the harness hits).
            self._send(404, json.dumps({"error": f"unsupported path {self.path}"}).encode())
            return
        try:
            body = json.loads(data)
        except Exception:
            self._send(400, json.dumps({"error": "invalid json body"}).encode())
            return
        # model rewrite (KB hardcodes "default" -> real served name)
        if PROXY_SERVED_NAME and body.get("model") != PROXY_SERVED_NAME:
            body["model"] = PROXY_SERVED_NAME
        model = body.get("model") or PROXY_SERVED_NAME or "unknown"
        anth = oai_to_anth(body)
        if anth is None:
            # Not a translatable chat body — return a clean OpenAI error so the
            # bench row is marked failed instead of silently empty.
            self._send(400, json.dumps({"error": "no messages in request",
                                         "model": model}).encode())
            return
        status, resp = self._forward_anth(anth)
        if status == 200:
            for empty_attempt in range(PROXY_RETRY_EMPTY + 1):
                try:
                    anth_obj = json.loads(resp)
                except Exception as e:
                    self._send(502, json.dumps({"error": f"translate failed: {e}",
                                                 "upstream": resp[:500].decode(errors="replace")}).encode())
                    return
                # Does this response carry a non-empty text block? opus+thinking
                # sometimes returns only thinking+tool_use blocks (no text) ->
                # empty content -> silent bench failure. Retry upstream for a
                # text answer (temp=1 -> non-deterministic, usually recovers).
                blocks = anth_obj.get("content")
                has_text = False
                if isinstance(blocks, list):
                    has_text = any(isinstance(b, dict) and b.get("type") == "text"
                                   and (b.get("text") or "").strip() for b in blocks)
                elif isinstance(blocks, str) and blocks.strip():
                    has_text = True
                if has_text or empty_attempt >= PROXY_RETRY_EMPTY:
                    break
                print(f"[anthro-proxy] empty-content (no text block) retry "
                      f"attempt={empty_attempt + 1}/{PROXY_RETRY_EMPTY} "
                      f"stop_reason={anth_obj.get('stop_reason')}", file=sys.stderr)
                status, resp = self._forward_anth(anth)
                if status != 200:
                    break
            try:
                oai = anth_to_oai(json.loads(resp), model)
                resp = json.dumps(oai).encode()
            except Exception as e:
                self._send(502, json.dumps({"error": f"translate failed: {e}",
                                             "upstream": resp[:500].decode(errors="replace")}).encode())
                return
        self._send(status, resp)

    def _forward_anth(self, anth_body):
        """POST the Anthropic request to the upstream with retry/backoff.
        Returns (status, body_bytes). On terminal failure passes the REAL
        upstream status + body through."""
        url = UPSTREAM.rstrip("/") + "/v1/messages"
        data = json.dumps(anth_body).encode()
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            # Some Anthropic gateways 502-reject the default
            # `Python-urllib/x.y` User-Agent with "Upstream access forbidden" on
            # EVERY request -> the proxy retries 5x (~61s) then surfaces the 502
            # -> KB gen produces 0 kernels (every call fails). Any explicit UA
            # is accepted (curl/8.0, anthro-proxy, etc. all pass). So always set
            # one.
            "User-Agent": os.environ.get("PROXY_USER_AGENT", "anthro-proxy/1.0"),
        }
        # Anthropic auth = x-api-key header (not Bearer). Some Anthropic gateways
        # ALSO accept Authorization: Bearer; send both for max compat.
        if PROXY_API_KEY:
            headers["x-api-key"] = PROXY_API_KEY
            headers["Authorization"] = f"Bearer {PROXY_API_KEY}"
        max_retries = PROXY_MAX_RETRIES
        last_status, last_body = 502, b""
        for attempt in range(max_retries):
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=PROXY_UPSTREAM_TIMEOUT) as r:
                    return 200, r.read()
            except HTTPError as e:
                last_status = e.code
                last_body = e.read()
                retryable, kind = _classify(e)
            except TimeoutError as e:
                last_status = 504
                last_body = json.dumps({"error": f"upstream read timeout after "
                                                 f"{PROXY_UPSTREAM_TIMEOUT}s"}).encode()
                return last_status, last_body
            except (URLError, TimeoutError, ConnectionError, OSError) as e:
                last_status = 502
                last_body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
                retryable, kind = _classify(e)
            except Exception as e:
                last_status = 502
                last_body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
                retryable, kind = False, "other"
            cap = PROXY_RETRY_429_MAX if kind == "quota" else max_retries
            if not retryable or attempt + 1 >= cap:
                return last_status, last_body
            delay = _backoff_delay(kind, attempt)
            print(f"[anthro-proxy] upstream {kind} status={last_status} "
                  f"attempt={attempt + 1}/{cap} sleep={delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
        return last_status, last_body


if __name__ == "__main__":
    print(f"[anthro-proxy] :{PORT} think={THINK} retries={PROXY_MAX_RETRIES} "
          f"-> {UPSTREAM} (served={PROXY_SERVED_NAME}, key={'set' if PROXY_API_KEY else 'unset'})",
          flush=True)
    if not UPSTREAM:
        print("[anthro-proxy] !! no upstream (pass UPSTREAM as argv[2] or "
              "PROXY_UPSTREAM_URL)", file=sys.stderr)
    HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
