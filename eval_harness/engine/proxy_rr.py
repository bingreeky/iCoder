#!/usr/bin/env python3
"""Round-robin proxy forwarding /v1/* POSTs across N backend OpenAI servers.

Two modes:
  - local vLLM (default): BACKENDS = N single-GPU vLLM URLs started by
    serve_vllm.sh. Lets verilog-eval's `make -jN` fan out across backends.
  - external API: BACKENDS = one external gateway URL (e.g.
    http://<your-gateway>:<port>/v1). serve_vllm.sh api sets PROXY_API_KEY +
    PROXY_SERVED_NAME so the proxy injects Authorization and rewrites the
    request `model` field — needed because (a) external gateways require a
    real bearer key (clients send a dummy), and (b) KernelBench's local
    query_server hardcodes model="default" which the gateway rejects.

Reliability: requests are retried with backoff on transient upstream faults.
On terminal failure the proxy preserves the upstream status code and body so
clients can distinguish provider, quota, and server failures. Backoff mirrors
the shared client policy in expand/llm.py.
"""
import itertools
import json
import os
import re
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError

BACKENDS = sys.argv[2:] if len(sys.argv) > 2 else []
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
_rr = itertools.cycle(BACKENDS)

# THINK=1 (default): let the model reason. We DON'T force enable_thinking=false,
# and we strip <think>...</think> from the response content so the downstream
# bench code-extractors see only the post-reasoning answer.
# THINK=0: force enable_thinking=false (old non-thinking behavior).
THINK = os.environ.get("THINK", "1") not in ("0", "false", "False", "")

# External-API mode: when PROXY_API_KEY is set, inject Authorization on every
# forwarded request (overrides the client's dummy key). When PROXY_SERVED_NAME
# is set, rewrite the request body `model` field so KB's hardcoded "default"
# reaches the gateway as the real served name. No-op for local vLLM (unset).
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "").strip()
PROXY_SERVED_NAME = os.environ.get("PROXY_SERVED_NAME", "").strip()

# Retry and backoff settings mirror the shared client policy. Transient faults
# are retried; exhausted requests preserve the upstream status and body so the
# caller can classify the failure through a trusted control-plane channel.
PROXY_MAX_RETRIES = int(os.environ.get("PROXY_MAX_RETRIES", "5"))
PROXY_RETRY_BACKOFF_CAP = int(os.environ.get("PROXY_RETRY_BACKOFF_CAP", "20"))
# Quota/429 gets a longer token-bucket-style backoff (30/60/90/120s) — the
# upstream refills on a ~minute window, so short retries just burn the budget.
PROXY_RETRY_429_MAX = int(os.environ.get("PROXY_RETRY_429_MAX",
                                          str(PROXY_MAX_RETRIES)))
# Per-request upstream urlopen timeout (seconds). This bounds an unresponsive
# engine even when a downstream client does not configure its own timeout. Set
# PROXY_UPSTREAM_TIMEOUT=None only when an external supervisor provides the
# required lifecycle boundary.
_upstream_to = os.environ.get("PROXY_UPSTREAM_TIMEOUT", "86400")
PROXY_UPSTREAM_TIMEOUT = None if _upstream_to == "None" else int(_upstream_to)


def strip_think(text):
    if not text:
        return text
    # remove well-formed <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Qwen chat template injects the opening <think> itself, so the returned
    # content usually starts mid-reasoning and only emits a closing </think>.
    # Strip everything up to and including the first orphan </think>.
    text = re.sub(r"\A.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.lstrip("\n")


def _classify(err):
    """Classify an upstream error. Returns (retryable, kind).
    kind in {"quota","membership","transient","client","other"}.
    Mirrors expand/llm.py:184-203."""
    code = getattr(err, "code", None)  # HTTPError carries .code; others don't
    low = str(err).lower()
    is_quota = code == 429 or "quota" in low or "rate limit" in low
    is_membership = code == 402 or "membership" in low
    is_transient = (
        is_quota or is_membership
        or code in (500, 502, 503, 504)
        or "bad gateway" in low
        or "service unavailable" in low
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
    """Delay in seconds for retry #attempt (0-indexed). Mirrors llm.py."""
    if kind == "quota":
        return 30.0 * (attempt + 1)
    return float(min(3 * (2 ** attempt), PROXY_RETRY_BACKOFF_CAP))


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
        # Health check only — do NOT echo the backend list (info disclosure).
        self._send(200, json.dumps({"status": "ok"}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        is_chat = self.path.endswith("/chat/completions")
        is_completions = self.path.endswith("/completions")
        # External-API model rewrite: KB's local query_server sends model=
        # "default"; the gateway needs the real served name. Rewrite the
        # body `model` field on chat/completions endpoints. (No-op for
        # runners that already send the right name.)
        if (is_chat or is_completions) and (PROXY_SERVED_NAME or PROXY_API_KEY):
            try:
                body = json.loads(data)
                changed = False
                if PROXY_SERVED_NAME and body.get("model") != PROXY_SERVED_NAME:
                    body["model"] = PROXY_SERVED_NAME
                    changed = True
                # External gateway (PROXY_API_KEY set): strip the vLLM-only
                # chat_template_kwargs field. Some Anthropic-native / reasoning
                # gateways REJECT any request carrying it (proxy_unavailable —
                # true/false alike, proven 3/3 A/B), so leaving it in zeroes the
                # whole bench. A reasoning model reasons by default on a bare
                # request, so thinking stays ON without the flag. Local vLLM (no
                # PROXY_API_KEY) keeps the field — its chat template needs it.
                if PROXY_API_KEY and "chat_template_kwargs" in body:
                    del body["chat_template_kwargs"]
                    changed = True
                if changed:
                    data = json.dumps(body).encode()
            except Exception:
                pass
        # THINK=0: force enable_thinking=false so <think> never appears (old
        # non-thinking behavior). THINK=1: leave thinking on (chat template
        # default) and strip <think> from the response instead.
        if is_chat and not THINK and not PROXY_API_KEY:
            try:
                body = json.loads(data)
                kw = body.setdefault("chat_template_kwargs", {})
                kw.setdefault("enable_thinking", False)
                data = json.dumps(body).encode()
            except Exception:
                pass
        backend = next(_rr)
        url = backend.rstrip("/") + self.path
        # Inject Authorization for external-API mode (overrides dummy client
        # key). Local vLLM ignores the header either way.
        headers = {"Content-Type": "application/json"}
        if PROXY_API_KEY:
            headers["Authorization"] = f"Bearer {PROXY_API_KEY}"
        status, resp = self._forward(url, data, headers)
        if status == 200 and (is_chat or is_completions) and THINK:
            # strip <think> from each choice so bench extractors see clean code
            try:
                obj = json.loads(resp)
                for ch in obj.get("choices", []):
                    msg = ch.get("message")
                    if msg and isinstance(msg.get("content"), str):
                        msg["content"] = strip_think(msg["content"])
                    # /completions-style choices carry .text
                    if isinstance(ch.get("text"), str):
                        ch["text"] = strip_think(ch["text"])
                resp = json.dumps(obj).encode()
            except Exception:
                pass
        # On non-200, pass through the upstream's REAL status + body verbatim
        # (no synthetic 500) so clients/preflight can distinguish e.g. 503
        # "No active provider key" from a genuine 500.
        self._send(status, resp)

    def _forward(self, url, data, headers):
        """Forward one request with retry/backoff. Returns (status, body).
        On success: (200, body). On terminal failure: passthrough the real
        upstream status code + body (or 502 for connection-level faults)."""
        max_retries = PROXY_MAX_RETRIES
        last_status, last_body = 502, b""
        for attempt in range(max_retries):
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                # Per-request upstream timeout. KB's query_server uses an OpenAI
                # client with timeout=None (utils.py) — a single hung/runaway
                # generation would block a whole KernelBench level forever (seen
                # the model stall level1 at 99/100 for 40min+, engine idle). Cap
                # here so a stuck upstream request returns an error the bench can
                # handle (marked infra/dropped) instead of hanging the job.
                # Override via PROXY_UPSTREAM_TIMEOUT.
                with urllib.request.urlopen(req, timeout=PROXY_UPSTREAM_TIMEOUT) as r:
                    return 200, r.read()
            except HTTPError as e:
                last_status = e.code
                last_body = e.read()  # upstream error JSON (e.g. 503 body)
                retryable, kind = _classify(e)
            except TimeoutError as e:
                # OUR urlopen READ timeout — the upstream is generating too slowly
                # (runaway to max_tokens on a slow local model). Retrying just
                # re-runs the same slow generation (5x600s=50min wasted; seen
                # the model KB gen crawl). Fail FAST: return immediately so the
                # bench marks the sample dropped/infra and moves on.
                last_status = 504
                last_body = json.dumps(
                    {"error": f"upstream read timeout after "
                              f"{PROXY_UPSTREAM_TIMEOUT}s"}).encode()
                return last_status, last_body
            except (URLError, TimeoutError, ConnectionError, OSError) as e:
                last_status = 502
                last_body = json.dumps(
                    {"error": f"{type(e).__name__}: {e}"}).encode()
                retryable, kind = _classify(e)
            except Exception as e:  # any other unexpected fault
                last_status = 502
                last_body = json.dumps(
                    {"error": f"{type(e).__name__}: {e}"}).encode()
                retryable, kind = False, "other"
            # Per-class cap: quota may use a shorter retry budget.
            cap = PROXY_RETRY_429_MAX if kind == "quota" else max_retries
            if not retryable or attempt + 1 >= cap:
                return last_status, last_body
            delay = _backoff_delay(kind, attempt)
            print(f"[proxy] upstream {kind} status={last_status} "
                  f"attempt={attempt + 1}/{cap} sleep={delay:.0f}s "
                  f"path={self.path}", file=sys.stderr)
            time.sleep(delay)
        return last_status, last_body


if __name__ == "__main__":
    HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
    print(f"[proxy] {HOST}:{PORT} think={THINK} retries={PROXY_MAX_RETRIES} "
          f"({len(BACKENDS)} backends)", flush=True)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
