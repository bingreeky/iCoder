#!/usr/bin/env python3
# ⚠️ Diagnostic tool — NOT part of the standard eval pipeline. This is a
#    recovery/rescore helper (not invoked by run_all.sh / summarize.sh).
"""_rtllm_selfmatch_smoke.py — golden self-match smoke for verify/icarus.

After build_profiles has produced a verdict profile, pick the first task whose
golden reference produced a stdout hash, re-run it through execute_icarus, and
judge it against the profile. A correct pipeline MUST mark the golden as
correct (its hash is in the profile by construction). Any other outcome =
verify/icarus.py or build_profiles is broken.

Usage: python _rtllm_selfmatch_smoke.py <rtllm_root> <profile.json>
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verify.icarus import execute_icarus, judge_against_profile  # noqa: E402


def main() -> int:
    rtllm_root, profile_path = sys.argv[1], sys.argv[2]
    prof = json.load(open(profile_path))
    print(f"[selfmatch] profile has {len(prof)} tasks")
    # RTLLM designs nest at depth 3 (Category/Subcategory/design/); os.walk in
    # build_profiles reaches all depths, so the profile has them — this glob must
    # reach depth 3 too, plus the shallow forms for flatter layouts.
    for d in sorted(glob.glob(os.path.join(rtllm_root, "*", "*", "*"))
                    + glob.glob(os.path.join(rtllm_root, "*", "*"))
                    + glob.glob(os.path.join(rtllm_root, "*"))):
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        if "design_description.txt" not in files or "testbench.v" not in files:
            continue
        name = os.path.basename(d)
        if name not in prof or not prof[name]:
            continue
        refs = [v for v in glob.glob(os.path.join(d, "*.v"))
                if os.path.basename(v) != "testbench.v"
                and not os.path.basename(v).startswith("testbench")]
        if not refs:
            continue
        src = open(refs[0], encoding="utf-8", errors="replace").read()
        r = execute_icarus(src, testbench_path=os.path.join(d, "testbench.v"),
                           timeout=60, eval_backend="rtllm")
        r = judge_against_profile(r, frozenset(prof[name]))
        print(f"[selfmatch] {name}: compiled={r.get('compiled')} "
              f"correct={r.get('correct')} info={r.get('info')} "
              f"stage={(r.get('verify_meta') or {}).get('stage')} "
              f"hash={str(r.get('stdout_sha256') or '')[:12]}")
        # the golden MUST self-match (its hash is in the profile)
        return 0 if r.get("correct") else 2
    print("[selfmatch] no runnable golden task with a profile hash found")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
