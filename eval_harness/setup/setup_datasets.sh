#!/bin/bash
# setup_datasets.sh — clone the public bench datasets into a benchmark root.
# Output: $1 (default ./benchmarks), with the layout config.local.sh expects:
#   verilog-eval/verilog-eval/{configure,dataset_spec-to-rtl,...}
#   rtllm/RTLLM/<Cat>/<Family>/<design>/{design_description.txt,testbench.v}
#   KernelBench/KernelBench/level{1,2,3}
#   TritonBench/
#   ArchXBench/level-{0..6}/<design>/{problem-description.txt,tb.v}
#   RealBench/{aes,sdc,e203_hbirdv2,run_verify.py}  (+ `make decrypt` for .md.gpg)
set -euo pipefail
ROOT=${1:-./benchmarks}
mkdir -p "$ROOT"; cd "$ROOT"

clone() {  # name url dest
  if [ -d "$3" ]; then echo "[datasets] $1: exists, skip"; else
    echo "[datasets] $1: cloning $2"; git clone --depth 1 "$2" "$3"; fi
}

clone KernelBench  https://github.com/ScalingIntelligence/KernelBench.git KernelBench
clone TritonBench  https://github.com/thunlp/TritonBench.git         TritonBench
clone ArchXBench   https://github.com/sureshpurini/ArchXBench.git     ArchXBench
clone RealBench    https://github.com/IPRC-DIP/RealBench.git          RealBench
clone RTLLM        https://github.com/hkust-zhiyao/RTLLM.git          rtllm
# verilog-eval: NVIDIA's repo ships the harness (configure/sv-generate) under
# verilog-eval/verilog-eval/. If the upstream URL changes, adjust here.
clone verilog-eval https://github.com/NVlabs/verilog-eval.git         verilog-eval

# RealBench publishes an upstream target for preparing its encrypted files.
if [ -d RealBench ] && ! find RealBench -name '*.md' -not -name 'README*' | grep -q .; then
  echo "[datasets] RealBench: running upstream decrypt target..."
  ( cd RealBench && make decrypt >/tmp/rb_decrypt.log 2>&1 || tail -5 /tmp/rb_decrypt.log )
fi

# Apply the 3 KB patches the harness needs (chat endpoint, env port/suffix)
EH="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$EH/setup/apply_kb_patches.py" "$ROOT/KernelBench"
python3 "$EH/setup/apply_veval_patches.py" "$ROOT/verilog-eval/verilog-eval" 2>/dev/null || echo "[datasets] veval patch skipped (verilog-eval layout?)"

cat <<EOF
[datasets] done in $ROOT. Set in config.local.sh:
  export CODERBENCH_ROOT=$ROOT
  export VERILOGEVAL_ROOT=$ROOT/verilog-eval/verilog-eval
  export RTLLM_ROOT=$ROOT/rtllm/RTLLM
  export KERNELBENCH_ROOT=$ROOT/KernelBench
  export TRITONBENCH_ROOT=$ROOT/TritonBench
  export ARCHX_ROOT=$ROOT/ArchXBench
  export RB_ROOT=$ROOT/RealBench
EOF

# CVDP factory patch (if CVDP_local is mounted separately)
# python3 "$EH/setup/apply_cvdp_patches.py" /path/to/CVDP_local
