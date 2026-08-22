#!/bin/bash
# build_venv_cu121.sh — build a GDN-correct venv (.venv-vllm-cu121) for a
# GDN-architecture model, NON-destructive (new dir; does not touch .venv-vllm).
# vllm 0.20 has native GDN support (user spec); let it pull its torch, then
# install flashinfer (matching cu/torch wheel index) + fla 0.5.1 +
# causal_conv1d 1.6.2.post1 (prebuilt wheels where possible, else source w/ nvcc).
# The cu130/.venv-vllm (torch2.11+cu130) can't install GDN packages (host nvcc
# 12.1 ≠ torch cu130; no cu130 prebuilt wheels). This cu121-ish venv should.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
SFT_ROOT="${SFT_ROOT:-$REPO_ROOT}"
V="$REPO_ROOT/.venv-vllm-cu121/bin/python"
LOG="$RESULTS_DIR/_audit_cu121/cu121_build.log"
echo "=== $(date) start ===" > $LOG
echo "=== 1) install vllm==0.20.0 (pulls torch + cuda) ===" >> $LOG
$V -m pip install 'vllm==0.20.0' >> $LOG 2>&1 || echo "VLLM INSTALL FAILED" >> $LOG
echo "=== 2) torch version pulled ===" >> $LOG
$V -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)" >> $LOG 2>&1 || echo "torch import failed" >> $LOG
TORCH=$($V -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null)
CUDA=$($V -c "import torch;print(torch.version.cuda)" 2>/dev/null)
CUSHORT=$(echo "$CUDA" | tr -d .)
TMAJORMINOR=$(echo "$TORCH" | cut -d. -f1,2)
echo "resolved: torch=$TORCH cuda=$CUDA cushort=$CUSHORT tmajmin=$TMAJORMINOR" >> $LOG
echo "=== 3) install flashinfer (index cu$CUSHORT/torch$TMAJORMINOR) ===" >> $LOG
$V -m pip install flashinfer --index-url "https://flashinfer.ai/whl/cu${CUSHORT}/torch${TMAJORMINOR}/" >> $LOG 2>&1 || echo "FLASHINFER INSTALL FAILED (try default PyPI)" >> $LOG
if ! $V -c "import flashinfer" 2>/dev/null; then
  $V -m pip install flashinfer >> $LOG 2>&1 || echo "flashinfer default also failed" >> $LOG
fi
echo "=== 4) install fla 0.5.1 + causal_conv1d 1.6.2.post1 ===" >> $LOG
$V -m pip install 'flash-linear-attention==0.5.1' 'causal-conv1d==1.6.2.post1' >> $LOG 2>&1 || echo "FLA/CC1D INSTALL FAILED" >> $LOG
echo "=== 5) verify imports ===" >> $LOG
$V -c "import vllm,torch,fla,causal_conv1d,flashinfer; print('vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda); print('ALL GDN PACKAGES OK')" >> $LOG 2>&1 || echo "VERIFY FAILED (some import missing)" >> $LOG
echo "=== $(date) DONE ===" >> $LOG
