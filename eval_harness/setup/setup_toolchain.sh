#!/bin/bash
# setup_toolchain.sh — stage iverilog v12 + verilator/yosys (oss-cad-suite).
# These are NOT in the eval_harness repo (binaries too big). This script builds
# iverilog v12 from source (the v14-devel in latest oss-cad-suite has a
# $dumpvars forward-ref bug that zeroes VEval/RTLLM pass_rate) and downloads
# oss-cad-suite for verilator/yosys (RealBench).
#
# Output: $1 (default /opt/eval-toolchain), with:
#   iverilog12/bin/{iverilog,vvp,iverilog-vpi}
#   oss-cad-suite/oss-cad-suite/bin/{verilator,yosys,...}
set -euo pipefail
OUT=${1:-/opt/eval-toolchain}
mkdir -p "$OUT"

# --- iverilog v12 (stable) from source ---
IV="$OUT/iverilog12"
if [ -x "$IV/bin/iverilog" ] && "$IV/bin/iverilog" -V 2>&1 | grep -q 'version 12'; then
  echo "[toolchain] iverilog v12 already built at $IV"
else
  echo "[toolchain] building iverilog v12 from source..."
  apt-get update -y && apt-get install -y bison flex gperf g++ make autoconf git
  TMP=$(mktemp -d); cd "$TMP"
  curl -sL -o iv.tgz https://api.github.com/repos/steveicarus/iverilog/tarball/v12_0
  tar xzf iv.tgz && cd steveicarus-iverilog-*
  sh ./autoconf.sh && ./configure --prefix="$IV" && make -j"$(nproc)" && make install
  "$IV/bin/iverilog" -V | head -1
  cd / && rm -rf "$TMP"
fi

# --- oss-cad-suite (verilator + yosys; its iverilog is v14-devel, DO NOT USE) ---
OSS="$OUT/oss-cad-suite"
if [ -x "$OSS/oss-cad-suite/bin/verilator" ]; then
  echo "[toolchain] oss-cad-suite already at $OSS"
else
  echo "[toolchain] downloading oss-cad-suite (for verilator/yosys)..."
  mkdir -p "$OSS"; cd "$OSS"
  TAG=$(curl -sL https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest | \
        python3 -c "import sys,json;print(json.load(sys.stdin)['tag_name'])")
  curl -sL -o ocs.tgz "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$TAG/oss-cad-suite-linux-x64-${TAG//-/}.tgz"
  tar xzf ocs.tgz && rm ocs.tgz
  "$OSS/oss-cad-suite/bin/verilator" --version | head -1
fi

# --- setup_env.sh (adds oss-cad-suite bin+lib to PATH; iverilog12 prepended by config) ---
cat > "$OUT/setup_env.sh" <<EOF
#!/bin/bash
OSS="$OUT/oss-cad-suite/oss-cad-suite"
[ -d "\$OSS/bin" ] && case ":\$PATH:" in *":\$OSS/bin:"*) ;; *) export PATH="\$OSS/bin:\$PATH";; esac
[ -d "\$OSS/lib" ] && case ":\${LD_LIBRARY_PATH:-}:" in *":\$OSS/lib:"*) ;; *) export LD_LIBRARY_PATH="\$OSS/lib:\${LD_LIBRARY_PATH:-}";; esac
EOF
chmod +x "$OUT/setup_env.sh"
echo "[toolchain] done. Set in config.local.sh:"
echo "  export IVERILOG12_BIN=$IV/bin"
echo "  export SETUP_ENV=$OUT/setup_env.sh"
