# Dockerfile for eval_harness — one-click reproducible eval environment.
# Build:  docker build -t eval-harness .
# Run:    docker run --gpus all -v $PWD:/harness -w /harness eval-harness ...
#
# Contains: iverilog v12 + verilator + yosys (built via setup/setup_toolchain.sh)
# + torch + triton + KB pip deps (litellm pydra-config ...). Datasets are
# NOT bundled — mount or run setup/setup_datasets.sh.
FROM python:3.11-slim AS toolchain-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bison flex gperf g++ make autoconf git curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
# Build iverilog v12 (v14-devel in oss-cad-suite has a $dumpvars bug -> pass_rate=0)
RUN cd /tmp && curl -sL -o iv.tgz https://api.github.com/repos/steveicarus/iverilog/tarball/v12_0 \
    && tar xzf iv.tgz && cd steveicarus-iverilog-* \
    && sh ./autoconf.sh && ./configure --prefix=/opt/iverilog12 && make -j"$(nproc)" && make install \
    && /opt/iverilog12/bin/iverilog -V | head -1
# oss-cad-suite for verilator + yosys (its iverilog is v14-devel — DO NOT rely on it)
RUN cd /opt && TAG=$(curl -sL https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest | \
        python3 -c "import sys,json;print(json.load(sys.stdin)['tag_name'])") \
    && curl -sL -o ocs.tgz "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/$TAG/oss-cad-suite-linux-x64-${TAG//-/}.tgz" \
    && tar xzf ocs.tgz -C /opt && rm ocs.tgz

FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
# iverilog v12 + verilator + yosys from builder
COPY --from=toolchain-builder /opt/iverilog12 /opt/iverilog12
COPY --from=toolchain-builder /opt/oss-cad-suite /opt/oss-cad-suite
ENV PATH=/opt/iverilog12/bin:/opt/oss-cad-suite/oss-cad-suite/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/oss-cad-suite/oss-cad-suite/lib:$LD_LIBRARY_PATH
# torch + triton + KB pip deps (CUDA runtime must be on the host; use a torch
# wheel matching your CUDA. Default cu121.)
RUN pip install --no-cache-dir \
    torch triton \
    litellm pydra-config tomli tabulate einops python-dotenv ninja openai \
    cocotb langchain-openai langchain-community \
    && rm -rf /root/.cache
# iverilog sanity ($dumpvars must produce VCD, not the v14-devel forward-ref bug)
RUN iverilog -V | head -1 && verilator --version | head -1 && yosys --version | head -1
# Drop privileges: run evals as a non-root user (mounts/results stay unprivileged)
RUN useradd -u 1000 -m -d /home/harness harness
USER 1000:1000
WORKDIR /harness
