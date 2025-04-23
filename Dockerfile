FROM nvcr.io/nvidia/tritonserver:24.04-py3-min as base
ARG PYTHON_VERSION=3.9
ARG MAMBA_VERSION=23.1.0-1
ARG TARGETPLATFORM

ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

RUN chmod 777 -R /tmp && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    curl \
    g++ \
    make \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
    *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -o ~/mambaforge.sh -v "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh" && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  exit 1 ;; \
    *)              /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" ;; \
    esac && \
    /opt/conda/bin/conda clean -ya


WORKDIR /root

RUN pip install --no-cache-dir --ignore-installed --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.5.1

RUN git clone https://github.com/Dao-AILab/flash-attention.git -b v2.7.4.post1
RUN cd flash-attention/hopper && MAX_JOBS=1 NVCC_THREADS=1 FLASH_ATTN_CUDA_ARCHS=90 python setup.py bdist_wheel
RUN mkdir -p /out && cp flash-attention/hopper/dist/*.whl /out/
