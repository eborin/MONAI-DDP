# ==================================================================
# Image: Base
# ==================================================================
FROM nvidia/cuda:11.6.0-base-ubuntu20.04 AS base

ENV LANG C.UTF-8
ENV APT_INSTALL "apt-get install -y --no-install-recommends"
ENV PIP_INSTALL "python -m pip --no-cache-dir install --upgrade"

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update

# ------------------------------------------------------------------
# Installs Python 3.8
# ------------------------------------------------------------------
RUN $APT_INSTALL \
        wget \
        python3.8 \
        python3-distutils && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py && \
    rm -rf ~/get-pip.py && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python

# ------------------------------------------------------------------
# Installs Pytorch
# ------------------------------------------------------------------
RUN $APT_INSTALL \
        python3-dev \
        gcc && \
    $PIP_INSTALL \
        --pre torch torchvision torchaudio -f \
        https://download.pytorch.org/whl/nightly/cu116/torch_nightly.html

# ------------------------------------------------------------------
# Installs project dependencies
# ------------------------------------------------------------------
COPY requirements.txt /
RUN $PIP_INSTALL \
        -r /requirements.txt && \
    rm -rf /requirements.txt

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
RUN ldconfig & \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# ==================================================================
# Image: Sagemaker
# ==================================================================
FROM base
ENV PIP_INSTALL "python -m pip --no-cache-dir install --upgrade"

# ------------------------------------------------------------------
# Installs Sagemaker
# ------------------------------------------------------------------

RUN $PIP_INSTALL \
    sagemaker-training