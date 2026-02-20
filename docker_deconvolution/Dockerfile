# -----------------------
# Base Image
# -----------------------
FROM --platform=linux/amd64 ubuntu:22.04

# -----------------------
# RCP CaaS User Mapping
# -----------------------
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

USER root
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} \
    && useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# -----------------------
# System dependencies
# -----------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    ocl-icd-opencl-dev \
    libopencl1 \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Clone your GitHub repo
# -----------------------
USER ${LDAP_USERNAME}
WORKDIR /home/${LDAP_USERNAME}
RUN git clone https://github.com/feyzanar/docker.git deconvolution_repo

# -----------------------
# Python dependencies
# -----------------------
RUN pip install --upgrade pip
RUN pip install \
    numpy \
    tifffile \
    tqdm \
    dask \
    pyclesperanto-prototype \
    clij2-fft

# -----------------------
# Set working directory to repo
# -----------------------
WORKDIR /home/${LDAP_USERNAME}/deconvolution_repo/docker_deconvolution

# -----------------------
# Default entrypoint
# -----------------------
ENTRYPOINT ["python3", "3d_deconvolution.py"]
