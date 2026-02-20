# -----------------------
# Base Image
# -----------------------
FROM --platform=linux/amd64 ubuntu:22.04

# -----------------------
# RCP CaaS Storage User Mapping
# -----------------------
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} \
 && useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# -----------------------
# Switch to user
# -----------------------
USER ${LDAP_USERNAME}
WORKDIR /home/${LDAP_USERNAME}

# -----------------------
# Clone your fixed GitHub repo
# -----------------------
RUN git clone https://github.com/feyzanar/docker.git deconvolution_repo

# -----------------------
# System dependencies
# -----------------------
USER root
RUN apt update && apt install -y \
    python3-pip \
    git \
    wget \
    ocl-icd-opencl-dev \
    libopencl1 \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Python dependencies
# -----------------------
USER ${LDAP_USERNAME}
RUN pip install --upgrade pip
RUN pip install \
    numpy \
    tifffile \
    tqdm \
    dask \
    pyclesperanto-prototype \
    clij2-fft

# -----------------------
# Set working directory to cloned repo
# -----------------------
WORKDIR /home/${LDAP_USERNAME}/deconvolution_repo/docker_deconvolution

# -----------------------
# Optional ENTRYPOINT (runs your fixed script by default)
# -----------------------
ENTRYPOINT ["python3", "3d_deconvolution.py"]
