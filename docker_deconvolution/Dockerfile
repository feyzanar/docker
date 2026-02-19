# -----------------------
# Base Image
# -----------------------
FROM ubuntu:22.04

# -----------------------
# RCP CaaS User Mapping
# -----------------------
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

# Create group and user matching the cluster LDAP
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} \
    && useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# -----------------------
# System dependencies
# -----------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    openjdk-11-jre \
    libopencl1 \
    ocl-icd-opencl-dev \
    clinfo \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Setup working directories for RCP scratch
# -----------------------
RUN mkdir -p /scratch/input /scratch/output \
    && chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /scratch

WORKDIR /scratch

# -----------------------
# Copy only the deconvolution script
# -----------------------
USER ${LDAP_USERNAME}
COPY ./3d_deconvolution.py /home/${LDAP_USERNAME}/3d_deconvolution.py
RUN chown ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}/3d_deconvolution.py

# -----------------------
# Python environment
# -----------------------
RUN pip install --upgrade pip
RUN pip install \
    numpy \
    scipy \
    tifffile \
    tqdm \
    dask \
    clij2-fft \
    pyclesperanto-prototype

# -----------------------
# Default entrypoint: run 3D deconvolution
# -----------------------
ENTRYPOINT ["python3", "/home/fnarslan/3d_deconvolution.py", "/scratch/input", "/scratch/output"]
