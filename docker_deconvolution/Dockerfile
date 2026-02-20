# -----------------------
# Base Image
# -----------------------
FROM --platform=linux/amd64 ubuntu:22.04

# -----------------------
# Optional RCP CaaS User Mapping
# -----------------------
# These build args are required on the cluster
# Can be left empty for local Mac builds
ARG LDAP_USERNAME=fnarslan
ARG LDAP_UID=503
ARG LDAP_GROUPNAME=staff
ARG LDAP_GID=20

USER root

# Only create user/group if build args are not empty
RUN if [ -n "$LDAP_USERNAME" ] && [ -n "$LDAP_UID" ] && [ -n "$LDAP_GROUPNAME" ] && [ -n "$LDAP_GID" ]; then \
        groupadd -f -g $LDAP_GID $LDAP_GROUPNAME && \
        id -u $LDAP_USERNAME || useradd -m -s /bin/bash -g $LDAP_GROUPNAME -u $LDAP_UID $LDAP_USERNAME ; \
    fi

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
# Clone GitHub repo
# -----------------------
USER $LDAP_USERNAME
WORKDIR /home/$LDAP_USERNAME
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
# Set working directory to repo folder
# -----------------------
WORKDIR /home/$LDAP_USERNAME/deconvolution_repo/docker_deconvolution

# -----------------------
# Default entrypoint
# -----------------------
ENTRYPOINT ["python3", "3d_deconvolution.py"]
