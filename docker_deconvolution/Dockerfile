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
RUN if [ -n "$LDAP_USERNAME" ] && [ -n "$LDAP_UID" ] && [ -n "$LDAP_GROUPNAME" ] && [ -n "$LDAP_GID" ]; then \
        groupadd -f -g $LDAP_GID $LDAP_GROUPNAME && \
        id -u $LDAP_USERNAME || useradd -m -s /bin/bash -g $LDAP_GROUPNAME -u $LDAP_UID $LDAP_USERNAME; \
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
# Clone repo into /scratch (writable on RCP)
# -----------------------
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}
WORKDIR /home/${LDAP_USERNAME}

# -----------------------
# Python packages
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
# Switch to user
# -----------------------
USER ${LDAP_USERNAME}

# -----------------------
# Default entrypoint
# -----------------------
# ENTRYPOINT ["python3", "/scratch/deconvolution_repo/docker_deconvolution/3d_deconvolution.py"]
