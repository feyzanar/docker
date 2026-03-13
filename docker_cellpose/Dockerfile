FROM --platform=linux/amd64 nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

USER root

# Create group and user safely (guards against missing build args)
RUN if [ -n "$LDAP_USERNAME" ] && [ -n "$LDAP_UID" ] && [ -n "$LDAP_GROUPNAME" ] && [ -n "$LDAP_GID" ]; then \
        groupadd -f -g $LDAP_GID $LDAP_GROUPNAME && \
        id -u $LDAP_USERNAME || useradd -m -s /bin/bash -g $LDAP_GROUPNAME -u $LDAP_UID $LDAP_USERNAME; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        git \
        wget \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Download the cellpose segmentation script from GitHub
RUN mkdir -p /opt/scripts
RUN wget -O /opt/scripts/3d_cellpose.py \
    https://raw.githubusercontent.com/feyzanar/docker/main/docker_cellpose/3d_cellpose.py
RUN chmod +x /opt/scripts/3d_cellpose.py

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install numpy tifffile
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install cellpose==3.1.0

# Fix ownership
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /opt/scripts

USER ${LDAP_USERNAME}
WORKDIR /home/${LDAP_USERNAME}
