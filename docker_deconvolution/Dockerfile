# Base Ubuntu 22.04
FROM ubuntu:22.04

# -----------------------
# RCP CaaS User Mapping
# -----------------------
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

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
# Copy your code
# -----------------------
RUN mkdir -p /home/${LDAP_USERNAME}/public
COPY ./ /home/${LDAP_USERNAME}/public

# Set ownership
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# -----------------------
# Switch to user
# -----------------------
WORKDIR /home/${LDAP_USERNAME}
USER ${LDAP_USERNAME}

# -----------------------
# Clone your GitHub repo (optional, if you want latest)
# -----------------------
RUN git clone https://github.com/feyzanar/public.git

# -----------------------
# Install Python packages
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
# Entrypoint
# -----------------------
ENTRYPOINT ["python3", "public/3d_deconvolution.py"]
