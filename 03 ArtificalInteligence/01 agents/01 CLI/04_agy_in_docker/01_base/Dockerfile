FROM ubuntu:24.04

# Install dependencies: ca-certificates, curl, git, docker client
RUN apt update && apt install -y ca-certificates curl git docker.io && \
    rm -rf /var/lib/apt/lists/*

# Install agy via official installer (binary ends up in /root/.local/bin)
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash && \
    ln -s /root/.local/bin/agy /usr/local/bin/agy

# Ensure the install location is in PATH
ENV PATH="${PATH}:/root/.local/bin"

WORKDIR /workspace

# Default command launches a bash shell
CMD ["bash"]
