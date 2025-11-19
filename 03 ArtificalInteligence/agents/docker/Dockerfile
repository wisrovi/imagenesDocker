FROM node:22

ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Madrid LANG=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TERM=xterm HOME=/root

# ============= SYSTEM SETUP =============
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y ca-certificates curl gnupg software-properties-common && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# ============= DOCKER REPO =============
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# ============= NVIDIA/CUDA SETUP =============
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update

# ============= PYTHON & SYSTEM PACKAGES =============
RUN apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    git \
    unzip \
    curl \
    wget \
    zsh \
    btop \
    htop \
    nano \
    tree \
    jq \
    net-tools \
    iputils-ping \
    docker-ce-cli \
    docker-compose-plugin \
    cuda-toolkit-12-3 \
    cmake \
    libncurses5-dev \
    libncursesw5-dev \
    libudev-dev \
    libdrm-dev \
    libicu-dev \
    libicu72 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
    bandit \
    safety \
    ruff \
    black \
    watchdog \
    ipython \
    httpie  \
    visidata \
    scapy \
    glances \
    pre-commit \
    py-spy

# ============= NVTOP INSTALLATION =============
RUN git clone https://github.com/Syllo/nvtop.git /tmp/nvtop && \
    cd /tmp/nvtop && mkdir build && cd build && \
    cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=OFF -DINTEL_SUPPORT=OFF -DV3D_SUPPORT=OFF && \
    make -j$(nproc) && make install && \
    cd / && rm -rf /tmp/nvtop && \
    apt-get remove -y cmake git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /scripts
COPY scripts /scripts


WORKDIR /app

# ============= RUST INSTALLATION =============
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . "$HOME/.cargo/env"

# ============= NODE.JS TOOLS =============
RUN curl -fsSL https://opencode.ai/install | bash && \
    npm install -g @github/copilot nodemon && \
    npm install -g @google/gemini-cli && \
    npm install -g rimraf

# ============= DOCKER TOOLS =============
RUN curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash && \
    (mv /root/bin/lazydocker /usr/local/bin/lazydocker 2>/dev/null || find /root -name "lazydocker" -exec mv {} /usr/local/bin/ \; 2>/dev/null || true)

# ============= ZSH CONFIGURATION =============
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- -t aussiegeek && \
    wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true && \
    git config --global user.name "wisrovi" && git config --global user.email "wisrovi.rodriguez@gmail.com" && \
    echo "alias wisrovi='echo \"https://es.linkedin.com/in/wisrovi-rodriguez\"'\nalias ll='ls -la'\nalias la='ls -A'\nalias l='ls -CF'\nalias ..='cd ..'\nalias ...='cd ../..'" >> ~/.zshrc && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-completions && \
    sed -i '/^plugins=/s/)/ zsh-autosuggestions zsh-syntax-highlighting zsh-completions z docker docker-compose)/' ~/.zshrc && \
    sed -i 's/^ZSH_THEME="[^"]*"$/ZSH_THEME="aussiegeek"/' /root/.zshrc && \
    echo /bin/zsh >> /etc/shells

# ============= MICROSOFT EDIT =============
RUN apt-get update && apt-get install -y zstd && \
    wget https://github.com/microsoft/edit/releases/download/v1.2.1/edit-1.2.0-x86_64-linux-gnu.tar.zst && \
    tar -I zstd -xf edit-1.2.0-x86_64-linux-gnu.tar.zst && \
    mv edit /usr/local/bin/edit && \
    rm edit-1.2.0-x86_64-linux-gnu.tar.zst && \
    rm -rf /var/lib/apt/lists/*

# ============= FINAL CONFIGURATION =============
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && npm cache clean --force && \
    echo "wAgents" > /etc/hostname && node -v && npm -v

ENV PATH="/root/.cargo/bin:/root/.opencode/bin:/usr/local/bin:$PATH"

CMD ["tail", "-f", "/dev/null"]