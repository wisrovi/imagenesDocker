FROM node:22

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TERM=xterm
ENV HOME=/root

# Update and upgrade system packages
RUN apt-get update && apt-get upgrade -y

# Add Docker's official GPG key and repository
RUN apt-get install -y ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update

# Add NVIDIA CUDA repository
RUN apt-get install -y software-properties-common \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update

# Install required packages
RUN apt-get install -y \
    curl \
    wget \
    build-essential \
    git \
    unzip \
    python3 \
    python3-pip \
    zsh \
    btop \
    htop \
    nano \
    tree \
    jq \
    net-tools \
    iputils-ping \
    docker-ce-cli \
    cuda-toolkit-12-3 \
    && rm -rf /var/lib/apt/lists/*

# Install nvtop from source to avoid dependency issues
RUN apt-get update && apt-get install -y cmake libncurses5-dev libncursesw5-dev git \
    && git clone https://github.com/Syllo/nvtop.git /tmp/nvtop \
    && cd /tmp/nvtop \
    && mkdir build && cd build \
    && cmake .. -DNVIDIA_SUPPORT=ON \
    && make -j$(nproc) \
    && make install \
    && cd / && rm -rf /tmp/nvtop \
    && apt-get remove -y cmake git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Configure timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set working directory
WORKDIR /app

# Install OpenCode
RUN curl -fsSL https://opencode.ai/install | bash

# Install GitHub Copilot CLI
RUN npm install -g @github/copilot

# Skip Google Gemini CLI due to installation timeout
# RUN npm install -g @google/gemini-cli

# Install nodemon for development
RUN npm install -g nodemon

# Install lazydocker
RUN curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash \
    && mv /root/bin/lazydocker /usr/local/bin/lazydocker 2>/dev/null || true \
    && find /root -name "lazydocker" -exec mv {} /usr/local/bin/ \; 2>/dev/null || true

# Skip Microsoft Edit installation (release not found)
# RUN wget https://github.com/microsoft/edit/releases/download/v1.2.1/edit-1.2.1-x86_64-unknown-linux-gnu.tar.gz \
#     && tar -xzf edit-1.2.1-x86_64-unknown-linux-gnu.tar.gz \
#     && mv edit /usr/local/bin/edit \
#     && rm edit-1.2.1-x86_64-unknown-linux-gnu.tar.gz

# ------------------- CONFIGURACIÓN DEL ZSH ----------------------
# Uses "aussiegeek" theme (original Oh My Zsh theme), with no plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t aussiegeek

# oh-my-zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

# Configure git with user data
RUN git config --global user.name "wisrovi" \
    && git config --global user.email "wisrovi.rodriguez@gmail.com"

# Add custom aliases
RUN echo "alias wisrovi='echo \"https://es.linkedin.com/in/wisrovi-rodriguez\"'" >> ~/.zshrc \
    && echo "alias ll='ls -la'" >> ~/.zshrc \
    && echo "alias la='ls -A'" >> ~/.zshrc \
    && echo "alias l='ls -CF'" >> ~/.zshrc \
    && echo "alias ..='cd ..'" >> ~/.zshrc \
    && echo "alias ...='cd ../..'" >> ~/.zshrc

# Plugin installation
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
RUN git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-completions

# Plugin activation
RUN sed -i '/^plugins=/s/)/ zsh-autosuggestions)/' ~/.zshrc
RUN sed -i '/^plugins=/s/)/ zsh-syntax-highlighting)/' ~/.zshrc
RUN sed -i '/^plugins=/s/)/ zsh-completions)/' ~/.zshrc
RUN sed -i '/^plugins=/s/)/ z)/' ~/.zshrc
RUN sed -i '/^plugins=/s/)/ docker docker-compose)/' ~/.zshrc

# Set theme
RUN sed -i 's/^ZSH_THEME="[^"]*"$/ZSH_THEME="aussiegeek"/' /root/.zshrc

# Set zsh as default shell
RUN echo /bin/zsh >> /etc/shells

# Verify installation
RUN node -v && npm -v

# Add opencode and edit to PATH
ENV PATH="/root/.opencode/bin:/usr/local/bin:$PATH"

# Set hostname
RUN echo "wAgents" > /etc/hostname

# Clean up cache and temporary files
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    npm cache clean --force

CMD ["zsh"]