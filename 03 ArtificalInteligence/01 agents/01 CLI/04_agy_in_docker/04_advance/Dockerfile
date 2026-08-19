FROM wisrovi/agy-edc:v1.1.0

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=xterm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    zsh \
    figlet \
    && sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- -t aussiegeek \
    && wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true \
    && echo "alias ll='ls -l'" >> ~/.zshrc \
    && echo "figlet wagy" >> ~/.zshrc \
    && echo "figlet wagy" >> ~/.bashrc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/cache/apk/*

ENTRYPOINT []