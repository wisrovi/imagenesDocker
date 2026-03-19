# Use the official Jenkins image with Alpine as the base
FROm jenkins/jenkins:alpine

# Switch to the root user to install packages
USER root

# Update the package list and upgrade existing packages
RUN apk update
RUN apk upgrade

# Install Docker, OpenRC, Git, and Docker Compose
RUN apk add --update docker openrc
RUN apk add git
RUN apk add docker-compose
RUN apk add make
RUN apk add nano

RUN apk add zsh
RUN which zsh
RUN apk add git
RUN apk add oh-my-zsh
RUN cp /usr/share/oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
RUN apk add shadow
RUN apk add zsh-syntax-highlighting
RUN apk add zsh-autosuggestions
RUN mkdir -p ~/.oh-my-zsh/custom/plugins
# Enlace para zsh-syntax-highlighting
RUN ln -s /usr/share/zsh/plugins/zsh-syntax-highlighting ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting
# Enlace para zsh-autosuggestions
RUN ln -s /usr/share/zsh/plugins/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions
RUN chsh -s /bin/zsh

RUN apk add nodejs npm
RUN npm install -g n
RUN n 22
RUN n use 22
RUN apk add bash curl build-base
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash






# Install build dependencies for Python
RUN apk add build-base openssl-dev bzip2-dev libffi-dev readline-dev zlib-dev sqlite-dev ncurses-dev gcc make musl-dev

# Download and install a specific version of Python
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz 
RUN tar -xf Python-3.10.14.tgz
RUN cd Python-3.10.14 && ./configure --enable-optimizations --with-ensurepip=install --prefix=/usr/local --enable-loadable-sqlite-extensions && make -j$(nproc) && make altinstall

# Verify the Python and pip versions
RUN /usr/local/bin/python3.10 --version
RUN /usr/local/bin/pip3.10 --version
