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

# Install build dependencies for Python
RUN apk add build-base openssl-dev bzip2-dev libffi-dev readline-dev zlib-dev sqlite-dev ncurses-dev gcc make musl-dev

# Download and install a specific version of Python
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz 
RUN tar -xf Python-3.10.14.tgz
RUN cd Python-3.10.14 && ./configure --enable-optimizations --with-ensurepip=install --prefix=/usr/local --enable-loadable-sqlite-extensions && make -j$(nproc) && make altinstall

# Verify the Python and pip versions
RUN /usr/local/bin/python3.10 --version
RUN /usr/local/bin/pip3.10 --version
