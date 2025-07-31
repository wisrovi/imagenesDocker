FROm jenkins/jenkins:alpine

USER root

RUN apk update
RUN apk upgrade
RUN apk add --update docker openrc
RUN apk add git
RUN apk add docker-compose
RUN apk add build-base openssl-dev bzip2-dev libffi-dev readline-dev zlib-dev sqlite-dev ncurses-dev gcc make musl-dev
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz 
RUN tar -xf Python-3.10.14.tgz
RUN cd Python-3.10.14 && ./configure --enable-optimizations --with-ensurepip=install --prefix=/usr/local --enable-loadable-sqlite-extensions && make -j$(nproc) && make altinstall
RUN /usr/local/bin/python3.10 --version
RUN /usr/local/bin/pip3.10 --version




