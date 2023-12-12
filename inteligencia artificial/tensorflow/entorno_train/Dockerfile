FROM tensorflow/tensorflow:latest-gpu-jupyter
FROM --platform=linux/amd64 python:3.10.12-slim-bullseye

USER root

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DOCKER 1

# is necessary for opencv
RUN apt-get update -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y iputils-ping

# ZSH
RUN apt-get update && apt-get install -y zsh
RUN apt-get install -y wget
# Uses "robbyrussell" theme (original Oh My Zsh theme), with no plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t aussiegeek

# customizations
RUN apt-get install figlet -y
RUN echo "alias ll='ls -l'" >> ~/.zshrc



WORKDIR /tmp
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools pip
RUN pip install jupyterlab
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt 

WORKDIR /models
WORKDIR /datasets
WORKDIR /library
WORKDIR /content
WORKDIR /tf


RUN apt-get update && apt-get install -y \
    figlet \
    && rm -rf /var/lib/apt/lists/*

ENV GRANT_SUDO=yes
ENV NB_GID=100
ENV GEN_CERT=yes

EXPOSE 8888

RUN echo "figlet Custom Jupyter" >> ~/.zshrc
RUN echo "figlet Custom Jupyter" >> ~/.bashrc


CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]
