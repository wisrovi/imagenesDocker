FROM ubuntu:22.04

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get -y install net-tools
RUN apt install iputils-ping -y


# ********************************************************
# ************************* ssh **************************
# ********************************************************
USER root

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:wZMqvW6aGt2omtedxz7s' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

EXPOSE 22


# ********************************************************
# ********************** docker in docker **********************
# ********************************************************

# Instala los paquetes necesarios para usar repositorios sobre HTTPS
RUN apt-get install -y ca-certificates curl gnupg

# Añade la clave GPG oficial de Docker
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Configura el repositorio estable de Docker
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Actualiza la lista de paquetes con el nuevo repositorio
RUN apt-get update

# Instala el cliente de Docker (la CLI)
RUN apt-get install -y docker-ce-cli


WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1




CMD ["/usr/sbin/sshd", "-D"]
