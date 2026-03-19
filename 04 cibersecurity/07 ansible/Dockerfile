# Usa una imagen base de Ubuntu
FROM ubuntu:22.04

# Actualiza la lista de paquetes e instala las dependencias necesarias
# Ansible requiere Python, que se instala aquí.
RUN apt-get update && \
    apt-get install -y python3 python3-pip openssh-client && \
    apt-get clean

# Instala Ansible usando pip
RUN pip3 install ansible

# Crea un usuario no root para buenas prácticas de seguridad
RUN useradd -m ansible && \
    mkdir -p /home/ansible/.ssh && \
    chown -R ansible:ansible /home/ansible

# Establece el directorio de trabajo
WORKDIR /home/ansible

# Define un comando por defecto para iniciar el contenedor
# Esto te dejará en un shell listo para usar Ansible
CMD ["/bin/bash"]
