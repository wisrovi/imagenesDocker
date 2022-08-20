# 1) indicamos la imagen base a usar
FROM faucet/python3

#Author and Maintainer
MAINTAINER wisrovi.rodriguez@gmail.com

# 2) creamos una carpeta para alojar los archivos del proyecto
WORKDIR /api_send_email

RUN echo face_recognition

# 3) instalamos sudo y actualizamos
RUN apt-get update -y

# 4) instalar dependencias del SO
#RUN apt-get -y install python3-pip

# 5) instalar dependencias de python
RUN pip3 install flask

# 6) copiamos la carpeta del codigo y todos sus recursos
COPY src .

# 7) le damos permisos a la carpeta donde se alojan los archivos del proyecto para que los archivos python puedan trabajar sin problemas
RUN sudo chmod -R +777 /face_recognition

# 8) le decimos que archivo ejecutar cuando se lance el container
#CMD [ "tail" ,"-f", "/etc/hosts" ]
CMD ["python3", "./api_send_email.py" ]