# 1) indicamos la imagen base a usar
FROM python:3.8

#Author and Maintainer
LABEL MAINTAINER wisrovi.rodriguez@gmail.com

# 2) creamos una carpeta para alojar los archivos del proyecto
WORKDIR /api

RUN pip install uvicorn[standard]
RUN pip install fastapi
RUN pip install pymongo
RUN pip install Werkzeug

RUN pip freeze

# 6) copiamos la carpeta del codigo y todos sus recursos
COPY src .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80" ]
