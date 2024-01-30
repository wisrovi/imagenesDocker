# Utiliza una imagen base de Python
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DOCKER 1

RUN apt-get update && apt-get install -y \
    figlet \
    && rm -rf /var/lib/apt/lists/*

RUN echo "figlet Train eyesDcar" >> ~/.zshrc
RUN echo "figlet Train eyesDcar" >> ~/.bashrc

# Copia los archivos necesarios al contenedor
COPY ./ /app

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r /app/requirements.txt
