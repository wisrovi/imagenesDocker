FROM ultralytics/ultralytics

WORKDIR /app

RUN apt-get install -y ffmpeg libx264-dev

# Copia los archivos de requisitos e instálalos primero para aprovechar el cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación de Gradio
COPY api.py .


# Comando para ejecutar la aplicación de Gradio
# Gradio por defecto usa el puerto 7860 y escucha en 0.0.0.0 cuando se lanza.
CMD ["python", "api.py"]
