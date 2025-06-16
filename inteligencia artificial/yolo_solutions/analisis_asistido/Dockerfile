FROM ultralytics/ultralytics

WORKDIR /app

RUN apt-get install -y ffmpeg libx264-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ .


CMD ["python", "api.py"]