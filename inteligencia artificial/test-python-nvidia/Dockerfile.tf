FROM tensorflow/tensorflow:latest-gpu

WORKDIR /tmp

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r requirements.txt