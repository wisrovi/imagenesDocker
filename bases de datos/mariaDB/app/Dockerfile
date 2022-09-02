FROM python:3
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
RUN apt-get update -y
RUN apt-get install nano -y
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
