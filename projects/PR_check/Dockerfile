FROM python:3.10-slim

RUN apt update && apt install git openssh-client nano -y
# RUN chown -R root:root /root/.ssh && chmod 700 /root/.ssh && chmod 600 /root/.ssh/config && chmod 600 /root/.ssh/id_rsa && chmod 644 /root/.ssh/id_rsa.pub  && clear
RUN pip install pylint pandas requests python-dotenv

WORKDIR /app


WORKDIR /report
WORKDIR /scripts

COPY PR_get.py .
COPY python_pylint.py .
COPY config.py .
COPY get_branches_opened.py .
COPY ssh_permision.sh .
COPY analize_complete.py .
COPY secrets.env .


