#https://gist.github.com/VictorGob/f40cb7ab08499ce7d854c1929e3ca98d.

FROM postgres:13-bullseye

ENV TERM=xterm

LABEL maintainer="wisrovi.rodriguez@gmail.com"

# Create extensions when the db starts.
RUN echo 'CREATE EXTENSION IF NOT EXISTS plpython3u;' > /docker-entrypoint-initdb.d/py3.sql && \
    echo 'CREATE EXTENSION IF NOT EXISTS pg_stat_statements;' > /docker-entrypoint-initdb.d/pg_stat.sql && \
    echo 'CREATE EXTENSION IF NOT EXISTS postgis;' > /docker-entrypoint-initdb.d/postgis.sql

# System update and package installation
ARG PG_MAJOR=13
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    postgresql-plpython3-${PG_MAJOR} \
    postgresql-13-mysql-fdw \
    python3-pip \
    postgis \
    postgresql-13-postgis-3 \
    postgresql-13-postgis-3-scripts && \
    apt-get clean && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Install python packages
RUN pip3 install requests

EXPOSE 5432