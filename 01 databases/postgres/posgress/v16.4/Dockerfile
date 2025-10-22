# Custom PostgreSQL Dockerfile based on postgres:16.4-bullseye
# Reference: https://gist.github.com/VictorGob/f40cb7ab08499ce7d854c1929e3ca98d

FROM postgres:16.4-bullseye

# Set terminal for better compatibility
ENV TERM=xterm
# Prevent Python from creating __pycache__ directories
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1

# Maintainer label
LABEL maintainer="wisrovi.rodriguez@gmail.com"

# Create extension initialization scripts to be executed on first startup
RUN echo 'CREATE EXTENSION IF NOT EXISTS plpython3u;' > /docker-entrypoint-initdb.d/py3.sql && \
    echo 'CREATE EXTENSION IF NOT EXISTS pg_stat_statements;' > /docker-entrypoint-initdb.d/pg_stat.sql && \
    echo 'CREATE EXTENSION IF NOT EXISTS postgis;' > /docker-entrypoint-initdb.d/postgis.sql && \
    echo 'CREATE EXTENSION IF NOT EXISTS pg_cron;' > /docker-entrypoint-initdb.d/pg_cron.sql && \
    echo 'CREATE EXTENSION IF NOT EXISTS pg_buffercache;' > /docker-entrypoint-initdb.d/pg_buffercache.sql

# Copy seeding script
COPY config/seed.sql /docker-entrypoint-initdb.d/seed.sql

# Install system packages and dependencies
# Use ARG for PostgreSQL major version
ARG PG_MAJOR=16
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    postgresql-plpython3-${PG_MAJOR} \
    postgresql-16-mysql-fdw \
    python3-pip \
    postgresql-16-postgis-3 \
    postgresql-16-postgis-3-scripts \
    postgresql-16-pg-cron \
    postgresql-16-pg-buffercache && \
    # Clean up to reduce image size
    apt-get clean && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Install Python packages with specific versions and no cache
RUN pip3 install --no-cache-dir requests==2.31.0 && \
    # Remove pip to reduce image size after installation
    apt-get purge -y python3-pip && \
    apt-get autoremove -y

# Expose PostgreSQL port
EXPOSE 5432

# Health check to verify PostgreSQL is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pg_isready -U perseus -d perseus