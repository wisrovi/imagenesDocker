Setup
=====

Prerequisites
-------------

- Docker
- Docker Compose

Installation
------------

1. Clone the repository
2. Copy .env.example to .env and configure
3. Run ``docker-compose up --build``

Services
--------

- PostgreSQL: Port 5433
- PgBouncer: Port 6432
- PostgREST: Port 3001
- pgAdmin: Port 5050
- Prometheus: Port 9090
- Grafana: Port 3000
- Redis: Port 6379