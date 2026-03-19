Configuración Completa
======================

Esta sección detalla la configuración completa de los **10 contenedores** de la plataforma, incluyendo variables de entorno, volúmenes, redes y parámetros de seguridad.

Variables de Entorno
--------------------

Archivo `.env` Principal
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # === CONFIGURACIÓN SSH ===
    SSH_PASSWORD=Ch@ng3M3N0w!2024
    SSH_PORT=50422
    SSH_USER=root
    SSH_ALLOW_ROOT_LOGIN=yes
    SSH_PASSWORD_AUTHENTICATION=yes

    # === CONFIGURACIÓN PORTAINER ===
    PORTAINER_ADMIN_PASSWORD=Adm1nP@ssw0rd!
    PORTAINER_ADMIN_USERNAME=admin

    # === CONFIGURACIÓN SSL/TLS ===
    SSL_CERT_PATH=/etc/ssl/certs
    SSL_KEY_PATH=/etc/ssl/private
    LETS_ENCRYPT_EMAIL=admin@example.com
    DOMAIN_NAME=localhost

    # === CONFIGURACIÓN DE SEGURIDAD ===
    FIREWALL_ENABLED=true
    RATE_LIMIT_REQUESTS_PER_MINUTE=100
    SESSION_TIMEOUT_MINUTES=30

    # === CONFIGURACIÓN DOCKER ===
    DOCKER_TLS_CERTDIR=
    DOCKER_DRIVER=overlay2
    DOCKER_REGISTRY_MIRROR=

    # === CONFIGURACIÓN DE MONITOREO ===
    PROMETHEUS_ENABLED=true
    GRAFANA_ADMIN_PASSWORD=Gr@f@n@Adm1n!
    METRICS_RETENTION_DAYS=30

    # === CONFIGURACIÓN DE BACKUPS ===
    BACKUP_ENABLED=true
    BACKUP_SCHEDULE=0 2 * * *
    BACKUP_RETENTION_DAYS=7

    # === CONFIGURACIÓN DE BASE DE DATOS ===
    POSTGRES_PASSWORD=P0stgr3sS3cur3!
    POSTGRES_DB=dind_db
    POSTGRES_USER=dind_user

    # === CONFIGURACIÓN DE NOTIFICACIONES ===
    SLACK_WEBHOOK_URL=https://hooks.slack.com/...
    EMAIL_SMTP_SERVER=smtp.gmail.com
    EMAIL_SMTP_PORT=587
    EMAIL_USERNAME=alerts@example.com
    EMAIL_PASSWORD=secure_password

    # === CONFIGURACIÓN DE APLICACIÓN ===
    NODE_ENV=production
    DEBUG=false
    LOG_LEVEL=info

Configuración por Contenedor
----------------------------

1. 🏠 Contenedor DinD (dind)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    dind:
      build:
        context: .
        dockerfile: ./docker/Dockerfile
      privileged: true
      ports:
        - "9003:9000"    # Portainer HTTP
        - "9443:9443"    # Portainer HTTPS
        - "50422:50422"  # SSH
        - "80:80"        # HTTP
        - "443:443"      # HTTPS
      environment:
        - DOCKER_TLS_CERTDIR=
        - SSH_PASSWORD=${SSH_PASSWORD}
        - PORTAINER_ADMIN_PASSWORD=${PORTAINER_ADMIN_PASSWORD}
      volumes:
        - ./volumes/dind-data:/var/lib/docker
        - ./volumes/ssl:/etc/ssl:ro
        - ./volumes/logs:/var/log
        - ./volumes/backups:/backups
      deploy:
        resources:
          limits:
            cpus: '2.0'
            memory: 4G
          reservations:
            cpus: '0.5'
            memory: 1G
      healthcheck:
        test: ["CMD-SHELL", "docker ps >/dev/null"]
        interval: 30s
        timeout: 10s
        retries: 3
        start_period: 60s

2. 📚 Documentación (docs-server)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    docs-server:
      build:
        context: ./docs
      ports:
        - "8082:80"   # HTTP
        - "8443:443"  # HTTPS
      volumes:
        - ./volumes/ssl:/etc/ssl:ro
      depends_on:
        dind:
          condition: service_healthy
      deploy:
        resources:
          limits:
            cpus: '0.5'
            memory: 512M
          reservations:
            cpus: '0.1'
            memory: 128M
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost/"]
        interval: 30s
        timeout: 10s
        retries: 3

3. 📊 Prometheus (prometheus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    prometheus:
      image: prom/prometheus:latest
      ports:
        - "9090:9090"
      volumes:
        - ./config/prometheus.yml:/etc/prometheus/prometheus.yml:ro
        - ./volumes/prometheus:/prometheus
      command:
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
        - '--web.enable-lifecycle'
      depends_on:
        - dind
      deploy:
        resources:
          limits:
            cpus: '0.5'
            memory: 1G
          reservations:
            cpus: '0.1'
            memory: 256M

4. 📈 Grafana (grafana)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    grafana:
      image: grafana/grafana:latest
      ports:
        - "3000:3000"
      environment:
        - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
        - GF_USERS_ALLOW_SIGN_UP=false
      volumes:
        - ./volumes/grafana:/var/lib/grafana
        - ./config/grafana/provisioning:/etc/grafana/provisioning:ro
      depends_on:
        - prometheus
      deploy:
        resources:
          limits:
            cpus: '0.5'
            memory: 512M
          reservations:
            cpus: '0.1'
            memory: 128M

5. 🔍 cAdvisor (cadvisor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    cadvisor:
      image: gcr.io/cadvisor/cadvisor:latest
      ports:
        - "8080:8080"
      volumes:
        - /:/rootfs:ro
        - /var/run:/var/run:ro
        - /sys:/sys:ro
        - /var/lib/docker/:/var/lib/docker:ro
      privileged: true
      deploy:
        resources:
          limits:
            cpus: '0.2'
            memory: 256M
          reservations:
            cpus: '0.05'
            memory: 64M

6. 📊 Node Exporter (node-exporter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    node-exporter:
      image: prom/node-exporter:latest
      ports:
        - "9100:9100"
      volumes:
        - /proc:/host/proc:ro
        - /sys:/host/sys:ro
        - /:/rootfs:ro
      command:
        - '--path.procfs=/host/proc'
        - '--path.rootfs=/rootfs'
        - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
      deploy:
        resources:
          limits:
            cpus: '0.1'
            memory: 128M
          reservations:
            cpus: '0.02'
            memory: 32M

7. 📝 Loki (loki)
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    loki:
      image: grafana/loki:latest
      ports:
        - "3100:3100"
      volumes:
        - ./volumes/loki:/loki
      command: -config.file=/etc/loki/local-config.yaml
      deploy:
        resources:
          limits:
            cpus: '0.3'
            memory: 512M
          reservations:
            cpus: '0.05'
            memory: 128M

8. 📤 Promtail (promtail)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    promtail:
      image: grafana/promtail:latest
      volumes:
        - /var/log:/var/log
        - ./config/promtail.yml:/etc/promtail/config.yml
      command: -config.file=/etc/promtail/config.yml
      deploy:
        resources:
          limits:
            cpus: '0.1'
            memory: 128M
          reservations:
            cpus: '0.02'
            memory: 32M

9. 🚨 Alertmanager (alertmanager)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    alertmanager:
      image: prom/alertmanager:latest
      ports:
        - "9093:9093"
      volumes:
        - ./config/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      command:
        - '--config.file=/etc/alertmanager/alertmanager.yml'
        - '--storage.path=/alertmanager'
      deploy:
        resources:
          limits:
            cpus: '0.2'
            memory: 256M
          reservations:
            cpus: '0.05'
            memory: 64M

10. 🔧 API REST (dind-api)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    dind-api:
      build:
        context: .
        dockerfile: scripts/Dockerfile.api
      ports:
        - "5000:5000"
      environment:
        - API_PORT=5000
        - DEBUG=false
      depends_on:
        - dind
      volumes:
        - /var/run/docker.sock:/var/run/docker.sock:ro
      deploy:
        resources:
          limits:
            cpus: '0.5'
            memory: 256M
          reservations:
            cpus: '0.1'
            memory: 64M
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
        interval: 30s
        timeout: 10s
        retries: 3

Arquitectura de Volúmenes
--------------------------

Estructura Completa de Volúmenes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    volumes/
    ├── dind-data/          # Docker persistente (imágenes, contenedores)
    │   ├── builder/        # Build cache
    │   ├── containers/     # Container metadata
    │   ├── image/         # Image layers
    │   └── volumes/       # Named volumes
    ├── portainer_data/     # Portainer configuration
    ├── grafana/           # Grafana dashboards y configuración
    ├── prometheus/        # Time-series metrics
    ├── loki/             # Centralized logs
    ├── ssl/              # SSL certificates
    │   ├── certs/        # Public certificates
    │   └── private/      # Private keys
    ├── logs/             # Application logs
    │   ├── supervisor/   # Process manager logs
    │   ├── nginx/       # Web server logs
    │   └── app/         # Application logs
    └── backups/          # Automated backups
        ├── daily/        # Daily backups
        ├── weekly/       # Weekly backups
        └── monthly/      # Monthly backups

Configuración de Red
--------------------

Redes Docker
~~~~~~~~~~~~

.. code-block:: yaml

    networks:
      dind-network:
        driver: bridge
        driver_opts:
          com.docker.network.bridge.name: dind-net
        ipam:
          config:
            - subnet: 172.20.0.0/16
              gateway: 172.20.0.1
        labels:
          - "com.docker.compose.network=dind-network"

Configuración de Seguridad
--------------------------

Firewall (UFW)
~~~~~~~~~~~~~~

.. code-block:: bash

    # Reglas activadas automáticamente
    ufw allow 50422/tcp   # SSH
    ufw allow 9003/tcp    # Portainer HTTP
    ufw allow 9443/tcp    # Portainer HTTPS
    ufw allow 80/tcp      # HTTP
    ufw allow 443/tcp     # HTTPS
    ufw allow 8082/tcp    # Documentation
    ufw allow 9090/tcp    # Prometheus
    ufw allow 3000/tcp    # Grafana
    ufw allow 8080/tcp    # cAdvisor
    ufw allow 9100/tcp    # Node Exporter

Fail2Ban
~~~~~~~~~

.. code-block:: ini

    [DEFAULT]
    bantime = 3600
    findtime = 600
    maxretry = 3

    [sshd]
    enabled = true
    port = 50422
    logpath = /var/log/auth.log

Headers de Seguridad (Nginx)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: nginx

    # Security headers aplicados automáticamente
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; ..." always;
    add_header Strict-Transport-Security "max-age=31536000; ..." always;

Configuración de Monitoreo
---------------------------

Prometheus Scrape Config
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    scrape_configs:
      - job_name: 'dind'
        static_configs:
          - targets: ['dind:9100']
        scrape_interval: 5s

      - job_name: 'node-exporter'
        static_configs:
          - targets: ['node-exporter:9100']

      - job_name: 'cadvisor'
        static_configs:
          - targets: ['cadvisor:8080']

      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']

Reglas de Alerta
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    groups:
    - name: docker
      rules:
      - alert: ContainerDown
        expr: up == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Container {{ $labels.instance }} is down"

    - name: system
      rules:
      - alert: HighCpuUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning

Configuración de Logs
----------------------

Logrotate
~~~~~~~~~

.. code-block:: text

    /var/log/*.log {
        daily
        missingok
        rotate 52
        compress
        delaycompress
        notifempty
        create 644 root root
        postrotate
            supervisorctl reload
        endscript
    }

Loki Config
~~~~~~~~~~~

.. code-block:: yaml

    auth_enabled: false
    server:
      http_listen_port: 3100
    ingester:
      lifecycler:
        ring:
          kvstore:
            store: inmemory
          replication_factor: 1

Promtail Config
~~~~~~~~~~~~~~~

.. code-block:: yaml

    server:
      http_listen_port: 9080
    positions:
      filename: /tmp/positions.yaml
    clients:
      - url: http://loki:3100/loki/api/v1/push
    scrape_configs:
      - job_name: system
        static_configs:
          - targets: ['localhost']
            labels:
              job: varlogs
              __path__: /var/log/*log

Configuración de Backups
-------------------------

Cron Jobs
~~~~~~~~~

.. code-block:: bash

    # Backup diario a las 2 AM
    0 2 * * * /usr/local/bin/backup.sh

    # Rotación de logs cada hora
    0 * * * * /usr/sbin/logrotate /etc/logrotate.d/app

    # Limpieza semanal de imágenes no utilizadas
    0 3 * * 0 docker system prune -f >/dev/null 2>&1

Backup Script Config
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    BACKUP_DIR=${BACKUP_DIR:-/backups}
    BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-7}
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_NAME="dind_backup_$TIMESTAMP"

Configuración de SSL
---------------------

Certbot (Let's Encrypt)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Configuración automática para dominio personalizado
    DOMAIN_NAME=${DOMAIN_NAME:-localhost}
    LETS_ENCRYPT_EMAIL=${EMAIL:-admin@example.com}

    # Generación de certificados self-signed para desarrollo
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN_NAME"

Nginx SSL Config
~~~~~~~~~~~~~~~~

.. code-block:: nginx

    server {
        listen 443 ssl http2;
        server_name localhost;

        ssl_certificate /etc/ssl/certs/server.crt;
        ssl_certificate_key /etc/ssl/private/server.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:...;

        # Configuración SSL adicional...
    }