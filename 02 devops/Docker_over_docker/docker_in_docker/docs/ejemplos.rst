Ejemplos Prácticos
==================

.. _ejemplos:

Esta sección proporciona ejemplos completos de uso de los **10 contenedores especializados** de la plataforma, desde desarrollo básico hasta operaciones avanzadas.

Ejemplo 1: Aplicación Web con Monitoreo Completo
--------------------------------------------------

**Objetivo:** Desplegar una aplicación web completa con base de datos, monitoreo y logging.

**Estructura del Proyecto:**

.. code-block:: text

   volumes/files/my-web-app/
   ├── docker-compose.yml
   ├── app/
   │   ├── server.js
   │   ├── package.json
   │   └── Dockerfile
   ├── db/
   │   └── init.sql
   └── monitoring/
       └── prometheus.yml

**1. Aplicación Node.js (app/server.js):**

.. code-block:: javascript

   const express = require('express');
   const { Pool } = require('pg');
   const app = express();

   // Conexión a base de datos
   const pool = new Pool({
     connectionString: process.env.DATABASE_URL,
   });

   app.get('/', async (req, res) => {
     try {
       const result = await pool.query('SELECT NOW()');
       res.json({
         message: 'Hello from DinD!',
         timestamp: result.rows[0].now,
         container: process.env.HOSTNAME
       });
     } catch (err) {
       res.status(500).json({ error: err.message });
     }
   });

   app.get('/health', (req, res) => {
     res.json({ status: 'healthy', uptime: process.uptime() });
   });

   const port = process.env.PORT || 3000;
   app.listen(port, () => {
     console.log(`Server running on port ${port}`);
   });

**2. Dockerfile de la aplicación:**

.. code-block:: dockerfile

   FROM node:18-alpine
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci --only=production
   COPY . .
   EXPOSE 3000
   CMD ["node", "server.js"]

**3. Configuración de base de datos (db/init.sql):**

.. code-block:: sql

   CREATE TABLE visits (
     id SERIAL PRIMARY KEY,
     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
     ip_address INET,
     user_agent TEXT
   );

   CREATE INDEX idx_visits_timestamp ON visits(timestamp);

**4. Docker Compose completo:**

.. code-block:: yaml

   version: '3.8'
   services:
     web:
       build: ./app
       ports:
         - "3000:3000"
       depends_on:
         - db
       environment:
         - DATABASE_URL=postgres://myuser:mypass@db:5432/myapp
         - NODE_ENV=production
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
         interval: 30s
         timeout: 10s
         retries: 3

     db:
       image: postgres:15-alpine
       environment:
         - POSTGRES_DB=myapp
         - POSTGRES_USER=myuser
         - POSTGRES_PASSWORD=mypass
       volumes:
         - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
         - db_data:/var/lib/postgresql/data
       healthcheck:
         test: ["CMD-SHELL", "pg_isready -U myuser -d myapp"]
         interval: 30s
         timeout: 10s
         retries: 3

   volumes:
     db_data:

**Despliegue y Monitoreo:**

.. code-block:: bash

   # Desplegar la aplicación
   docker-compose exec dind sh
   cd /app/my-web-app
   docker-compose up -d

   # Verificar funcionamiento
   curl http://localhost:3000/

   # Monitorear en Grafana
   open http://localhost:3000  # Dashboard de aplicación

   # Ver logs en Loki
   open http://localhost:3100  # Buscar logs de la app

Ejemplo 2: CI/CD Pipeline Interno
-----------------------------------

**Objetivo:** Crear un pipeline de CI/CD que se ejecuta dentro del entorno DinD.

**Estructura:**

.. code-block:: text

   volumes/files/ci-pipeline/
   ├── docker-compose.yml
   ├── jenkins/
   │   ├── Dockerfile
   │   └── jobs/
   │       └── build-job.xml
   └── app/
       ├── src/
       └── Dockerfile

**Jenkins con Docker (jenkins/Dockerfile):**

.. code-block:: dockerfile

   FROM jenkins/jenkins:lts-alpine

   USER root
   RUN apk add --no-cache docker-cli
   RUN echo "jenkins ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

   USER jenkins
   COPY jobs/ /var/jenkins_home/jobs/

**Pipeline de Jenkins:**

.. code-block:: groovy

   pipeline {
       agent any

       stages {
           stage('Checkout') {
               steps {
                   git 'https://github.com/myorg/my-app.git'
               }
           }

           stage('Build') {
               steps {
                   sh 'docker build -t my-app:${BUILD_NUMBER} .'
               }
           }

           stage('Test') {
               steps {
                   sh 'docker run --rm my-app:${BUILD_NUMBER} npm test'
               }
           }

           stage('Deploy') {
               steps {
                   sh 'docker tag my-app:${BUILD_NUMBER} my-app:latest'
                   sh 'docker-compose up -d web'
               }
           }
       }

       post {
           always {
               sh 'docker-compose logs web'
           }
           success {
               echo 'Pipeline succeeded!'
           }
           failure {
               echo 'Pipeline failed!'
           }
       }
   }

Ejemplo 3: Monitoreo Avanzado con Alertas
-------------------------------------------

**Objetivo:** Configurar monitoreo completo con alertas personalizadas.

**Configuración de Prometheus (monitoring/prometheus.yml):**

.. code-block:: yaml

   global:
     scrape_interval: 15s
     evaluation_interval: 15s

   rule_files:
     - "alert_rules.yml"

   alerting:
     alertmanagers:
       - static_configs:
           - targets:
             - alertmanager:9093

   scrape_configs:
     - job_name: 'my-app'
       static_configs:
         - targets: ['web:3000']
       metrics_path: '/metrics'

Docker Playground Examples
==========================

Try these examples interactively using Docker Playground:

**Example 1: Basic DinD Setup**

`🔗 Try it on Docker Playground <https://labs.play-with-docker.com/?stack=https://raw.githubusercontent.com/your-repo/docker-dind-portainer/main/docker-compose.yaml>`_

This playground provides:
- Pre-configured DinD environment
- Portainer web interface
- SSH access ready
- Sample containers to manage

**Example 2: Monitoring Stack**

`🔗 Try monitoring on Docker Playground <https://labs.play-with-docker.com/?stack=https://raw.githubusercontent.com/your-repo/docker-dind-portainer/main/docker-compose.yaml>`_

Interactive monitoring setup with:
- Prometheus metrics collection
- Grafana dashboards
- Alertmanager configuration
- Real-time container monitoring

**Example 3: API Development**

`🔗 Try API development on Docker Playground <https://labs.play-with-docker.com/?stack=https://raw.githubusercontent.com/your-repo/docker-dind-portainer/main/docker-compose.yaml>`_

REST API playground featuring:
- Container management endpoints
- Health checks
- Log retrieval
- Backup operations

**Getting Started with Playground**

1. Click any "Try it" link above
2. Wait for the environment to load
3. Access services via the provided URLs
4. Experiment with container operations
5. View logs and metrics in real-time

.. note::
   Docker Playground sessions are temporary. Changes are lost when the session ends.

     - job_name: 'database'
       static_configs:
         - targets: ['db:5432']

**Reglas de alerta (monitoring/alert_rules.yml):**

.. code-block:: yaml

   groups:
   - name: myapp
     rules:
     - alert: HighResponseTime
       expr: http_request_duration_seconds{quantile="0.5"} > 0.5
       for: 5m
       labels:
         severity: warning
       annotations:
         summary: "High response time on {{ $labels.instance }}"
         description: "Response time is {{ $value }}s for 5 minutes."

     - alert: DatabaseDown
       expr: up{job="database"} == 0
       for: 1m
       labels:
         severity: critical
       annotations:
         summary: "Database is down"
         description: "Database has been down for more than 1 minute."

**Configuración de Alertmanager:**

.. code-block:: yaml

   global:
     smtp_smarthost: 'smtp.gmail.com:587'
     smtp_from: 'alerts@example.com'

   route:
     group_by: ['alertname']
     group_wait: 10s
     group_interval: 10s
     repeat_interval: 1h
     receiver: 'email-and-slack'
     routes:
     - match:
         severity: critical
       receiver: 'critical-email'

   receivers:
   - name: 'email-and-slack'
     email_configs:
     - to: 'team@example.com'
     slack_configs:
     - api_url: 'https://hooks.slack.com/services/...'

   - name: 'critical-email'
     email_configs:
     - to: 'admin@example.com'

Ejemplo 4: Logging Centralizado con Loki
------------------------------------------

**Objetivo:** Configurar logging avanzado con consultas y visualización.

**Configuración de aplicación para logging estructurado:**

.. code-block:: javascript

   const winston = require('winston');

   const logger = winston.createLogger({
     level: 'info',
     format: winston.format.combine(
       winston.format.timestamp(),
       winston.format.json()
     ),
     defaultMeta: { service: 'my-app' },
     transports: [
       new winston.transports.Console(),
       new winston.transports.File({ filename: '/var/log/app.log' })
     ],
   });

   // Uso del logger
   app.get('/api/users', (req, res) => {
     logger.info('Fetching users', {
       userId: req.user?.id,
       ip: req.ip,
       userAgent: req.get('User-Agent')
     });

     // ... lógica de la API
   });

**Consultas en Loki:**

.. code-block:: logql

   # Todos los logs de error
   {service="my-app", level="error"}

   # Logs de un usuario específico
   {service="my-app"} |= "userId=123"

   # Errores en las últimas 24 horas
   {service="my-app", level="error"} [24h]

   # Tasa de errores por minuto
   rate({service="my-app", level="error"} [5m])

Ejemplo 5: Backup y Disaster Recovery
---------------------------------------

**Objetivo:** Implementar estrategia completa de backup y recuperación.

**Script de backup personalizado:**

.. code-block:: bash

   #!/bin/bash
   # Custom backup script

   BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$BACKUP_DIR"

   echo "Starting custom backup..."

   # Backup de base de datos
   docker exec db pg_dump -U myuser myapp > "$BACKUP_DIR/database.sql"

   # Backup de archivos de aplicación
   docker run --rm -v myapp_data:/source -v "$BACKUP_DIR:/backup" alpine \
     tar czf /backup/app-data.tar.gz -C /source .

   # Backup de configuración
   cp /app/docker-compose.yml "$BACKUP_DIR/"
   cp /app/.env "$BACKUP_DIR/"

   # Crear manifest
   cat > "$BACKUP_DIR/manifest.json" << EOF
   {
     "timestamp": "$(date -Iseconds)",
     "type": "full-backup",
     "services": ["web", "db"],
     "volumes": ["myapp_data"],
     "size": "$(du -sh "$BACKUP_DIR" | cut -f1)"
   }
   EOF

   echo "Backup completed: $BACKUP_DIR"

**Script de restauración:**

.. code-block:: bash

   #!/bin/bash
   # Restore script

   BACKUP_DIR="$1"

   if [ -z "$BACKUP_DIR" ]; then
     echo "Usage: $0 <backup-directory>"
     exit 1
   fi

   echo "Restoring from $BACKUP_DIR..."

   # Detener servicios
   docker-compose down

   # Restaurar base de datos
   docker-compose up -d db
   sleep 10
   docker exec -i db psql -U myuser myapp < "$BACKUP_DIR/database.sql"

   # Restaurar archivos
   docker run --rm -v myapp_data:/dest -v "$BACKUP_DIR:/backup" alpine \
     tar xzf /backup/app-data.tar.gz -C /dest

   # Reiniciar servicios
   docker-compose up -d

   echo "Restore completed!"

Ejemplo 6: API Automation con Python
-------------------------------------

**Objetivo:** Automatizar operaciones usando la API REST.

**Cliente Python para la API:**

.. code-block:: python

   import requests
   import time
   from typing import Dict, List

   class DindAPI:
       def __init__(self, base_url: str = "http://localhost:5000"):
           self.base_url = base_url
           self.session = requests.Session()

       def health_check(self) -> Dict:
           """Verificar estado de la plataforma"""
           response = self.session.get(f"{self.base_url}/api/health")
           return response.json()

       def list_containers(self) -> List[Dict]:
           """Listar todos los contenedores"""
           response = self.session.get(f"{self.base_url}/api/containers")
           return response.json().get('containers', [])

       def start_container(self, container_id: str) -> Dict:
           """Iniciar un contenedor"""
           response = self.session.post(
               f"{self.base_url}/api/containers/{container_id}/start"
           )
           return response.json()

       def get_logs(self, container_id: str, lines: int = 100) -> Dict:
           """Obtener logs de un contenedor"""
           response = self.session.get(
               f"{self.base_url}/api/containers/{container_id}/logs",
               params={'lines': lines}
           )
           return response.json()

       def create_backup(self) -> Dict:
           """Crear backup de volúmenes"""
           response = self.session.post(f"{self.base_url}/api/backup")
           return response.json()

       def wait_for_healthy(self, timeout: int = 300) -> bool:
           """Esperar a que la plataforma esté saludable"""
           start_time = time.time()
           while time.time() - start_time < timeout:
               try:
                   health = self.health_check()
                   if health.get('status') == 'healthy':
                       return True
               except:
                   pass
               time.sleep(5)
           return False

   # Uso del cliente
   if __name__ == "__main__":
       api = DindAPI()

       # Verificar estado
       print("Health check:", api.health_check())

       # Listar contenedores
       containers = api.list_containers()
       print(f"Found {len(containers)} containers")

       # Backup automático
       backup_result = api.create_backup()
       print("Backup result:", backup_result)

Ejemplo 7: Testing con Chaos Engineering
------------------------------------------

**Objetivo:** Implementar pruebas de caos para validar resiliencia.

**Experimento de caos personalizado:**

.. code-block:: json

   {
     "version": "1.0.0",
     "title": "Web Application Chaos Experiment",
     "description": "Test application resilience under failure conditions",
     "steady-state-hypothesis": {
       "title": "Application is healthy",
       "probes": [
         {
           "type": "probe",
           "name": "app_responds",
           "tolerance": true,
           "provider": {
             "type": "http",
             "url": "http://web:3000/health",
             "method": "GET",
             "timeout": 5
           }
         },
         {
           "type": "probe",
           "name": "db_responds",
           "tolerance": true,
           "provider": {
             "type": "http",
             "url": "http://db:5432",
             "method": "GET",
             "timeout": 5
           }
         }
       ]
     },
     "method": [
       {
         "type": "action",
         "name": "kill_web_container",
         "provider": {
           "type": "python",
           "module": "chaosdocker.actions",
           "func": "stop_container",
           "arguments": {
             "container_name": "web"
           }
         },
         "pauses": {
           "after": 30
         }
       },
       {
         "type": "action",
         "name": "simulate_high_load",
         "provider": {
           "type": "process",
           "path": "stress-ng",
           "arguments": ["--cpu", "2", "--timeout", "60s"],
           "background": true
         },
         "pauses": {
           "after": 65
         }
       }
     ],
     "rollbacks": [
       {
         "type": "action",
         "name": "restart_web",
         "provider": {
           "type": "python",
           "module": "chaosdocker.actions",
           "func": "start_container",
           "arguments": {
             "container_name": "web"
           }
         }
       }
     ]
   }

Ejemplo 8: Multi-tenancy con Namespaces
----------------------------------------

**Objetivo:** Aislar aplicaciones usando namespaces de Docker.

**Configuración multi-tenant:**

.. code-block:: bash

   # Crear networks separadas para tenants
   docker network create tenant-a --driver bridge --subnet 172.20.10.0/24
   docker network create tenant-b --driver bridge --subnet 172.20.20.0/24

   # Aplicación del tenant A
   docker run -d --name app-a \
     --network tenant-a \
     --label tenant=tenant-a \
     my-app:latest

   # Base de datos del tenant A
   docker run -d --name db-a \
     --network tenant-a \
     --label tenant=tenant-a \
     postgres:15

   # Aplicación del tenant B
   docker run -d --name app-b \
     --network tenant-b \
     --label tenant=tenant-b \
     my-app:latest

**Monitoreo por tenant:**

.. code-block:: yaml

   # Prometheus scrape config por tenant
   scrape_configs:
     - job_name: 'tenant-a'
       static_configs:
         - targets: ['app-a:3000', 'db-a:5432']
       labels:
         tenant: 'tenant-a'

     - job_name: 'tenant-b'
       static_configs:
         - targets: ['app-b:3000', 'db-b:5432']
       labels:
         tenant: 'tenant-b'

Ejemplo 9: Integración con Terraform
-------------------------------------

**Objetivo:** Gestionar la infraestructura como código.

**Configuración Terraform (terraform/main.tf):**

.. code-block:: hcl

   terraform {
     required_providers {
       docker = {
         source  = "kreuzwerker/docker"
         version = "~> 3.0"
       }
     }
   }

   provider "docker" {
     host = "unix:///var/run/docker.sock"
   }

   # Aplicación containerizada
   resource "docker_image" "my_app" {
     name = "my-app:latest"
     build {
       context = "../volumes/files/my-app"
     }
   }

   resource "docker_container" "my_app" {
     name  = "my-app"
     image = docker_image.my_app.image_id

     ports {
       internal = 3000
       external = 3000
     }

     env = [
       "NODE_ENV=production",
       "DATABASE_URL=${var.database_url}"
     ]

     depends_on = [docker_container.database]
   }

   # Base de datos
   resource "docker_container" "database" {
     name  = "my-db"
     image = "postgres:15"

     env = [
       "POSTGRES_DB=myapp",
       "POSTGRES_USER=myuser",
       "POSTGRES_PASSWORD=${var.db_password}"
     ]

     volumes {
       host_path      = "${path.module}/../volumes/files/db/init.sql"
       container_path = "/docker-entrypoint-initdb.d/init.sql"
     }
   }

   # Network personalizada
   resource "docker_network" "app_network" {
     name = "my-app-network"
   }

Ejemplo 10: Load Testing con JMeter
------------------------------------

**Objetivo:** Realizar pruebas de carga automatizadas.

**Plan de pruebas JMeter (test/performance_test.jmx):**

.. code-block:: xml

   <?xml version="1.0" encoding="UTF-8"?>
   <jmeterTestPlan version="1.2" properties="5.0" jmeter="5.6.3">
       <hashTree>
           <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="Load Test">
               <boolProp name="TestPlan.functional_mode">false</boolProp>
           </TestPlan>
           <hashTree>
               <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="Load Test Group">
                   <stringProp name="ThreadGroup.num_threads">50</stringProp>
                   <stringProp name="ThreadGroup.ramp_time">30</stringProp>
                   <longProp name="ThreadGroup.duration">300</longProp>
               </ThreadGroup>
               <hashTree>
                   <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="Home Page">
                       <stringProp name="HTTPSampler.domain">${__P(BASE_URL,localhost)}</stringProp>
                       <stringProp name="HTTPSampler.port">3000</stringProp>
                       <stringProp name="HTTPSampler.path">/</stringProp>
                       <stringProp name="HTTPSampler.method">GET</stringProp>
                   </HTTPSamplerProxy>
                   <hashTree/>
                   <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="API Health">
                       <stringProp name="HTTPSampler.domain">${__P(BASE_URL,localhost)}</stringProp>
                       <stringProp name="HTTPSampler.port">3000</stringProp>
                       <stringProp name="HTTPSampler.path">/health</stringProp>
                       <stringProp name="HTTPSampler.method">GET</stringProp>
                   </HTTPSamplerProxy>
                   <hashTree/>
               </hashTree>
           </hashTree>
       </hashTree>
   </jmeterTestPlan>

**Ejecución de pruebas:**

.. code-block:: bash

   # Ejecutar pruebas de carga
   make load-test

   # Ver resultados
   open test/report/index.html

   # Métricas de rendimiento en Grafana
   open http://localhost:3000/d/load-test-dashboard

Resumen de Ejemplos
-------------------

Estos ejemplos demuestran el uso completo de la plataforma:

1. **Aplicación Web**: Desarrollo full-stack con monitoreo
2. **CI/CD**: Pipelines automatizados internos
3. **Monitoreo**: Alertas y dashboards personalizados
4. **Logging**: Consultas avanzadas y centralización
5. **Backup**: Estrategias de recuperación de desastres
6. **API**: Automatización programática
7. **Chaos**: Validación de resiliencia
8. **Multi-tenancy**: Aislamiento de aplicaciones
9. **IaC**: Gestión de infraestructura
10. **Testing**: Validación de rendimiento

Cada ejemplo puede ser adaptado y combinado según las necesidades específicas del proyecto.

   .. code-block:: javascript

      const express = require('express');
      const { Client } = require('pg');

      const app = express();
      const client = new Client({
        connectionString: process.env.DATABASE_URL,
      });

      app.get('/', async (req, res) => {
        try {
          await client.connect();
          const result = await client.query('SELECT NOW()');
          res.json({ message: 'Hello from DinD!', time: result.rows[0] });
        } catch (err) {
          res.status(500).json({ error: err.message });
        }
      });

      app.listen(3000, () => {
        console.log('Server running on port 3000');
      });

4. **Despliegue:**

   .. code-block:: bash

      # Dentro del contenedor
      cd /app
      docker-compose up -d

   Acceda en: http://localhost:51080

Microservicios con Balanceo de Carga
------------------------------------

**Objetivo:** Arquitectura de microservicios con nginx como proxy reverso.

1. **docker-compose.yml:**

   .. code-block:: yaml

      version: '3.8'
      services:
        nginx:
          image: nginx:alpine
          ports:
            - "80:80"
          volumes:
            - ./nginx.conf:/etc/nginx/nginx.conf
          depends_on:
            - api1
            - api2

        api1:
          build: ./api
          environment:
            - PORT=3001

        api2:
          build: ./api
          environment:
            - PORT=3002

2. **nginx.conf:**

   .. code-block:: nginx

      events {
        worker_connections 1024;
      }

      http {
        upstream api_backend {
          server api1:3001;
          server api2:3002;
        }

        server {
          listen 80;

          location / {
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
          }
        }
      }

CI/CD Pipeline Simulado
-----------------------

**Objetivo:** Simular un pipeline de CI/CD completo.

1. **Script de pipeline:**

   .. code-block:: bash

      #!/bin/bash
      set -e

      echo "=== CI/CD Pipeline ==="

      # Build
      echo "Building application..."
      docker build -t myapp:${BUILD_NUMBER:-latest} .

      # Test
      echo "Running tests..."
      docker run --rm myapp:${BUILD_NUMBER:-latest} npm test

      # Security scan
      echo "Security scanning..."
      docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \\
        goodwithtech/dockle:latest myapp:${BUILD_NUMBER:-latest}

      # Deploy
      echo "Deploying..."
      docker tag myapp:${BUILD_NUMBER:-latest} myapp:stable
      docker-compose up -d --scale web=3

      echo "Pipeline completed successfully!"

2. **Ejecución:**

   .. code-block:: bash

      cd /app
      chmod +x pipeline.sh
      ./pipeline.sh

Testing con Selenium
--------------------

**Objetivo:** Testing de UI con Selenium en contenedores.

1. **docker-compose.yml:**

   .. code-block:: yaml

      version: '3.8'
      services:
        selenium:
          image: selenium/standalone-chrome
          ports:
            - "4444:4444"
            - "7900:7900"

        test:
          build: .
          depends_on:
            - selenium
          environment:
            - SELENIUM_URL=http://selenium:4444/wd/hub

2. **Test script:**

   .. code-block:: python

      from selenium import webdriver
      from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

      # Configurar driver remoto
      driver = webdriver.Remote(
          command_executor='http://selenium:4444/wd/hub',
          desired_capabilities=DesiredCapabilities.CHROME
      )

      # Ejecutar test
      driver.get('http://web-app')
      assert 'Welcome' in driver.title

      driver.quit()

Desarrollo con Hot Reload
-------------------------

**Objetivo:** Desarrollo con recarga automática de código.

1. **docker-compose.yml:**

   .. code-block:: yaml

      version: '3.8'
      services:
        app:
          build: .
          ports:
            - "3000:3000"
          volumes:
            - .:/app
            - /app/node_modules
          command: npm run dev
          environment:
            - CHOKIDAR_USEPOLLING=true

2. **package.json:**

   .. code-block:: json

      {
        "scripts": {
          "dev": "nodemon server.js"
        },
        "dependencies": {
          "express": "^4.18.0",
          "nodemon": "^2.0.0"
        }
      }

3. **Uso:**

   .. code-block:: bash

      # Cambios en código se reflejan automáticamente
      echo "Modificando server.js..."
      # La aplicación se recarga sola

Backup Automatizado
-------------------

**Objetivo:** Sistema de backup automático de datos.

1. **Script de backup:**

   .. code-block:: bash

      #!/bin/bash

      BACKUP_DIR="/backups"
      TIMESTAMP=$(date +%Y%m%d_%H%M%S)

      # Backup de base de datos
      docker exec db pg_dump -U user mydb > ${BACKUP_DIR}/db_${TIMESTAMP}.sql

      # Backup de volúmenes
      docker run --rm -v myapp_data:/data -v ${BACKUP_DIR}:/backup \\
        alpine tar czf /backup/volumes_${TIMESTAMP}.tar.gz -C / data

      # Limpiar backups antiguos (mantener 7 días)
      find ${BACKUP_DIR} -name "*.sql" -mtime +7 -delete
      find ${BACKUP_DIR} -name "*.tar.gz" -mtime +7 -delete

      echo "Backup completed: ${TIMESTAMP}"

2. **Programar con cron:**

   .. code-block:: bash

      # Añadir al crontab
      0 2 * * * /path/to/backup.sh

Monitoreo con Prometheus
------------------------

**Objetivo:** Monitoreo completo del entorno.

1. **docker-compose.yml:**

   .. code-block:: yaml

      version: '3.8'
      services:
        prometheus:
          image: prom/prometheus
          ports:
            - "9090:9090"
          volumes:
            - ./prometheus.yml:/etc/prometheus/prometheus.yml

        grafana:
          image: grafana/grafana
          ports:
            - "3001:3000"
          depends_on:
            - prometheus

        node-exporter:
          image: prom/node-exporter
          ports:
            - "9100:9100"

2. **prometheus.yml:**

   .. code-block:: yaml

      global:
        scrape_interval: 15s

      scrape_configs:
        - job_name: 'docker'
          static_configs:
            - targets: ['host.docker.internal:9100']

Ejemplos Avanzados
------------------

**Kubernetes Local:**

Simule un cluster Kubernetes con minikube dentro de DinD.

**Machine Learning:**

Ejecute notebooks Jupyter con GPU passthrough.

**Base de Datos Distribuida:**

Configure un cluster de bases de datos con replicación.

**Serverless:**

Implemente funciones serverless con OpenFaaS.

Cada ejemplo incluye archivos completos y instrucciones detalladas en el repositorio.