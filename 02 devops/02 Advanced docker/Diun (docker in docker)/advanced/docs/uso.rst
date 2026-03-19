Uso de la Plataforma
=====================

.. _uso:

Esta guía explica cómo utilizar los **10 contenedores especializados** de la plataforma Docker-in-Docker para desarrollo, monitoreo y operaciones.

Acceso a Servicios
------------------

Dashboard Principal
~~~~~~~~~~~~~~~~~~~~

+--------------------------------+------------------+--------------------------------+
| Servicio                       | URL              | Credenciales                   |
+================================+==================+================================+
| **Portainer** (Docker GUI)     | localhost:9003   | admin / Ver .env               |
+--------------------------------+------------------+--------------------------------+
| **Grafana** (Dashboards)       | localhost:3000   | admin / Ver .env               |
+--------------------------------+------------------+--------------------------------+
| **Prometheus** (Métricas)      | localhost:9090   | -                              |
+--------------------------------+------------------+--------------------------------+
| **Documentación** (Sphinx)     | localhost:8082   | -                              |
+--------------------------------+------------------+--------------------------------+
| **API REST** (Automatización)  | localhost:5000   | -                              |
+--------------------------------+------------------+--------------------------------+
| **cAdvisor** (Contenedores)    | localhost:8080   | -                              |
+--------------------------------+------------------+--------------------------------+
| **Node Exporter** (Sistema)    | localhost:9100   | -                              |
+--------------------------------+------------------+--------------------------------+
| **Loki** (Logs)               | localhost:3100   | -                              |
+--------------------------------+------------------+--------------------------------+
| **Alertmanager** (Alertas)    | localhost:9093   | -                              |
+--------------------------------+------------------+--------------------------------+

Acceso por Línea de Comandos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SSH al Contenedor Principal:**

.. code-block:: bash

   # Conectar al entorno DinD
   ssh root@localhost -p 50422
   # Password: Ver variable SSH_PASSWORD en .env

   # Una vez dentro, comandos Docker normales
   docker ps                    # Ver contenedores
   docker images               # Ver imágenes
   docker run hello-world      # Probar funcionamiento

**Ejecución Directa en Contenedores:**

.. code-block:: bash

   # Acceso al shell del contenedor principal
   docker-compose exec dind sh

   # Ver logs de todos los servicios
   docker-compose logs -f

   # Ejecutar comandos en servicios específicos
   docker-compose exec prometheus promtool check config /etc/prometheus/prometheus.yml

Gestión de Contenedores Docker
-------------------------------

Operaciones Básicas
~~~~~~~~~~~~~~~~~~~

**Desde Portainer (Interfaz Web):**

1. Acceda a http://localhost:9003
2. Vaya a "Containers" en el menú lateral
3. Use los botones para: Start, Stop, Restart, Remove
4. Para nuevos contenedores: "Add container"
5. Para imágenes: "Images" → "Pull" o "Build"

**Desde Línea de Comandos:**

.. code-block:: bash

   # Dentro del contenedor DinD
   docker run -d --name mi-app nginx:alpine    # Crear contenedor
   docker ps                                   # Ver contenedores activos
   docker stop mi-app                         # Detener contenedor
   docker start mi-app                        # Iniciar contenedor
   docker logs mi-app                         # Ver logs
   docker exec -it mi-app sh                  # Acceder al shell

**Desde API REST:**

.. code-block:: bash

   # Ver todos los contenedores
   curl http://localhost:5000/api/containers

   # Iniciar un contenedor específico
   curl -X POST http://localhost:5000/api/containers/mi-app/start

   # Ver logs de un contenedor
   curl http://localhost:5000/api/containers/mi-app/logs

Gestión de Imágenes
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Descargar imagen
   docker pull nginx:alpine

   # Ver imágenes disponibles
   docker images

   # Construir imagen personalizada
   docker build -t mi-app:latest .

   # Subir a registry
   docker tag mi-app:latest registry.example.com/mi-app:latest
   docker push registry.example.com/mi-app:latest

Gestión de Volúmenes
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Crear volumen persistente
   docker volume create mi-volumen

   # Usar volumen en contenedor
   docker run -d -v mi-volumen:/data nginx:alpine

   # Ver volúmenes
   docker volume ls

   # Respaldar volumen
   docker run --rm -v mi-volumen:/source -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz -C /source .

Monitoreo y Observabilidad
---------------------------

Sistema de Métricas
~~~~~~~~~~~~~~~~~~~

**Prometheus - Recolección de Métricas:**

- URL: http://localhost:9090
- Métricas disponibles:
  - Uso de CPU y memoria por contenedor
  - Rendimiento del sistema host
  - Estado de servicios
  - Latencia de red

**Consultas Básicas en Prometheus:**

.. code-block:: promql

   # CPU usage
   rate(container_cpu_usage_seconds_total[5m])

   # Memory usage
   container_memory_usage_bytes

   # System load
   node_load1

**Grafana - Dashboards Visuales:**

- URL: http://localhost:3000
- Dashboards incluidos:
  - **Docker Monitoring**: CPU, memoria, red de contenedores
  - **System Metrics**: Load, disco, procesos del host
  - **Application Performance**: Latencia y throughput

**cAdvisor - Monitoreo Detallado:**

- URL: http://localhost:8080
- Información por contenedor:
  - Uso de CPU por core
  - Memoria (RSS, cache, swap)
  - I/O de red y disco
  - Sistema de archivos

Sistema de Logs
~~~~~~~~~~~~~~~

**Loki - Almacenamiento Centralizado:**

- URL: http://localhost:3100
- Explore logs con consultas como:
  - ``{job="docker"}`` - Todos los logs de Docker
  - ``{container_name="mi-app"}`` - Logs de contenedor específico
  - ``{level="error"}`` - Solo errores

**Promtail - Recolección Automática:**

- Recolecta logs automáticamente de:
  - ``/var/log/*.log`` - Logs del sistema
  - ``/var/log/docker.log`` - Logs de Docker
  - Contenedores con labels específicas

**Alertmanager - Notificaciones:**

- URL: http://localhost:9093
- Configurado para enviar alertas por:
  - Email (SMTP)
  - Slack webhooks
  - PagerDuty

Alertas Configuradas
^^^^^^^^^^^^^^^^^^^^

- **Container Down**: Contenedor caído por más de 5 minutos
- **High CPU Usage**: CPU > 80% por más de 5 minutos
- **Low Memory**: Memoria libre < 10%
- **Disk Full**: Disco > 90% usado
- **Service Unhealthy**: Health checks fallando

Operaciones de Mantenimiento
-----------------------------

Backups Automáticos
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Crear backup manual
   make backup

   # Ver backups disponibles
   ls -la volumes/backups/

   # Restaurar desde backup
   # (Proceso manual: extraer archivos al directorio volumes/)

Backups programados se ejecutan diariamente a las 2:00 AM y mantienen 7 días de retención.

Gestión de Recursos
~~~~~~~~~~~~~~~~~~~

**Límites de Recursos por Contenedor:**

+------------------+-------+--------+-------------+
| Contenedor       | CPU   | RAM    | Disco       |
+==================+=======+========+=============+
| DinD             | 2.0   | 4GB    | Persistente |
+------------------+-------+--------+-------------+
| Monitoreo        | 0.5   | 512MB  | Persistente |
+------------------+-------+--------+-------------+
| Documentación    | 0.5   | 512MB  | Read-only   |
+------------------+-------+--------+-------------+
| API              | 0.5   | 256MB  | -           |
+------------------+-------+--------+-------------+

**Monitoreo de Recursos:**

.. code-block:: bash

   # Ver uso de recursos en tiempo real
   docker stats

   # Ver métricas detalladas en Grafana
   open http://localhost:3000

Limpieza y Mantenimiento
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Limpiar contenedores detenidos
   docker container prune -f

   # Limpiar imágenes no utilizadas
   docker image prune -f

   # Limpiar volúmenes huérfanos
   docker volume prune -f

   # Limpieza completa del sistema
   docker system prune -a --volumes

Tareas Automatizadas (Cron)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Diario**: Backups, rotación de logs
- **Semanal**: Limpieza de imágenes no utilizadas
- **Cada hora**: Verificación de health checks

Desarrollo y Testing
---------------------

Entorno de Desarrollo
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Desarrollar dentro del contenedor
   docker-compose exec dind sh
   cd /app

   # Instalar dependencias
   apk add --no-cache git curl wget

   # Ejecutar scripts de desarrollo
   ./scripts/setup.sh

Testing Automatizado
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Ejecutar todos los tests
   make test

   # Tests específicos
   make integration-test    # Tests de integración
   make load-test          # Tests de carga
   make security-test      # Tests de seguridad
   make chaos-test         # Tests de caos

   # Ver resultados de tests
   cat test-results/junit.xml

Despliegue de Aplicaciones
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Desarrollo Local:**

.. code-block:: bash

   # Crear aplicación de ejemplo
   mkdir -p volumes/files/my-app
   cat > volumes/files/my-app/app.js << 'EOF'
   const http = require('http');
   const server = http.createServer((req, res) => {
     res.statusCode = 200;
     res.setHeader('Content-Type', 'text/plain');
     res.end('Hello from DinD!\n');
   });
   server.listen(3000, () => {
     console.log('App running on port 3000');
   });
   EOF

   # Ejecutar dentro del DinD
   docker-compose exec dind sh
   cd /app/my-app
   docker run -d -p 3000:3000 --name my-app node:alpine node app.js

**Producción:**

.. code-block:: bash

   # Usar Docker Compose para aplicaciones complejas
   docker-compose exec dind sh
   cd /app
   docker-compose up -d

Integración con CI/CD
~~~~~~~~~~~~~~~~~~~~~~

La plataforma incluye configuración completa para CI/CD:

.. code-block:: yaml

   # .github/workflows/ci.yml incluye:
   # - Tests automatizados
   # - Security scanning
   # - Build optimization
   # - Deployment stages

**Comandos de CI/CD:**

.. code-block:: bash

   # Construir imagen optimizada
   make build

   # Ejecutar pipeline completo
   make test

   # Desplegar a staging
   make deploy-staging

   # Desplegar a producción
   make deploy-production

Solución de Problemas
----------------------

Diagnóstico Rápido
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Ver estado de todos los servicios
   docker-compose ps

   # Ver logs de servicios con problemas
   docker-compose logs dind
   docker-compose logs prometheus

   # Verificar conectividad de red
   docker-compose exec dind ping -c 3 prometheus

   # Verificar health checks
   curl http://localhost:5000/api/health

Problemas Comunes
~~~~~~~~~~~~~~~~~~

**Contenedor no inicia:**

.. code-block:: bash

   # Ver logs detallados
   docker-compose logs --tail=100 <service-name>

   # Verificar configuración
   docker-compose config

   # Reiniciar servicio específico
   docker-compose restart <service-name>

**Problemas de rendimiento:**

.. code-block:: bash

   # Ver uso de recursos
   docker stats

   # Ver métricas en Grafana
   open http://localhost:3000

   # Ajustar límites de recursos
   # Editar docker-compose.yaml

**Problemas de conectividad:**

.. code-block:: bash

   # Verificar puertos abiertos
   netstat -tlnp | grep LISTEN

   # Test de conectividad
   curl -v http://localhost:9090/-/healthy

   # Verificar firewall
   docker-compose exec dind ufw status

**Problemas de logs:**

.. code-block:: bash

   # Ver logs centralizados
   open http://localhost:3100

   # Buscar errores específicos
   # En Loki: {level="error"}

   # Ver logs de servicios
   docker-compose logs --tail=50 --follow

Comandos de Emergencia
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Reinicio completo de la plataforma
   docker-compose down
   docker-compose up -d

   # Limpieza de emergencia
   docker system prune -a --volumes -f

   # Restaurar desde backup
   # (Extraer backup a volumes/ y reiniciar)

Documentación Adicional
~~~~~~~~~~~~~~~~~~~~~~~

- **API Reference**: http://localhost:8082/api.html
- **Troubleshooting**: http://localhost:8082/troubleshooting.html
- **Arquitectura**: http://localhost:8082/arquitectura.html
- **Configuración**: http://localhost:8082/configuracion.html

Gestión de Aplicaciones
-----------------------

Estructura de Archivos
~~~~~~~~~~~~~~~~~~~~~~

Coloque sus aplicaciones en ``./volumes/files/``:

.. code-block:: text

   volumes/
   └── files/
       ├── app.js
       ├── Dockerfile
       └── docker-compose.yml

Estos archivos estarán disponibles en ``/app`` dentro del contenedor.

Ejemplo Práctico: Aplicación Node.js
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Crear aplicación** (``./volumes/files/app.js``):

   .. code-block:: javascript

      const http = require('http');
      const os = require('os');

      const server = http.createServer((req, res) => {
        res.writeHead(200, {'Content-Type': 'text/plain'});
        res.end(`Hello from ${os.hostname()}!\\n`);
      });

      server.listen(3000, () => {
        console.log('Server running on port 3000');
      });

2. **Crear Dockerfile** (``./volumes/files/Dockerfile``):

   .. code-block:: dockerfile

      FROM node:18-alpine
      WORKDIR /app
      COPY app.js .
      CMD ["node", "app.js"]

3. **Construir y ejecutar**:

   .. code-block:: bash

      # Dentro del contenedor DinD
      cd /app
      docker build -t my-app .
      docker run -d -p 3000:3000 --name my-app my-app

4. **Acceder**: http://localhost:51080

Gestión de Contenedores con Portainer
-------------------------------------

Portainer ofrece una interfaz intuitiva para:

* **Dashboard**: Vista general del sistema
* **Contenedores**: Crear, iniciar, detener, eliminar
* **Imágenes**: Gestionar imágenes Docker
* **Volúmenes**: Administrar almacenamiento persistente
* **Redes**: Configurar conectividad
* **Registros**: Ver logs y métricas

Flujos de Trabajo Comunes
-------------------------

Desarrollo Local
~~~~~~~~~~~~~~~~

1. Desarrolle en ``./volumes/files/``
2. Use Portainer para testing rápido
3. Acceda vía SSH para debugging avanzado

CI/CD Pipeline
~~~~~~~~~~~~~~

1. Monte código fuente como volumen
2. Ejecute builds dentro del contenedor
3. Use Docker interno para testing

Testing de Infraestructura
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Simule entornos complejos
2. Test configuraciones de red
3. Valide compatibilidad entre servicios

Monitoreo y Logs
----------------

**Ver logs del sistema:**

.. code-block:: bash

   docker-compose logs -f dind

**Monitoreo de recursos:**

.. code-block:: bash

   docker stats dind

**Health checks:**

.. code-block:: bash

   docker inspect dind | grep -A 10 "Health"

Backup y Restauración
---------------------

**Backup de datos:**

.. code-block:: bash

   # Backup de volúmenes
   tar -czf backup.tar.gz volumes/

**Restauración:**

.. code-block:: bash

   # Detener contenedores
   docker-compose down

   # Restaurar volúmenes
   tar -xzf backup.tar.gz

   # Reiniciar
   docker-compose up -d

Comandos Útiles
---------------

**Reinicio completo:**

.. code-block:: bash

   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d

**Limpieza:**

.. code-block:: bash

   docker-compose down -v
   docker system prune -f

**Debugging:**

.. code-block:: bash

   # Acceso directo
   docker exec -it dind sh

   # Ver procesos
   docker exec dind ps aux

   # Ver redes
   docker exec dind docker network ls