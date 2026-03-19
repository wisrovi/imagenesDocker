Detailed Architecture
====================

.. _arquitectura:

This page describes in detail the **10 specialized containers** that make up the Docker-in-Docker platform, their functions, configurations, and interactions.

Architecture Overview
---------------------

The platform follows a microservices architecture with clearly defined responsibilities:

.. image:: _static/arquitectura.png
   :alt: Platform architecture
   :align: center
   :scale: 75%

Main Containers
---------------

1. 🏠 Docker-in-Docker (dind)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Main Function**: Provides the complete Docker-in-Docker environment

**Features**:

* **Base Image**: Official ``docker:dind`` with optimizations
* **Mode**: Privileged with host user namespace
* **Ports**:
  * ``9003``: Portainer HTTP
  * ``9443``: Portainer HTTPS
  * ``50422``: SSH
  * ``80/443``: Web services
* **Volumes**:
  * ``dind-data``: Persistent Docker
  * ``ssl``: SSL certificates
  * ``logs``: Application logs
  * ``backups``: Automatic backups
* **Resources**: 2 CPUs, 4GB RAM
* **Health Check**: Docker daemon status

**Integrated Services**:

* **Portainer**: Web container management
* **SSH Server**: Secure remote access
* **Node Exporter**: Container metrics
* **Fail2Ban**: SSH protection
* **UFW**: Firewall
* **Cron**: Scheduled tasks

2. 📚 Documentación (docs-server)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Función Principal**: Servidor web para documentación Sphinx

**Características**:

* **Imagen Base**: ``nginx:alpine`` con Sphinx
* **Puertos**: ``8082`` (HTTP), ``8443`` (HTTPS)
* **SSL**: Certificados automáticos
* **Funciones**:
  * Documentación completa
  * Búsqueda full-text
  * Tema responsive
  * Headers de seguridad
* **Recursos**: 0.5 CPUs, 512MB RAM

Sistema de Monitoreo
---------------------

3. 📊 Prometheus (prometheus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Función Principal**: Recolección y almacenamiento de métricas

**Características**:

* **Imagen**: ``prom/prometheus:latest``
* **Puerto**: ``9090``
* **Configuración**:
  * Service discovery automático
  * Reglas de alerting
  * Retención 200 horas
  * Compresión de datos
* **Métricas Recolectadas**:
  * Todos los contenedores
  * Sistema operativo
  * Aplicaciones personalizadas
* **Recursos**: 0.5 CPUs, 1GB RAM

4. 📈 Grafana (grafana)
~~~~~~~~~~~~~~~~~~~~~~~~

**Función Principal**: Visualización y dashboards de métricas

**Características**:

* **Imagen**: ``grafana/grafana:latest``
* **Puerto**: ``3000``
* **Funciones**:
  * Dashboards pre-configurados
  * Alertas integradas
  * Usuarios y permisos
  * Plugins adicionales
* **Dashboards Incluidos**:
  * Docker monitoring
  * System metrics
  * Application performance
* **Recursos**: 0.5 CPUs, 512MB RAM

5. 🔍 cAdvisor (cadvisor)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Función Principal**: Monitoreo detallado de contenedores

**Características**:

* **Imagen**: ``gcr.io/cadvisor/cadvisor:latest``
* **Puerto**: ``8080``
* **Métricas**:
  * CPU por contenedor
  * Memoria y swap
  * Uso de red
  * I/O de disco
  * Sistema de archivos
* **Privilegiado**: Acceso completo al host
* **Recursos**: 0.2 CPUs, 256MB RAM

6. 📊 Node Exporter (node-exporter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Función Principal**: Métricas del sistema operativo host

**Características**:

* **Imagen**: ``prom/node-exporter:latest``
* **Puerto**: ``9100``
* **Métricas del Sistema**:
  * CPU (usage, load, cores)
  * Memoria (RAM, swap)
  * Disco (I/O, usage, inodes)
  * Red (interfaces, traffic)
  * Sistema de archivos
  * Procesos y threads
* **Volúmenes**: Acceso readonly al host
* **Recursos**: 0.1 CPUs, 128MB RAM

Sistema de Logs
---------------

7. 📝 Loki (loki)
~~~~~~~~~~~~~~~~~~

**Función Principal**: Almacenamiento centralizado de logs

**Características**:

* **Imagen**: ``grafana/loki:latest``
* **Puerto**: ``3100``
* **Características**:
  * Indexación por etiquetas
  * Compresión eficiente
  * Consultas LogQL
  * Integración Grafana
* **Almacenamiento**: Volumen persistente
* **Recursos**: 0.3 CPUs, 512MB RAM

8. 📤 Promtail (promtail)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Función Principal**: Recolección y envío de logs a Loki

**Características**:

* **Imagen**: ``grafana/promtail:latest``
* **Funciones**:
  * Descubrimiento automático de logs
  * Etiquetado inteligente
  * Envío eficiente a Loki
  * Soporte múltiples formatos
* **Volúmenes**: Acceso a ``/var/log``
* **Configuración**: YAML con targets dinámicos
* **Recursos**: 0.1 CPUs, 128MB RAM

Sistema de Alertas
------------------

9. 🚨 Alertmanager (alertmanager)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Función Principal**: Gestión inteligente de alertas

**Características**:

* **Imagen**: ``prom/alertmanager:latest``
* **Puerto**: ``9093``
* **Funciones**:
  * Agrupación de alertas
  * Deduplicación automática
  * Envío multi-canal
  * Silenciamiento programado
  * Inhibición de alertas
* **Canales Soportados**:
  * Email SMTP
  * Slack webhooks
  * PagerDuty
  * Webhooks genéricos
* **Recursos**: 0.2 CPUs, 256MB RAM

API y Automatización
--------------------

10. 🔧 API REST (dind-api)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Función Principal**: API programática para gestión de contenedores

**Características**:

* **Framework**: Flask + Flask-CORS
* **Puerto**: ``5000``
* **Endpoints**:
  * ``GET /api/health``: Estado del servicio
  * ``GET /api/containers``: Lista contenedores
  * ``POST /api/containers/{id}/start``: Iniciar contenedor
  * ``POST /api/containers/{id}/stop``: Detener contenedor
  * ``GET /api/containers/{id}/logs``: Obtener logs
  * ``GET /api/images``: Lista imágenes
  * ``GET /api/volumes``: Lista volúmenes
  * ``GET /api/system/info``: Info del sistema
  * ``POST /api/backup``: Crear respaldo
* **Autenticación**: API keys (futuro)
* **Documentación**: OpenAPI/Swagger integrada
* **Recursos**: 0.5 CPUs, 256MB RAM

Interacciones entre Contenedores
---------------------------------

Flujo de Datos
~~~~~~~~~~~~~~

.. code-block:: text

    Aplicaciones → cAdvisor → Prometheus → Grafana
                       ↓
    Sistema → Node Exporter → Prometheus → Alertmanager → Notificaciones
                       ↓
    Logs → Promtail → Loki → Grafana

Dependencias de Inicio
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    dind:
      # Contenedor base, no depende de otros

    docs-server:
      depends_on:
        dind:
          condition: service_healthy

    prometheus:
      depends_on:
        - dind

    grafana:
      depends_on:
        - prometheus

    cadvisor, node-exporter, loki, promtail, alertmanager:
      depends_on:
        - dind

    dind-api:
      depends_on:
        - dind

Redes y Comunicación
~~~~~~~~~~~~~~~~~~~~

* **Red Principal**: ``dind-network`` (172.20.0.0/16)
* **DNS**: Resolución automática entre contenedores
* **Seguridad**: Comunicación interna cifrada donde aplica
* **Balanceo**: No requerido (servicios directos)

Volúmenes Compartidos
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    volumes/
    ├── dind-data/      # Docker persistente
    ├── ssl/           # Certificados SSL
    ├── logs/          # Logs centralizados
    ├── backups/       # Respaldos
    ├── grafana/       # Config Grafana
    ├── prometheus/    # Métricas Prometheus
    └── loki/          # Logs Loki

Monitoreo de Arquitectura
-------------------------

Health Checks
~~~~~~~~~~~~~

Cada contenedor incluye health checks específicos:

* **DinD**: ``docker ps`` cada 30s
* **Documentación**: ``curl localhost`` cada 30s
* **API**: ``curl /api/health`` cada 30s
* **Prometheus/Grafana**: HTTP status checks
* **Monitoreo**: Métricas de disponibilidad

Métricas de Rendimiento
~~~~~~~~~~~~~~~~~~~~~~~

* **CPU**: Uso por contenedor y global
* **Memoria**: RAM y swap por servicio
* **Red**: Tráfico interno y externo
* **Disco**: I/O y uso de volúmenes
* **Latencia**: Response times de APIs

Alertas Configuradas
~~~~~~~~~~~~~~~~~~~~

* Contenedor caído o unhealthy
* Uso alto de CPU (>80%)
* Memoria baja (<10% free)
* Disco lleno (>90%)
* Errores de red
* Fallos de backup

Escalabilidad
-------------

Escalado Horizontal
~~~~~~~~~~~~~~~~~~~

* **Monitoreo**: Múltiples instancias con balanceo
* **Logs**: Sharding por tiempo/tenant
* **API**: Load balancing con Nginx
* **Base de datos**: Clustering para métricas

Escalado Vertical
~~~~~~~~~~~~~~~~~

* **Límites de Recursos**: Configurables por contenedor
* **Auto-scaling**: Basado en métricas de uso
* **Resource Quotas**: Por namespace/usuario

Limitaciones Actuales
~~~~~~~~~~~~~~~~~~~~~

* Monitoreo: Instancia única (no cluster)
* Logs: Retención limitada por disco
* API: Sin autenticación avanzada
* Backups: Locales (no cloud)

Mantenimiento y Operaciones
---------------------------

Rutinas de Mantenimiento
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Diario**: Rotación de logs, limpieza de imágenes
* **Semanal**: Backups completos, actualización de imágenes
* **Mensual**: Revisión de configuraciones, optimización

Monitoreo de Operaciones
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Disponibilidad**: 99.9% uptime objetivo
* **Latencia**: <100ms para APIs críticas
* **Recursos**: <80% uso promedio
* **Backups**: Verificación diaria de integridad

Recuperación de Desastres
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Backups**: Automáticos con retención 7 días
* **Restauración**: Scripts automatizados
* **Failover**: Configuración preparada
* **Documentación**: Runbooks detallados