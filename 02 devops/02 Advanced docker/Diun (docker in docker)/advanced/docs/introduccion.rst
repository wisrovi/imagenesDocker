Introduction
============

.. _introduction:

What is Docker-in-Docker?
-------------------------

Docker-in-Docker (DinD) is a technique that allows running the Docker daemon inside a Docker container. This is useful for scenarios requiring nested containerization, such as CI/CD pipelines, Docker tool development, or complex testing environments.

.. warning::
    DinD requires privileged mode and may have security implications. Use it only in controlled environments.

About the Project
-----------------

This enterprise-ready platform provides a **fully automated Docker-in-Docker environment** with **10 specialized containers**:

### 🏠 Core Containers
* **DinD**: Complete Docker-in-Docker environment with internal daemon
* **Portainer**: Advanced web interface for container management
* **SSH**: Secure remote access with automatic configuration
* **REST API**: Programmatic container control
* **Documentation**: Sphinx with integrated search and SSL

### 📊 Complete Monitoring System
* **Prometheus**: Centralized metrics collection
* **Grafana**: Professional dashboards and alerts
* **cAdvisor**: Detailed container monitoring
* **Node Exporter**: Operating system metrics

### 📝 Observability System
* **Loki**: Centralized log storage
* **Promtail**: Intelligent log collection
* **Alertmanager**: Intelligent notification system

### 🔒 Security Features
* **SSL/TLS**: Automatic certificates with Let's Encrypt
* **Firewall**: UFW with service-specific rules
* **Fail2Ban**: Automatic SSH attack protection
* **Security Headers**: CSP, HSTS, X-Frame-Options
* **Rate Limiting**: Protection against API abuse

### 💾 Operational Features
* **Automatic Backups**: Backup system with retention
* **Health Checks**: Continuous service monitoring
* **Resource Limits**: Precise CPU and memory control
* **Structured Logging**: Centralized and rotated logs
* **Cron Jobs**: Automated maintenance

Arquitectura Completa
--------------------

.. code-block:: text

    +---------------------------------------------------+
    |                HOST SYSTEM                        |
    |                                                   |
    |  🐳 Docker Engine                    🌐 Usuario   |
    +---------------------------------------------------+
                    │
                    ▼
    +---------------------------------------------------+
    |           PLATAFORMA DOCKER-IN-DOCKER            |
    +---------------------------------------------------+
    │                                                   │
    │  🏠 DIN D          📊 MONITORING         📝 LOGS   │
    │  ├─ Docker Daemon  ├─ Prometheus         ├─ Loki   │
    │  ├─ Portainer      ├─ Grafana           ├─ Promtail│
    │  ├─ SSH Server     ├─ cAdvisor          └─ AlertMgr│
    │  └─ API REST      └─ Node Exporter               │
    │                                                   │
    │  📚 DOCUMENTACIÓN     🔧 SERVICIOS                 │
    │  └─ Sphinx Server     └─ Cron Jobs                │
    │                        └─ Backups                 │
    +---------------------------------------------------+
                    │
                    ▼
    +---------------------------------------------------+
    |           CONTENEDORES APLICACIÓN                 |
    │  (Gestionados por Portainer/DinD)                │
    +---------------------------------------------------+

Componentes Principales
-----------------------

🏠 Núcleo Docker-in-Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Contenedor DinD (dind)**
    * Docker daemon completo ejecutándose en contenedor
    * Modo privilegiado para acceso total al sistema de archivos
    * Health checks automáticos cada 30 segundos
    * Persistencia completa de imágenes, contenedores y volúmenes
    * Puertos: 9003 (Portainer), 50422 (SSH), 80/443 (web)

**Portainer (integrado en dind)**
    * Interfaz web moderna y responsive
    * Gestión completa: contenedores, imágenes, volúmenes, redes
    * Autenticación con usuarios y equipos
    * API REST completa para automatización
    * Stacks y compose integrados

**SSH Server (integrado en dind)**
    * OpenSSH con configuración de seguridad avanzada
    * Acceso root configurable con contraseña fuerte
    * Puerto personalizado 50422 para evitar conflictos
    * Sesiones tmux para estabilidad de conexiones
    * Fail2ban integrado para protección contra ataques

📊 Sistema de Monitoreo
~~~~~~~~~~~~~~~~~~~~~~~~

**Prometheus (prometheus:9090)**
    * Recolección centralizada de métricas de todos los servicios
    * Almacenamiento eficiente con compresión
    * Lenguaje de consulta PromQL avanzado
    * Service discovery automático
    * Retención configurable de métricas

**Grafana (grafana:3000)**
    * Dashboards profesionales con visualizaciones avanzadas
    * Alertas integradas con Prometheus
    * Plantillas y themes personalizables
    * Usuarios, equipos y permisos granulares
    * Integración con múltiples fuentes de datos

**cAdvisor (cadvisor:8080)**
    * Monitoreo detallado de rendimiento de contenedores
    * Métricas de CPU, memoria, red y disco por contenedor
    * Jerarquía completa de contenedores
    * API REST para integración
    * Soporte para Docker y Kubernetes

**Node Exporter (node-exporter:9100)**
    * Métricas detalladas del sistema operativo host
    * CPU, memoria, disco, red, sistema de archivos
    * Métricas de hardware y procesos
    * Configurable para diferentes sistemas operativos

📝 Sistema de Logs
~~~~~~~~~~~~~~~~~~~

**Loki (loki:3100)**
    * Almacenamiento eficiente de logs etiquetados
    * Indexación por etiquetas en lugar de texto completo
    * Consultas LogQL similares a PromQL
    * Integración nativa con Grafana
    * Compresión y retención configurables

**Promtail (promtail)**
    * Recolección inteligente de logs de contenedores
    * Descubrimiento automático de archivos de log
    * Etiquetado automático con metadatos de Docker
    * Envío eficiente a Loki
    * Soporte para múltiples formatos

🚨 Sistema de Alertas
~~~~~~~~~~~~~~~~~~~~~~

**Alertmanager (alertmanager:9093)**
    * Gestión inteligente de alertas de Prometheus
    * Agrupación y deduplicación automática
    * Envío a múltiples canales (email, Slack, PagerDuty)
    * Silenciamiento y mantenimiento programado
    * Inhibición de alertas relacionadas

🔧 Servicios Adicionales
~~~~~~~~~~~~~~~~~~~~~~~~~

**API REST (dind-api:5000)**
    * API programática para gestión de contenedores
    * Endpoints para CRUD de contenedores, imágenes, volúmenes
    * Autenticación y autorización
    * Documentación OpenAPI integrada
    * SDKs para Python y JavaScript

**Documentación (docs-server:8082)**
    * Servidor Sphinx con documentación completa
    * Búsqueda full-text integrada
    * SSL/TLS automático con Let's Encrypt
    * Tema responsive y moderno
    * Versionado y control de cambios

Casos de Uso
------------

Desarrollo y Testing
~~~~~~~~~~~~~~~~~~~~

* Entornos de desarrollo aislados
* Testing de aplicaciones contenerizadas
* Simulación de infraestructuras complejas

CI/CD Pipelines
~~~~~~~~~~~~~~~

* Construcción de imágenes dentro de pipelines
* Testing de contenedores anidados
* Validación de configuraciones Docker

Educación y Aprendizaje
~~~~~~~~~~~~~~~~~~~~~~~

* Aprendizaje de Docker sin afectar el host
* Experimentación segura con contenedores
* Demostraciones y talleres

Requisitos del Sistema
----------------------

Hardware Mínimo
~~~~~~~~~~~~~~~

* **Docker**: Versión 24.0 o superior
* **Docker Compose**: Versión 2.20 o superior
* **RAM**: 8GB (mínimo para monitoreo básico)
* **CPU**: 2 cores con virtualización habilitada
* **Disco**: 50GB SSD (para imágenes y métricas)
* **Red**: Conexión estable para alertas externas

Hardware Recomendado
~~~~~~~~~~~~~~~~~~~~~

* **Docker**: Versión 26.0+ (última estable)
* **Docker Compose**: Versión 2.24+ (última)
* **RAM**: 16GB+ (para cargas de trabajo intensivas)
* **CPU**: 4+ cores dedicados
* **Disco**: 100GB+ NVMe (alta IOPS para métricas)
* **Red**: 1Gbps+ con baja latencia

Sistemas Operativos Soportados
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+, Debian 11+
* **macOS**: 12.0+ con Docker Desktop
* **Windows**: 10/11 Pro con WSL2 y Docker Desktop
* **Cloud**: AWS, GCP, Azure con instancias optimizadas

Dependencias Adicionales
~~~~~~~~~~~~~~~~~~~~~~~~~

* **curl/wget**: Para health checks y descargas
* **openssl**: Para gestión de certificados SSL
* **git**: Para control de versiones
* **make**: Para automatización de builds
* **python3**: Para scripts de automatización (opcional)

Limitaciones
------------

* Requiere modo privilegiado
* No compatible con Docker Desktop en algunos casos
* Mayor uso de recursos que contenedores normales
* Riesgos de seguridad si no se configura correctamente