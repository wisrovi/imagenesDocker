# URLs de Servicios Levantados

Lista de URLs de los servicios que están corriendo en el proyecto Docker-in-Docker:

## 🌐 Servicios Web Principales
- **[Portainer](http://localhost:9003)** - Interfaz web para gestión de contenedores Docker
- **[Grafana](http://localhost:3000)** - Dashboards de monitoreo y visualización (usuario: admin)
- **[Documentación Sphinx](http://localhost:8082)** - Documentación completa del proyecto
- **[Kibana](http://localhost:5601)** - Interfaz de visualización para logs ELK

## 📊 Monitoreo y Métricas
- **[Prometheus](http://localhost:9090)** - Sistema de métricas y consultas PromQL
- **[Alertmanager](http://localhost:9093)** - Gestión y notificaciones de alertas
- **[cAdvisor](http://localhost:8081)** - Monitoreo detallado de contenedores
- **[Node Exporter](http://localhost:9100)** - Métricas del sistema host

## 🔍 Logging y Análisis
- **[Loki](http://localhost:3100)** - Almacenamiento centralizado de logs
- **[Elasticsearch](http://localhost:9200)** - Motor de búsqueda y análisis de datos
- **[Logstash](http://localhost:9600)** - Pipeline de procesamiento de logs

## 🔧 API y Automatización
- **[API REST](http://localhost:5000)** - API programática para control de contenedores
- **[API Health Check](http://localhost:5000/api/health)** - Verificación de estado de la API

---

*Proyecto: Docker-in-Docker con Monitoreo Completo*  
*Verifica que todos los contenedores estén corriendo con `docker-compose ps`*