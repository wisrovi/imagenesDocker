Mejores Prácticas
=================

.. _mejores-practicas:

Esta guía presenta las mejores prácticas para usar el entorno DinD de manera efectiva y segura.

Organización del Proyecto
-------------------------

Estructura Recomendada
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   my-dind-project/
   ├── docker-compose.yaml
   ├── scripts/
   │   ├── install_ssh.sh
   │   ├── install_portainer.sh
   │   └── start.sh
   ├── volumes/
   │   ├── files/
   │   └── dind-data/
   ├── docs/
   │   ├── index.rst
   │   └── ...
   ├── .gitignore
   └── README.md

Versionado
~~~~~~~~~~

- Usa Git para control de versiones
- Incluye ``.gitignore`` para excluir datos sensibles
- Etiqueta releases importantes
- Documenta cambios en ``CHANGELOG.md``

Seguridad
---------

Configuración Segura
~~~~~~~~~~~~~~~~~~~~

**Cambia contraseñas por defecto:**

.. code-block:: bash

   # En scripts/install_ssh.sh
   echo "root:$(openssl rand -base64 12)" | chpasswd

**Usa variables de entorno:**

.. code-block:: bash

   # docker-compose.yaml
   environment:
     - SSH_PASSWORD=${SSH_PASSWORD}
     - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

**Limita acceso a red:**

.. code-block:: yaml

   services:
     dind:
       networks:
         - internal
       ports:
         - "127.0.0.1:9003:9000"  # Solo localhost

**Auditoría de logs:**

.. code-block:: bash

   # Monitoreo de logs
   docker-compose logs -f dind | grep -i "error\|warning"

Gestión de Recursos
-------------------

Monitoreo de Recursos
~~~~~~~~~~~~~~~~~~~~~

**Límites de recursos:**

.. code-block:: yaml

   services:
     dind:
       deploy:
         resources:
           limits:
             cpus: '2.0'
             memory: 4G
           reservations:
             cpus: '0.5'
             memory: 1G

**Monitoreo continuo:**

.. code-block:: bash

   # Script de monitoreo
   #!/bin/bash
   while true; do
     echo "$(date): CPU $(docker stats --no-stream --format 'table {{.CPUPerc}}' dind | tail -1)"
     sleep 60
   done

Limpieza Automática
~~~~~~~~~~~~~~~~~~~

**Eliminar contenedores no utilizados:**

.. code-block:: bash

   # Cron job para limpieza
   0 2 * * * docker system prune -f --volumes

**Rotación de logs:**

.. code-block:: bash

   # En docker-compose.yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"

Desarrollo
----------

Flujo de Trabajo
~~~~~~~~~~~~~~~~

1. **Desarrollo local:**

   .. code-block:: bash

      # Montar código fuente
      volumes:
        - ./src:/app/src

2. **Testing:**

   .. code-block:: bash

      # Ejecutar tests dentro del contenedor
      docker exec dind npm test

3. **Debugging:**

   .. code-block:: bash

      # Acceso interactivo
      docker exec -it dind sh

      # Logs en tiempo real
      docker-compose logs -f

Control de Versiones
~~~~~~~~~~~~~~~~~~~~

**Git workflow:**

.. code-block:: bash

   # Ramas por feature
   git checkout -b feature/nueva-funcionalidad

   # Commits descriptivos
   git commit -m "feat: añadir autenticación JWT

   - Implementar middleware de auth
   - Añadir tests unitarios
   - Actualizar documentación"

**Conventional commits:**

- ``feat:`` nuevas características
- ``fix:`` corrección de bugs
- ``docs:`` cambios en documentación
- ``style:`` cambios de formato
- ``refactor:`` refactorización
- ``test:`` añadir tests
- ``chore:`` tareas de mantenimiento

Testing
-------

Estrategias de Testing
~~~~~~~~~~~~~~~~~~~~~~

**Tests unitarios:**

.. code-block:: bash

   # Dentro del contenedor
   docker exec dind python -m pytest tests/unit/

**Tests de integración:**

.. code-block:: bash

   # Testing de servicios
   docker exec dind ./test-integration.sh

**Tests de carga:**

.. code-block:: bash

   # Usando herramientas como k6
   docker run --rm -v $(pwd)/tests:/tests loadimpact/k6 run /tests/load-test.js

**Tests de seguridad:**

.. code-block:: bash

   # Escaneo de vulnerabilidades
   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \\
     goodwithtech/dockle:latest my-image

CI/CD
-----

Pipelines Recomendados
~~~~~~~~~~~~~~~~~~~~~~~

**GitHub Actions:**

.. code-block:: yaml

   name: CI/CD
   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Setup DinD
           run: docker-compose up -d
         - name: Run tests
           run: docker exec dind ./run-tests.sh
         - name: Build and push
           run: docker exec dind ./build-and-push.sh

**Jenkins Pipeline:**

.. code-block:: groovy

   pipeline {
       agent any
       stages {
           stage('Setup') {
               steps {
                   sh 'docker-compose up -d'
               }
           }
           stage('Test') {
               steps {
                   sh 'docker exec dind ./run-tests.sh'
               }
           }
           stage('Deploy') {
               steps {
                   sh 'docker exec dind ./deploy.sh'
               }
           }
       }
       post {
           always {
               sh 'docker-compose down'
           }
       }
   }

Despliegue
----------

Estrategias de Despliegue
~~~~~~~~~~~~~~~~~~~~~~~~~

**Blue-Green Deployment:**

.. code-block:: bash

   # Crear nueva versión
   docker tag myapp:latest myapp:v2
   docker run -d --name myapp-v2 -p 8081:80 myapp:v2

   # Verificar health
   curl -f http://localhost:8081/health || exit 1

   # Cambiar tráfico (usando nginx o load balancer)
   # ...

   # Remover versión anterior
   docker stop myapp-v1
   docker rm myapp-v1

**Rolling Updates:**

.. code-block:: bash

   # Actualizar con zero-downtime
   docker-compose up -d --scale web=2
   docker-compose up -d --scale web=1

**Canary Releases:**

.. code-block:: bash

   # Desplegar 10% del tráfico a nueva versión
   docker run -d --name canary -p 8082:80 myapp:new-version

   # Configurar load balancer para 10% del tráfico
   # Monitorear métricas
   # Si OK, aumentar tráfico gradualmente

Backup y Recuperación
---------------------

Estrategias de Backup
~~~~~~~~~~~~~~~~~~~~~

**Backup de datos:**

.. code-block:: bash

   # Backup automático diario
   #!/bin/bash
   DATE=$(date +%Y%m%d_%H%M%S)
   docker run --rm -v myapp_data:/data -v $(pwd)/backups:/backup \\
     alpine tar czf /backup/backup_${DATE}.tar.gz -C / /data

**Backup de configuraciones:**

.. code-block:: bash

   # Backup de docker-compose y scripts
   tar czf config_backup.tar.gz docker-compose.yaml scripts/ docs/

**Backup de imágenes:**

.. code-block:: bash

   # Guardar imágenes importantes
   docker save myapp:latest > myapp_latest.tar

Recuperación de Desastres
~~~~~~~~~~~~~~~~~~~~~~~~~

**Plan de recuperación:**

1. **Identificar el problema**
2. **Aislar el sistema afectado**
3. **Restaurar desde backup**
4. **Verificar integridad**
5. **Reanudar operaciones**

**Tiempos objetivo:**

- RTO (Recovery Time Objective): 4 horas
- RPO (Recovery Point Objective): 1 hora

Documentación
-------------

Mantención de Documentación
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Actualización automática:**

.. code-block:: bash

   # Hook de pre-commit para verificar docs
   #!/bin/bash
   make docs
   git add docs/_build/

**Revisión de documentación:**

- Actualizar con cada cambio significativo
- Revisar ortografía y gramática
- Verificar enlaces rotos
- Mantener ejemplos actualizados

**Herramientas:**

- Sphinx para documentación técnica
- Read the Docs para hosting
- Markdown para documentación simple
- PlantUML para diagramas

Colaboración
------------

Trabajo en Equipo
~~~~~~~~~~~~~~~~~

**Roles definidos:**

- **DevOps:** Mantenimiento de infraestructura
- **Developers:** Desarrollo de aplicaciones
- **QA:** Testing y calidad
- **DevRel:** Documentación y soporte

**Comunicación:**

- Issues y PRs bien documentados
- Reuniones de sincronización regulares
- Canales de Slack/Teams para soporte rápido
- Wiki interna para conocimiento compartido

**Code Reviews:**

- Al menos 2 aprobaciones por PR
- Checklist de revisión estándar
- Tests obligatorios
- Documentación actualizada

Métricas y KPIs
---------------

Monitoreo de Éxito
~~~~~~~~~~~~~~~~~~

**Métricas técnicas:**

- Uptime del sistema: >99.9%
- Tiempo de respuesta: <500ms
- Tasa de error: <1%
- Cobertura de tests: >80%

**Métricas de negocio:**

- Tiempo de desarrollo reducido
- Menos bugs en producción
- Mayor satisfacción del equipo
- ROI positivo

**Monitoreo continuo:**

.. code-block:: bash

   # Dashboard de métricas
   docker run -d -p 3000:3000 \\
     -v prometheus_data:/var/lib/prometheus \\
     grafana/grafana

Conclusión
----------

Seguir estas mejores prácticas asegura que el entorno DinD sea:

- **Seguro:** Minimizando riesgos y vulnerabilidades
- **Escalable:** Capaz de crecer con las necesidades
- **Mantenible:** Fácil de actualizar y mantener
- **Productivo:** Maximizando la eficiencia del equipo

La implementación consistente de estas prácticas es la diferencia entre un entorno útil y uno problemático.