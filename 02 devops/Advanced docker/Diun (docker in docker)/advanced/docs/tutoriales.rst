Tutoriales
==========

.. _tutoriales:

Esta sección contiene tutoriales paso a paso para aprender a usar el entorno DinD.

Tutorial 1: Primera Aplicación
------------------------------

**Objetivo:** Crear y desplegar tu primera aplicación en el entorno DinD.

**Tiempo estimado:** 15 minutos

**Prerrequisitos:**

- Entorno DinD funcionando
- Acceso básico a terminal

Paso 1: Preparar el código
~~~~~~~~~~~~~~~~~~~~~~~~~~

Crea un directorio para tu aplicación:

.. code-block:: bash

   mkdir -p volumes/files/my-first-app
   cd volumes/files/my-first-app

Crea un archivo ``app.py``:

.. code-block:: python

   from flask import Flask
   import socket

   app = Flask(__name__)

   @app.route('/')
   def hello():
       hostname = socket.gethostname()
       return f'<h1>Hello from {hostname}!</h1><p>Running in DinD environment</p>'

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)

Crea ``requirements.txt``:

.. code-block::

   Flask==2.3.3

Crea ``Dockerfile``:

.. code-block:: dockerfile

   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY app.py .

   CMD ["python", "app.py"]

Paso 2: Construir la imagen
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dentro del contenedor DinD:

.. code-block:: bash

   cd /app
   docker build -t my-first-app .

Verifica que la imagen se creó:

.. code-block:: bash

   docker images

Paso 3: Ejecutar el contenedor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker run -d -p 5000:5000 --name my-app my-first-app

Paso 4: Verificar funcionamiento
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accede desde tu navegador: http://localhost:51080

Deberías ver: "Hello from [container-id]! Running in DinD environment"

Paso 5: Limpieza
~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker stop my-app
   docker rm my-app
   docker rmi my-first-app

¡Felicitaciones! Has completado tu primer tutorial.

Tutorial 2: Configuración de Red
---------------------------------

**Objetivo:** Aprender a configurar redes personalizadas entre contenedores.

**Tiempo estimado:** 20 minutos

Paso 1: Crear red personalizada
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker network create my-network

Paso 2: Ejecutar contenedores en la red
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Base de datos
   docker run -d --name db --network my-network -e POSTGRES_PASSWORD=secret postgres:13

   # Aplicación
   docker run -d --name web --network my-network -p 8080:80 nginx

Paso 3: Verificar conectividad
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Desde el contenedor web
   docker exec web ping db

   # Ver configuración de red
   docker network inspect my-network

Paso 4: Acceso desde host
~~~~~~~~~~~~~~~~~~~~~~~~~

La aplicación web estará disponible en: http://localhost:51080

Tutorial 3: Persistencia de Datos
---------------------------------

**Objetivo:** Configurar volúmenes persistentes para bases de datos.

**Tiempo estimado:** 25 minutos

Paso 1: Crear volumen nombrado
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker volume create my-db-data

Paso 2: Ejecutar base de datos con volumen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker run -d --name postgres \\
     -v my-db-data:/var/lib/postgresql/data \\
     -e POSTGRES_PASSWORD=mypassword \\
     -e POSTGRES_DB=mydb \\
     postgres:13

Paso 3: Crear datos de prueba
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker exec -it postgres psql -U postgres -d mydb -c "
   CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(50));
   INSERT INTO users (name) VALUES ('Alice'), ('Bob'), ('Charlie');
   "

Paso 4: Verificar persistencia
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Detener contenedor
   docker stop postgres

   # Reiniciar
   docker start postgres

   # Verificar datos
   docker exec -it postgres psql -U postgres -d mydb -c "SELECT * FROM users;"

Los datos deberían persistir.

Tutorial 4: Monitoreo Básico
----------------------------

**Objetivo:** Configurar monitoreo básico de contenedores.

**Tiempo estimado:** 30 minutos

Paso 1: Instalar herramientas de monitoreo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Dentro del contenedor DinD
   apk add htop iotop

Paso 2: Monitorear recursos
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Ver uso de CPU y memoria
   docker stats

   # Monitoreo interactivo
   htop

   # I/O de disco
   iotop

Paso 3: Logs de contenedores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Ver logs en tiempo real
   docker logs -f my-container

   # Logs con timestamps
   docker logs --timestamps my-container

Tutorial 5: Backup y Restauración
----------------------------------

**Objetivo:** Aprender a hacer backup y restaurar datos.

**Tiempo estimado:** 35 minutos

Paso 1: Crear datos para backup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Crear archivo de prueba
   echo "Important data" > volumes/files/important.txt

   # Ejecutar contenedor con datos
   docker run -d --name data-container -v $(pwd)/volumes/files:/data alpine sleep 3600

Paso 2: Hacer backup
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Backup del volumen
   docker run --rm -v $(pwd)/volumes/files:/source -v $(pwd)/backups:/backup \\
     alpine tar czf /backup/files_backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /source .

Paso 3: Simular pérdida de datos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # "Perder" datos
   rm volumes/files/important.txt

Paso 4: Restaurar desde backup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Encontrar el backup más reciente
   LATEST_BACKUP=$(ls -t backups/*.tar.gz | head -1)

   # Restaurar
   docker run --rm -v $(pwd)/volumes/files:/target -v $(pwd)/backups:/backup \\
     alpine tar xzf /backup/$(basename $LATEST_BACKUP) -C /target

Verifica que los datos se restauraron.

Tutorial Avanzado: CI/CD Pipeline
----------------------------------

**Objetivo:** Crear un pipeline completo de CI/CD.

**Tiempo estimado:** 45 minutos

Paso 1: Preparar aplicación
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Crea una aplicación Node.js con tests:

``package.json``:

.. code-block:: json

   {
     "name": "my-app",
     "scripts": {
       "test": "jest",
       "build": "echo 'Building...'",
       "start": "node server.js"
     },
     "devDependencies": {
       "jest": "^29.0.0"
     }
   }

``server.js``:

.. code-block:: javascript

   const http = require('http');
   http.createServer((req, res) => {
     res.end('Hello CI/CD!');
   }).listen(3000);

``__tests__/app.test.js``:

.. code-block:: javascript

   test('dummy test', () => {
     expect(1 + 1).toBe(2);
   });

Paso 2: Crear script de CI/CD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pipeline.sh``:

.. code-block:: bash

   #!/bin/bash
   set -e

   echo "=== CI/CD Pipeline Started ==="

   # Install dependencies
   npm install

   # Run tests
   npm test

   # Build
   npm run build

   # Build Docker image
   docker build -t my-app:$BUILD_NUMBER .

   # Run container
   docker run -d -p 3000:3000 --name my-app my-app:$BUILD_NUMBER

   # Health check
   sleep 5
   curl -f http://localhost:3000 || exit 1

   echo "=== Pipeline Completed Successfully ==="

Paso 3: Ejecutar pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd /app
   chmod +x pipeline.sh
   BUILD_NUMBER=$(date +%s) ./pipeline.sh

Paso 4: Verificar despliegue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accede a http://localhost:51080 para verificar que la aplicación funciona.

¡Has completado todos los tutoriales! Ahora tienes una base sólida para trabajar con Docker-in-Docker.