Instalación
===========

.. _instalacion:

Prerrequisitos del Sistema
--------------------------

Antes de proceder con la instalación, verifique que su sistema cumpla con los siguientes requisitos:

**Software Requerido:**

* **Docker Engine**: Versión 20.10.0 o superior

  .. code-block:: bash

     docker --version

* **Docker Compose**: Versión 1.29.0 o superior

  .. code-block:: bash

     docker-compose --version

**Requisitos de Hardware:**

* **Memoria RAM**: Mínimo 4GB, recomendado 8GB
* **Espacio en Disco**: Mínimo 10GB libres
* **CPU**: 2 núcleos o más (recomendado)

**Sistemas Operativos Soportados:**

* Linux (Ubuntu, CentOS, Debian, etc.)
* macOS (con Docker Desktop)
* Windows (con WSL2 + Docker Desktop)

Verificación de Prerrequisitos
------------------------------

Ejecute estos comandos para verificar su instalación:

.. code-block:: bash

   # Verificar Docker
   docker run hello-world

   # Verificar Docker Compose
   docker-compose --version

   # Verificar espacio en disco
   df -h

   # Verificar memoria
   free -h

Instalación Paso a Paso
-----------------------

Paso 1: Obtener el Proyecto
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone el repositorio o descargue los archivos:

.. code-block:: bash

   git clone https://github.com/your-repo/docker-dind-portainer.git
   cd docker-dind-portainer

O descargue el ZIP desde el repositorio.

Paso 2: Configuración Inicial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No se requiere configuración adicional. Los archivos están pre-configurados para funcionar out-of-the-box.

Sin embargo, puede personalizar:

* Puertos en ``docker-compose.yaml``
* Contraseña SSH en ``scripts/install_ssh.sh``
* Configuración de Portainer

Paso 3: Construir e Iniciar
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ejecute el comando principal:

.. code-block:: bash

   docker-compose up -d

Este comando:

1. **Descarga imágenes**: Obtiene ``docker:dind`` y dependencias
2. **Construye contenedor**: Crea la imagen personalizada con SSH y Portainer
3. **Configura volúmenes**: Crea directorios para persistencia de datos
4. **Inicia servicios**: Levanta el contenedor con todos los servicios

.. note::
   La primera ejecución puede tomar varios minutos debido a la descarga de imágenes.

Paso 4: Verificar Instalación
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compruebe que todo esté funcionando:

.. code-block:: bash

   # Ver estado de contenedores
   docker-compose ps

   # Ver logs
   docker-compose logs -f dind

Debería ver algo como:

.. code-block:: text

   NAME      IMAGE                   COMMAND                  SERVICE   STATUS
   dind      docker_in_docker-dind   "/usr/local/bin/star…"   dind      Up 2 minutes (healthy)

Acceso a los Servicios
----------------------

Una vez instalado, acceda a los servicios:

**Portainer (Interfaz Web):**

* URL: http://localhost:9003
* Usuario: admin (primera vez)
* Contraseña: Configure en la interfaz

**SSH (Acceso Terminal):**

.. code-block:: bash

   ssh root@localhost -p 50422
   Password: password

**Docker API (Interna):**

.. code-block:: bash

   export DOCKER_HOST=tcp://localhost:2375
   docker ps

Instalación Avanzada
--------------------

Modo Desarrollo
~~~~~~~~~~~~~~~

Para desarrollo, puede montar el código fuente:

.. code-block:: yaml

   volumes:
     - ./scripts:/app/scripts
     - ./docs:/app/docs

Modo Producción
~~~~~~~~~~~~~~~

Para producción, considere:

* Cambiar contraseñas por defecto
* Usar secrets de Docker
* Configurar logging persistente
* Implementar backups automáticos

Solución de Problemas en Instalación
------------------------------------

**Error: "Port already in use"**

.. code-block:: bash

   # Cambiar puertos en docker-compose.yaml
   ports:
     - "9004:9000"  # Portainer
     - "50423:50422"  # SSH

**Error: "Permission denied"**

.. code-block:: bash

   # Añadir usuario al grupo docker
   sudo usermod -aG docker $USER
   # Reiniciar sesión

**Error: "No space left on device"**

.. code-block:: bash

   # Limpiar Docker
   docker system prune -a
   # Liberar espacio en disco

Desinstalación
--------------

Para remover completamente:

.. code-block:: bash

   # Detener y remover contenedores
   docker-compose down -v

   # Remover imágenes
   docker rmi docker_in_docker-dind

   # Limpiar volúmenes (opcional)
   sudo rm -rf volumes/