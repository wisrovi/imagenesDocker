Uso
===

Inicio Rápido
-------------

1. Verifica los prerrequisitos:

   .. code-block:: bash

      make check

2. Inicia el inspector:

   .. code-block:: bash

      make run

3. Accede a la interfaz web en http://localhost:6274

Comandos Disponibles
--------------------

El proyecto incluye varios comandos para facilitar su uso:

make check
    Verifica que Docker y Docker Compose estén instalados y funcionando

make run
    Ejecuta el inspector con eliminación automática del contenedor

make up
    Ejecuta el inspector en segundo plano

make down
    Detiene los contenedores en ejecución

make logs
    Muestra los logs en tiempo real

make clean
    Limpia contenedores, imágenes y volúmenes no utilizados

Uso con Configuración Personalizada
------------------------------------

Para usar una configuración personalizada de servidores MCP:

.. code-block:: bash

   docker-compose run --rm -v $(pwd)/mi-config.json:/app/config.json mcp-inspector --config /app/config.json --server mi-servidor

Interfaz Web
------------

La interfaz web proporciona:

- **Configuración del servidor**: Selecciona el tipo de transporte y configura parámetros
- **Lista de herramientas**: Visualiza las herramientas disponibles en el servidor
- **Lista de recursos**: Explora los recursos expuestos por el servidor
- **Lista de prompts**: Gestiona los prompts disponibles
- **Historial de solicitudes**: Revisa las interacciones anteriores
- **Exportación de configuración**: Genera archivos de configuración para otros clientes

Modo CLI
---------

El inspector también puede usarse desde la línea de comandos:

.. code-block:: bash

   docker-compose run --rm mcp-inspector --cli node build/index.js --method tools/list

Solución de Problemas
---------------------

Problemas Comunes
^^^^^^^^^^^^^^^^^

**El contenedor no inicia**
    Verifica que los puertos 6274 y 6277 estén disponibles

**No puedo acceder a la interfaz web**
    Comprueba que el contenedor esté ejecutándose con ``make logs``

**Error de conexión con servidor MCP**
    Verifica la configuración del servidor y el tipo de transporte

**Timeout en solicitudes**
    Ajusta las variables de timeout en el archivo ``.env``

Logs y Depuración
^^^^^^^^^^^^^^^^^

Para ver los logs detallados:

.. code-block:: bash

   make logs

Los logs incluyen información sobre:
- Inicio del servidor proxy
- Conexiones de clientes
- Errores de comunicación
- Tokens de autenticación