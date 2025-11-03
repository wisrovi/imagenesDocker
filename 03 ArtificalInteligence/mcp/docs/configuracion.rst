Configuración
=============

Variables de Entorno
--------------------

El proyecto utiliza variables de entorno para configurar el comportamiento del inspector. Estas se definen en el archivo ``.env``:

.. code-block:: bash

   CLIENT_PORT=6274
   SERVER_PORT=6277
   MCP_SERVER_REQUEST_TIMEOUT=300000
   MCP_REQUEST_TIMEOUT_RESET_ON_PROGRESS=true
   MCP_REQUEST_MAX_TOTAL_TIMEOUT=60000
   MCP_AUTO_OPEN_ENABLED=true

Descripción de Variables
------------------------

CLIENT_PORT
    Puerto para la interfaz web del inspector (por defecto: 6274)

SERVER_PORT
    Puerto para el servidor proxy (por defecto: 6277)

MCP_SERVER_REQUEST_TIMEOUT
    Timeout en milisegundos para solicitudes al servidor MCP (por defecto: 300000)

MCP_REQUEST_TIMEOUT_RESET_ON_PROGRESS
    Si se debe resetear el timeout cuando hay notificaciones de progreso (true/false)

MCP_REQUEST_MAX_TOTAL_TIMEOUT
    Timeout máximo total en milisegundos (por defecto: 60000)

MCP_AUTO_OPEN_ENABLED
    Si se debe abrir automáticamente el navegador al iniciar (true/false)

Configuración de Servidores MCP
-------------------------------

Para configurar servidores MCP personalizados, utiliza el archivo ``sample-config.json`` como base:

.. code-block:: json

   {
     "mcpServers": {
       "mi-servidor": {
         "command": "node",
         "args": ["build/index.js", "--debug"],
         "env": {
           "API_KEY": "tu-api-key",
           "DEBUG": "true"
         }
       }
     }
   }

Tipos de Transporte
-------------------

El inspector soporta diferentes tipos de transporte:

STDIO
    Comunicación a través de entrada/salida estándar

SSE (Server-Sent Events)
    Eventos enviados por el servidor

Streamable HTTP
    Comunicación HTTP con streaming

Configuración Avanzada
-----------------------

Para configuraciones más avanzadas, consulta la documentación oficial del MCP Inspector en https://modelcontextprotocol.io/docs/tools/inspector