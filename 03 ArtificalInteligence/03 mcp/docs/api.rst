API Reference
=============

Esta sección documenta la API del MCP Inspector cuando se ejecuta en modo servidor.

Endpoints
---------

/health
    Verifica el estado del servicio

    **Método**: GET

    **Respuesta**: JSON con estado del servicio

/sse
    Endpoint para conexiones Server-Sent Events

    **Método**: GET

    **Parámetros**: token (opcional para autenticación)

/mcp
    Endpoint para conexiones Streamable HTTP

    **Método**: POST

    **Headers**: Authorization (Bearer token)

Configuración Programática
---------------------------

Variables de Entorno
^^^^^^^^^^^^^^^^^^^^

CLIENT_PORT
    Puerto del cliente web

SERVER_PORT
    Puerto del servidor proxy

HOST
    Dirección de enlace (por defecto: localhost)

MCP_PROXY_AUTH_TOKEN
    Token de autenticación para el proxy

DANGEROUSLY_OMIT_AUTH
    Deshabilitar autenticación (NO RECOMENDADO)

Configuración de Timeouts
^^^^^^^^^^^^^^^^^^^^^^^^^

MCP_SERVER_REQUEST_TIMEOUT
    Timeout para solicitudes individuales

MCP_REQUEST_MAX_TOTAL_TIMEOUT
    Timeout máximo total

MCP_REQUEST_TIMEOUT_RESET_ON_PROGRESS
    Reset timeout en progreso

Ejemplos de Uso Programático
-----------------------------

Conexión STDIO
^^^^^^^^^^^^^^^

.. code-block:: javascript

   const { spawn } = require('child_process');

   const server = spawn('node', ['build/index.js'], {
     stdio: ['pipe', 'pipe', 'pipe']
   });

Conexión SSE
^^^^^^^^^^^^

.. code-block:: javascript

   const eventSource = new EventSource('http://localhost:6277/sse?token=your-token');

Conexión Streamable HTTP
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: javascript

   const response = await fetch('http://localhost:6277/mcp', {
     method: 'POST',
     headers: {
       'Authorization': 'Bearer your-token',
       'Content-Type': 'application/json'
     },
     body: JSON.stringify(request)
   });

Códigos de Error
----------------

400 Bad Request
    Solicitud malformada

401 Unauthorized
    Token de autenticación inválido o faltante

404 Not Found
    Endpoint no encontrado

500 Internal Server Error
    Error interno del servidor

Seguridad
---------

Consideraciones de Seguridad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- El proxy puede ejecutar procesos locales
- No exponer a redes no confiables
- Usar autenticación cuando sea posible
- Mantener tokens seguros

Protecciones Implementadas
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Autenticación por token
- Enlace solo a localhost por defecto
- Validación de origen para DNS rebinding
- Timeouts configurables