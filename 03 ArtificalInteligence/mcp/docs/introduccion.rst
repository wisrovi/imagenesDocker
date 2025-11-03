Introducción
============

El MCP Inspector es una herramienta de desarrollo para probar y depurar servidores MCP (Model Context Protocol). Este proyecto proporciona una configuración Docker Compose simplificada para ejecutar el inspector de manera local.

¿Qué es MCP?
------------

El Model Context Protocol (MCP) es un protocolo abierto que permite conectar aplicaciones de IA a sistemas externos de manera estandarizada. Facilita la integración con:

- Fuentes de datos (bases de datos, archivos locales)
- Herramientas (motores de búsqueda, calculadoras)
- Flujos de trabajo (prompts especializados)

Arquitectura del Inspector
--------------------------

El MCP Inspector consta de dos componentes principales:

1. **Cliente MCP Inspector (MCPI)**: Interfaz web React para interactuar con servidores MCP
2. **Proxy MCP (MCPP)**: Servidor Node.js que actúa como puente entre la UI y los servidores MCP

Beneficios
----------

- **Desarrollo simplificado**: Reduce el tiempo y complejidad al construir o integrar aplicaciones de IA
- **Ecosistema amplio**: Acceso a un catálogo de fuentes de datos, herramientas y aplicaciones
- **Mejor experiencia**: Mejora las capacidades y la experiencia del usuario final