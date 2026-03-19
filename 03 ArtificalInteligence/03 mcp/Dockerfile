# Dockerfile personalizado para MCP Inspector (opcional)
# Útil para desarrollo o builds personalizados

FROM ghcr.io/modelcontextprotocol/inspector:latest

# Etiquetas
LABEL maintainer="William Rodriguez"
LABEL description="MCP Inspector con configuraciones personalizadas"
LABEL version="1.0.0"

# Variables de entorno por defecto
ENV CLIENT_PORT=6274
ENV SERVER_PORT=6277
ENV MCP_SERVER_REQUEST_TIMEOUT=300000
ENV MCP_AUTO_OPEN_ENABLED=true

# Puerto expuesto
EXPOSE 6274 6277

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${CLIENT_PORT} || exit 1

# Comando por defecto
CMD ["npm", "start"]