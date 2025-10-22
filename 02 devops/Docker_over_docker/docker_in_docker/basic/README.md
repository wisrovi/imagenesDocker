# Docker-in-Docker Básico

Versión simplificada de Docker-in-Docker con SSH y Portainer.

## 🚀 Inicio Rápido

```bash
# Copiar configuración
cp .env.example .env

# Construir y ejecutar
docker-compose up -d

# Acceder
# Portainer: http://localhost:50421 (usuario: admin, contraseña: admin)
# SSH: ssh root@localhost -p 50422 (contraseña: password)
# Terminal Web: http://localhost:50423
```

## 📋 Servicios

- **Docker-in-Docker**: Entorno Docker anidado
- **SSH**: Acceso remoto al contenedor (puerto 50422)
- **Portainer**: Interfaz web para gestión de Docker (puerto 50421)
- **Terminal Web**: Terminal en el navegador via ttyd (puerto 50423)

## 🔧 Configuración

Edita `.env` para cambiar:
- `SSH_PASSWORD`: Contraseña para SSH (default: password)

## 🛠️ Uso

```bash
# Ver logs
docker-compose logs -f

# Acceder al contenedor
docker-compose exec dind-basic sh

# Detener
docker-compose down
```