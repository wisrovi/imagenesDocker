# Docker Registry con UI

Este proyecto configura un registro Docker local con una interfaz web para gestión.

## Servicios

- **Registry**: Puerto 40231 (basado en registry:2.8.2)
- **UI**: Puerto 40232 HTTPS (basado en joxit/docker-registry-ui:main, con nginx proxy)
- **Nginx**: Proxy SSL para UI con rate limiting

## Inicio

```bash
docker-compose up -d
```

## Autenticación

El registry requiere autenticación básica. Usuario: `wisrovi`, Contraseña: `nJ6OPitYMidApj8ebk4h`.

La interfaz web requiere login. Usuario: `registry`, Contraseña: `wisrovi`.

Para acceder al registry desde Docker CLI:
```bash
docker login localhost:40231
# Usuario: wisrovi, Contraseña: nJ6OPitYMidApj8ebk4h
```

## HTTP

El registry usa HTTP interno; la UI está en HTTP para simplicidad.



## Scripts de Prueba

- `scripts/push_image.sh <origen> <destino>`: Subir imagen al registry
- `scripts/pull_image.sh <imagen>`: Descargar imagen del registry
- `scripts/list_images.sh`: Listar imágenes en el registry
- `scripts/test_registry.sh`: Probar conectividad al registry
- `scripts/test_frontend.sh`: Probar funcionalidades del frontend
- `scripts/test_integration.sh`: Pruebas de integración (push, pull, list)
- `scripts/backup.sh`: Crear backup del registry-data

## Uso

1. Subir imágenes: `./scripts/push_image.sh hello-world localhost:40231/hello-world:v1`
2. Acceder a la UI: http://localhost (login: registry / password)
3. Gestionar imágenes desde la interfaz web

## Limpieza

```bash
docker-compose down
```

## Notas

- DELETE_IMAGES habilitado para borrar imágenes desde la UI
- Volúmenes persistentes en `./registry-data`
- Configuraciones en `.env`
- Health checks configurados para reinicio automático
- Backups automáticos con `scripts/backup.sh`