# 01 · Autenticación (Basic Auth)

Proxy nginx con autenticación básica (usuario/contraseña) usando `htpasswd`.

## Servicios

| Servicio | Descripción |
|---|---|
| `server1` | Backend nginx que sirve la página |
| `nginx_auth` | Proxy nginx que exige credenciales (puerto host 8081) |

## Archivos

- `auth/htpasswdp` — archivo de usuarios/contraseñas (BCrypt). Generar con `auth/generar.sh`.
- `auth/generar.sh` — agrega un usuario al `htpasswdp`: `htpasswd -Bbn admin 12345678 >> htpasswdp`
- `config/default.conf` — server block con `auth_basic` y `auth_basic_user_file`; `/public` y `/health` sin auth.
- `html/index.html` — contenido servido por el backend.

## Uso

```bash
make up        # levanta los servicios (puerto 8081)
make curl      # petición con credenciales -> 200
make curl-noauth  # sin credenciales -> 401
make curl-auth    # igual que curl
make logs      # logs en vivo
make down      # detiene los contenedores
```

## Rutas

| Ruta | Acceso |
|---|---|
| `/` | Requiere auth (401 sin credenciales) |
| `/public` | Abierta |
| `/health` | Abierta, devuelve `OK` |

## Credenciales de ejemplo

```
usuario: admin
password: 12345678
```
