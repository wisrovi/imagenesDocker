# 07 · Balanceo de carga con Nginx

Colección de setups de **nginx** para balanceo de carga, autenticación y SSL/TLS, cada uno en su propia carpeta aislada e independiente (compilan y corren por separado, sin pisarse).

## Estructura

| Carpeta | Propósito | HTTP | HTTPS |
|---|---|---|---|
| [01 authentication](01%20authentication/README.md) | Proxy nginx con basic auth (htpasswd) | 8081 | — |
| [02 load_balancer](02%20load_balancer/README.md) | Balanceador round-robin entre 3 backends | 8080 | — |
| [04 letsencrypt](04%20letsencrypt/README.md) | HTTPS con certificados reales de Let's Encrypt | 8082 | 8443 |
| [04 nginx_with_openssl](04%20nginx_with_openssl/README.md) | HTTPS con certificados self-signed (OpenSSL) | 8083 | 8444 |
| [05 advance](05%20advance/README.md) | Auto-detección: jwilder/nginx-proxy + letsencrypt companion | 8084 | 8445 |

## Requisitos

- Docker + Docker Compose v2
- `make`

## Cómo usar cada carpeta

Cada carpeta es autónoma: entra, levanta y prueba. Todas exponen los mismos objetivos de `Makefile`:

```bash
cd "02  load_balancer"
make up        # levanta los contenedores
make curl      # petición al server (por SSL cuando aplica)
make logs      # logs en vivo
make down      # detiene todo
```

Para conocer los objetivos de una carpeta:

```bash
make help
```

## Probar varias a la vez

Cada carpeta usa `container_name` y puertos host únicos, así que pueden correr en paralelo sin conflictos.

## Curl por SSL

Donde aplica SSL (`04 letsencrypt`, `04 nginx_with_openssl`, `05 advance`), el `make curl` hace una petición HTTPS:

- **letsencrypt / advance** → curl validando el certificado real (`--resolve dominio:puerto:127.0.0.1`).
- **nginx_with_openssl** → curl con `-k` (cert self-signed).
