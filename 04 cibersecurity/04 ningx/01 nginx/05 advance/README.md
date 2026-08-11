# 05 · Avanzado: nginx-proxy + Let's Encrypt (auto-detección)

`jwilder/nginx-proxy` + `jrcs/letsencrypt-nginx-proxy-companion`: el proxy detecta los contenedores automáticamente vía variables `VIRTUAL_HOST` y emite/renueva certificados Let's Encrypt solo.

## Servicios

| Servicio | Descripción |
|---|---|
| `nginx-proxy` | Reverse proxy que auto-configura vhosts según labels/env (puertos host 8084 http / 8445 https) |
| `letsencrypt` | Companion que emite y renueva certs automáticamente |
| `server1` | Backend que expone `VIRTUAL_HOST`/`LETSENCRYPT_HOST` |

## Cómo funciona

- `nginx-proxy` monta `/var/run/docker.sock` y detecta contenedores con `VIRTUAL_HOST`.
- `server1` define `VIRTUAL_HOST=wisrovi.duckdns.org`, `LETSENCRYPT_HOST` y `LETSENCRYPT_EMAIL`.
- `letsencrypt` (companion) genera el certificado para ese host y nginx-proxy lo sirve por HTTPS.

## Uso

```bash
make up        # levanta proxy + letsencrypt + server1 (https 8445)
make curl      # curl https://wisrovi.duckdns.org:8445 validando el certificado
make curl-ssl  # igual que curl
make curl-http # http:8084 -> redirige a https
make logs      # logs en vivo (útil para ver la emisión del cert)
make down      # detiene los contenedores
```

## Notas

- Los certs se guardan en `certs/` (jwilder layout: `DOMINIO.crt`, `DOMINIO.key`, etc.).
- `vhost.d/` permite sobrescribir configs por vhost.
- `config/default.conf` y `config/nginx-custom.conf` se montan como configs globales.
- Al primer arranque el cert puede tardar unos segundos en emitirse; revisa `make logs`.
