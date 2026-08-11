# 04 · Certificados con Let's Encrypt

nginx sirviendo HTTPS con certificados reales de Let's Encrypt.

## Servicios

| Servicio | Descripción |
|---|---|
| `server1` | Backend nginx que sirve la página |
| `nginx_ssl` | Proxy HTTPS con los certs (puertos host 8082 http / 8443 https) |
| `certbot` | One-shot: renueva los certificados |

## Archivos

- `certs/` — certificados de Let's Encrypt (`fullchain.pem`, `key.pem`, etc.) para `wisrovi.duckdns.org`.
- `nginx/conf.d/default.conf` — server block 443 SSL + redirect 80→443.
- `html/index.html` — contenido servido por el backend.

## Uso

```bash
make up        # levanta los servicios (https 8443)
make curl      # curl https://wisrovi.duckdns.org:8443 validando el certificado
make curl-ssl  # igual que curl
make curl-http # http:8082 -> redirige a https
make certbot   # renueva los certificados
make logs      # logs en vivo
make down      # detiene los contenedores
```

El curl usa `--resolve wisrovi.duckdns.org:8443:127.0.0.1` para probar en local apuntando el dominio al host sin tocar DNS.

## Notas

- El certificado se valida (sin `-k`) porque es un cert real de Let's Encrypt.
- Para renovar: `make certbot` ejecuta `docker compose run --rm certbot renew`.
