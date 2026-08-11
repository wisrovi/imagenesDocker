# 04 · Certificados self-signed con OpenSSL

nginx sirviendo HTTPS con certificados autofirmados generados por OpenSSL.

## Servicios

| Servicio | Descripción |
|---|---|
| `openssl` | One-shot: genera los certs self-signed con `openssl req` |
| `server1` | Backend nginx que sirve la página |
| `nginx_ssl` | Proxy HTTPS con los certs (puertos host 8083 http / 8444 https) |

## Archivos

- `nginx/Dockerfile` — imagen Debian con OpenSSL.
- `nginx/conf/openssl_wisrovi.cnf` — config del cert (CN, SAN DNS para `www.vault.ecapturedtech.com`).
- `nginx/conf.d/default.conf` — server block 443 SSL + redirect 80→443.
- `html/index.html` — contenido servido por el backend.
- `nginx/certs/` — se genera en `make up`, no está en el repo.

## Uso

```bash
make up        # genera certs (openssl) y levanta nginx (https 8444)
make curl      # curl https://www.vault.ecapturedtech.com:8444 con -k
make curl-ssl  # igual que curl
make cert      # regenera los certificados
make logs      # logs en vivo
make down      # detiene los contenedores
```

El curl usa `-k` porque el certificado es self-signed (no hay CA que lo valide) y `--resolve` para apuntar el dominio a `127.0.0.1`.

## Notas

- `nginx_ssl` espera a que `openssl` termine (`condition: service_completed_successfully`) antes de arrancar.
- El cert se emite por 825 días (configurado en el comando de `docker-compose.yaml`).
