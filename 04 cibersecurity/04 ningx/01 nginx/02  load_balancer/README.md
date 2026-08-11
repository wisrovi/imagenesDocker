# 02 · Balanceo de carga

Balanceador nginx que reparte el tráfico entre 3 backends con round-robin.

## Servicios

| Servicio | Descripción |
|---|---|
| `server1`, `server2`, `server3` | Backends nginx que sirven `index.1/2/3.html` |
| `nginx` | Balanceador con upstream `backend` (puerto host 8080) |

## Archivos

- `config/default.conf` — upstream `backend` con los 3 servidores y modos de balanceo comentados (`least_conn`, `ip_hash`). El default es round-robin. Incluye rutas `/health`, `/cached` y `/nochached`.
- `config/nginx-custom.conf` — ajustes extra (`client_max_body_size 0`).
- `html/index.1|2|3.html` — página distinta por servidor para ver el reparto.

## Uso

```bash
make up        # levanta los 4 contenedores (puerto 8080)
make curl      # una petición al balanceador
make curl-loop # 10 peticiones -> se ve Server 1/2/3 repartidos
make health    # check /health
make logs      # logs en vivo
make down      # detiene los contenedores
```

## Verificar el balanceo

Cada servidor devuelve `Server 1`, `Server 2` o `Server 3` según a quién le tocó:

```bash
for i in $(seq 1 9); do curl -s localhost:8080/ | grep -o "Server [0-9]"; done
```

Resultado esperado (round-robin):

```
Server 1
Server 2
Server 3
Server 1
...
```

## Rutas

| Ruta | Descripción |
|---|---|
| `/` | Balanceado entre los 3 backends |
| `/health` | Devuelve `OK`, sin logs |
| `/cached` | Proxy a backend (config de cache comentada) |
| `/nochached` | Proxy a backend sin cache |
| `/serv1/` | Enruta directo a `server1` |
