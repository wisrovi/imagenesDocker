# Usage & Test Documentation

## 1. Test Execution Evidence

This section documents the verification tests performed to validate the BunkerWeb deployment.

### 1.1 Service Health Verification

```bash
# Command: docker compose ps
# Result: All services running with healthy status

NAME                         IMAGE                                 SERVICE        CREATED         STATUS                   PORTS
03bunkerweb-bunkerweb-1      bunkerity/bunkerweb:1.6.9             bunkerweb      8 minutes ago   Up 8 minutes (healthy)   80/tcp, 0.0.0.0:80->8080/tcp
03bunkerweb-bw-db-1          mariadb:11                            bw-db          8 minutes ago   Up 8 minutes             3306/tcp
03bunkerweb-bw-scheduler-1   bunkerity/bunkerweb-scheduler:1.6.9   bw-scheduler   8 minutes ago   Up 8 minutes (healthy)   
03bunkerweb-bw-ui-1          bunkerity/bunkerweb-ui:1.6.9          bw-ui          8 minutes ago   Up 8 minutes (healthy)   0.0.0.0:7000->7000/tcp
03bunkerweb-redis-1          redis:8-alpine                        redis          8 minutes ago   Up 8 minutes             6379/tcp
```

### 1.2 HTTP Endpoint Test

```bash
# Command: curl -s -o /dev/null -w "HTTP %{http_code}" http://localhost:80
# Result: HTTP 200
```

### 1.3 HTTPS Endpoint Test

```bash
# Command: curl -sk -o /dev/null -w "HTTP %{http_code}" https://localhost:443
# Result: HTTP 200
```

### 1.4 UI Endpoint Test

```bash
# Command: curl -s -o /dev/null -w "HTTP %{http_code}" http://localhost:7000
# Result: HTTP 303 (Redirect to setup/login)
```

### 1.5 HTTP Response Content

```bash
# Command: curl -s http://localhost:80
# Result: BunkerWeb default page HTML
```

```html
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BunkerWeb</title>
    <meta name="description" content="Nothing to see here...">
</head>
<body>
    <article><h3>Nothing to see here ...</h3></article>
    <footer>
        <a href="https://www.bunkerweb.io/">BunkerWeb</a>
    </footer>
</body>
</html>
```

### 1.6 WAF API Health Check

```bash
# Command: docker compose logs bunkerweb --tail=30
# Evidence: Health check calls from scheduler

2026/03/24 05:26:09 [notice] [API] validated access from IP 10.20.30.5
2026/03/24 05:26:09 [notice] [API] successful call from IP 10.20.30.5 on /health : ok
bunkerweb-1  | bwapi 10.20.30.5 - GET /health HTTP/1.1 200 43
```

## 2. Usage Guide

### 2.1 Starting the Stack

```bash
# Navigate to deployment directory
cd "/home/wisrovi/Documentos/imagenesDocker/04 cibersecurity/04 ningx/03 bunkerweb"

# Start all services
docker compose up -d

# Verify deployment
docker compose ps
```

### 2.2 Stopping the Stack

```bash
# Stop all services (preserves data)
docker compose down

# Stop and remove volumes (data loss)
docker compose down -v
```

### 2.3 Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f bunkerweb
docker compose logs -f bw-ui
docker compose logs -f bw-scheduler
docker compose logs -f bw-db
docker compose logs -f redis
```

### 2.4 Service Management

```bash
# Restart specific service
docker compose restart bw-ui

# Rebuild and restart
docker compose up -d --build

# View resource usage
docker stats
```

### 2.5 Accessing Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Web UI | http://localhost:7000 | Setup on first access |
| HTTP Proxy | http://localhost:80 | N/A |
| HTTPS Proxy | https://localhost:443 | Accept self-signed cert |

## 3. Configuration Tests

### 3.1 Testing Threat Blocking

```bash
# Test SQL injection detection
curl -s "http://localhost:80/?id=1' OR '1'='1"

# Test XSS detection  
curl -s "http://localhost:80/?q=<script>alert(1)</script>"

# Both should return 403 Forbidden
```

### 3.2 Testing API Access

```bash
# Internal health check (from within container network)
docker compose exec bunkerweb wget -q -O- http://localhost:5000/health

# Expected response: {"status": "ok", "version": "1.6.9"}
```

### 3.3 Testing Database Connection

```bash
# Check MariaDB connectivity
docker compose exec bw-db mysql -u bunkerweb -pchangeme -e "SELECT VERSION();"

# Expected: MariaDB version string
```

### 3.4 Testing Redis Connection

```bash
# Test Redis ping
docker compose exec redis redis-cli ping

# Expected: PONG
```

## 4. Integration Tests

### 4.1 End-to-End Request Flow

```bash
# 1. Client request to WAF
curl -v http://localhost:80 2>&1 | head -20

# Response should include:
# - HTTP/1.1 200 OK
# - Server: nginx (BunkerWeb)
# - X-BunkerWeb header
```

### 4.2 Scheduler Configuration Push

```bash
# Check scheduler logs for configuration updates
docker compose logs bw-scheduler --tail=50

# Expected entries:
# - [SCHEDULER] [ℹ️ ] - Generator successfully executed
# - [API.CALLER] [ℹ️ ] - Successfully sent API request to http://bunkerweb:5000/confs
```

### 4.3 UI Dashboard Access

```bash
# Access login page
curl -s http://localhost:7000/login | grep -o "<title>.*</title>"

# Expected: <title>Login - BunkerWeb</title>
```

## 5. Troubleshooting Tests

### 5.1 Container Health Issues

```bash
# Check container health status
docker inspect --format='{{.State.Health.Status}}' 03bunkerweb-bunkerweb-1

# Check failed health checks
docker inspect --format='{{.LastHealthFailure}}' 03bunkerweb-bunkerweb-1
```

### 5.2 Network Connectivity

```bash
# Test inter-container connectivity
docker compose exec bunkerweb ping -c 3 bw-db
docker compose exec bunkerweb ping -c 3 redis
docker compose exec bw-ui ping -c 3 bunkerweb
```

### 5.3 Port Binding Verification

```bash
# Check exposed ports
docker port 03bunkerweb-bunkerweb-1
docker port 03bunkerweb-bw-ui-1

# Verify with netstat
netstat -tlnp | grep -E "80|443|7000"
```

## 6. Performance Tests

### 6.1 Concurrent Request Handling

```bash
# Test with multiple concurrent requests
for i in {1..50}; do
    curl -s -o /dev/null http://localhost:80 &
done
wait

# All should return HTTP 200
```

### 6.2 Request Rate Limiting

```bash
# Send rapid requests to test rate limiting
time for i in {1..100}; do
    curl -s -o /dev/null http://localhost:80
done

# Check logs for rate limit entries
docker compose logs bunkerweb | grep -i "rate"
```

## 7. Backup and Restore Tests

### 7.1 Database Backup

```bash
# Create database backup
docker run --rm \
    -v 03bunkerweb_bw-data:/data \
    -v $(pwd):/backup \
    busybox tar czf /backup/bw-db-backup.tar.gz /data

# Verify backup file
ls -lh bw-db-backup.tar.gz
```

### 7.2 Configuration Export

```bash
# Export current configuration
docker compose exec bw-db mysqldump -u bunkerweb -pchangeme db > config_backup.sql
```
