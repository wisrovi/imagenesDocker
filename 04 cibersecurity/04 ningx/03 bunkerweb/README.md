# BunkerWeb WAF Deployment

Enterprise-grade Web Application Firewall (WAF) built on Nginx, engineered to protect web applications from common attack vectors including SQL injection, cross-site scripting (XSS), DDoS attempts, brute-force attacks, and other OWASP Top 10 vulnerabilities.

## Overview

This repository contains a production-ready Docker Compose configuration for deploying BunkerWeb, an open-source WAF solution developed by bunkerity. The deployment architecture encompasses a complete security stack with MariaDB for persistent configuration storage, Redis for high-performance caching and session management, a web-based administration UI, and an automated scheduler for security list updates.

## Key Features

- **Web Application Firewall**: Real-time traffic filtering, threat detection, and mitigation
- **Modular Plugin Architecture**: Extensible plugin system for custom security policies
- **Automated Blacklist Management**: Scheduled IP and user-agent blocklist updates from multiple threat intelligence sources
- **Web Administration Dashboard**: Visual interface for configuration, monitoring, and analytics
- **Multi-site Support**: Manage multiple protected sites from a unified control plane
- **Integrated Health Monitoring**: Built-in health checks with automated recovery for all components
- **QUIC/HTTP3 Protocol Support**: Modern transport layer for reduced latency and improved performance
- **Database-driven Configuration**: Persistent settings via MariaDB with automatic migrations
- **Redis Session Backend**: High-performance distributed session storage

## Technical Stack

| Component | Technology | Version |
|-----------|------------|---------|
| WAF Engine | BunkerWeb | 1.6.9 |
| Reverse Proxy | Nginx | Latest (via BunkerWeb) |
| Database | MariaDB | 11 |
| Cache & Sessions | Redis | 8-alpine |
| Administration UI | BunkerWeb UI | 1.6.9 |
| Job Scheduler | BunkerWeb Scheduler | 1.6.9 |
| Container Runtime | Docker Compose | v2 |

### Infrastructure Technologies

- **Container Orchestration**: Docker with custom network isolation
- **Load Balancing**: Integrated Nginx with upstream health checking
- **Logging**: Containerized logging with structured output
- **Security Scanning**: Tripwire integration for file integrity monitoring

---

## 1. 🚶 Diagram Walkthrough

High-level execution flow showing how a client request traverses the BunkerWeb security stack.

```mermaid
flowchart TD
    A[Client Request] --> B{Protocol}
    B -->|HTTP| C[Port 80]
    B -->|HTTPS| D[Port 443]
    
    C --> E[BunkerWeb WAF]
    D --> E
    
    E --> F{Security Check}
    
    F -->|Threat Detected| G[403 Blocked]
    F -->|Valid Request| H[ModSecurity Rules]
    
    H --> I{Rule Match}
    I -->|Match| J[Block Request]
    I -->|No Match| K[Forward to Backend]
    
    G --> L[Log to Redis]
    J --> L
    K --> M[Backend Application]
    
    M --> N[Response]
    N --> A
    
    subgraph "Scheduler Loop"
    S[Scheduler] --> S1[Download Blacklists]
    S1 --> S2[Parse IP Lists]
    S2 --> S3[Push to WAF]
    S3 --> S
    end
    
    E -.-> S
```

---

## 2. 🗺️ System Workflow

Detailed sequence diagram showing the interaction between components during request processing and administrative operations.

```mermaid
sequenceDiagram
    participant Client as Client Browser
    participant WAF as BunkerWeb<br/>:80, :443
    participant Redis as Redis<br/>:6379
    participant DB as MariaDB<br/>:3306
    participant Scheduler as Scheduler<br/>bg-blacklist
    participant UI as Web UI<br/>:7000

    Note over Client,WAF: === REQUEST PHASE ===
    Client->>WAF: GET /protected-resource
    
    rect rgb(40, 40, 40)
        Note over WAF: ModSecurity Inspection
        alt Attack Detected
            WAF->>Client: 403 Forbidden
            WAF->>Redis: INCR blocked_count
            WAF->>Redis: SET blocked:{ip} timestamp
        else Clean Request
            WAF->>WAF: Apply security rules
            WAF->>WAF: Log access
        end
    end
    
    Note over Client,WAF: === SCHEDULER PHASE ===
    loop Every 60 seconds
        Scheduler->>Redis: GET last_update
        Scheduler->>Scheduler: Check if update needed
        Scheduler->>External: Download blacklists
        Scheduler->>External: Fetch user-agent lists
        Scheduler->>DB: Read API whitelist config
        Scheduler->>WAF: POST /confs (new rules)
        WAF->>WAF: Reload Nginx config
        WAF->>Scheduler: 200 OK
        Scheduler->>Redis: SET last_update timestamp
    end
    
    Note over Client,UI: === ADMIN PHASE ===
    Client->>UI: GET /login
    UI->>Redis: GET session
    Redis->>UI: Return session data
    UI->>Client: Login Form
    
    Client->>UI: POST /login (credentials)
    UI->>DB: Validate credentials
    DB->>UI: User authenticated
    UI->>Redis: CREATE session
    UI->>Client: Redirect /home
    
    Client->>UI: GET /home (with session)
    UI->>DB: FETCH settings
    UI->>UI: Render dashboard
    UI->>Client: Dashboard HTML
```

---

## 3. 🏗️ Architecture Components

Static structure diagram showing the main modules, services, and their dependencies.

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
        API[API Client]
    end

    subgraph "BunkerWeb Security Stack"
        WAF[BunkerWeb Core<br/>WAF Engine]
        API_Server[API Server<br/>:5000]
        ModSec[ModSecurity<br/>Rules Engine]
        Nginx[Nginx<br/>Reverse Proxy]
    end

    subgraph "Scheduler Layer"
        Scheduler[Scheduler<br/>Job Executor]
        Blacklist[Blacklist<br/>Downloader]
        ConfigGen[Config<br/>Generator]
    end

    subgraph "UI Layer"
        WebUI[Web Administration<br/>:7000]
        Flask[Flask Application]
        Gunicorn[Gunicorn<br/>WSGI]
    end

    subgraph "Data Layer"
        Redis[Redis<br/>Cache/Sessions]
        MariaDB[MariaDB<br/>Configuration]
    end

    subgraph "Output Layer"
        Logs[Container Logs]
        Stats[Statistics]
    end

    Browser --> WAF
    API --> WAF
    
    WAF --> Nginx
    Nginx --> ModSec
    ModSec --> API_Server
    
    Scheduler --> Blacklist
    Scheduler --> ConfigGen
    Blacklist --> MariaDB
    ConfigGen --> WAF
    
    WebUI --> Flask
    Flask --> Gunicorn
    Gunicorn --> MariaDB
    Gunicorn --> Redis
    
    WAF --> Redis
    WAF --> MariaDB
    WAF --> Logs
    Scheduler --> Logs
    WebUI --> Stats
```

---

## 4. ⚙️ Container Lifecycle

### 4a. Build Process

The BunkerWeb images are pre-built and pulled from Docker Hub. The build process involves:

1. **Image Pull**: `docker pull bunkerity/bunkerweb:1.6.9`
   - Base image with Nginx + ModSecurity + Python
   - Pre-installed security plugins
   - Health check scripts included

2. **Volume Initialization**:
   - `bw-data`: MariaDB database files
   - `bw-storage`: Cache, backups, and configuration
   - `redis-data`: Redis persistence (AOF + RDB)

3. **Network Creation**:
   - `bw-universe` (10.20.30.0/24): Main communication network
   - `bw-services`: Backend application connectivity
   - `bw-db`: Isolated database network

### 4b. Runtime Process

Each container follows a specific initialization sequence:

| Service | Startup Sequence |
|---------|------------------|
| **bw-db** | 1. Init MariaDB → 2. Create database → 3. Create user → 4. Ready |
| **redis** | 1. Load config → 2. Start Redis server → 3. Enable AOF → 4. Ready |
| **bunkerweb** | 1. Validate config → 2. Start Nginx → 3. Health check endpoint → 4. Ready |
| **bw-scheduler** | 1. Connect to DB → 2. Connect to Redis → 3. Load plugins → 4. Start cron → 5. Ready |
| **bw-ui** | 1. Temp UI startup → 2. DB migration → 3. Generate secrets → 4. Main UI → 5. Ready |

```mermaid
stateDiagram-v2
    [*] --> Pulled
    Pulled --> Created: docker compose create
    Created --> Starting: docker compose start
    Starting --> Running: Container health OK
    Running --> Healthy: Healthcheck passes
    Healthy --> Restarting: Failure detected
    Restarting --> Running: Container restarts
    Running --> Stopping: docker compose stop
    Stopping --> Stopped: Container stopped
    Stopped --> Starting: docker compose start
    Stopped --> [*]: docker compose down
```

---

## 5. 📂 File-by-File Guide

| File/Folder | Purpose | Description |
|-------------|---------|-------------|
| `docker-compose.yaml` | Main Configuration | Complete deployment configuration with all 5 services, networks, and volumes |
| `README.md` | Documentation | This technical documentation file |
| `example/` | Integration Examples | Folder containing example configurations |
| `example/docker-compose.yaml` | GPU API Example | Sample configuration for GPU-enabled API service |

### Configuration Details

| Element | Type | Purpose |
|---------|------|---------|
| `x-bw-env` | YAML Anchor | Shared environment variables for all services |
| `services.bunkerweb` | Service | Main WAF engine container |
| `services.bw-scheduler` | Service | Background job scheduler for updates |
| `services.bw-ui` | Service | Web administration interface |
| `services.bw-db` | Service | MariaDB database container |
| `services.redis` | Service | Redis cache container |
| `volumes.bw-data` | Volume | Persistent MariaDB data |
| `volumes.bw-storage` | Volume | Cache and backup storage |
| `volumes.redis-data` | Volume | Redis persistence |
| `networks.bw-universe` | Network | Main inter-service network (10.20.30.0/24) |
| `networks.bw-services` | Network | Backend application network |
| `networks.bw-db` | Network | Isolated database network |

---

## Architecture & Workflow

### Network Topology

```mermaid
graph TB
    subgraph "External Zone"
        Client["Client Traffic<br/>HTTP/HTTPS"]
    end
    
    subgraph "bw-universe (10.20.30.0/24)"
        BW["BunkerWeb WAF<br/>:80, :443"]
        Scheduler["Scheduler<br/>Blacklist Updates"]
        UI["Web UI<br/>:7000"]
        Redis["Redis<br/>:6379"]
    end
    
    subgraph "bw-db"
        DB["MariaDB<br/>:3306"]
    end
    
    subgraph "bw-services"
        App["Backend Application"]
    end
    
    Client --> BW
    BW --> App
    BW --> Redis
    Scheduler --> BW
    Scheduler --> DB
    UI --> DB
    UI --> Redis
```

### Request Processing Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant W as BunkerWeb WAF
    participant S as Scheduler
    participant D as MariaDB
    participant R as Redis
    participant U as Web UI
    
    Note over C,W: Incoming HTTP/HTTPS Request
    C->>W: Request arrives on :80 or :443
    
    Note over W: ModSecurity Inspection
    alt Threat Detected
        W->>C: 403 Forbidden Blocked
        W->>R: Log blocked request
    else Valid Request
        W->>W: Apply security rules
        W->>App: Forward to backend
        
        Note over S: Periodic security updates
        S->>D: Fetch config updates
        S->>S: Download blacklists
        S->>W: Push new rules
        S->>R: Update cache
    end
    
    Note over U: Administrative access
    U->>D: Read/Write settings
    U->>R: Session management
```

## File Structure

```
03 bunkerweb/
├── docker-compose.yaml          # Main deployment configuration
├── README.md                    # This documentation
└── example/
    └── docker-compose.yaml      # GPU API integration example
```

## Installation & Setup

### Prerequisites

| Requirement | Minimum Version |
|-------------|-----------------|
| Docker Engine | 20.10+ |
| Docker Compose | v2 |
| System RAM | 4GB |
| Disk Space | 20GB |

### Deployment Steps

1. **Navigate to project directory**:

```bash
cd "/home/wisrovi/Documentos/imagenesDocker/04 cibersecurity/04 ningx/03 bunkerweb"
```

2. **Review and customize environment variables**:

```bash
vim docker-compose.yaml
```

3. **Deploy the security stack**:

```bash
docker compose up -d
```

4. **Verify service health**:

```bash
docker compose ps
```

5. **Access the administration UI** (first-time setup required):

```bash
# Web UI
open http://localhost:7000

# Protected web application
open http://localhost:80
```

### Service Endpoints

| Service | Protocol | Port | Purpose |
|---------|----------|------|---------|
| Web UI | HTTP | 7000 | Administration interface |
| WAF | HTTP | 80 | HTTP traffic proxy |
| WAF | HTTPS | 443 | HTTPS traffic proxy |
| MariaDB | TCP | 3306 | Configuration storage |
| Redis | TCP | 6379 | Cache & sessions |

## Configuration

### Environment Variables Reference

```yaml
services:
  bunkerweb:
    environment:
      # API access control - restrict to trusted networks
      API_WHITELIST_IP: "127.0.0.0/8 10.20.30.0/24"
      
      # Database connection string
      DATABASE_URI: "mariadb+pymysql://bunkerweb:changeme@bw-db:3306/db"
      
      # Enable Redis for session management
      USE_REDIS: "yes"
      REDIS_HOST: "redis"
```

### Network Configuration

| Network Name | Subnet | Isolation Level | Connected Services |
|--------------|--------|-----------------|-------------------|
| bw-universe | 10.20.30.0/24 | Internal | bunkerweb, scheduler, ui, redis |
| bw-services | Auto (bridge) | Application | bunkerweb, backend apps |
| bw-db | Auto (bridge) | Database | scheduler, ui, mariadb |

### Security Hardening Recommendations

- **Credential Management**: Change default `MYSQL_PASSWORD` before production deployment
- **API Access Control**: Restrict `API_WHITELIST_IP` to management network only
- **TLS Certificates**: Replace self-signed certificates with valid CA-signed certificates
- **Network Segmentation**: Implement firewall rules to limit external exposure
- **File Integrity**: Enable Tripwire plugin for monitoring configuration changes
- **Logging & Auditing**: Configure centralized logging for security event correlation

## Usage

### Docker Compose Operations

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View real-time logs (all services)
docker compose logs -f

# View logs for specific service
docker compose logs -f bunkerweb
docker compose logs -f bw-ui
docker compose logs -f bw-scheduler
docker compose logs -f bw-db
docker compose logs -f redis

# Restart individual service
docker compose restart bw-ui

# Rebuild and restart
docker compose up -d --build

# View resource usage
docker stats
```

### Service Health Checks

```bash
# Check all containers status
docker compose ps

# Inspect container health
docker inspect --format='{{.State.Health.Status}}' 03bunkerweb-bunkerweb-1

# API health endpoint
curl http://localhost:5000/health
```

### Backup & Restore

```bash
# Backup database volume
docker run --rm -v 03bunkerweb_bw-data:/data -v $(pwd):/backup busybox tar czf /backup/bw-db-backup.tar.gz /data

# Restore database volume
docker run --rm -v 03bunkerweb_bw-data:/data -v $(pwd):/backup busybox tar xzf /backup/bw-db-backup.tar.gz -C /
```

## Troubleshooting Guide

### Container Failures

```bash
# Check container logs
docker compose logs <service-name> --tail=100

# Inspect container state
docker inspect <container-name>

# Check resource constraints
docker stats <container-name>
```

### Network Connectivity

```bash
# List BunkerWeb networks
docker network ls | grep bw-

# Test connectivity between containers
docker compose exec bunkerweb ping -c 3 bw-db
docker compose exec bw-ui ping -c 3 redis
```

### UI Access Issues

```bash
# Verify port binding
netstat -tlnp | grep 7000

# Check UI container status
docker compose ps bw-ui
docker compose logs bw-ui --tail=20
```

### Database Connection

```bash
# Test database connectivity
docker compose exec bw-db mysql -u bunkerweb -p -e "SELECT VERSION();"

# Check database initialization
docker compose logs bw-db --tail=30
```

---

## License

This project is provided as-is for educational and deployment purposes. BunkerWeb and its components are licensed under their respective open-source licenses.

---

## Author

**William Rodríguez** - [wisrovi](https://www.linkedin.com/in/wisrovi)

*Technology Evangelist & Security Architect* specializing in enterprise infrastructure automation, cybersecurity engineering, DevSecOps practices, and cloud-native security architecture.
