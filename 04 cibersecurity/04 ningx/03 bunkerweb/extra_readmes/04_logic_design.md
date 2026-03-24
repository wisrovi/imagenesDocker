# Logic Design

## 1. Business Logic Overview

This document describes the core business logic of the BunkerWeb WAF system, including security enforcement, configuration management, and administrative workflows.

## 2. Security Enforcement Logic

### 2.1 Request Filtering Pipeline

The core security logic processes every HTTP request through multiple stages:

```mermaid
flowchart TD
    A[HTTP Request] --> B{Valid HTTP?}
    B -->|No| C[400 Bad Request]
    B -->|Yes| D[Parse Headers]
    
    D --> E{IP Whitelisted?}
    E -->|Yes| F[Skip Rules]
    E -->|No| G[Check Blacklists]
    
    G --> H{IP Blocked?}
    H -->|Yes| I[403 Forbidden]
    H -->|No| J[Check User-Agent]
    
    J --> K{UA Blocked?}
    K -->|Yes| I
    K -->|No| L[Check Request Body]
    
    L --> M{SQL Injection?}
    M -->|Yes| I
    M -->|No| N{XSS Attempt?}
    
    N -->|Yes| I
    N -->|No| O[Check Rate Limit]
    
    O -->|Exceeded| P[429 Too Many]
    O -->|OK| Q[Forward to Backend]
    
    C --> R[Log Result]
    I --> R
    P --> R
    Q --> R
```

### 2.2 ModSecurity Rule Evaluation

Rules are evaluated in the following order:

1. **Phase 1 - Request Headers**
   - IP reputation checks
   - User-agent validation
   - Request method validation
   - URL encoding validation

2. **Phase 2 - Request Body**
   - SQL injection detection
   - XSS pattern matching
   - Command injection prevention
   - File inclusion prevention

3. **Phase 3 - Response Headers**
   - Response header validation
   - Server information leakage prevention

4. **Phase 4 - Response Body**
   - Information leakage detection
   - Error message filtering

### 2.3 Threat Scoring

Each request receives a threat score:

| Score | Action |
|-------|--------|
| 0 | Allow |
| 1-5 | Log only |
| 6-10 | Log + Warn |
| 11+ | Block (403) |

## 3. Configuration Management Logic

### 3.1 Settings Storage

```mermaid
erDiagram
    SETTINGS {
        int id PK
        string key UK
        string value
        timestamp created_at
        timestamp updated_at
    }
    
    USERS {
        int id PK
        string username UK
        string password_hash
        string email
        int role
        timestamp created_at
    }
    
    SESSIONS {
        string session_id PK
        int user_id FK
        string ip_address
        timestamp expires_at
    }
    
    BLACKLIST {
        int id PK
        string type "ip|useragent"
        string value
        string source
        timestamp created_at
    }
    
    SETTINGS ||--o{ USERS : references
    USERS ||--o{ SESSIONS : has
```

### 3.2 Configuration Update Flow

```mermaid
sequenceDiagram
    participant Admin as Administrator
    participant UI as Web UI
    participant DB as MariaDB
    participant Cache as Redis
    participant WAF as BunkerWeb
    participant Scheduler as Scheduler

    Admin->>UI: Update WAF settings
    UI->>DB: BEGIN TRANSACTION
    UI->>DB: UPDATE settings SET value=?
    DB->>UI: COMMIT
    UI->>Cache: DEL cache:settings
    UI->>Admin: Success
    
    loop Scheduler Cycle (60s)
        Scheduler->>DB: SELECT * FROM settings
        Scheduler->>Cache: GET cache:settings
        alt Cache Miss
            Scheduler->>Scheduler: Generate nginx.conf
            Scheduler->>WAF: POST /confs
            WAF->>WAF: nginx -t
            WAF->>WAF: nginx -s reload
            WAF->>Scheduler: 200 OK
            Scheduler->>Cache: SET cache:last_update
        end
    end
```

## 4. Blacklist Management Logic

### 4.1 Blacklist Sources

The scheduler downloads blacklists from multiple sources:

| Source | Type | Update Frequency |
|--------|------|-----------------|
| dan.me.uk/torlist | IP (Tor exits) | Every 6 hours |
| github.com/bad-bot-blocker | User-Agent | Every 24 hours |
| Local database | Custom rules | On-demand |

### 4.2 Blacklist Processing Pipeline

```mermaid
flowchart LR
    A[Download Sources] --> B[Parse Lists]
    B --> C[Validate Format]
    C --> D[Remove Duplicates]
    D --> E[Merge with Database]
    E --> F[Push to WAF]
    F --> G[Test Config]
    G --> H[Reload Nginx]
```

### 4.3 IP Reputation Logic

```mermaid
flowchart TD
    A[Incoming Request] --> B{IP in Allowlist?}
    B -->|Yes| C[Allow]
    B -->|No| D{IP in Blocklist?}
    
    D -->|Yes| E[Block + Log]
    D -->|No| F{Check Reputation}
    
    F --> G{GeoIP Blocked?}
    G -->|Yes| E
    G -->|No| H{Tor Exit Node?}
    
    H -->|Yes| I[Score +10]
    H -->|No| J{Known Bad?}
    
    J -->|Yes| I
    J -->|No| K{Score > Threshold?}
    
    I --> L{Score > 10}
    L -->|Yes| E
    L -->|No| C
    
    K -->|Yes| E
    K -->|No| C
```

## 5. User Authentication Logic

### 5.1 Login Flow

```mermaid
flowchart TD
    A[Login Form] --> B{Valid Input?}
    B -->|No| C[Show Validation Error]
    B -->|Yes| D[Hash Password]
    
    D --> E[Query Database]
    E --> F{User Found?}
    F -->|No| G[Show Error]
    F -->|Yes| H{Valid Password?}
    
    H -->|No| G
    H -->|Yes| I[Generate Session Token]
    
    I --> J[Store in Redis]
    J --> K[Set HTTP Cookie]
    K --> L[Redirect to Dashboard]
```

### 5.2 Session Validation

```mermaid
flowchart TD
    A[Request] --> B{Session Cookie?}
    B -->|No| C[Redirect to Login]
    B -->|Yes| D[Lookup Session]
    
    D --> E{Found?}
    E -->|No| C
    E -->|Yes| F{Expired?}
    
    F -->|Yes| G[Delete Session]
    F -->|No| H{IP Changed?}
    
    G --> C
    H -->|Yes| I[Regenerate Token]
    H -->|No| J[Allow Request]
    
    I --> J
```

## 6. Rate Limiting Logic

### 6.1 Token Bucket Algorithm

```
requests_per_minute = 60
bucket_capacity = 100
refill_rate = 1.67/second
```

### 6.2 Rate Limit Evaluation

```mermaid
flowchart TD
    A[Request] --> B{Get IP Bucket}
    B --> C{Bucket Exists?}
    C -->|No| D[Create Bucket]
    C -->|Yes| E[Check Tokens]
    
    D --> F[Initialize: 100 tokens]
    F --> E
    
    E --> G{Tokens Available?}
    G -->|No| H[429 Too Many Requests]
    G -->|Yes| I[Decrement Tokens]
    
    I --> J[Process Request]
    J --> K[Increment Request Count]
    H --> L[Log Rate Limit]
    L --> K
```

## 7. Health Monitoring Logic

### 7.1 Health Check Types

| Check | Interval | Timeout | Action |
|-------|----------|---------|--------|
| Docker Health | 30s | 10s | Restart container |
| API Health | 30s | 5s | Log error |
| Database | 60s | 10s | Alert |
| Redis | 60s | 5s | Alert |

### 7.2 Self-Healing Flow

```mermaid
flowchart TD
    A[Health Check Failed] --> B{Check Count < 3?}
    B -->|No| C[Container Unhealthy]
    B -->|Yes| D[Wait 10s]
    
    D --> E[Retry Health Check]
    E --> F{Pass?}
    F -->|Yes| G[Mark Healthy]
    F -->|No| B
    
    C --> H{Restart Container}
    H --> I[Wait for Start]
    I --> J[Check Health]
    J --> K{Healthy?}
    K -->|Yes| G
    K -->|No| L[Alert On-Call]
```
