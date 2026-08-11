COMPOSE := docker compose

.PHONY: up down logs ps curl curl-auth curl-noauth help

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# Sin credenciales: debe responder 401
curl-noauth:
	curl -i http://localhost:8081/

# Con credenciales del htpasswdp (admin:12345678)
curl-auth:
	curl -i -u admin:12345678 http://localhost:8081/

curl: curl-auth

help:
	@echo "Objetivos:"
	@echo "  up          levanta nginx + server1 con basic auth (puerto 8081)"
	@echo "  down        detiene los contenedores"
	@echo "  curl        prueba el server con credenciales (200)"
	@echo "  curl-auth   igual que curl"
	@echo "  curl-noauth sin credenciales (401)"
	@echo "  logs | ps   logs / estado"
