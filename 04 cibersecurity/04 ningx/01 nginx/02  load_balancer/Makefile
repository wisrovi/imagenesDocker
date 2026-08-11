COMPOSE := docker compose
URL     := http://localhost:8080

.PHONY: up down logs ps curl curl-loop health help

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# Peticion unica al balanceador
curl:
	curl -s $(URL)/

# 10 peticiones: se debe ver como reparte entre Server 1/2/3 (least_conn)
curl-loop:
	@for i in $$(seq 1 10); do curl -s $(URL)/; done

health:
	curl -s $(URL)/health

help:
	@echo "Objetivos:"
	@echo "  up        levanta 3 servidores + nginx balanceador (puerto 8080)"
	@echo "  down      detiene los contenedores"
	@echo "  curl      una peticion al balanceador"
	@echo "  curl-loop 10 peticiones para ver el balanceo entre servidores"
	@echo "  health    check /health"
	@echo "  logs | ps logs / estado"
