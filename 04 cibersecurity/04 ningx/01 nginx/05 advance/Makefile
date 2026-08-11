COMPOSE := docker compose
DOMAIN  := wisrovi.duckdns.org

.PHONY: up down logs ps curl curl-ssl curl-http help

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# Curl por HTTPS: nginx-proxy + letsencrypt companion emiten el cert (puerto 8445)
curl-ssl:
	curl --resolve $(DOMAIN):8445:127.0.0.1 https://$(DOMAIN):8445/

curl: curl-ssl

# El puerto 80 (8084) redirige a HTTPS
curl-http:
	curl -i --resolve $(DOMAIN):8084:127.0.0.1 http://$(DOMAIN):8084/

help:
	@echo "Objetivos:"
	@echo "  up        levanta nginx-proxy + letsencrypt + server1 (auto-detection, https 8445)"
	@echo "  down      detiene los contenedores"
	@echo "  curl-ssl  curl https://$(DOMAIN):8445 validando el certificado"
	@echo "  curl-http curl http:8084 (redirige a https)"
	@echo "  logs | ps logs / estado"
