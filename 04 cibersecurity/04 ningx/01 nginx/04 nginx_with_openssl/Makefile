COMPOSE := docker compose
DOMAIN  := www.vault.ecapturedtech.com

.PHONY: up down logs ps cert curl curl-ssl curl-http help

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# Genera los certificados self-signed con openssl
cert:
	$(COMPOSE) up openssl

# Curl por HTTPS (self-signed: se usa -k para no validar la CA) (puerto 8444)
curl-ssl:
	curl -k --resolve $(DOMAIN):8444:127.0.0.1 https://$(DOMAIN):8444/

curl: curl-ssl

# El puerto 80 (8083) redirige a HTTPS
curl-http:
	curl -i --resolve $(DOMAIN):8083:127.0.0.1 http://$(DOMAIN):8083/

help:
	@echo "Objetivos:"
	@echo "  up        levanta openssl + server1 + nginx_ssl (https 8444)"
	@echo "  cert      genera los certificados self-signed"
	@echo "  down      detiene los contenedores"
	@echo "  curl-ssl  curl https://$(DOMAIN):8444 con -k (self-signed)"
	@echo "  curl-http curl http:8083 (redirige a https)"
	@echo "  logs | ps logs / estado"
