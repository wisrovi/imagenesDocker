COMPOSE := docker compose
DOMAIN  := wisrovi.duckdns.org

.PHONY: up down logs ps curl curl-ssl curl-http certbot help

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# Curl por HTTPS validando el certificado real de Let's Encrypt (puerto 8443)
curl-ssl:
	curl --resolve $(DOMAIN):8443:127.0.0.1 https://$(DOMAIN):8443/

curl: curl-ssl

# El puerto 80 (8082) redirige a HTTPS
curl-http:
	curl -i --resolve $(DOMAIN):8082:127.0.0.1 http://$(DOMAIN):8082/

# Renovar los certificados con certbot
certbot:
	$(COMPOSE) run --rm certbot renew

help:
	@echo "Objetivos:"
	@echo "  up        levanta server1 + nginx_ssl (certs Let's Encrypt, https 8443)"
	@echo "  down      detiene los contenedores"
	@echo "  curl-ssl  curl https://$(DOMAIN):8443 validando el certificado"
	@echo "  curl-http curl http:8082 (redirige a https)"
	@echo "  certbot   renueva los certificados"
	@echo "  logs | ps logs / estado"
