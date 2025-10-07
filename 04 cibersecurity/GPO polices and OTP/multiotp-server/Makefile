

start:
	docker-compose up -d
	cp base_config.ini data/config/multiotp.ini
	docker-compose restart

sync:
	docker-compose exec -it multiotp /bin/bash
	>> multiotp -debug -display-log -ldap-users-sync

chage_secretkey:
	docker-compose exec -it multiotp /bin/bash
	>> multiotp -config server_secret=tAXgoNQwpsOZCw8pfoTC
	docker-compose restart
