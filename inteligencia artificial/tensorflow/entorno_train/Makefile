MAKEFLAGS += --always-make




start:
	docker-compose -f docker-compose.yml --compatibility up -d --build --force-recreate --no-deps

stop:
	docker-compose -f docker-compose.yml down

build:
	docker-compose -f docker-compose.yml build

into:
	docker-compose -f docker-compose.yml exec jupyter zsh

log:
	docker-compose -f docker-compose.yml logs -f jupyter