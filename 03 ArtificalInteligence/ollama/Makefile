.PHONY: start start-ollama1 stop build logs clean docs

start:
	docker-compose up -d

start-ollama1:
	docker-compose up -d ollama_1

stop:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f

clean:
	docker-compose down -v

docs:
	cd docs && make html
	python -c "import webbrowser; webbrowser.open('docs/_build/html/index.html')"