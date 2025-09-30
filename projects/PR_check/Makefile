.PHONY: build lint test run-analysis test-in-container clean debug-pylint generate-csv-report

# Variables
REPO_OWNER ?= cimacorporate
REPO_NAME ?= 001-AREPO
IMAGE_NAME = wisrovi/pr_analizer:latest

clean:
	@echo "Cleaning docker cache..."
	@docker builder prune -f

build: clean
	@echo "Building docker image..."
	@sh scripts/build_image.sh

lint:
	@echo "Linting source code..."
	@pylint src/pr_analyzer

test: test-in-container

stop:
	@echo "Stopping pr_reviewer container..."
	@docker stop pr_reviewer

base:
	@echo "Creating base docker image..."
	@docker run -d --name pr_reviewer wisrovi/pr_analizer:latest bash -c "tail -f /dev/null"

test-in-container:
	@echo "Running tests inside docker container..."
	@docker run --rm $(IMAGE_NAME) pytest

run-analysis:
	@echo "Running analysis for $(REPO_OWNER)/$(REPO_NAME)..."
	@sh scripts/run_analysis.sh $(REPO_OWNER) $(REPO_NAME)

debug-pylint:
	@echo "Debugging pylint inside docker container..."
	@docker run --rm $(IMAGE_NAME) bash -c "printf 'a = 1\nb = 2\n' > test_pylint_file.py && pylint test_pylint_file.py && rm test_pylint_file.py"

generate-csv-report:
	@echo "Generating global CSV report..."
	@docker run --rm -v $(pwd)/report:/report $(IMAGE_NAME) python /usr/src/app/scripts/generate_csv_report.py