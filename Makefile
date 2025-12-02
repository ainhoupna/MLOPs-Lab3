install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest tests/ -vv --cov=mylib --cov=api --cov=cli 

format:	
	uv run black mylib/*.py cli/*.py api/*.py

lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*\.py mylib/*.py cli/*.py api/*.py 

refactor: format lint

all: install format lint test

# Docker targets
docker-build:
	docker build -t mlops-lab3:latest .

docker-run:
	docker run -d -p 8000:8000 --name mlops-lab3-container mlops-lab3:latest

docker-stop:
	docker stop mlops-lab3-container || true
	docker rm mlops-lab3-container || true

docker-clean: docker-stop
	docker rmi mlops-lab3:latest || true

docker-logs:
	docker logs mlops-lab3-container

docker-test:
	@echo "Testing API endpoints..."
	@curl -s http://localhost:8000/ > /dev/null && echo "✓ Home page accessible" || echo "✗ Home page failed"
	@curl -s -X POST -F "file=@image.jpg" http://localhost:8000/predict > /dev/null && echo "✓ Predict endpoint working" || echo "✗ Predict endpoint failed"