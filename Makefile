.PHONY: install install-dev train test lint run docker-build docker-run

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python -m src.train_cnn_lstm

test:
	pytest -v

lint:
	ruff check .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t vqa-api:local .

docker-run:
	docker run --rm -p 8000:8000 vqa-api:local
