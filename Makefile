.PHONY: install download prepare train-anomaly train evaluate active-learning test api docker-up docker-down lint

install:
	python3.12 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

download:
	python scripts/download_data.py

train-anomaly:
	python scripts/train_anomaly.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

active-learning:
	python scripts/run_active_learning.py

test:
	pytest

lint:
	ruff check src tests scripts

api:
	uvicorn automotive_vision.api.main:app --reload

docker-up:
	docker compose up --build

docker-down:
	docker compose down
