install:
	pip install -r requirements.txt

train:
	python src/models/train_model.py

start-api:
	uvicorn src.api.app:app --reload

test:
	pytest tests/

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache