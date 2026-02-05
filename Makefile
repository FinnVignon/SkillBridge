PYTHON ?= python3

run:
	FLASK_SECRET_KEY="your-strong-secret" $(PYTHON) app/app.py

run-dev:
	FLASK_ENV=development FLASK_SECRET_KEY="your-strong-secret" $(PYTHON) app/app.py

git:
	rm -rf .venv venv env __pycache__
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f .env .env.local
