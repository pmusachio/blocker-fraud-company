.PHONY: install install-api install-app profile train predict test api app bot

## Install core dependencies (data prep, training, CLI)
install:
	python -m pip install -r requirements.txt

## Install extras for the FastAPI service
install-api:
	python -m pip install -r requirements.txt -r requirements-api.txt

## Install extras for the Streamlit dashboard and Telegram bot
install-app:
	python -m pip install -r requirements.txt -r requirements-app.txt

## Write reports/data_profile.json with a quick data profile
profile:
	PYTHONPATH=src python -m blocker_fraud_company.cli profile

## Train the fraud classifier and write models/model.joblib + reports/metrics.json
train:
	PYTHONPATH=src python -m blocker_fraud_company.cli train

## Score a CSV in batch: make predict INPUT=path/to.csv [OUTPUT=path/to.csv]
predict:
	PYTHONPATH=src python -m blocker_fraud_company.cli predict --input $(INPUT) $(if $(OUTPUT),--output $(OUTPUT),)

## Run the test suite
test:
	python -m pytest

## Serve the trained model through FastAPI (http://127.0.0.1:8000)
api:
	PYTHONPATH=src uvicorn blocker_fraud_company.api:app --reload

## Launch the Streamlit dashboard (http://127.0.0.1:8501)
app:
	PYTHONPATH=src streamlit run app/streamlit_app.py

## Run the Telegram bot webhook (requires TELEGRAM_TOKEN and FRAUD_API_URL)
bot:
	python app/telegram_bot.py
