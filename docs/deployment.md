# Deployment and Consumption

The project exposes the fraud score for operational consumption through three channels:
a REST API, a Streamlit dashboard, and a Telegram bot.

## Channels

- **FastAPI:** `/predict` endpoint that scores transactions sent as JSON.
- **Streamlit:** [`app/streamlit_app.py`](../app/streamlit_app.py) — score CSV files or single
  transaction payloads from a browser.
- **Telegram Bot:** [`app/telegram_bot.py`](../app/telegram_bot.py) — score a transaction sent
  as a JSON message.

All channels read the trained artifact from `models/model.joblib` (or directly call the API,
in the case of Streamlit and the Telegram bot), so make sure `make train` has produced that
file before starting any of them.

## REST API

```bash
python -m pip install -r requirements.txt -r requirements-api.txt
make train
make api
```

Test it from another terminal:

```bash
PYTHONPATH=src python scripts/sample_api_request.py
```

`POST /predict` expects `{"records": [{...transaction fields...}]}` and returns the
predicted label and, when available, the fraud probability score.

## Streamlit

```bash
python -m pip install -r requirements-app.txt
make app
```

Open the URL printed in the terminal (usually `http://127.0.0.1:8501`), point it at a
running API instance, and either upload a CSV of transactions or paste a single
transaction as JSON.

## Telegram

```bash
export TELEGRAM_TOKEN="<your-telegram-bot-token>"
export FRAUD_API_URL="http://127.0.0.1:8000/predict"
python app/telegram_bot.py
```

Point your bot's public webhook to `/blocker/bot`. The bot expects each message to
contain a single transaction as JSON and replies with the prediction and the fraud
score (when the model exposes `predict_proba`).
