"""Telegram webhook for fraud scoring.

Configure TELEGRAM_TOKEN and FRAUD_API_URL before exposing the webhook.
The bot expects each message to contain one transaction as JSON.
"""

from __future__ import annotations

import json
import os

import requests
from flask import Flask, request


TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
FRAUD_API_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000/predict")

app = Flask(__name__)


def send_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_TOKEN:
        return
    requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=20)


def score_transaction(record: dict) -> str:
    response = requests.post(FRAUD_API_URL, json={"records": [record]}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    score = payload.get("score", [None])[0]
    prediction = payload.get("prediction", [None])[0]
    if score is None:
        return f"Prediction: {prediction}"
    return f"Prediction: {prediction}\nFraud score: {float(score):.2%}"


@app.post("/blocker/bot")
def webhook() -> tuple[dict, int]:
    update = request.get_json(force=True)
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if not chat_id:
        return {"ok": True}, 200

    try:
        record = json.loads(text)
        answer = score_transaction(record)
    except Exception as exc:  # noqa: BLE001
        answer = f"Send one transaction as JSON. Error: {exc}"

    send_message(chat_id, answer)
    return {"ok": True}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
