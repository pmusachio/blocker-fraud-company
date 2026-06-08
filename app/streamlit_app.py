"""Streamlit console for fraud scoring."""

from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Blocker Fraud Company", layout="wide")
st.title("Blocker Fraud Company")

api_url = st.text_input("API URL", "http://127.0.0.1:8000/predict")
upload_tab, json_tab = st.tabs(["CSV scoring", "Single payload"])

with upload_tab:
    file = st.file_uploader("Transaction CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)
        if st.button("Score transactions", type="primary"):
            response = requests.post(api_url, json={"records": df.to_dict(orient="records")}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            result = df.copy()
            for key, values in payload.items():
                if isinstance(values, list) and len(values) == len(result):
                    result[key] = values
            st.dataframe(result.sort_values(result.columns[-1], ascending=False), use_container_width=True, hide_index=True)

with json_tab:
    example = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 10000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    payload_text = st.text_area("Transaction JSON", json.dumps(example, indent=2), height=220)
    if st.button("Score single transaction"):
        record = json.loads(payload_text)
        response = requests.post(api_url, json={"records": [record]}, timeout=30)
        response.raise_for_status()
        st.json(response.json())
