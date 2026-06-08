"""Feature engineering utilities for the fraud detection workflow."""

from __future__ import annotations

from .data import normalize_columns


def target_name(config: dict) -> str | None:
    target = config.get("data", {}).get("target")
    if not target:
        return None
    return _normalize_name(target)


def _normalize_name(name: str) -> str:
    import pandas as pd

    return normalize_columns(pd.DataFrame(columns=[name])).columns[0]


def add_fraud_features(df):
    """Engineer domain features for the PaySim transactions dataset."""
    if "step" in df.columns:
        df["transaction_day"] = (df["step"] // 24 + 1).astype("Int64")
        df["is_weekend_cycle"] = (df["transaction_day"] % 7).isin([0, 6]).astype(int)
    if {"new_balance_orig", "old_balance_orig"}.issubset(df.columns):
        df["origin_balance_delta"] = df["new_balance_orig"] - df["old_balance_orig"]
    if {"new_balance_dest", "old_balance_dest"}.issubset(df.columns):
        df["destination_balance_delta"] = df["new_balance_dest"] - df["old_balance_dest"]
    if {"amount", "old_balance_orig"}.issubset(df.columns):
        df["amount_to_origin_balance"] = df["amount"] / (df["old_balance_orig"].abs() + 1)
    return df


def prepare_features(df, config: dict, training: bool = True):
    df = normalize_columns(df)
    return add_fraud_features(df)


def model_matrix(df, config: dict, training: bool = True):
    df = prepare_features(df, config, training=training)
    target = _normalize_name(config.get("data", {}).get("target", "")) if config.get("data", {}).get("target") else None
    drop_columns = {_normalize_name(c) for c in config.get("data", {}).get("drop_columns", [])}
    id_columns = {_normalize_name(c) for c in config.get("data", {}).get("id_columns", [])}

    y = None
    if target and target in df.columns:
        y = df[target]
        drop_columns.add(target)

    X = df.drop(columns=[c for c in drop_columns.union(id_columns) if c in df.columns], errors="ignore")
    return X, y, df
