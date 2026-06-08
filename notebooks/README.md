# Notebooks

Suggested order to review the analysis and capture portfolio screenshots:

1. [`00_business_understanding.ipynb`](00_business_understanding.ipynb): problem framing, assumptions and solution strategy.
2. [`01_data_understanding.ipynb`](01_data_understanding.ipynb): schema, missing values, sample and data quality checks.
3. [`02_exploratory_analysis.ipynb`](02_exploratory_analysis.ipynb): distributions, hypotheses and charts.
4. [`03_feature_engineering.ipynb`](03_feature_engineering.ipynb): transformations and engineered features.
5. [`04_modeling_and_business_results.ipynb`](04_modeling_and_business_results.ipynb): metrics, artifacts and business interpretation.
6. [`05_deployment_and_consumption.ipynb`](05_deployment_and_consumption.ipynb): API, Streamlit dashboard and Telegram bot for consuming the fraud score.

The notebooks reuse the same code from the `src/` package, keeping the visual narrative
aligned with the reproducible pipeline used in production.
