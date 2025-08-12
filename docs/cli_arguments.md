# CLI Arguments & Data Requirements

This document provides detailed descriptions of the command-line arguments for both `train` and `predict` commands, along with dataset formatting requirements.

---

## Train Command

| Argument        | Required | Default       | Description                                                               |
|----------------|----------|---------------|---------------------------------------------------------------------------|
| `--data`        | Yes      | —             | Path to the input CSV file with raw training data*.                       |
| `--market`      | No       | `None`        | Market to filter by (e.g., country code). If not provided, uses all data. |
| `--market-col`  | No       | `country`     | Name of the column indicating market.                                     |
| `--target`      | No       | `churn`       | Name of the binary target column.                                         |
| `--outdir`      | No       | `artifacts`   | Directory to store trained model, metrics, and predictions.               |

### *Training Data Requirements

- File must **not be empty**.
- The **target column** must be **binary** and contain values in one of the following formats:
  - `"yes"` / `"no"`
  - `"true"` / `"false"`
  - `1` / `0`
- Index column is optional and will be handled automatically if present.

---

## Predict Command

| Argument        | Required | Default       | Description                                                                 |
|----------------|----------|---------------|-----------------------------------------------------------------------------|
| `--data`        | Yes      | —             | Path to the CSV file for prediction.                                       |
| `--model`       | Yes      | —             | Path to the previously trained `.joblib` model file.                        |
| `--market`      | No       | `None`        | Market to filter by (same logic as in training).                           |
| `--market-col`  | No       | `country`     | Name of the column indicating market.                                      |
| `--target`      | No       | `churn`       | Name of the target column. If present, it will be dropped before scoring.  |
| `--outdir`      | No       | `artifacts`   | Directory to save predictions.                                              |
