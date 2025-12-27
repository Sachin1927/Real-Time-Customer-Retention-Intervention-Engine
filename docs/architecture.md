# System Architecture

## Data Pipeline
The data pipeline is managed by **Feast** (Feature Store).
* **Raw Data:** Parquet files in `data/raw/` containing historical customer logs.
* **Offline Store:** Used for generating training datasets (point-in-time correct).
* **Online Store:** SQLite database used for low-latency (<50ms) retrieval during inference.

## Model Lifecycle
1.  **Training:** XGBoost model trained on 80% of historical data.
2.  **Evaluation:** Model must achieve AUC > 0.75 to be promoted.
3.  **Serving:** The model is wrapped in a **FastAPI** application (`src/api/app.py`).

## Deployment
* **Container:** Dockerized using `python:3.10-slim`.
* **CI/CD:** GitHub Actions triggers `pytest` on every push to the `main` branch.