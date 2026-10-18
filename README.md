# Musanze Harvest Intelligence Service

| Field | Value |
|---|---|
| Full name | Doris-Kelly Atemnkeng |
| Registration number | 24/23813 |
| Institutional username | ug2423813@ines.ac.rw |
| Verification code | AI-A2-3813 |
| Repository URL | To be added after creating your own GitHub repository |
| Final commit hash | To be added after the final commit |
| Python version tested | 3.14.6 |
| Operating system tested | macOS, Apple Silicon |

**Synthetic data notice:** This project uses synthetic classroom data.
Its metrics do not establish real-world agricultural accuracy.

## Problem

The Musanze Harvest Intelligence Service is a machine-learning
prototype for estimating farm yield, classifying production risk,
and grouping farms with similar input conditions. It provides a
command-line prediction interface and a FastAPI service.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py --student-id ug2423813@ines.ac.rw --seed 3813
 

```bash
python predict.py --json samples/valid_farm.json
python predict.py --json samples/invalid_farm.json
pytest -q
```

To start the API server:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API documentation.

## Test results

The automated test run completed with 19 passed tests and one dependency deprecation warning.

- Regression: ridge regression; RMSE 0.988
- Classification: random forest; F1 0.939; AUC 0.977
- Clustering: six clusters

These results come from synthetic classroom data and do not establish performance on real farms.

## Generated outputs

- `data/AI_A2_2423813.csv`
- `artifacts/regression_metrics.json`
- `artifacts/classification_metrics.json`
- `artifacts/clustering_report.json`
- `artifacts/data_quality_report.json`
- `artifacts/run_metadata.json`
- `artifacts/test_predictions.csv`
- `artifacts/data_fingerprint.txt`
- `artifacts/plots/`
- `models/`

## API endpoints

- `GET /health` — service health
- `GET /model-info` — model and dataset information
- `POST /predict` — validated farm prediction

Manual API tests returned HTTP 200 for health, model information and valid predictions. Invalid input returned HTTP 422.

## Responsible AI and limitations

The dataset is synthetic and does not represent real farms or farmers. Real-world accuracy has not been established. Every prediction requires human review. This prototype must not replace an agronomist or determine access to finance, insurance or public services.

## Repository

The repository URL and final commit hash will be added after creating and committing this project to your own Git repository.

