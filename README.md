# Health Assistant (Streamlit)

An AI-assisted health helper with three ML predictors (diabetes, heart disease, Parkinson's) and a chat assistant. This app is for information only and is not a substitute for professional medical advice.

## Use the app

- Open the live app: https://health-assistant-app-58hi.onrender.com

## Deployment

- Deployed on Render: https://health-assistant-app-58hi.onrender.com

## Architecture

- `app.py` – Streamlit UI router for 4 sections and admin panel
- `ml/models.py` – cached loading of pickled models
- `chat/client.py` – chat wrapper with retry/backoff, model selection, logging
- `app_data/analytics.py` – visits tracking (SQLite by default, Postgres when `DATABASE_URL` is set)
- `scripts/train/` – training scripts for diabetes, heart disease, and Parkinson's models
- `data/raw/` – input CSV datasets used by the training scripts
- `.github/workflows/ci.yml` – CI with ruff, black, pytest, smoke run

## Secrets

- `OPENAI_API_KEY` – required for chat
- `ADMIN_PASSWORD` – unlocks admin panel
- `DATABASE_URL` – optional Postgres for persistent analytics

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
black .
pytest
```

Code style is configured in `pyproject.toml`.

## Safety notes

- The assistant provides general information only. For medical issues, consult a clinician. For emergencies, call your local emergency number.
- Chat temperature is kept low (0.2) to reduce hallucinations.
- High-risk queries should be treated with caution; consider adding elevated warnings.

## Troubleshooting

- scikit-learn build failing on cloud: ensure Python 3.11 (runtime.txt) and current wheels (`numpy==1.26.4`, `scikit-learn==1.5.2`).
- App cannot start due to missing key: set `OPENAI_API_KEY` in Secrets or environment.
- Analytics not persisting on Streamlit Cloud: set `DATABASE_URL` to a managed Postgres.
- Admin login fails: set `ADMIN_PASSWORD` in Secrets.

## License

MIT