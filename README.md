# Health Assistant (Streamlit)

An AI-assisted health helper with three ML predictors (diabetes, heart disease, Parkinson's) and a chat assistant. This app is for information only and is not a substitute for professional medical advice.

## Quick start (local)

1. Python 3.11 recommended (runtime is pinned for cloud)
2. Create venv and install deps
   ```bash
   python3.11 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Set secrets (at least `OPENAI_API_KEY`)
   ```bash
   export OPENAI_API_KEY=sk-...
   # Optional admin
   export ADMIN_PASSWORD=your_password
   # Optional analytics persistence
   export DATABASE_URL=postgresql://user:pass@host:5432/db
   ```
4. Run
   ```bash
   streamlit run app.py
   ```

## Deploy (Streamlit Community Cloud)

1. Push to GitHub (already connected).
2. On Streamlit Cloud, create an app from this repo.
3. Ensure runtime is Python 3.11 (provided by `runtime.txt`).
4. Configure Secrets in the app Settings:
   ```toml
   OPENAI_API_KEY = "sk-..."
   # optional
   ADMIN_PASSWORD = "your_password"
   DATABASE_URL = "postgresql://user:pass@host:5432/db"
   ```
5. Deploy. First build may take a few minutes.

## Architecture

- `app.py` – Streamlit UI router for 4 sections and admin panel
- `ml/models.py` – cached loading of pickled models
- `chat/client.py` – chat wrapper with retry/backoff, model selection, logging
- `app_data/analytics.py` – visits tracking (SQLite by default, Postgres when `DATABASE_URL` is set)
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