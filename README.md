# Banking Risk & Fraud Investigation Prototype

An end-to-end learning project for customer-behavior fraud detection. It generates synthetic transactions, trains and saves a model, adds explainable risk scoring, stores scored cases, and provides a Streamlit analyst console with a policy-grounded RAG assistant.


## What the application does

- Generates a reproducible dataset with 100,000 transactions across 5,000 synthetic customers.
- Builds behavior features from each customer's prior transactions only.
- Trains Logistic Regression and Random Forest and selects the best model by holdout PR-AUC.
- Produces model output, a behavior anomaly score, a combined 0–100 review score, a risk band, and human-readable factors.
- Loads the scored chronological holdout into SQLite by default or PostgreSQL when configured.
- Serves a FastAPI for dashboard, transaction, customer, investigation-note, and policy-assistant operations.
- Provides a Streamlit dashboard, transaction explorer, alert queue, customer view, case notes, and policy assistant.
- Retrieves evidence from the synthetic policy documents using local TF-IDF. OpenAI Responses API generation is optional; the local retrieval answer works without an API key.

The model output is not calibrated as a real-world fraud probability. The combined risk score is an investigation-priority aid, not a decision policy.

## Run locally on Windows

Open PowerShell in the repository folder (`C:\Users\HARSH\Desktop\Python Project`). Python 3.11–3.13 is recommended.

### 1. Create an environment and install packages

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell says `python` is not recognized, use the installed interpreter directly. For the Python 3.13 installation used with this project:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If script activation is blocked in the current PowerShell session, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once, then activate the environment again.

### 2. Build the model, risk scores, and database

From the repository root, with the environment active:

```powershell
python -m src.pipeline
```

The pipeline uses the included `data/raw/transactions.csv`. If that file is absent, it generates it. It trains a model only if `models/fraud_model.joblib` is absent; it then scores the latest 20% by timestamp and imports those holdout rows into `data/banking_fraud.db`.

To train a fresh model and refresh the score/database outputs:

```powershell
python -m src.pipeline --force-train
```

Expected generated files:

- `models/fraud_model.joblib` — selected model pipeline and feature metadata
- `models/metrics.json` — holdout PR-AUC, ROC-AUC, and classification reports
- `data/processed/test_risk_scores.csv` — scores and reasons for the chronological holdout
- `data/banking_fraud.db` — local SQLite database

### 3. Start the API

In the same activated environment, run:

```powershell
python -m uvicorn app.api:app --reload --port 8000
```

Check `http://localhost:8000/api/health`. Interactive API docs are at `http://localhost:8000/docs`.

### 4. Start the dashboard

Open a second PowerShell window in the repository, activate `.venv`, and run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. The app pages are **Dashboard**, **Transaction Explorer**, **Fraud Alerts**, **Customer Risk**, **AI Investigation Assistant**, and **Investigation Report**. Investigation status and analyst notes are stored in the configured database.

### 5. Optional OpenAI generation for RAG answers

The assistant always retrieves relevant synthetic policies locally. Without credentials it returns policy evidence with a grounded local response. To turn on model-generated answers, edit `.env` and set:

```text
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=the_text_model_available_to_your_account
```

Restart the API after changing `.env`. The app uses the OpenAI Responses API with retrieved policy passages and warns the model to cite those passages and avoid treating the score as proof of fraud. No API key means no OpenAI call.

## Run with Docker and PostgreSQL

Install and start Docker Desktop. From the repository root:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

On the first start, the API container creates the synthetic data if needed, trains the model if no saved model is mounted, scores the holdout, loads PostgreSQL, and then starts FastAPI. The dashboard starts after the API health check succeeds.

- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

Stop the services with `Ctrl+C`, then run `docker compose down`. PostgreSQL data is kept in a named Docker volume. To remove that database volume and reset stored cases, run `docker compose down -v`.

The Compose database credentials are development-only. Do not expose this Compose setup as a public or production deployment.

## Notebook and command-line workflows

To inspect model training and feature engineering interactively:

```powershell
jupyter notebook notebooks/01_fraud_detection.ipynb
```

Run cells from top to bottom. The notebook trains and selects a model, then scores the holdout. You can separately regenerate scores from the saved model with:

```powershell
python -m src.risk.score_dataset
```

To regenerate the synthetic dataset with different size or seed:

```powershell
python -m src.data.generate_synthetic --customers 5000 --transactions-per-customer 20 --seed 42
python -m src.pipeline --force-train
```

The generator emits 20 transactions per customer by default, including early normal transactions to establish history. Fraud is injected at approximately 5% after warm-up and is intentionally correlated with unusual behavior so the learning exercise is detectable.

## Project layout

```text
app/
  api.py                 FastAPI endpoints
  streamlit_app.py       analyst dashboard
data/
  raw/                   synthetic source transactions
  processed/             model scores and reason codes
documents/               synthetic policy documents used by RAG
models/                  saved model and evaluation summary
notebooks/               interactive ML walkthrough
src/
  data/                  synthetic data generator
  features/              past-only customer behavior features
  models/                baseline training and selection
  rag/                   TF-IDF retrieval and grounded answer generation
  risk/                  score, bands, factors, and scoring CLI
  storage/               SQLAlchemy SQLite/PostgreSQL persistence
  pipeline.py            end-to-end local preparation command
Dockerfile
docker-compose.yml
requirements.txt
```

## API endpoints

- `GET /api/health` — service and database readiness
- `GET /api/dashboard` — totals and risk bands
- `GET /api/dashboard/activity` — daily and location chart aggregates
- `GET /api/transactions?query=...&risk_level=HIGH&limit=100` — search scored transactions
- `GET /api/transactions/{transaction_id}` — transaction and case notes
- `GET /api/alerts?risk_level=HIGH` — sorted risk queue
- `GET /api/customers/{customer_id}` — activity in the scored window
- `POST /api/assistant/ask` — policy retrieval and investigation guidance
- `POST /api/transactions/{transaction_id}/report` — structured investigation report with retrieved sources
- `PUT /api/transactions/{transaction_id}/investigation` — save case status and analyst notes

## Configuration

Settings are read from environment variables; `.env` is loaded for local runs.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | local SQLite file | SQLAlchemy URL; use `postgresql+psycopg://...` for PostgreSQL |
| `OPENAI_API_KEY` | unset | Enables optional OpenAI text generation |
| `OPENAI_MODEL` | unset | Model name to use when the key is set |
| `API_BASE_URL` | `http://localhost:8000` | API URL used by Streamlit |
| `POLICY_DOCUMENTS_DIR` | `documents/` | Folder searched for Markdown policy references |

## Learning roadmap

The complete prototype follows the original learning sequence: banking/fraud fundamentals; synthetic dataset and SQL; data preparation and EDA; model and feature engineering; risk scoring; policy RAG; Streamlit; then testing, Docker, and deployment. PostgreSQL, FastAPI, RAG, Streamlit, and Docker are now represented in the project, while the dataset and policies remain explicitly synthetic and the deployment target is local development.
