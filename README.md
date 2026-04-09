---
title: IncidentCommanderEnv
sdk: docker
app_port: 7860
license: mit
short_description: OpenEnv incident response RL env with deterministic grading.
---

# IncidentCommanderEnv

IncidentCommanderEnv is an OpenEnv-compatible RL-style environment for incident response.
An agent acts as an on-call Incident Commander and must investigate, diagnose, remediate,
and resolve failures in a simulated microservice system.

## What This Project Includes

- OpenEnv environment implementation with `reset`, `step`, `state`, and `grade`
- Deterministic simulation engine with service dependencies and cascading failures
- Three tasks with increasing complexity: easy, medium, hard
- Dense reward shaping and deterministic grading in range `[0.0, 1.0]`
- FastAPI service for API-based interaction
- Inference runner that calls Hugging Face Inference Providers through an OpenAI-compatible client
- Docker support for local containerized runs and Hugging Face Spaces style deployment

## Service Topology

The environment simulates these dependencies:

- `frontend -> api`
- `api -> auth, db`
- `auth -> db, redis`

When upstream services degrade or fail, dependent services can degrade or fail as well.

## Project Structure

```text
incident_env/
  app.py                 # FastAPI app exposing environment endpoints
  inference.py           # LLM inference loop with strict stdout format
  openenv.yaml           # OpenEnv metadata/spec file
  Dockerfile             # Container image for serving app.py
  requirements.txt       # Python dependencies
  .env                   # Local runtime secrets/config (not for commit)
  env/
    __init__.py
    models.py            # Pydantic models
    scenarios.py         # easy/medium/hard scenario definitions
    simulator.py         # Core simulation and reward logic
    grader.py            # Deterministic scoring
    environment.py       # OpenEnv wrapper class used by app/inference
  scripts/
    validate-submission.sh
```

## OpenEnv Interface

The environment class is `env.IncidentCommanderEnv` and supports:

- `reset(task_id)` -> initial observation
- `step(action)` -> observation, reward, done, info
- `state()` -> current observation
- `grade()` -> final grade breakdown

See `openenv.yaml` for interface metadata.

## Tasks

The available tasks are:

1. `easy_single_failure`
2. `medium_cascade`
3. `hard_multi_root`

Behavior summary:

- Easy: single root cause (`db`), straightforward diagnosis/remediation
- Medium: cascading degradation from `redis`
- Hard: multi-root incident with delayed failure and misleading signals

## Action Space

Supported actions:

- `query_logs`
- `query_metrics`
- `check_deploys`
- `restart_service`
- `rollback_service`
- `scale_service`
- `declare_root_cause`
- `resolve_incident`

Valid services: `frontend`, `api`, `auth`, `db`, `redis`

## Rewards And Grading

Reward design in `env/simulator.py` includes:

- Dense step rewards
- Progressive time penalty
- Diminishing returns for repeated queries
- Penalties for premature or incorrect actions
- Bonuses for evidence-based investigation and correct remediation

Grading in `env/grader.py` is deterministic and weighted:

- Root cause accuracy: 25%
- Remediation correctness: 25%
- Investigation quality: 20%
- Efficiency: 15%
- Penalty avoidance: 15%

Final score is clamped to strict open interval `(0, 1)` (implemented as `[0.01, 0.99]`).

## Local Setup

### 1) Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Configure `.env`

Create `.env` in the project root:

```env
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

Notes:

- `HF_TOKEN` is required for inference calls.
- `API_BASE_URL` defaults to the router endpoint in `inference.py`.
- `MODEL_NAME` should be a chat-capable model for the chat completions API.

## Run The API Server

```powershell
python app.py
```

Default URL: `http://localhost:7860`

Useful endpoints:

- `GET /health`
- `GET /tasks`
- `GET /actions`
- `POST /reset`
- `POST /step`
- `POST /state`
- `POST /grade`

## Run Inference

Single task:

```powershell
python inference.py --task easy_single_failure --max-steps 20
```

All tasks:

```powershell
python inference.py --all-tasks --max-steps 20
```

Expected stdout format from `inference.py`:

- `[START] task_id`
- `[STEP] step|action|reward|cumulative`
- `[END] score`

Non-protocol logs are written to stderr.

## Docker

Build image:

```powershell
docker build -t incident-commander-env .
```

Run container:

```powershell
docker run --rm -p 7860:7860 incident-commander-env
```

## Submission Validation Script

This repo includes a pre-submission validator at `scripts/validate-submission.sh`.

It checks:

- Hugging Face Space liveness (`POST /reset`)
- Docker build success
- `openenv validate` pass

Local usage:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://kevin2976-incident-commanderenv.hf.space
```

Remote usage:

```bash
curl -fsSL https://raw.githubusercontent.com/itskris2283/incident_env/main/scripts/validate-submission.sh | bash -s -- https://kevin2976-incident-commanderenv.hf.space
```

Submission links:

- GitHub: https://github.com/itskris2283/incident_env
- Hugging Face Space: https://huggingface.co/spaces/Kevin2976/incident-commanderenv

## Determinism

- Environment instances are created with a fixed seed (`42`) by default.
- Given the same task and action sequence, state transitions and grading are deterministic.

## Operational Notes

- This project uses in-memory session storage in `app.py`.
- For production-scale serving, add external session storage and authentication.
- Hugging Face provider quota and credits can limit long inference runs.

## Security

- Never commit real API tokens.
- Keep `.env` in `.gitignore`.
