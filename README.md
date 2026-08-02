# Autonomous Agent Prediction — Kaggle-in-Kaggle Local Eval Harness

A local evaluation and submission toolset for Kaggle-in-Kaggle autonomous agent prediction competitions. It provides a lightweight evaluation harness, submission compilation/validation utilities, and model-pricing configuration so participants can test and validate agent submissions locally before uploading to Kaggle.

- Primary use: authors and participants of Kaggle-in-Kaggle style competitions who build autonomous agents that submit predictions.
- Key features: local evaluation loop, submission validation, model pricing table, sample submission layout, data layout for multiple train folds.

---

## Quick links
- Run local evaluation: python run_local_eval.py
- Validate a submission: python validate_submission.py
- Model pricing/config: models.yaml
- Example submission layout: sample_submission/agent.yaml
- Environment template: .env.example
- Python deps: requirements.txt

---

## Stack
- Language(s): Python 3.10+ (script style + modern typing)
- Runtime / Framework: plain Python scripts using the ADK (adk_submission) and kaggle_kaggle helper libraries
- Notable libraries:
  - litellm (LLM routing)
  - pydantic / pyyaml / python-dotenv
  - pandas, numpy, scikit-learn (local metric evaluation)
  - adk_submission, kaggle_kaggle (provided as local wheels in ./wheels)

---

## Repository layout (top-level)
```
.env.example               # env template for provider/proxy keys
models.yaml                # model pricing / model id -> path / prices
requirements.txt           # pip requirements (includes local wheel references)
run_local_eval.py          # local evaluation harness (main entrypoint)
validate_submission.py     # pre-flight submission linter / dry-run compiler
data/                      # datasets (train_01, train_02, ..., each expected to contain train.csv/test.csv/sample_submission.csv/solution.csv)
sample_submission/         # example agent submission layout
  agent.yaml               # minimal sample agent config
  configs/                 # sampling / generation configs (referenced by agent.yaml)
  prompts/                 # included prompt files (referenced by agent.yaml)
kaggle-kaggle-skill/       # skill / helpers (competition-specific)
wheels/                    # local wheels referenced by requirements (adk_submission, kaggle_kaggle)
submissions/               # (optional) structured submissions layout
output/                    # default output directory for traces (created by scripts)
```

How it fits together:
- The evaluation harness (run_local_eval.py) wires up a ProblemConfig (dataset paths), BudgetConfig (tool/LLM usage limits), a ModelRegistry (from models.yaml), and then compiles and runs a participant submission (the agent directory). The harness uses the ADK submission compiler to create a runnable agent and then runs it inside an Evaluation loop, producing a trace and final metric.
- validate_submission.py performs pre-flight checks: parses agent.yaml (and any !include prompt files), verifies requested models are allowed via models.yaml, and does a dry-run compilation with a dummy executor to catch errors before Kaggle upload.

---

## Getting started — install dependencies

1. Clone the repo:
   ```
   git clone https://github.com/Arya920/autonomous-agent-prediction.git
   cd autonomous-agent-prediction
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   .venv\Scripts\activate       # Windows
   ```

3. Install Python dependencies (requirements.txt references local wheels in ./wheels):
   ```
   pip install -r requirements.txt
   ```
   If pip cannot find the local wheels, ensure the files under `wheels/` exist (adk_submission and kaggle_kaggle). You can also install those wheels manually:
   ```
   pip install ./wheels/adk_submission-0.1.0-py3-none-any.whl ./wheels/kaggle_kaggle-0.1.0-py3-none-any.whl
   ```

4. Copy the environment template and set provider keys:
   ```
   cp .env.example .env
   # edit .env and set GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
   # OR set MODEL_PROXY_URL and MODEL_PROXY_API_KEY if using a model proxy
   ```

Notes:
- The harness supports two modes:
  - Direct Provider Mode: set GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY for direct LLM provider access.
  - Model Proxy Mode (competition runs): set MODEL_PROXY_URL and MODEL_PROXY_API_KEY. If one proxy var is set without the other, scripts will error (both are required).

---

## How to run (local evaluation)

Basic validation of a submission directory:
```
python validate_submission.py --agent-dir sample_submission
```
This performs:
- agent.yaml existence & YAML parsing (resolves !include)
- ensures requested models are listed in models.yaml
- dry-run compilation of the ADK agent using a dummy executor

Run a local evaluation:
```
python run_local_eval.py \
  --submission-dir sample_submission \
  --dataset train_01 \
  --metric roc_auc
```

Key options (see script help for full list):
- --submission-dir: path to agent submission directory (default: sample_submission)
- --dataset: which dataset folder under data/ to use (default: train_01)
- --metric: evaluation metric name (default: roc_auc)
- --output-dir: directory for trace files (defaults to output/)
- Budget / runtime limits: --max-tool-calls, --max-submissions, --max-exec-seconds, --max-budget-usd, etc.
- Retry / caching / compaction options for evaluation internals are also available.

Example short run:
```
python run_local_eval.py --submission-dir sample_submission --dataset train_01
```

After a run:
- The harness prints results and saves a trace (JSON/CSV) under the configured output directory. The harness will also print saved trace file paths if available.

---

## models.yaml (model pricing / allowed models)
- models.yaml maps model IDs to provider paths, friendly names, and per-million-token pricing parameters.
- The evaluation harness and validate script read models.yaml to:
  - build a PricingTable and ModelRegistry used for budgeting LLM calls
  - validate that a submission requests only allowed models
- If models.yaml is not present, default pricing/config may be used by the helper libraries, but providing a repo-local models.yaml ensures consistent, reproducible budgeting.

---

## Data layout
- Each dataset folder under data/ (e.g., data/train_01, data/train_02, ...) is expected to contain:
  - train.csv
  - test.csv
  - sample_submission.csv
  - solution.csv
- run_local_eval.py will abort if any required file is missing. Use the sample dataset layout to test the pipeline.

---

## Example submission (sample_submission/agent.yaml)
The sample agent demonstrates the minimal keys used by the harness:
```yaml
name: sample_submission
model: gemini-3.5-flash
instruction: !include prompts/system.md
tools:
  - submit_predictions
generate_content_config: !include configs/sampling.yaml
```
- Submissions must include an agent.yaml and referenced prompt/config files (use !include).
- The validator resolves includes and checks that referenced models and tools are permitted.

---

## Troubleshooting & common errors
- Missing submission dir or dataset: run_local_eval.py and validate_submission.py check for the presence of submission and dataset files and exit with a helpful message.
- MODEL_PROXY_URL set but not MODEL_PROXY_API_KEY (or vice versa): scripts will raise a RuntimeError. Either set both for proxy mode or unset both to operate in Direct Provider Mode.
- Missing local wheels: requirements.txt references local wheels. If those wheel files are not present, pip install -r requirements.txt may fail. Place the wheel files into ./wheels or install replacements from your environment.

---

## Contributing
- Issues and PRs welcome. If you add new datasets, examples, or change the submission schema, update validate_submission.py and the sample_submission example to keep the validator in sync.

---

## License & contact
- LICENSE: haven't decided yet ⚠️
- Author / contact: Tamal & Arya
