# SQUiD Reproduction Project

**Course:** CSCI/DASC 6010 — Big Data Analytics and Management  
**Institution:** East Carolina University  
**Team:** Curtis Moore · Jaret Logan · Zach Ellington

Reproduction and noise sensitivity analysis of [SQUiD: Synthesizing Relational Databases from Unstructured Text](https://arxiv.org/abs/2505.19025) (Sadia et al., EMNLP 2025).

---

## Overview

SQUiD is a neurosymbolic framework that converts raw unstructured text into a fully formed SQLite relational database using a four-stage pipeline:

```
Schema Generation → Value Identification → Table Population → Database Materialization
```

This project:
1. **Reproduces** the paper's core results, verifying two hypotheses about SQUiD's performance and integrity compared to a single-shot LLM baseline.
2. **Extends** the paper with a noise sensitivity analysis — testing how SQUiD and the baseline degrade when the input text contains contradictory facts, missing values, duplicate entities, or structural errors.

### Research Hypotheses

- **Hypothesis 1 (Performance):** The decomposed SQUiD pipeline outperforms end-to-end LLMs in both schema accuracy and data correctness.
- **Hypothesis 2 (Integrity):** A neurosymbolic approach maintains primary and foreign key relationships more reliably than a single-shot LLM prompt.

---

## Environment Notes

**Local development is strongly recommended over Google Colab for this project.**

SQUiD's compute happens through LLM API calls — not GPU workloads. Colab's session resets (~90 min inactivity) will wipe your environment and any unsaved outputs mid-pipeline, which is a serious risk for a multi-stage run. Run everything locally in the virtual environment described below.

### Tested Environments

**Curtis:**
- macOS (Apple Silicon M-series)
- Python 3.13
- `claude-haiku-4-5-20251001` via Anthropic API
- Dataset: BIRD (`bird_dataset.json`, 384 entries)

**Jaret:**
- Windows 11 (x64)-
- Python 3.13
- `claude-haiku-4-5-20251001` via Anthropic API
- Dataset: BIRD (`bird_dataset.json`, 192 entries)

**Zach:**
- *(fill in after setup)*

> **Deviation from paper:** The paper used `claude-3-7-sonnet-20241022`. This model was unavailable on the team's account tier. We used `claude-haiku-4-5-20251001` instead. Documented in Section 4.3 of the report.

---

## Repository Setup

### Prerequisites

- Python 3.10+ (tested on 3.13)
- Git
- An Anthropic API key with credits — get one at [console.anthropic.com](https://console.anthropic.com)

> **Note:** Do NOT `pip install sqlite3` — it is part of Python's standard library. Verify with: `python -c "import sqlite3; print(sqlite3.version)"`

> **Model flexibility:** Setup notes are written for Claude via the Anthropic API. If you use a different model (DeepSeek, OpenAI, etc.), update the model string in `SQUiD/src/model.py` and record the exact string in Table 1 of the report.

---

### Step 1 — Clone this repo

Use the alias so your local folder name is short and matches all path references in the scripts:

```bash
git clone https://github.com/ckmoore2/CSCI-6040---Final-Project---SQUiD-Reproduction.git squid-reproduction
cd squid-reproduction
```

> Share this exact command with teammates so everyone's local folder is named `squid-reproduction`.

---

### Step 2 — Clone the official SQUiD repo

```bash
git clone https://github.com/Mushtari-Sadia/SQUiD.git
```

Record the commit hash immediately — this goes into Table 1 of the report:

```bash
cd SQUiD && git log --oneline -1
cd ..
```

> **Important:** Remove SQUiD's own `.git` folder so Git doesn't treat it as a submodule:
> ```bash
> rm -rf SQUiD/.git
> ```

---

### Step 3 — Create the virtual environment

```bash
python -m venv squid-env
source squid-env/bin/activate      # Mac / Linux
squid-env\Scripts\activate         # Windows
```

Your terminal prompt should show `(squid-env)` before proceeding.

---

### Step 4 — Install dependencies

SQUiD's `requirements.txt` was written for Linux with NVIDIA GPU support. On Mac, filter out incompatible packages first:

```bash
grep -v -E "nvidia|triton" SQUiD/requirements.txt > requirements_mac.txt
pip install -r requirements_mac.txt
pip install pyyaml anthropic openai python-dotenv pandas matplotlib seaborn
```

> Several packages are missing from `requirements.txt` but required at runtime — install them as errors appear:
> ```bash
> pip install json-repair pyyaml anthropic
> ```

Verify core packages installed:

```bash
python -c "import anthropic, pandas, matplotlib, yaml, json_repair; print('all ok')"
python -c "import sqlite3; print(sqlite3.version)"
```

---

### Step 5 — Configure your API key

**Never paste your API key directly into a script or commit it to the repo.**

Create your `.env` file from the template:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your actual key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Verify it loads correctly:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.environ.get('ANTHROPIC_API_KEY', 'NOT SET')[:20])"
```

**If using Google Colab instead of local development:**

1. Click the key icon (Secrets) in the left sidebar
2. Add a new secret: Name = `ANTHROPIC_API_KEY`, Value = your key
3. Access it in your notebook:

```python
from google.colab import userdata
import os
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

---

### Step 6 — Update model.py to load the API key from .env

Open `SQUiD/src/model.py`. The `load_dotenv()` call needs an explicit path since SQUiD runs from inside its own subdirectory. Replace the top two lines:

```python
# Replace this:
from dotenv import load_dotenv
load_dotenv()

# With this:
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
```

Also update the Anthropic client line in `__init__`:

```python
# Replace this:
self.client = Anthropic(api_key="")

# With this:
self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
```

---

### Step 7 — Configure SQUiD

The config file at `SQUiD/configs/config.yaml` has placeholder paths that must be filled in. Update all `/add_path/` entries to point to your local paths:

```yaml
schema_generation:
    datapath: "BIRD/bird_dataset"
    num_of_entries: 192
    model_name: "claude"
    prompt_type: "cot"
    method: "text"
    base_data_path: "/your/path/to/squid-reproduction/SQUiD/dataset"
    results_dir: "/your/path/to/squid-reproduction/results/reproduction/stage1"
    logs_dir: "/your/path/to/squid-reproduction/results/reproduction/logs/schema"
```

Repeat for `value_identification`, `value_population`, `database_generation`, and `baseline` — updating `base_data_path`, `results_dir`, and `logs_dir` for each.

---

### Step 8 — Create output directories

```bash
mkdir -p results/reproduction/logs/schema
mkdir -p results/reproduction/logs/values
mkdir -p results/reproduction/logs/population
mkdir -p results/reproduction/logs/database
mkdir -p results/reproduction/logs/baseline
mkdir -p results/reproduction/stage1/BIRD/bird_dataset
mkdir -p results/reproduction/stage2/BIRD/bird_dataset
mkdir -p results/reproduction/stage3/BIRD/bird_dataset
mkdir -p results/reproduction/stage4/BIRD/bird_dataset
mkdir -p results/reproduction/baseline/BIRD/bird_dataset
mkdir -p results/sensitivity
```

---

## Project Stages

All pipeline commands must be run from inside the `SQUiD/` directory:

```bash
cd SQUiD
```

---

### Stage 1 — Schema Generation

Generates a relational schema (tables, columns, PKs, FKs) from the input text.

```bash
python src/schema_generation.py --model_name claude --prompt_type cot --method text
```

**Output:** `results/reproduction/stage1/BIRD/bird_dataset/text_cot_claude.json`  
**Status:** ✅ Complete — 384 entries, `claude-haiku-4-5-20251001`

---

### Stage 2 — Value Identification

Extracts values from the text and maps them to the schema produced in Stage 1.

```bash
python src/value_identification.py --model_name claude
```

**Input:** Stage 1 output  
**Output:** `results/reproduction/stage2/`  
**Status:** 🔲 Pending Stage 1

---

### Stage 3 — Table Population
Organizes extracted values into relational tuples with correct PK/FK assignments.

```bash
python src/value_population.py --model_name claude
```

**Input:** Stage 2 output  
**Output:** `results/reproduction/stage3/`  
**Status:** 🔲 Pending Stage 2

---

### Stage 4 — Database Materialization

Converts the populated tables into a valid, executable SQLite database.

```bash
python src/database_generation.py --model_name claude
```

**Input:** Stage 3 output  
**Output:** `results/reproduction/stage4/*.db`  
**Status:** 🔲 Pending Stage 3

---

### Baseline — LLM Single-Shot

Runs the end-to-end LLM baseline (single prompt, no decomposition) on the same dataset for comparison.

```bash
python src/baseline.py --model_name claude
```

**Output:** `results/reproduction/baseline/`

---

### Verify a completed stage

After each stage, verify the output file exists and has content:

```bash
ls -lh results/reproduction/stage1/BIRD/bird_dataset/

python -c "
import json
with open('results/reproduction/stage1/BIRD/bird_dataset/text_cot_claude.json') as f:
    data = json.load(f)
print(f'Entries: {len(data)}')
print(f'Keys: {list(data[0].keys())}')
"
```

---

## Datasets

The benchmark datasets are included inside the SQUiD repo — no external downloads needed.

| Dataset | Location | Domain | Notes |
|---|---|---|---|
| BIRD | `SQUiD/dataset/BIRD/bird_dataset.json` | Mixed (BIRD Text2SQL benchmark) | Used for reproduction run |
| SyntheticText | `SQUiD/dataset/SyntheticText/merged_dataset2.json` | Synthetic travel/booking narrative | Paper's primary examples |
| RealDocuments | `SQUiD/dataset/RealDocuments/` | News (CNN, COVID, Wikipedia) | Real-world text domain |

The datasets are pre-generated natural language text files — the authors converted existing relational databases into narrative text using LLMs. They are not raw CSV files.

---

## Folder Structure

```
squid-reproduction/
│
├── SQUiD/                              # Official SQUiD codebase
│   ├── configs/
│   │   └── config.yaml                 # Pipeline config — fill in your local paths
│   ├── dataset/
│   │   ├── BIRD/                       # BIRD benchmark dataset
│   │   ├── SyntheticText/              # Synthetic travel/booking text
│   │   └── RealDocuments/             # CNN, COVID, Wikipedia docs
│   └── src/
│       ├── schema_generation.py        # Stage 1
│       ├── value_identification.py     # Stage 2
│       ├── value_population.py         # Stage 3
│       ├── database_generation.py      # Stage 4
│       ├── baseline.py                 # LLM baseline
│       ├── model.py                    # LLM client (updated for .env loading)
│       └── utils.py                    # Shared utilities
│
├── data/
│   ├── clean/                          # Reserved for noise experiment base inputs
│   └── noisy/                          # 12 noisy variants (4 types × 3 levels)
│
├── scripts/
│   ├── noise_injector.py               # Generates noisy dataset variants (Curtis)
│   ├── run_squid_batch.py              # Batch SQUiD runs on noisy inputs (Jaret)
│   ├── run_baseline_batch.py           # Batch baseline runs on noisy inputs (Zach)
│   └── compute_sensitivity_metrics.py  # Aggregates metrics across variants (Zach)
│
├── results/
│   ├── reproduction/
│   │   ├── stage1/                     # Schema Generation output ✅
│   │   ├── stage2/                     # Value Identification output
│   │   ├── stage3/                     # Table Population output
│   │   ├── stage4/                     # SQLite databases
│   │   ├── baseline/                   # Baseline LLM output
│   │   └── logs/                       # Per-stage run logs
│   └── sensitivity/                    # Noise experiment outputs
│
├── .env                                # Your API key — NEVER commit this
├── .env.example                        # Template — copy to .env
├── .gitignore
├── requirements_mac.txt                # Mac-compatible dependency list
└── README.md
```

---

## Task Ownership

| Area | Owner |
|---|---|
| Dataset access and verification |  |
| Stage 1 — Schema Generation |  |
| Stage 2 — Value Identification |  |
| Stage 3 — Table Population |  |
| Stage 4 — Database Materialization |  |
| All four reproduction metrics |  |
| LLM baseline |  |
| Noise injection script (`scripts/noise_injector.py`) |  |
| SQUiD batch runs on noisy variants |  |
| Baseline batch runs on noisy variants |  |
| Metrics aggregation and charts |  |
| Report Sections 1, 2, 3 |  |
| Report Sections 4, 5, 6, 7 | |
| Presentation slides |  |

---

## Task Dependencies

```
 Stage 1 complete
        │
        ▼
 Stage 2 — Value Identification
        │
        ▼
 Stage 3 → Stage 4 → compute metrics
        │
        ▼
 noise_injector.py → 12 noisy files generated
        │
        ├──▶SQUiD batch runs on all 12 variants
        └──▶baseline batch runs on all 12 variants
                        │
                        ▼
                aggregate metrics + build charts
                        │
                        ▼
                write Results + Discussion sections
```

---

## Reproducibility Notes

- **LLM model (Curtis):** `claude-haiku-4-5-20251001` — record in Table 1 of the report. The paper used `claude-3-7-sonnet-20241022`, unavailable on our account tier.
- **Random seed:** The noise injection script uses `RANDOM_SEED = 42`. Do not change this — must be consistent across all runs.
- **SQUiD commit hash:** Run `git log --oneline -1` inside the `SQUiD/` folder and record the hash in the report.
- **Mac users:** Use `requirements_mac.txt` not `requirements.txt` — the full file includes NVIDIA/CUDA packages that cannot install on Apple Silicon.
- **Python version:** Tested on 3.13. Some packages specify `<3.13` — install individually at the latest compatible version if they fail.
- **Save outputs after each stage:** Don't rely on terminal history — write results to disk and commit before moving to the next stage.

---

## Common Issues

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'yaml'` | `pip install pyyaml` |
| `ModuleNotFoundError: No module named 'anthropic'` | `pip install anthropic` |
| `ModuleNotFoundError: No module named 'json_repair'` | `pip install json-repair` |
| `nvidia-cublas-cu12` install failure | Use `requirements_mac.txt` — filters out all NVIDIA packages |
| `triton` not found | `grep -v -E "nvidia\|triton" requirements.txt > requirements_mac.txt` |
| `FileNotFoundError: configs/config.yaml` | Run from inside `SQUiD/` not project root |
| `FileNotFoundError` on output path | Create missing dir with `mkdir -p` |
| `model: claude-3-7-sonnet not found` | Update model string in `SQUiD/src/model.py` to an available model — run `python -c "import anthropic, os; from dotenv import load_dotenv; load_dotenv(); client = anthropic.Anthropic(); [print(m.id) for m in client.models.list().data]"` to see available models |
| `credit balance too low` | Add credits at console.anthropic.com → Settings → Billing |
| API key `NOT SET` | Check `.env` has `ANTHROPIC_API_KEY=...` and `model.py` uses the explicit Path load |
| SQUiD appears as submodule in Git | `rm -rf SQUiD/.git` then `git add SQUiD/` |
| `.idea/` folder in Git | Add `.idea/` to `.gitignore`, then `git rm -r --cached .idea` |
| `sqlite3` install error | Do not pip install — it is stdlib. Verify: `python -c "import sqlite3; print(sqlite3.version)"` |

---

## What NOT to Commit

The `.gitignore` already covers these:

- `.env` — your API key; never commit it
- `*.db` — SQLite databases
- `squid-env/` — your local virtual environment
- `__pycache__/` and `*.pyc` — Python bytecode
- `.idea/` — IDE project settings

If you accidentally stage any of these:

```bash
git rm --cached <filename>
git commit -m "chore: remove accidentally staged file"
```

---

## Key Resources

| Resource | Link |
|---|---|
| SQUiD paper (arXiv) | https://arxiv.org/abs/2505.19025 |
| SQUiD official repo | https://github.com/Mushtari-Sadia/SQUiD |
| Anthropic console | https://console.anthropic.com |

---

## Questions

Post in the team group chat. If you're stuck on something for more than 30 minutes, flag it — don't block the pipeline silently.
