# SQUiD Reproduction Project

**Course:** CSCI/DASC 6040, Computational Analysis of Natural Languages
**Institution:** East Carolina University
**Team:** Curtis Moore, Jaret Logan, Zach Ellington

Reproduction and noise sensitivity analysis of [SQUiD: Synthesizing Relational Databases from Unstructured Text](https://arxiv.org/abs/2505.19025) (Sadia et al., EMNLP 2025).

---

## Overview

SQUiD is a neurosymbolic framework that converts raw unstructured text into an executable SQLite relational database using a four-stage pipeline:

```
Schema Generation → Value Identification → Table Population → Database Materialization
```

This project does two things:

1. **Reproduces** the paper's core results. We verify two hypotheses about SQUiD's performance and integrity compared to a single-shot LLM baseline.
2. **Extends** the paper with a noise sensitivity analysis. We test how SQUiD and the baseline degrade when the input text contains contradictory facts, missing values, duplicate entities, or structural errors.

### Research Hypotheses

- **Hypothesis 1 (Performance):** The decomposed SQUiD pipeline outperforms end-to-end LLMs in both schema accuracy and data correctness.
- **Hypothesis 2 (Integrity):** A neurosymbolic approach maintains primary and foreign key relationships more reliably than a single-shot LLM prompt.
- **Extension (Sensitivity):** SQUiD's staged decomposition should make it more robust to noisy input than the single-shot baseline.

---

## Environment Notes

Run locally, not on Google Colab. SQUiD makes LLM API calls rather than GPU workloads, so you don't need Colab for compute. Meanwhile Colab's session resets (about 90 minutes of inactivity) will wipe your environment partway through a multi-stage run and you will lose work.

### Tested Environments

**Curtis:**
- macOS (Apple Silicon M-series)
- Python 3.13
- `claude-haiku-4-5-20251001` via Anthropic API
- Dataset: BIRD (`bird_dataset.json`, 192 entries)

**Jaret:**
- Windows 11 (x64)
- Python 3.13
- `claude-haiku-4-5-20251001` via Anthropic API
- Dataset: BIRD (`bird_dataset.json`, 192 entries)

**Zach:**
- macOS (Apple Silicon M-series)
- Python 3.13
- `claude-haiku-4-5-20251001` via Anthropic API
- Dataset: BIRD, full 192 entries for all 12 noisy variants
- Clean reference: 149 entries (inherited from Curtis's reproduction run to conserve API budget)

> **Deviation from paper:** The paper used `claude-3-7-sonnet-20241022`. That model was unavailable on our account tier, so we used `claude-haiku-4-5-20251001`. See Section 4.3 of the report.

---

## Repository Setup

### Prerequisites

- Python 3.10+ (tested on 3.13)
- Git
- An Anthropic API key with credits. Get one at [console.anthropic.com](https://console.anthropic.com).

> **Note:** Do not `pip install sqlite3`. It's part of Python's standard library. Verify with `python -c "import sqlite3; print(sqlite3.version)"`.

> **Model flexibility:** Setup notes assume Claude via the Anthropic API. If you use a different model (DeepSeek, OpenAI, etc.), update the model string in `SQUiD/src/model.py` and record the exact string in Table 1 of the report.

---

### Step 1: Clone this repo

Use the alias so your local folder name is short and matches all path references in the scripts:

```bash
git clone https://github.com/ckmoore2/CSCI-6040---Final-Project---SQUiD-Reproduction.git squid-reproduction
cd squid-reproduction
```

---

### Step 2: The SQUiD codebase is already in the repo

The `SQUiD/` folder is already tracked. No separate clone needed. It contains the full codebase, datasets, and configs.

Record the SQUiD commit hash for your report (Table 1):

```bash
cd SQUiD && git log --oneline -1
cd ..
```

---

### Step 3: Create the virtual environment

```bash
python -m venv squid-env
source squid-env/bin/activate      # Mac / Linux
```

Your terminal prompt should show `(squid-env)` before proceeding.

---

### Step 4: Install dependencies

SQUiD's `requirements.txt` was written for Linux with NVIDIA GPU support. On Mac, filter out incompatible packages first:

```bash
grep -v -E "nvidia|triton" SQUiD/requirements.txt > requirements_mac.txt
pip install -r requirements_mac.txt
pip install pyyaml anthropic openai python-dotenv pandas matplotlib seaborn
```

For the sensitivity analysis you also need sentence-transformers (used for semantic similarity in the metric pipeline):

```bash
pip install sentence-transformers
```

> Additional packages sometimes missing from `requirements.txt`. Install as errors appear:
> ```bash
> pip install json-repair pyyaml anthropic
> ```

Verify:

```bash
python -c "import anthropic, pandas, matplotlib, yaml, json_repair; print('all ok')"
python -c "import sqlite3; print(sqlite3.version)"
python -c "import sentence_transformers; print('sentence-transformers ok')"
```

---

### Step 5: Configure your API key

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
2. Add a new secret. Name = `ANTHROPIC_API_KEY`, Value = your key
3. Access it in your notebook:

```python
from google.colab import userdata
import os
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

---

### Step 6: Update model.py to load the API key from .env

`SQUiD/src/model.py` uses an explicit dotenv path since SQUiD runs from inside its own subdirectory. The file in this repo is already updated, but verify the top of `model.py` contains:

```python
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
```

And that the Anthropic client uses:

```python
self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
```

---

### Step 7: Configure SQUiD

Update `SQUiD/configs/config.yaml`. Replace all `/add_path/` entries with your actual local paths:

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

Repeat for `value_identification`, `value_population`, `database_generation`, and `baseline`.

---

### Step 8: Create output directories

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
mkdir -p data/noisy
```

---

## Project Stages

All pipeline commands must be run from inside the `SQUiD/` directory:

```bash
cd SQUiD
```

---

### Stage 1: Schema Generation

Generates a relational schema (tables, columns, PKs, FKs) from the input text.

```bash
python src/schema_generation.py --model_name claude --prompt_type cot --method text
```

**Output:** `results/reproduction/stage1/BIRD/bird_dataset/text_cot_claude.json`

---

### Stage 2: Value Identification

Extracts values from the text and maps them to the schema produced in Stage 1.

```bash
python src/value_identification.py --model_name claude
```

**Input:** Stage 1 output
**Output:** `results/reproduction/stage2/`

---

### Stage 3: Table Population

Organizes extracted values into relational tuples with correct PK/FK assignments.

```bash
python src/value_population.py --model_name claude
```

**Input:** Stage 2 output
**Output:** `results/reproduction/stage3/`

---

### Stage 4: Database Materialization

Converts the populated tables into a valid, executable SQLite database.

```bash
python src/database_generation.py --model_name claude
```

**Input:** Stage 3 output
**Output:** `results/reproduction/stage4/*.db`

---

### Baseline: LLM Single-Shot

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

## Sensitivity Analysis

Tests how SQUiD handles noisy input. We generate 12 variants of the BIRD dataset by perturbing the text in four different ways, at three severity levels each (10%, 30%, and 60% of sentences affected).

### Noise types

1. **Contradictory facts.** Flip numeric values like years and zip codes so the same entity appears with conflicting attributes.
2. **Missing values.** Remove sentences entirely.
3. **Duplicate entities.** Shorten entity names so the same entity appears under multiple forms (for example, "Oakland High School" and "Oakland High").
4. **Structural errors.** Character swaps and dropped words inside sentences.

### Running it

Four steps, run in order:

```bash
# 1. Generate the 12 noisy datasets
python scripts/noise_injector.py

# 2. Run the SQUiD pipeline on all 12 variants
bash scripts/run_all_serial.sh

# 3. Run the baseline on all 12 variants.
#    Wait for step 2 to finish first. Both scripts write to
#    SQUiD/configs/config.yaml, and running them at the same time
#    causes a race condition where one overwrites the other's config.
python scripts/run_baseline_batch.py

# 4. Compute metrics and generate plots
python scripts/compute_sensitivity_metrics_v2.py
```

Step 2 takes about four hours. Step 3 takes about thirty minutes. Steps 1 and 4 are fast.

### Outputs

- `results/sensitivity/sensitivity_summary.csv`. One row per (variant, method) cell, 26 rows total.
- `results/sensitivity/plots/`. Five metric degradation curves and one heatmap.
- `results/sensitivity/report_findings.md`. Numeric summary and interpretation.

### Metrics

The metrics are computed directly from the Stage 4 JSON output and the generated SQLite databases. We do not use SQUiD's built-in evaluator because it has a schema mismatch bug. The join queries it generates in Stage 2 target idealized column names (like `school.id`) that do not match what Stage 3 actually creates in CREATE TABLE (like `school.school_id`). Every join fails with `no such column`, and the evaluator silently catches the error and returns zero. Our replacement metric script bypasses the join step and compares Stage 4 output against ground truth directly.

- **TC (tuple coverage).** Fraction of ground truth values whose semantic similarity to some cell in the database exceeds 0.7. Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings.
- **VC (value coverage).** Fraction of ground truth values that appear as a substring somewhere in the database.
- **CC (column consistency).** Fraction of ground truth values that end up in a column whose name matches their attribute.
- **RI (referential integrity).** Fraction of foreign key values that satisfy their parent primary key.
- **EX (executability).** Fraction of entries whose generated SQL runs without error.

### Key findings

SQUiD stays robust across most kinds of noise. TC ranges from 0.85 to 0.89 for contradictory facts, duplicate entities, and structural errors at every severity level tested. The one noise type that meaningfully degrades SQUiD is missing data at 60% perturbation, which drops TC to 0.786.

The baseline runs between TC 0.13 and 0.17 across every condition, clean or noisy. It does not get worse under noise, but only because it is already close to its performance floor. SQUiD outperforms the baseline on every variant tested.

### Limitations

- The clean reference point for SQUiD uses the earlier reproduction run at n=149 entries (Curtis's run). All 12 noisy variants ran at n=192. We kept the smaller clean run to conserve API budget. Within-noise comparisons use n=192 throughout and are unaffected by the mismatch.
- Two cells ran short because of mid-pipeline failures on heavily perturbed inputs: type 2 missing L1 (n=80) and type 3 duplicates L3 (n=147). Smaller n means wider confidence intervals for those two cells.
- Noise is injected at the sentence level with a fixed seed of 42. Results are deterministic for a given (seed, noise type, severity level).

---

## Datasets

The benchmark datasets are included inside the SQUiD repo. No external downloads needed.

| Dataset | Location | Domain | Notes |
|---|---|---|---|
| BIRD | `SQUiD/dataset/BIRD/bird_dataset.json` | Mixed (BIRD Text2SQL benchmark) | Used for reproduction and sensitivity runs |
| SyntheticText | `SQUiD/dataset/SyntheticText/merged_dataset2.json` | Synthetic travel/booking narrative | Paper's primary examples |
| RealDocuments | `SQUiD/dataset/RealDocuments/` | News (CNN, COVID, Wikipedia) | Real-world text domain |

The datasets are pre-generated natural language text files. The authors converted existing relational databases into narrative text using LLMs. They are not raw CSV files.

---

## Folder Structure

```
squid-reproduction/
│
├── SQUiD/                              # Official SQUiD codebase (included in repo)
│   ├── configs/config.yaml             # Pipeline config. Fill in your local paths.
│   ├── dataset/                        # All benchmark datasets
│   │   ├── BIRD/
│   │   ├── SyntheticText/
│   │   └── RealDocuments/
│   └── src/
│       ├── schema_generation.py        # Stage 1
│       ├── value_identification.py     # Stage 2
│       ├── value_population.py         # Stage 3
│       ├── database_generation.py      # Stage 4
│       ├── baseline.py                 # LLM baseline
│       ├── model.py                    # LLM client (updated to load from .env)
│       └── utils.py
│
├── data/
│   └── noisy/                          # 12 noisy variants (4 types x 3 levels)
│
├── scripts/
│   ├── noise_injector.py               # Generates the 12 noisy dataset variants
│   ├── run_squid_batch.py              # Batch SQUiD runs on noisy inputs
│   ├── run_all_serial.sh               # Serial wrapper (avoids config.yaml race)
│   ├── run_baseline_batch.py           # Batch baseline runs on noisy inputs
│   ├── compute_sensitivity_metrics.py  # First-pass metrics (broken evaluator)
│   └── compute_sensitivity_metrics_v2.py  # Working metrics pipeline
│
├── results/
│   ├── reproduction/
│   │   ├── stage1/                     # Schema Generation output
│   │   ├── stage2/                     # Value Identification output
│   │   ├── stage3/                     # Table Population output
│   │   ├── stage4/                     # SQLite databases
│   │   ├── baseline/                   # Baseline LLM output
│   │   └── logs/
│   └── sensitivity/
│       ├── sensitivity_summary.csv     # 26-row metric table
│       ├── plots/                      # 6 PNGs (5 degradation curves + 1 heatmap)
│       └── report_findings.md          # Numeric summary + interpretation
│
├── .env                                # Your API key. Never commit this.
├── .env.example                        # Template. Copy to .env
├── .gitignore
├── requirements_mac.txt                # Mac-compatible dependency list
└── README.md
```

---

## Task Ownership

| Area | Owner |
|---|---|
| Dataset access and verification | Curtis |
| Stage 1: Schema Generation | Curtis |
| Stage 2: Value Identification | Curtis |
| Stage 3: Table Population | Curtis |
| Stage 4: Database Materialization | Curtis |
| All four reproduction metrics | Curtis |
| LLM baseline | Curtis |
| Hypothesis 1 analysis (Performance) | Jaret |
| Hypothesis 2 analysis (Integrity) | Curtis |
| Noise injection script (`scripts/noise_injector.py`) | Zach |
| SQUiD batch runs on noisy variants | Zach |
| Baseline batch runs on noisy variants | Zach |
| Metrics aggregation and charts | Zach |
| Sensitivity analysis writeup | Zach |
| Presentation slides | Shared |

---

## Task Dependencies

```
 Stage 1 complete
        │
        ▼
 Stage 2: Value Identification
        │
        ▼
 Stage 3 → Stage 4 → compute metrics
        │
        ▼
 noise_injector.py → 12 noisy files generated
        │
        ├──▶ SQUiD batch runs on all 12 variants
        │
        ▼ (wait for SQUiD batch to finish)
        │
        ├──▶ baseline batch runs on all 12 variants
                        │
                        ▼
                aggregate metrics + build charts
                        │
                        ▼
                write Results and Discussion sections
```

---

## Reproducibility Notes

- **LLM model:** `claude-haiku-4-5-20251001`. Record in Table 1 of the report. The paper used `claude-3-7-sonnet-20241022`, which was unavailable on our account tier.
- **Random seed:** The noise injection script uses `RANDOM_SEED = 42`. Do not change this. It must be consistent across all runs.
- **SQUiD commit hash:** Run `git log --oneline -1` inside the `SQUiD/` folder and record the hash in the report.
- **Mac users:** Use `requirements_mac.txt`, not `requirements.txt`. The full file includes NVIDIA/CUDA packages that cannot install on Apple Silicon.
- **Python version:** Tested on 3.13. Some packages specify `<3.13`. Install individually at the latest compatible version if they fail.
- **Save outputs after each stage:** Do not rely on terminal history. Write results to disk and commit before moving to the next stage.

---

## Common Issues

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'yaml'` | `pip install pyyaml` |
| `ModuleNotFoundError: No module named 'anthropic'` | `pip install anthropic` |
| `ModuleNotFoundError: No module named 'json_repair'` | `pip install json-repair` |
| `ModuleNotFoundError: No module named 'sentence_transformers'` | `pip install sentence-transformers` |
| `nvidia-cublas-cu12` install failure | Use `requirements_mac.txt`. Filters out all NVIDIA packages. |
| `triton` not found | `grep -v -E "nvidia\|triton" requirements.txt > requirements_mac.txt` |
| `FileNotFoundError: configs/config.yaml` | Run from inside `SQUiD/`, not project root |
| `FileNotFoundError` on output path | Create missing dir with `mkdir -p` |
| `model: claude-3-7-sonnet not found` | Update model string in `SQUiD/src/model.py`. Run `python -c "import anthropic, os; from dotenv import load_dotenv; load_dotenv(); client = anthropic.Anthropic(); [print(m.id) for m in client.models.list().data]"` to see available models. |
| `credit balance too low` | Add credits at console.anthropic.com → Settings → Billing |
| API key shows `NOT SET` | Check `.env` has `ANTHROPIC_API_KEY=...` and `model.py` uses the explicit Path load |
| SQUiD appears as submodule in Git | `rm -rf SQUiD/.git` then `git add SQUiD/` |
| `.idea/` folder in Git | Add `.idea/` to `.gitignore`, then `git rm -r --cached .idea` |
| `sqlite3` install error | Do not pip install. It is standard library. Verify: `python -c "import sqlite3; print(sqlite3.version)"` |
| Baseline and SQUiD batch interfere with each other | Do not run them at the same time. They both write to `SQUiD/configs/config.yaml`. Run SQUiD batch to completion first, then baseline batch. |
| All SQUiD metrics return 0.000 | This is the Stage 2 vs Stage 3 schema mismatch bug in SQUiD's evaluator. Use `scripts/compute_sensitivity_metrics_v2.py` instead. |

---

## What NOT to Commit

The `.gitignore` already covers these:

- `.env`. Your API key. Never commit it.
- `*.db`. SQLite databases.
- `squid-env/`. Your local virtual environment.
- `__pycache__/` and `*.pyc`. Python bytecode.
- `.idea/`. IDE project settings.
- `SQUiD/configs/config.yaml`. Rewritten on every run with local paths.
- `results/sensitivity/*/stage*/`, `results/sensitivity/*/TS/`, `results/sensitivity/*/baseline/`. Large generated pipeline outputs.
- `SQUiD/databases/`. Generated SQLite files.

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

Post in the team group chat. If you are stuck on something for more than 30 minutes, flag it. Do not block the pipeline silently.
