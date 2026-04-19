"""
noise_injector.py  (v2 — per-entry, JSON output)
-------------------------------------------------
Generates 12 SQUiD-compatible noisy dataset variants for the noise sensitivity
experiment.

    4 noise types  x  3 noise levels  =  12 output JSON files

Each variant is a FULL BIRD dataset (192 entries) with the same JSON schema as
SQUiD/dataset/BIRD/bird_dataset.json:
    - text                     (PERTURBED)
    - ground_truth_entities    (preserved)
    - ground_truth_key_value   (preserved)
    - domain                   (preserved)
    - difficulty               (preserved)

Ground truth is intentionally FROZEN — not updated to match the perturbed text.
Rationale: the experiment tests the system's robustness to corrupted input.
A drop in metric against frozen GT = a drop in robustness. Documented in
report Section 5.2.

Usage:
    python scripts/noise_injector.py

Outputs (two copies of each file):
    data/noisy/bird_type1_contradictory_L1.json       (for inspection/report)
    SQUiD/dataset/BIRD/bird_type1_contradictory_L1.json   (for the pipeline)
    ... (12 variants)

Owner: Curtis Moore
Seed:  42 — do NOT change
"""

import json
import random
import re
import shutil
import sys
from pathlib import Path

# Configuration 

RANDOM_SEED = 42

# Resolve paths relative to the repo root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = REPO_ROOT / "SQUiD" / "dataset" / "BIRD" / "bird_dataset.json"

OUTPUT_DIR = REPO_ROOT / "data" / "noisy"
# Copy variants into SQUiD's dataset dir so the pipeline can read them
SQUID_BIRD_DIR = REPO_ROOT / "SQUiD" / "dataset" / "BIRD"

NOISE_LEVELS = {
    "L1": 0.10,
    "L2": 0.30,
    "L3": 0.60,
}

# Sentence splitting / rejoining

def split_sentences(text):
    """Split on period+whitespace, preserving the trailing period."""
    parts = re.split(r'(?<=\.)\s+', text)
    return [p.strip() for p in parts if p.strip()]


def join_sentences(sentences):
    """Rejoin sentences with a single space, add final period if missing."""
    joined = " ".join(s for s in sentences if s)
    if joined and not joined.endswith("."):
        joined += "."
    return joined


def sample_indices(n_sentences, noise_level, entry_idx):
    """Deterministic sentence-index sampler. Same (seed, entry_idx, level) = same set."""
    rng = random.Random(RANDOM_SEED + entry_idx)
    n = max(1, int(round(n_sentences * noise_level)))
    if n >= n_sentences:
        return set(range(n_sentences))
    return set(rng.sample(range(n_sentences), n))


# Type 1 - Contradictory Facts 

def inject_type1(sentences, noise_level, entry_idx):
    """
    Flip numeric values (years, zip codes, codes) to conflicting values.
    Targets Stages 1 & 3 — does SQUiD reconcile conflicting attributes
    for the same entity, or hallucinate/average them?
    """
    indices = sample_indices(len(sentences), noise_level, entry_idx)
    result = sentences.copy()
    rng = random.Random(RANDOM_SEED + entry_idx + 1000)

    for i in indices:
        s = result[i]

        # 1) 4-digit year (1900-2099)
        m = re.search(r'\b(19|20)\d{2}\b', s)
        if m:
            orig = m.group()
            new_val = str(int(orig) + rng.randint(1, 10))
            result[i] = s.replace(orig, new_val, 1)
            continue

        # 2) 5-digit zip / code
        m = re.search(r'\b\d{5}\b', s)
        if m:
            orig = m.group()
            new_val = str(int(orig) + rng.randint(10, 99))
            result[i] = s.replace(orig, new_val, 1)
            continue

        # 3) Any multi-digit number
        m = re.search(r'\b(\d{2,})\b', s)
        if m:
            orig = m.group()
            new_val = str(int(orig) + rng.randint(5, 50))
            result[i] = s.replace(orig, new_val, 1)

    return result


# Type 2 - Missing Values

def inject_type2(sentences, noise_level, entry_idx):
    """
    Remove sentences (simulates redacted or lost content).
    Targets Stage 2 — extraction cannot recover values not present in text.
    """
    indices = sample_indices(len(sentences), noise_level, entry_idx)
    result = [s for i, s in enumerate(sentences) if i not in indices]
    # Safety: keep at least 2 sentences so downstream code has something to parse
    if len(result) < 2:
        result = sentences[:2]
    return result


# Type 3 — Duplicate Entities (name variants) 

def inject_type3(sentences, noise_level, entry_idx):
    """
    Substitute entity names with shortened variants so the same entity appears
    under multiple forms. Targets FK/PK integrity (Hypothesis 2).
    """
    indices = sample_indices(len(sentences), noise_level, entry_idx)
    result = sentences.copy()
    rng = random.Random(RANDOM_SEED + entry_idx + 3000)

    for i in indices:
        s = result[i]
        # Drop honorifics
        for prefix in (r'\bMr\.\s+', r'\bMrs\.\s+', r'\bMs\.\s+',
                       r'\bDr\.\s+', r'\bProf\.\s+'):
            s = re.sub(prefix, '', s)
        # Institution-type suffixes
        s = re.sub(r'\bHigh School\b', 'High', s)
        s = re.sub(r'\bElementary School\b', 'Elementary', s)
        s = re.sub(r'\bMiddle School\b', 'Middle', s)
        s = re.sub(r'\bUnified School District\b', 'Unified', s)
        s = re.sub(r'\bSchool District\b', 'District', s)
        # Drop "County" suffix half the time
        if rng.random() < 0.5:
            s = re.sub(r'\b(\w+)\s+County\b', r'\1', s)
        result[i] = s

    return result


# Type 4 — Structural (typos + dropped words) 

def inject_type4(sentences, noise_level, entry_idx):
    """
    Character-swap + word-drop. Targets all stages (degrades NLP performance).
    """
    indices = sample_indices(len(sentences), noise_level, entry_idx)
    result = sentences.copy()

    for i in indices:
        words = result[i].split()
        if len(words) < 3:
            continue
        rng = random.Random(RANDOM_SEED + entry_idx + i + 4000)

        # Swap two adjacent chars inside a random word's alpha portion
        word_idx = rng.randint(0, len(words) - 1)
        chars = list(words[word_idx])
        alpha_end = len(chars)
        while alpha_end > 0 and not chars[alpha_end - 1].isalpha():
            alpha_end -= 1
        if alpha_end > 2:
            swap_pos = rng.randint(0, alpha_end - 2)
            chars[swap_pos], chars[swap_pos + 1] = chars[swap_pos + 1], chars[swap_pos]
        words[word_idx] = "".join(chars)

        # Drop a non-edge word
        if len(words) > 3:
            drop_idx = rng.randint(1, len(words) - 2)
            words.pop(drop_idx)

        result[i] = " ".join(words)

    return result


# Driver

NOISE_FUNCS = {
    "type1_contradictory": inject_type1,
    "type2_missing":       inject_type2,
    "type3_duplicates":    inject_type3,
    "type4_structural":    inject_type4,
}


def generate_variant(clean_dataset, noise_type, noise_level):
    """Produce one full noisy dataset for a given (type, level)."""
    inject_fn = NOISE_FUNCS[noise_type]
    out_entries = []
    n_total_sentences = 0
    n_modified = 0

    for idx, entry in enumerate(clean_dataset):
        sentences = split_sentences(entry["text"])
        n_total_sentences += len(sentences)

        noisy = inject_fn(sentences, noise_level, idx)

        if noise_type == "type2_missing":
            n_modified += len(sentences) - len(noisy)
        else:
            n_modified += sum(
                1 for a, b in zip(sentences, noisy) if a != b
            )

        new_entry = dict(entry)  # shallow copy — preserves GT, domain, difficulty
        new_entry["text"] = join_sentences(noisy)
        out_entries.append(new_entry)

    return out_entries, n_modified, n_total_sentences


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input dataset not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_FILE) as f:
        clean_dataset = json.load(f)
    print(f"Loaded clean dataset: {len(clean_dataset)} entries from {INPUT_FILE.name}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SQUID_BIRD_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for noise_type in NOISE_FUNCS:
        for level_name, level_val in NOISE_LEVELS.items():
            variant_name = f"bird_{noise_type}_{level_name}"
            variant, n_mod, n_total = generate_variant(
                clean_dataset, noise_type, level_val
            )

            out_path = OUTPUT_DIR / f"{variant_name}.json"
            with open(out_path, "w") as f:
                json.dump(variant, f, indent=2)

            # Mirror into SQUiD dataset dir so the pipeline can load it
            squid_path = SQUID_BIRD_DIR / f"{variant_name}.json"
            shutil.copy(out_path, squid_path)

            pct = (n_mod / n_total * 100) if n_total else 0
            summary_rows.append((variant_name, len(variant), n_mod, n_total, pct))
            print(
                f"  {variant_name:45s}  "
                f"{n_mod:5d}/{n_total:5d} sentences perturbed  "
                f"({pct:5.1f}%  target {int(level_val*100)}%)"
            )

    print(f"\nWrote {len(summary_rows)} variant JSONs to:")
    print(f"  {OUTPUT_DIR}")
    print(f"  {SQUID_BIRD_DIR}")
    print("\nEach file has 192 entries with perturbed text and FROZEN ground truth.")
    print("Next: python scripts/run_squid_batch.py")


if __name__ == "__main__":
    main()
