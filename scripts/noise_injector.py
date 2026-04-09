"""
noise_injector.py
-----------------
Generates 12 noisy dataset variants for the SQUiD noise sensitivity experiment.

4 noise types x 3 noise levels = 12 output files in data/noisy/

Usage:
    python scripts/noise_injector.py

Output files:
    data/noisy/type1_contradictory_L1.txt
    data/noisy/type1_contradictory_L2.txt
    data/noisy/type1_contradictory_L3.txt
    data/noisy/type2_missing_L1.txt
    ... (12 files total)

Owner: Curtis Moore
Random seed: 42 — DO NOT CHANGE. Documented in report Section 5.2.
"""

import random
import re
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

RANDOM_SEED = 42  # Fixed for reproducibility — document in report Section 5.2

# Update if using a different dataset
INPUT_FILE = "SQUiD/dataset/BIRD/bird_dataset.json"
OUTPUT_DIR = "data/noisy"

NOISE_LEVELS = {
    "L1": 0.10,  # 10% of sentences modified — low noise
    "L2": 0.30,  # 30% of sentences modified — moderate noise
    "L3": 0.60,  # 60% of sentences modified — high noise
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_text(filepath):
    """
    Load input text from the dataset file.
    SQUiD's BIRD dataset is a JSON file where each entry has a 'text' field.
    Adapt this function if using a plain .txt input instead.
    """
    import json
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}\n"
                                f"Make sure you are running from the project root (squid-reproduction/).")

    with open(path) as f:
        data = json.load(f)

    # Extract text from each entry and flatten into sentences
    all_sentences = []
    for entry in data:
        text = entry.get("text", "")
        # Split on period followed by space or end of string
        sentences = [s.strip() for s in re.split(r'\.\s+', text) if s.strip()]
        all_sentences.extend(sentences)

    print(
        f"Loaded {len(all_sentences)} sentences from {len(data)} entries in {filepath}")
    return all_sentences


def sample_indices(sentences, noise_level):
    """
    Select which sentence indices to perturb using a fixed random seed.
    Same seed + same level = same indices every run = reproducible results.
    """
    rng = random.Random(RANDOM_SEED)
    n = max(1, int(len(sentences) * noise_level))
    return set(rng.sample(range(len(sentences)), n))


def sentences_to_text(sentences):
    """Rejoin sentences into a single paragraph."""
    return ". ".join(s for s in sentences if s).rstrip(".") + "."


# ── Noise Type 1 — Contradictory Facts ───────────────────────────────────────

def inject_type1(sentences, noise_level):
    """
    Replace numeric values (ages, counts, years) with conflicting values.

    Target: Stages 1 and 3 — the schema must reconcile conflicting attribute
    values for the same entity. This tests whether SQUiD's symbolic constraints
    prevent hallucinated or averaged values from entering the database.

    IMPORTANT: The regex patterns below are tuned for the BIRD dataset which
    contains school/education data with ages, years, and numeric codes.
    If you switch datasets, inspect the text and adapt these patterns.
    """
    indices = sample_indices(sentences, noise_level)
    result = sentences.copy()

    for i in indices:
        s = result[i]
        modified = False

        # Pattern: numbers that look like years (4-digit starting with 19 or 20)
        year_match = re.search(r'\b(19|20)\d{2}\b', s)
        if year_match and not modified:
            orig = int(year_match.group())
            new_val = orig + random.Random(i).randint(1, 10)
            result[i] = s.replace(str(orig), str(new_val), 1)
            modified = True

        # Pattern: age-like numbers (2-digit standalone numbers)
        if not modified:
            age_match = re.search(r'\b(\d{2})\b', s)
            if age_match:
                orig = int(age_match.group())
                new_val = orig + random.Random(i + 1000).randint(3, 15)
                result[i] = s.replace(str(orig), str(new_val), 1)
                modified = True

        # Pattern: any remaining standalone number
        if not modified:
            num_match = re.search(r'\b(\d+)\b', s)
            if num_match:
                orig = int(num_match.group())
                new_val = orig + random.Random(i + 2000).randint(1, 50)
                result[i] = s.replace(str(orig), str(new_val), 1)

    return result


# ── Noise Type 2 — Missing Values ────────────────────────────────────────────

def inject_type2(sentences, noise_level):
    """
    Remove sentences entirely from the input text.

    Target: Stage 2 — Value Identification. Removed sentences mean the LLM
    cannot extract the values they contained, producing NULL or empty columns
    in the final database. Tests whether SQUiD gracefully handles incomplete input.

    Note: At L3 (60%), a significant portion of sentences are removed. For short
    inputs this may produce near-empty text. If any output file has fewer than
    3 sentences, cap the noise level at 50% and document in the report.
    """
    indices = sample_indices(sentences, noise_level)
    result = [s for i, s in enumerate(sentences) if i not in indices]

    # Safety check — ensure at least 3 sentences remain
    if len(result) < 3:
        print(f"  Warning: Type 2 at this noise level left only {len(result)} sentences. "
              f"Consider capping L3 at 50% for short inputs.")

    return result


# ── Noise Type 3 — Duplicate Entities ────────────────────────────────────────

def inject_type3(sentences, noise_level):
    """
    Replace entity names with alternate variants so the same entity appears
    under multiple names in the same document.

    Target: Stages 1 and 3 — FK/PK integrity. When the same entity appears
    as 'Fremont High' in one sentence and 'Fremont' in another, SQUiD may
    create two separate rows for the same entity, breaking FK chains.

    IMPORTANT: The entity patterns below are tuned for the BIRD education dataset.
    Common entity types: school names, district names, county names.
    Adapt the substitution rules to match entities in your actual dataset text.
    Run: head -5 SQUiD/dataset/BIRD/bird_dataset.json | python -c "import sys,json; [print(e['text'][:200]) for e in json.load(sys.stdin)[:2]]"
    to inspect what entities appear.
    """
    indices = sample_indices(sentences, noise_level)
    result = sentences.copy()
    rng = random.Random(RANDOM_SEED + 3000)

    for i in indices:
        s = result[i]

        # Drop title/honorific prefixes to create name variants
        s = re.sub(r'\bMr\.\s+', '', s)
        s = re.sub(r'\bMrs\.\s+', '', s)
        s = re.sub(r'\bDr\.\s+', '', s)
        s = re.sub(r'\bProf\.\s+', '', s)

        # Abbreviate "High School" to "High" to create school name variant
        s = re.sub(r'\bHigh School\b', 'High', s)

        # Abbreviate "Unified School District" to "Unified" to create district variant
        s = re.sub(r'\bUnified School District\b', 'Unified', s)

        # Drop "County" suffix from county names
        if rng.random() < 0.5:
            s = re.sub(r'\b(\w+)\s+County\b', r'\1', s)

        result[i] = s

    return result


# ── Noise Type 4 — Structural Noise ──────────────────────────────────────────

def inject_type4(sentences, noise_level):
    """
    Introduce character-level and word-level errors to degrade sentence structure.

    Target: All stages — structural noise tests the general robustness of SQUiD's
    NLP components. If a sentence is ungrammatical or has typos, information
    extraction at every stage becomes harder.

    Perturbations applied:
    - Swap two adjacent characters in a word (simulates typos)
    - Drop one word from the sentence (simulates transcription errors)
    """
    indices = sample_indices(sentences, noise_level)
    result = sentences.copy()

    for i in indices:
        words = result[i].split()
        if len(words) < 3:
            continue  # Skip very short sentences — not enough to perturb meaningfully

        rng = random.Random(RANDOM_SEED + i + 4000)

        # Perturbation 1: swap two adjacent characters in one word
        word_idx = rng.randint(0, len(words) - 1)
        chars = list(words[word_idx])
        if len(chars) > 2:
            swap_pos = rng.randint(0, len(chars) - 2)
            chars[swap_pos], chars[swap_pos +
                                   1] = chars[swap_pos + 1], chars[swap_pos]
        words[word_idx] = "".join(chars)

        # Perturbation 2: drop one word (not the first or last)
        if len(words) > 3:
            drop_idx = rng.randint(1, len(words) - 2)
            words.pop(drop_idx)

        result[i] = " ".join(words)

    return result


# ── Main Generation Loop ──────────────────────────────────────────────────────

def generate_all_variants(input_file=INPUT_FILE, output_dir=OUTPUT_DIR):
    """
    Generate all 12 noisy variants and save to output_dir.
    Prints a summary of what was generated.
    """
    sentences = load_text(input_file)

    noise_types = {
        "type1_contradictory": inject_type1,
        "type2_missing":       inject_type2,
        "type3_duplicates":    inject_type3,
        "type4_structural":    inject_type4,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    generated = []
    print(f"\nGenerating 12 noisy variants into {output_dir}/\n")

    for type_name, inject_fn in noise_types.items():
        for level_name, level_val in NOISE_LEVELS.items():
            noisy_sentences = inject_fn(sentences, level_val)
            out_text = sentences_to_text(noisy_sentences)
            out_path = Path(output_dir) / f"{type_name}_{level_name}.txt"

            with open(out_path, "w") as f:
                f.write(out_text)

            n_modified = abs(len(sentences) - len(noisy_sentences)) if type_name == "type2_missing" \
                else int(len(sentences) * level_val)

            print(f"  {out_path.name:45s}  {len(noisy_sentences):4d} sentences  "
                  f"(~{n_modified} modified at {int(level_val*100)}%)")
            generated.append(out_path)

    print(f"\nDone. {len(generated)} files created.")
    return generated


def verify_outputs(output_dir=OUTPUT_DIR):
    """
    Quick verification — check all 12 files exist and are non-empty.
    Prints a diff of clean vs. first noisy variant for each type.
    """
    import difflib

    print("\n── Verification ───────────────────────────────────────────────")
    expected = [
        f"{t}_{l}.txt"
        for t in ["type1_contradictory", "type2_missing", "type3_duplicates", "type4_structural"]
        for l in ["L1", "L2", "L3"]
    ]

    all_ok = True
    for fname in expected:
        path = Path(output_dir) / fname
        if not path.exists():
            print(f"  MISSING: {fname}")
            all_ok = False
        elif path.stat().st_size == 0:
            print(f"  EMPTY:   {fname}")
            all_ok = False
        else:
            size = path.stat().st_size
            print(f"  OK:      {fname:45s}  {size:8,d} bytes")

    if all_ok:
        print("\nAll 12 files present and non-empty.")
    else:
        print("\nSome files are missing or empty — check the errors above.")

    return all_ok


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Allow overriding input file and output dir from command line
    input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR

    print("SQUiD Noise Injector")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_dir}/")
    print(f"  Seed:   {RANDOM_SEED}")
    print(f"  Levels: L1=10%, L2=30%, L3=60%")

    generated = generate_all_variants(input_file, output_dir)
    verify_outputs(output_dir)

    print("\nNext steps:")
    print("  1. Inspect a few files: head -3 data/noisy/type1_contradictory_L2.txt")
    print("  2. Diff clean vs. noisy: diff <(head -10 SQUiD/dataset/BIRD/bird_dataset.json) data/noisy/type1_contradictory_L1.txt")
    print("  3. Tell Jaret and Zach — they can now run their batch scripts on data/noisy/")
