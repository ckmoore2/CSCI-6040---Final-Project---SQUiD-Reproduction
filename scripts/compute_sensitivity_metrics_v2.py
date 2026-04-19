"""
compute_sensitivity_metrics_v2.py
---------------------------------
Computes sensitivity metrics directly from Stage-4 outputs and built SQLite
DBs, bypassing SQUiD's broken join-based evaluator.

Metrics (all in [0,1], higher = better):
  EX  executability       fraction of entries whose sql_statements run cleanly
  RI  referential integ.  fraction of FK values whose parent PK exists
  TC  tuple coverage      semantic similarity >= 0.7 of GT values to DB cells
  VC  value coverage      exact substring presence of GT values in DB cells
  CC  column consistency  GT attr matches a DB column name containing its value

Usage: python scripts/compute_sensitivity_metrics_v2.py
"""

import csv
import json
import re
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SQUID_DIR = REPO_ROOT / "SQUiD"
RESULTS_DIR = REPO_ROOT / "results" / "sensitivity"
REPRODUCTION_DIR = REPO_ROOT / "results" / "reproduction"
DATABASES_DIR = SQUID_DIR / "databases" / "BIRD"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_NAMES = [
    "bird_type1_contradictory_L1", "bird_type1_contradictory_L2", "bird_type1_contradictory_L3",
    "bird_type2_missing_L1", "bird_type2_missing_L2", "bird_type2_missing_L3",
    "bird_type3_duplicates_L1", "bird_type3_duplicates_L2", "bird_type3_duplicates_L3",
    "bird_type4_structural_L1", "bird_type4_structural_L2", "bird_type4_structural_L3",
]


def squid_stage4_path(v):
    if v == "clean":
        return REPRODUCTION_DIR / "stage4TS" / "BIRD" / "bird_dataset" / "text_cot_claude.json"
    return RESULTS_DIR / v / "TS" / "BIRD" / v / "text_cot_claude.json"


def baseline_stage4_path(v):
    if v == "clean":
        return REPRODUCTION_DIR / "baseline" / "BIRD" / "bird_dataset" / "text_cot_claude.json"
    return RESULTS_DIR / v / "baseline" / "BIRD" / v / "text_cot_claude.json"


def squid_db_dir(v):
    if v == "clean":
        return DATABASES_DIR / "bird_dataset" / "text_cot_claude" / "TS"
    return DATABASES_DIR / v / "text_cot_claude" / "TS"


def baseline_db_dir(v):
    if v == "clean":
        return DATABASES_DIR / "bird_dataset" / "text_cot_claude" / "baseline"
    return DATABASES_DIR / v / "text_cot_claude" / "baseline"


# Sentence-transformer model

print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("  ready.")


def batch_best_match(needles, haystack):
    if not needles or not haystack:
        return [0.0] * len(needles)
    nv = MODEL.encode([str(x) for x in needles], convert_to_tensor=True, show_progress_bar=False)
    hv = MODEL.encode([str(x) for x in haystack], convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(nv, hv)
    return sims.max(dim=1).values.tolist()


# Ground truth flattening (handles deeply nested lists/dicts)

def _deep_flatten(v):
    """Recursively flatten any nested list/tuple/dict into scalar strings."""
    out = []
    if isinstance(v, (list, tuple)):
        for item in v:
            out.extend(_deep_flatten(item))
    elif isinstance(v, dict):
        for vv in v.values():
            out.extend(_deep_flatten(vv))
    elif v is None:
        pass
    else:
        out.append(str(v))
    return out


def ground_truth_values(entry):
    gt = entry.get("ground_truth_key_value", {})
    out = _deep_flatten(gt)
    return [x for x in out if x and x.lower() not in ("none", "null", "na")]


# executability 

def compute_ex_for_entry(entry):
    sql = entry.get("sql_statements", "")
    if not sql or not isinstance(sql, str):
        return False
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(sql)
        conn.close()
        return True
    except Exception:
        return False


# Referential integrity

_FK_PATTERN = re.compile(
    r"`?(\w+)`?\s+\w+.*?REFERENCES\s+`?(\w+)`?\s*\(\s*`?(\w+)`?\s*\)",
    re.IGNORECASE,
)


def extract_fks_from_sql(sql):
    fks = []
    blocks = re.findall(
        r"CREATE\s+TABLE\s+`?(\w+)`?\s*\((.*?)\)\s*;",
        sql, flags=re.IGNORECASE | re.DOTALL,
    )
    for table_name, block in blocks:
        for m in _FK_PATTERN.finditer(block):
            col, ref_table, ref_col = m.group(1), m.group(2), m.group(3)
            if col.upper() == "FOREIGN":
                continue
            fks.append((table_name, col, ref_table, ref_col))
    return fks


def compute_ri_for_entry(entry, db_path):
    sql = entry.get("sql_statements", "")
    if not sql:
        return 0.0
    fks = extract_fks_from_sql(sql)
    if not fks:
        return 1.0
    if not db_path.exists():
        return 0.0
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        per_fk = []
        for table, col, ref_table, ref_col in fks:
            try:
                cur.execute(f"SELECT `{col}` FROM `{table}` WHERE `{col}` IS NOT NULL")
                child_vals = [r[0] for r in cur.fetchall()]
                if not child_vals:
                    per_fk.append(1.0)
                    continue
                cur.execute(f"SELECT `{ref_col}` FROM `{ref_table}`")
                parent_vals = set(r[0] for r in cur.fetchall())
                ok = sum(1 for v in child_vals if v in parent_vals)
                per_fk.append(ok / len(child_vals))
            except sqlite3.Error:
                per_fk.append(0.0)
        conn.close()
        return sum(per_fk) / len(per_fk) if per_fk else 1.0
    except sqlite3.Error:
        return 0.0


# Extract rows from DB

def all_rows_from_db(db_path):
    if not db_path.exists():
        return []
    rows = []
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            try:
                cur.execute(f"SELECT * FROM `{t}`")
                cols = [d[0] for d in cur.description]
                for r in cur.fetchall():
                    for c, v in zip(cols, r):
                        if v is None:
                            continue
                        rows.append({"table": t, "column": c, "value": str(v)})
            except sqlite3.Error:
                continue
        conn.close()
    except sqlite3.Error:
        pass
    return rows


# TC / VC / CC

def compute_tc_for_entry(entry, db_path, threshold=0.7):
    gt = ground_truth_values(entry)
    if not gt:
        return 0.0
    db_rows = all_rows_from_db(db_path)
    if not db_rows:
        return 0.0
    db_values = list({r["value"] for r in db_rows})
    sims = batch_best_match(gt, db_values)
    return sum(1 for s in sims if s >= threshold) / len(gt)


def compute_vc_for_entry(entry, db_path):
    gt = ground_truth_values(entry)
    if not gt:
        return 0.0
    db_rows = all_rows_from_db(db_path)
    if not db_rows:
        return 0.0
    db_blob = " | ".join(r["value"].lower() for r in db_rows)
    return sum(1 for v in gt if v.lower() in db_blob) / len(gt)


def compute_cc_for_entry(entry, db_path, threshold=0.5):
    gt = entry.get("ground_truth_key_value", {})
    pairs = []

    def _collect(obj, key_hint=None):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _collect(v, key_hint=k)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect(item, key_hint=key_hint)
        elif obj is not None and key_hint is not None:
            s = str(obj)
            if s and s.lower() not in ("none", "null", "na"):
                pairs.append((str(key_hint), s))

    _collect(gt)
    if not pairs:
        return 0.0
    db_rows = all_rows_from_db(db_path)
    if not db_rows:
        return 0.0
    idx = {}
    for r in db_rows:
        idx.setdefault(r["value"].lower(), []).append(r["column"])
    ok = 0
    for attr, val in pairs:
        cols = idx.get(val.lower(), [])
        if not cols:
            for k, v in idx.items():
                if val.lower() in k or k in val.lower():
                    cols.extend(v)
        if not cols:
            continue
        sims = batch_best_match([attr], cols)
        if sims and sims[0] >= threshold:
            ok += 1
    return ok / len(pairs)


# Aggregate per cell
def evaluate_cell(variant_or_clean, method):
    if method == "squid":
        stage4_path = squid_stage4_path(variant_or_clean)
        db_dir = squid_db_dir(variant_or_clean)
    else:
        stage4_path = baseline_stage4_path(variant_or_clean)
        db_dir = baseline_db_dir(variant_or_clean)

    if not stage4_path.exists():
        print(f"    [skip] stage4 JSON not found: {stage4_path}")
        return {"tc": 0, "vc": 0, "cc": 0, "ri": 0, "ex": 0, "n": 0}

    with open(stage4_path) as f:
        entries = json.load(f)

    n = len(entries)
    ex_ok, ri_sum, tc_sum, vc_sum, cc_sum = 0, 0.0, 0.0, 0.0, 0.0
    for i, entry in enumerate(entries):
        if (i + 1) % 40 == 0:
            print(f"      entry {i+1}/{n}")
        db_name = entry.get("db_name", "")
        db_path = db_dir / f"{db_name}.db"

        if compute_ex_for_entry(entry):
            ex_ok += 1
        ri_sum += compute_ri_for_entry(entry, db_path)
        tc_sum += compute_tc_for_entry(entry, db_path)
        vc_sum += compute_vc_for_entry(entry, db_path)
        cc_sum += compute_cc_for_entry(entry, db_path)

    return {
        "tc": tc_sum / n, "vc": vc_sum / n, "cc": cc_sum / n,
        "ri": ri_sum / n, "ex": ex_ok / n, "n": n,
    }


# Plotting

NOISE_TYPES = ["type1_contradictory", "type2_missing", "type3_duplicates", "type4_structural"]
LEVELS = ["L1", "L2", "L3"]
LEVEL_PCT = {"L1": 10, "L2": 30, "L3": 60}


def plot_degradation(rows):
    by = {(r["variant"], r["method"]): r for r in rows}
    for metric in ["tc", "vc", "cc", "ri", "ex"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        cb = by.get(("clean", "baseline"), {}).get(metric, 0)
        cs = by.get(("clean", "squid"), {}).get(metric, 0)
        ax.axhline(cb, linestyle="--", alpha=0.4, color="gray", label=f"clean baseline = {cb:.3f}")
        ax.axhline(cs, linestyle=":", alpha=0.4, color="black", label=f"clean SQUiD = {cs:.3f}")
        colors = {"type1_contradictory": "#c0392b", "type2_missing": "#2980b9",
                  "type3_duplicates": "#27ae60", "type4_structural": "#8e44ad"}
        for nt in NOISE_TYPES:
            x = [LEVEL_PCT[l] for l in LEVELS]
            y_sq = [by.get((f"bird_{nt}_{l}", "squid"), {}).get(metric, 0) for l in LEVELS]
            y_bl = [by.get((f"bird_{nt}_{l}", "baseline"), {}).get(metric, 0) for l in LEVELS]
            ax.plot(x, y_sq, marker="o", color=colors[nt], label=f"{nt} (SQUiD)")
            ax.plot(x, y_bl, marker="s", linestyle="--", color=colors[nt], alpha=0.6, label=f"{nt} (baseline)")
        ax.set_xlabel("Noise level (% entries perturbed)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} degradation vs noise level")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        p = PLOTS_DIR / f"degradation_{metric}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"  wrote {p}")


def plot_heatmap(rows):
    by = {(r["variant"], r["method"]): r for r in rows}
    variants_ordered = ["clean"] + VARIANT_NAMES
    metrics = ["tc", "vc", "cc", "ri", "ex"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for col, method in enumerate(["squid", "baseline"]):
        M = np.array([
            [by.get((v, method), {}).get(m, 0) for m in metrics]
            for v in variants_ordered
        ])
        ax = axes[col]
        im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_yticks(range(len(variants_ordered)))
        if col == 0:
            ax.set_yticklabels([v.replace("bird_", "") for v in variants_ordered], fontsize=8)
        ax.set_title(f"{method.upper()}")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Metric heatmap: SQUiD vs Baseline across all variants")
    fig.tight_layout()
    p = PLOTS_DIR / "squid_vs_baseline_heatmap.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")


def write_findings(rows):
    by = {(r["variant"], r["method"]): r for r in rows}
    path = RESULTS_DIR / "report_findings.md"
    with open(path, "w") as f:
        f.write("# SQUiD Noise Sensitivity - Findings\n\n")
        f.write("## Methodology\n\n")
        f.write("Five metrics computed directly from Stage-4 outputs and the\n")
        f.write("built SQLite databases, bypassing SQUiD's internal\n")
        f.write("`database_evaluation.py` which fails silently due to a schema\n")
        f.write("mismatch between Stage 2 (join query) and Stage 3 (CREATE TABLE):\n\n")
        f.write("- **TC**  Tuple Coverage (semantic similarity of GT values to DB cells, threshold 0.7)\n")
        f.write("- **VC**  Value Coverage (exact substring presence in any DB cell)\n")
        f.write("- **CC**  Column Consistency (values landing in columns whose names match the GT attribute)\n")
        f.write("- **RI**  Referential Integrity (fraction of FK values satisfying their parent PK)\n")
        f.write("- **EX**  Executability (fraction of entries whose sql_statements run without error)\n\n")

        f.write("## Clean reference\n\n")
        cb = by.get(("clean", "baseline"), {})
        cs = by.get(("clean", "squid"), {})
        f.write(f"- Clean SQUiD  - TC={cs.get('tc',0):.3f}, VC={cs.get('vc',0):.3f}, "
                f"CC={cs.get('cc',0):.3f}, RI={cs.get('ri',0):.3f}, EX={cs.get('ex',0):.3f}\n")
        f.write(f"- Clean Baseline - TC={cb.get('tc',0):.3f}, VC={cb.get('vc',0):.3f}, "
                f"CC={cb.get('cc',0):.3f}, RI={cb.get('ri',0):.3f}, EX={cb.get('ex',0):.3f}\n\n")

        f.write("## Per-variant table\n\n")
        f.write("| Variant | Method | TC | VC | CC | RI | EX |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for v in ["clean"] + VARIANT_NAMES:
            for method in ["squid", "baseline"]:
                r = by.get((v, method), {})
                f.write(f"| {v} | {method} | {r.get('tc',0):.3f} | {r.get('vc',0):.3f} | "
                        f"{r.get('cc',0):.3f} | {r.get('ri',0):.3f} | {r.get('ex',0):.3f} |\n")

        f.write("\n## Degradation per noise type\n\n")
        for nt in NOISE_TYPES:
            f.write(f"### {nt}\n\n")
            for method in ["squid", "baseline"]:
                f.write(f"- **{method}**  ")
                deltas = []
                clean = by.get(("clean", method), {})
                for l in LEVELS:
                    cell = by.get((f"bird_{nt}_{l}", method), {})
                    delta_tc = cell.get('tc', 0) - clean.get('tc', 0)
                    deltas.append(f"{l}: dTC={delta_tc:+.3f}")
                f.write("  ".join(deltas) + "\n")
            f.write("\n")

        f.write("## Interpretation\n\n")
        f.write("_Fill in after seeing the numbers. Key questions:_\n")
        f.write("1. Which noise type causes the largest TC drop for SQUiD?\n")
        f.write("2. Is the baseline more or less robust than SQUiD?\n")
        f.write("3. Does degradation scale linearly with noise level (10% -> 30% -> 60%)?\n")
        f.write("4. Does structural noise affect RI more than value-level noise?\n")

    print(f"  wrote {path}")


# Main

def main():
    print()
    print("=" * 70)
    print("Computing metrics for all (variant x method) cells")
    print("=" * 70)

    rows = []
    all_cells = [("clean", "squid"), ("clean", "baseline")]
    for v in VARIANT_NAMES:
        all_cells.append((v, "squid"))
        all_cells.append((v, "baseline"))

    for i, (variant, method) in enumerate(all_cells, 1):
        print(f"\n[{i}/{len(all_cells)}] --- {variant} / {method} ---")
        metrics = evaluate_cell(variant, method)
        print(f"    TC={metrics['tc']:.3f}  VC={metrics['vc']:.3f}  CC={metrics['cc']:.3f}  "
              f"RI={metrics['ri']:.3f}  EX={metrics['ex']:.3f}  (n={metrics['n']})")
        rows.append({"variant": variant, "method": method, **metrics})

    csv_path = RESULTS_DIR / "sensitivity_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "method", "tc", "vc", "cc", "ri", "ex", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  wrote {csv_path}")

    plot_degradation(rows)
    plot_heatmap(rows)
    write_findings(rows)

    print("\n-- Preview --")
    print(f"  {'variant':40s}  {'method':8s}  {'TC':>6s}  {'VC':>6s}  {'CC':>6s}  {'RI':>6s}  {'EX':>6s}")
    for r in rows:
        print(f"  {r['variant']:40s}  {r['method']:8s}  "
              f"{r['tc']:6.3f}  {r['vc']:6.3f}  {r['cc']:6.3f}  {r['ri']:6.3f}  {r['ex']:6.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()