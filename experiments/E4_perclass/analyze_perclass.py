"""
E4 — Per-class CORAL breakdown analysis.

Addresses: DVkE concern on class-dependence in Table 7.
Shows: which classes benefit from CORAL, which degrade, and why.
No new model runs — pure analysis of existing results/T6_per_class_f1_hybrid.csv.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
OUT_DIR = Path("experiments/E4_perclass")
LOG_DIR = Path("experiments/logs/E4")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# 1. Load per-class data for all 3 strategies
# ----------------------------------------------------------

def load_perclass(strategy: str) -> pd.DataFrame:
    path = RESULTS / f"T6_per_class_f1_{strategy}.csv"
    if not path.exists():
        print(f"[MISSING] {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["strategy"] = strategy
    return df


dfs = []
for strat in ["hybrid", "Fuzzy", "sbert"]:
    d = load_perclass(strat)
    if not d.empty:
        dfs.append(d)

df_all = pd.concat(dfs, ignore_index=True)
print("Columns:", list(df_all.columns))
print(df_all.head(3).to_string())

# Standardize column names
col_map = {}
for c in df_all.columns:
    cl = c.lower()
    if "baseline" in cl and "f1" in cl:
        col_map[c] = "baseline_f1"
    elif "coral" in cl and "f1" in cl:
        col_map[c] = "coral_f1"
    elif "delta" in cl:
        col_map[c] = "delta_f1"
    elif "support" in cl or "n_" in cl or "count" in cl:
        col_map[c] = "support"
    elif "class" in cl and "id" in cl:
        col_map[c] = "class_id"
    elif "class" in cl and "name" in cl:
        col_map[c] = "class_name"

df_all = df_all.rename(columns=col_map)
print("\nAfter rename:", list(df_all.columns))


# ----------------------------------------------------------
# 2. Focus on hybrid strategy (main paper strategy)
# ----------------------------------------------------------

df_hybrid = df_all[df_all["strategy"] == "hybrid"].copy().reset_index(drop=True)
print(f"\nHybrid strategy: {len(df_hybrid)} classes")

# Ensure delta_f1 exists
if "delta_f1" not in df_hybrid.columns and "baseline_f1" in df_hybrid.columns and "coral_f1" in df_hybrid.columns:
    df_hybrid["delta_f1"] = df_hybrid["coral_f1"] - df_hybrid["baseline_f1"]

df_hybrid_sorted = df_hybrid.sort_values("delta_f1", ascending=False)

print("\n--- Hybrid per-class (sorted by delta) ---")
print(df_hybrid_sorted.to_string(index=False))


# ----------------------------------------------------------
# 3. Summary statistics
# ----------------------------------------------------------

improved = df_hybrid[df_hybrid["delta_f1"] > 0]
degraded = df_hybrid[df_hybrid["delta_f1"] < 0]
neutral  = df_hybrid[df_hybrid["delta_f1"] == 0]

print(f"\n=== Hybrid strategy: per-class CORAL summary ===")
print(f"Total classes: {len(df_hybrid)}")
print(f"Improved (Δ>0): {len(improved)}")
print(f"Degraded (Δ<0): {len(degraded)}")
print(f"Neutral (Δ=0):  {len(neutral)}")

if len(improved) > 0:
    print(f"\nImproved classes:")
    print(f"  Mean Δ: {improved['delta_f1'].mean():.4f}")
    print(f"  Max Δ:  {improved['delta_f1'].max():.4f}")
    if "support" in improved.columns:
        print(f"  Mean support: {improved['support'].mean():.1f}")

if len(degraded) > 0:
    print(f"\nDegraded classes:")
    print(f"  Mean Δ: {degraded['delta_f1'].mean():.4f}")
    print(f"  Min Δ:  {degraded['delta_f1'].min():.4f}")
    if "support" in degraded.columns:
        print(f"  Mean support: {degraded['support'].mean():.1f}")

# Low-support classes (< 50 target images)
if "support" in df_hybrid.columns:
    rare = df_hybrid[df_hybrid["support"] < 50]
    print(f"\nRare classes (support < 50): {len(rare)}")
    if len(rare) > 0:
        print(rare[["class_id", "baseline_f1", "coral_f1", "delta_f1", "support"]].to_string(index=False)
              if "class_id" in df_hybrid.columns else rare.to_string(index=False))

    # Correlation: support vs delta_f1
    corr = df_hybrid["support"].corr(df_hybrid["delta_f1"])
    print(f"\nCorrelation(support, Δ F1): {corr:.4f}")
    low_support = df_hybrid[df_hybrid["support"] < 50]["delta_f1"].mean() if len(rare) > 0 else float("nan")
    high_support = df_hybrid[df_hybrid["support"] >= 50]["delta_f1"].mean()
    print(f"Mean Δ F1 for low-support (<50): {low_support:.4f}")
    print(f"Mean Δ F1 for high-support (≥50): {high_support:.4f}")


# ----------------------------------------------------------
# 4. Compare across strategies
# ----------------------------------------------------------

print("\n\n=== Cross-strategy summary ===")
for strat in ["Fuzzy", "hybrid", "sbert"]:
    d = df_all[df_all["strategy"] == strat].copy()
    if d.empty:
        continue
    if "delta_f1" not in d.columns and "baseline_f1" in d.columns and "coral_f1" in d.columns:
        d["delta_f1"] = d["coral_f1"] - d["baseline_f1"]
    imp = (d["delta_f1"] > 0).sum()
    deg = (d["delta_f1"] < 0).sum()
    neu = (d["delta_f1"] == 0).sum()
    mean_delta = d["delta_f1"].mean()
    print(f"  {strat:8s}: {len(d)} classes | improved={imp} | degraded={deg} | neutral={neu} | mean_Δ={mean_delta:.4f}")


# ----------------------------------------------------------
# 5. Save outputs
# ----------------------------------------------------------

out_csv = OUT_DIR / "perclass_hybrid_full.csv"
df_hybrid_sorted.to_csv(out_csv, index=False)
print(f"\n[SAVED] {out_csv}")

# Summary for rebuttal
summary = {
    "strategy": "hybrid",
    "lambda": 0.1,
    "n_classes": len(df_hybrid),
    "n_improved": int(len(improved)),
    "n_degraded": int(len(degraded)),
    "n_neutral": int(len(neutral)),
    "mean_delta_improved": round(float(improved["delta_f1"].mean()), 4) if len(improved) > 0 else None,
    "mean_delta_degraded": round(float(degraded["delta_f1"].mean()), 4) if len(degraded) > 0 else None,
    "top3_improved": df_hybrid_sorted.head(3)[["class_id", "baseline_f1", "coral_f1", "delta_f1"]].to_dict("records")
        if "class_id" in df_hybrid_sorted.columns else [],
    "worst3_degraded": df_hybrid_sorted.tail(3)[["class_id", "baseline_f1", "coral_f1", "delta_f1"]].to_dict("records")
        if "class_id" in df_hybrid_sorted.columns else [],
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

# Markdown for rebuttal
md_lines = [
    "## E4 — Per-Class CORAL Breakdown (Hybrid, λ=0.1)",
    "",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| Total classes | {summary['n_classes']} |",
    f"| Improved (Δ>0) | {summary['n_improved']} |",
    f"| Degraded (Δ<0) | {summary['n_degraded']} |",
    f"| Neutral (Δ=0) | {summary['n_neutral']} |",
    f"| Mean Δ (improved) | {summary['mean_delta_improved']} |",
    f"| Mean Δ (degraded) | {summary['mean_delta_degraded']} |",
    "",
    "### Top-3 improved classes",
    "| class_id | Baseline F1 | CORAL F1 | Δ |",
    "|----------|------------|---------|---|",
]
for r in summary["top3_improved"]:
    md_lines.append(f"| {r.get('class_id','?')} | {r.get('baseline_f1',0):.3f} | {r.get('coral_f1',0):.3f} | {r.get('delta_f1',0):+.3f} |")

md_lines += [
    "",
    "### Worst-3 degraded classes",
    "| class_id | Baseline F1 | CORAL F1 | Δ |",
    "|----------|------------|---------|---|",
]
for r in summary["worst3_degraded"]:
    md_lines.append(f"| {r.get('class_id','?')} | {r.get('baseline_f1',0):.3f} | {r.get('coral_f1',0):.3f} | {r.get('delta_f1',0):+.3f} |")

(OUT_DIR / "perclass_analysis.md").write_text("\n".join(md_lines))

import subprocess, datetime
manifest = {
    "experiment": "E4_perclass",
    "description": "Per-class CORAL breakdown analysis (hybrid strategy)",
    "source_file": "results/T6_per_class_f1_hybrid.csv",
    "git_hash": subprocess.getoutput("git rev-parse --short HEAD"),
    "timestamp": datetime.datetime.now().isoformat(),
    "no_retraining": True,
}
(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
(LOG_DIR / "E4_run.log").write_text(
    f"E4 run at {manifest['timestamp']}\nGit: {manifest['git_hash']}\n"
    + "\n".join(md_lines)
)
print("[SAVED] summary.json, perclass_analysis.md, manifest.json, E4_run.log")
