"""
E1 — SBERT cosine threshold sensitivity sweep.

Addresses: fPLF (R1-M1) and DVkE (R2-m1) concerns that thresholds are manually tuned.

Approach:
1. Recompute SBERT pair similarities for all target→PV pairs
2. For each threshold in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}:
   - Count aligned classes (N_aligned)
   - Identify which classes are included at this threshold
3. For each threshold, evaluate downstream F1 using the existing sbert-strategy
   predictions (preds_target_resnet50_sbert.csv, preds_target_coral_lambda_*_sbert.csv)
   on the class subset accepted at that threshold.
   Note: The ResNet-50 model was trained on the 15-class SBERT split (threshold=0.65
   after manual inclusion). We evaluate it here on SUBSETS of those 15 classes, which
   is a valid lower bound on what each threshold achieves.

Key question: Is downstream F1 sensitive to threshold changes, or does it plateau?
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS = Path("artifacts")
CONFIGS = Path("configs")
RESULTS = Path("results")
OUT_DIR = Path("experiments/E1_threshold_sweep")
LOG_DIR = Path("experiments/logs/E1")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# 1. Load normalized class names
# ----------------------------------------------------------

def load_csv_col(path: Path, col: str) -> list[str]:
    with open(path) as f:
        return [row[col] for row in csv.DictReader(f)]


def load_csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


pv_rows = load_csv_rows(ARTIFACTS / "classes_pv_normalized.csv")
target_rows = load_csv_rows(ARTIFACTS / "classes_target_normalized.csv")

pv_normalized = [r["normalized"] for r in pv_rows]
pv_original = [r["original"] for r in pv_rows]
target_normalized = [r["normalized"] for r in target_rows]
target_original = [r["original"] for r in target_rows]

print(f"PV classes: {len(pv_normalized)}, Target classes: {len(target_normalized)}")


# ----------------------------------------------------------
# 2. Filter: same disease validity as original script
# ----------------------------------------------------------

INVALID = ["healthy", "spider", "mites", "pest"]
valid_target_idx = []
for i, (orig, norm) in enumerate(zip(target_original, target_normalized)):
    if not any(x in norm for x in INVALID) and len(norm.split()) > 1:
        valid_target_idx.append(i)

valid_target_original = [target_original[i] for i in valid_target_idx]
valid_target_normalized = [target_normalized[i] for i in valid_target_idx]
print(f"Valid target classes (non-healthy, has disease name): {len(valid_target_normalized)}")


# ----------------------------------------------------------
# 3. Compute SBERT embeddings and pairwise cosine similarity
# ----------------------------------------------------------

print("Loading SBERT model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

pv_emb = model.encode(pv_normalized, show_progress_bar=False, batch_size=64)
target_emb = model.encode(valid_target_normalized, show_progress_bar=False, batch_size=64)

sim_matrix = cosine_similarity(target_emb, pv_emb)  # (n_target, n_pv)

# Best PV match per target class
best_sims = []
for ti, tn_orig in enumerate(valid_target_original):
    best_pv_idx = int(np.argmax(sim_matrix[ti]))
    best_sim = float(sim_matrix[ti][best_pv_idx])
    best_sims.append({
        "target_original": tn_orig,
        "target_normalized": valid_target_normalized[ti],
        "pv_match": pv_original[best_pv_idx],
        "pv_normalized": pv_normalized[best_pv_idx],
        "cosine_sim": round(best_sim, 4),
    })

best_sims.sort(key=lambda x: -x["cosine_sim"])
print("\nAll valid target→PV pairs (sorted by sim):")
for p in best_sims:
    print(f"  {p['cosine_sim']:.4f}  {p['target_original']}  →  {p['pv_match']}")

# Save all pairs
(OUT_DIR / "all_pairs_with_sim.json").write_text(json.dumps(best_sims, indent=2))


# ----------------------------------------------------------
# 4. Threshold sweep: count aligned classes at each threshold
# ----------------------------------------------------------

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

print("\n=== Threshold sweep (alignment only) ===")
alignment_rows = []
for thr in THRESHOLDS:
    accepted = [p for p in best_sims if p["cosine_sim"] >= thr]
    mean_sim = np.mean([p["cosine_sim"] for p in accepted]) if accepted else 0.0
    alignment_rows.append({
        "threshold": thr,
        "n_aligned": len(accepted),
        "mean_sim_accepted": round(mean_sim, 4),
        "accepted_classes": [p["target_original"] for p in accepted],
    })
    print(f"  thr={thr:.2f}: {len(accepted):2d} classes, mean_sim={mean_sim:.4f}")
    for p in accepted:
        print(f"    {p['cosine_sim']:.4f}  {p['target_original']}")


# ----------------------------------------------------------
# 5. Downstream F1 evaluation per threshold
# ----------------------------------------------------------

# Load the SBERT strategy label map to get label IDs for each target class
lmap_sbert = pd.read_csv(CONFIGS / "label_map_sbert.csv")
print(f"\nlabel_map_sbert cols: {list(lmap_sbert.columns)}")
print(lmap_sbert.head(3).to_string())

# label_map_sbert.csv has fixed columns: label_id, canonical_name, pv_name, target_name
name_col = "target_name"
id_col_name = "label_id"
print(f"Using name_col='{name_col}', id_col='{id_col_name}'")

# Build target_name → label_id mapping for SBERT strategy
name_to_id = {}
for _, row in lmap_sbert.iterrows():
    name_to_id[str(row[name_col])] = int(row[id_col_name])

print(f"name_to_id sample: {dict(list(name_to_id.items())[:4])}")


def compute_f1_on_subset(pred_csv_path: Path, accepted_names: set) -> dict | None:
    if not pred_csv_path.exists():
        return None

    df = pd.read_csv(pred_csv_path)
    # auto-detect true/pred cols
    true_col = next((c for c in df.columns if c in ("y_true", "true", "label")), None)
    pred_col = next((c for c in df.columns if c in ("y_pred", "pred", "predicted")), None)
    if true_col is None or pred_col is None:
        return None

    # Map accepted class names to label IDs
    accepted_ids = {name_to_id[n] for n in accepted_names if n in name_to_id}
    if not accepted_ids:
        return None

    mask = df[true_col].isin(accepted_ids)
    df_filt = df[mask]
    if len(df_filt) == 0:
        return None

    f1 = f1_score(df_filt[true_col], df_filt[pred_col], average="macro", zero_division=0)
    acc = accuracy_score(df_filt[true_col], df_filt[pred_col])
    return {"macro_f1": round(f1, 4), "accuracy": round(acc, 4), "n_samples": len(df_filt)}


# Evaluate for each threshold
print("\n=== Downstream F1 per threshold (SBERT model, evaluated on threshold subset) ===")
result_rows = []
for row in alignment_rows:
    thr = row["threshold"]
    accepted_names = set(row["accepted_classes"])
    n_aligned = row["n_aligned"]

    baseline_res = compute_f1_on_subset(
        RESULTS / "preds_target_resnet50_sbert.csv", accepted_names
    )
    coral_res = compute_f1_on_subset(
        RESULTS / "preds_target_coral_lambda_0.01_sbert.csv", accepted_names
    )

    result_rows.append({
        "threshold": thr,
        "n_aligned": n_aligned,
        "mean_sim": row["mean_sim_accepted"],
        "baseline_macro_f1": baseline_res["macro_f1"] if baseline_res else None,
        "baseline_n_samples": baseline_res["n_samples"] if baseline_res else None,
        "coral_macro_f1": coral_res["macro_f1"] if coral_res else None,
        "coral_n_samples": coral_res["n_samples"] if coral_res else None,
    })

    b_f1 = f"{baseline_res['macro_f1']:.4f} (n={baseline_res['n_samples']})" if baseline_res else "N/A"
    c_f1 = f"{coral_res['macro_f1']:.4f} (n={coral_res['n_samples']})" if coral_res else "N/A"
    print(f"  thr={thr:.2f}: n_aligned={n_aligned:2d}, baseline_F1={b_f1}, coral_F1={c_f1}")


# ----------------------------------------------------------
# 6. Save results
# ----------------------------------------------------------

df_results = pd.DataFrame(result_rows)
out_csv = OUT_DIR / "threshold_sweep_results.csv"
df_results.to_csv(out_csv, index=False)
print(f"\n[SAVED] {out_csv}")
print(df_results.to_string(index=False))

import subprocess
import datetime

manifest = {
    "experiment": "E1_threshold_sweep",
    "description": "SBERT cosine threshold sensitivity: n_aligned and downstream F1",
    "thresholds_tested": THRESHOLDS,
    "model_used": "resnet50_pv_sbert_best.pt (existing, no retraining)",
    "git_hash": subprocess.getoutput("git rev-parse --short HEAD"),
    "timestamp": datetime.datetime.now().isoformat(),
    "no_retraining": True,
}
(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

log_content = (
    f"E1 run at {manifest['timestamp']}\n"
    f"Git: {manifest['git_hash']}\n\n"
    + df_results.to_string()
)
(LOG_DIR / "E1_run.log").write_text(log_content)
print("[SAVED] manifest.json, E1_run.log")
