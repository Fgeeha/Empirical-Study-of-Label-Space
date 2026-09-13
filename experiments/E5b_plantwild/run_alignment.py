"""E5b: Label alignment between PlantWild v2 and PlantVillage classes using SBERT."""
import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PLANTWILD_ZIP = Path(
    "/home/nkolesnikov/.cache/huggingface/hub/datasets--uqtwei2--PlantWild"
    "/snapshots/527a72eb8f00c95e41698bb09d982c7d4625dce3/plantwild_v2.zip"
)
PV_NORMALIZED_CSV = Path("artifacts/classes_pv_normalized.csv")
OUT_DIR = Path("experiments/E5b_plantwild")
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_plantwild_classes() -> dict[str, int]:
    z = zipfile.ZipFile(PLANTWILD_ZIP)
    counts: Counter = Counter()
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) >= 3 and parts[2]:
            counts[parts[1]] += 1
    return dict(counts)


def load_pv_classes() -> pd.DataFrame:
    return pd.read_csv(PV_NORMALIZED_CSV)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PlantWild v2 classes...")
    pw_classes = load_plantwild_classes()
    pw_names = sorted(pw_classes.keys())
    print(f"  {len(pw_names)} classes, {sum(pw_classes.values())} images")

    print("Loading PlantVillage normalized classes...")
    pv_df = load_pv_classes()
    pv_originals = pv_df["original"].tolist()
    pv_normalized = pv_df["normalized"].tolist()
    print(f"  {len(pv_normalized)} classes")

    print("Loading SBERT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    print("Encoding PlantWild v2 class names...")
    pw_emb = model.encode(pw_names, convert_to_tensor=True, normalize_embeddings=True)

    print("Encoding PlantVillage normalized names...")
    pv_emb = model.encode(pv_normalized, convert_to_tensor=True, normalize_embeddings=True)

    print("Computing cosine similarity matrix...")
    sim_matrix = (pw_emb @ pv_emb.T).cpu().numpy()  # (115, 38)

    # For each PW class, find best PV match
    results = []
    for i, pw_cls in enumerate(pw_names):
        best_j = sim_matrix[i].argmax()
        best_score = float(sim_matrix[i, best_j])
        results.append({
            "pw_class": pw_cls,
            "pw_images": pw_classes[pw_cls],
            "best_pv_match": pv_originals[best_j],
            "best_pv_normalized": pv_normalized[best_j],
            "cosine": best_score,
        })

    results_df = pd.DataFrame(results).sort_values("cosine", ascending=False)
    results_df.to_csv(OUT_DIR / "alignment_scores.csv", index=False)
    print(f"\nSaved alignment_scores.csv ({len(results_df)} rows)")

    print("\n=== Alignment Results by Threshold ===")
    threshold_summary = []
    for thr in THRESHOLDS:
        aligned = results_df[results_df["cosine"] >= thr]
        n_aligned = len(aligned)
        n_images = aligned["pw_images"].sum()
        mean_sim = aligned["cosine"].mean() if n_aligned > 0 else 0.0
        threshold_summary.append({
            "threshold": thr,
            "n_aligned": n_aligned,
            "n_images": int(n_images),
            "mean_cosine": round(float(mean_sim), 4),
        })
        print(
            f"  thr={thr:.2f}: {n_aligned:3d}/{len(pw_names)} classes aligned"
            f" | {n_images:5d} images | mean_sim={mean_sim:.4f}"
        )

    # Show top 20 alignments at threshold 0.65
    threshold_065 = results_df[results_df["cosine"] >= 0.65]
    print(f"\n=== Top alignments at threshold 0.65 ({len(threshold_065)} classes) ===")
    for _, row in threshold_065.iterrows():
        print(
            f"  [{row['cosine']:.4f}] {row['pw_class']!r:40s} -> {row['best_pv_match']!r}"
        )

    # Show bottom (rejected) classes
    rejected = results_df[results_df["cosine"] < 0.65].head(10)
    print(f"\n=== Lowest-scoring (rejected at 0.65) ===")
    for _, row in rejected.tail(10).iterrows():
        print(
            f"  [{row['cosine']:.4f}] {row['pw_class']!r:40s} -> {row['best_pv_match']!r}"
        )

    # Save manifest
    manifest = {
        "experiment": "E5b",
        "dataset": "PlantWild v2 (uqtwei2/PlantWild, plantwild_v2.zip)",
        "n_pw_classes": len(pw_names),
        "n_pw_images": sum(pw_classes.values()),
        "n_pv_classes": len(pv_normalized),
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "threshold_summary": threshold_summary,
        "aligned_at_065": threshold_065["pw_class"].tolist(),
        "n_aligned_at_065": len(threshold_065),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nManifest saved.")


if __name__ == "__main__":
    main()
