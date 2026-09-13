# Empirical Study of Label-Space Alignment, Fair-Protocol Domain Adaptation and Edge Deployment for Field Plant Disease Recognition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22734464.svg)](https://doi.org/10.5281/zenodo.22734464)

Reproducibility package for the manuscript submitted to the *AI* (MDPI) Special Issue
"Harvesting the Future: Transforming Agricultural Practices Through AI Application".
It contains everything needed to re-check every number in the paper and to re-run the
evaluation, quantization and edge-benchmark steps; raw image datasets and PyTorch
checkpoints are not included (see the Data Availability Statement). Archived release: https://doi.org/10.5281/zenodo.22734464.

## Layout

| Path | Content |
|---|---|
| `paper/mdpi_ai/main.md` | manuscript source (pandoc Markdown); `figures/` with generators; `submission/` with the Word file built into the MDPI template (`build_mdpi_docx.py`) |
| `paper/mdpi_ai/check_numbers.py` | 150+ checks that tie every number in the manuscript to a primary result file (`python3 paper/mdpi_ai/check_numbers.py`) |
| `configs/` | class mappings (fuzzy / SBERT / hybrid), label maps, training config |
| `splits/` | PlantVillage and PlantDoc split lists (paths relative to the data root) |
| `results/uda_benchmark/` | fair-protocol UDA runs: per-seed raw JSON, per-seed test predictions, paired bootstrap |
| `results/final/` | aggregated tables (T1 source-only, re-evaluated T1, T4/T5 edge latency, statistical validation) |
| `results/int8_accuracy/` | conversion-fidelity audit: PyTorch references vs TFLite float32 / INT8 (per-sample predictions) |
| `results/ub_audit/` | supervised reference: in-sample vs held-out evaluation of the fine-tuned checkpoint |
| `results/edgebench/` | Raspberry Pi 4 + Coral Edge TPU latency runs (February first-generation files and the 2026-09-13 re-measurement of the verified files) |
| `experiments/` | E1 threshold sweep, E2 legacy DANN, E3 shared 13-class control, E4 per-class, E5 beans, E5b PlantWild v2 |
| `export/` | verified INT8 TFLite models (ONNX → onnx2tf path; EfficientNet-Lite0 via Keras path) with Edge TPU compiler logs |
| `scripts/`, `src/`, `tests/` | pipeline scripts (numbered by phase), UDA implementations, unit tests (`pytest tests/`) |

## Re-running the checks

```bash
python3 paper/mdpi_ai/check_numbers.py        # numbers in main.md vs primary files
pytest tests/                                  # 20 unit tests of the UDA implementations
python3 scripts/14.6_summarize_int8.py         # rebuild INT8_ACCURACY_SUMMARY.md
python3 scripts/14.9_build_T5_reeval.py        # rebuild the edge table from raw runs
```

Latency re-measurement on the device (`scripts/14.8_rpi_latency_reeval.py`) needs the
edge-bench agent (`benchmark_tflite.py`) on the Raspberry Pi; accuracy evaluation of the
TFLite files (`scripts/14.1_eval_tflite_accuracy.py`) needs the datasets and TensorFlow 2.15.
