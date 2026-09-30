# IEI-5G: Dense-Autoencoder anomaly detection on 5G-NIDD, with explanations

Benign-only Dense Autoencoder for flow-level anomaly detection on the
[5G-NIDD dataset](https://www.kaggle.com/datasets/humera11/5g-nidd-dataset), with
per-feature reconstruction-error attribution, an offline KernelSHAP comparison,
and a failure analysis. Everything in the paper's tables/figures is produced by this code.

## Quick start (about 25-30 minutes on a laptop CPU)

```bash
pip install -r requirements.txt
# unzip the Kaggle archive so you have Combined.csv, then:
python run_all.py --input path/to/Combined.csv
```
Add `--skip-shap` to skip the KernelSHAP stage. All console output is also saved in `logs/`.

## Pipeline (what each file does)

| Step | File | Output |
|---|---|---|
| 1 | `data_prep.py` | split + scaler per seed in `data/` |
| 2 | `train_autoencoder.py` | `runs/model_seed*.pt`, `threshold_seed*.json`, `history_seed*.csv` |
| 3 | `evaluate_and_report.py` | `results/summary.txt`, `per_seed_overall.csv`, `per_seed_per_class.csv` |
| 4 | `analysis_figures.py` | ROC, score distributions, per-class recall, loss curves, UDP-vs-UDP-Flood table |
| 5 | `explain_attribution.py` | attribution heatmap + per-class example flows |
| 6 | `measure_latency.py` | `results/latency.txt` |
| 7 | `explain_kernelshap.py` | KernelSHAP agreement + timing |

`config.py` holds every setting (seeds, epochs, batch size, threshold percentile). They are
frozen; do not change them if you want to reproduce the reported numbers.

## Protocol

- Model: 20-32-16-8-16-32-20, ReLU, sigmoid output; Adam (lr 1e-3), batch 256, max 50 epochs,
  early stopping (patience 5, best-validation weights restored).
- Trained on benign flows only. Threshold tau = 95th percentile of validation-benign error.
- Split: benign 70/10/20 (train/val/test); all attack flows are test-only.
- Seeds: 42, 123, 256, 789, 1024. Reported: mean and 95% CI (t-distribution, n=5).
- Missing values (sVid ~91%, dTtl/dHops ~78% blank in the raw file) are filled with 0.

## Known limitations (stated in the paper)

- Results vary a lot between random seeds (one seed detects very little).
- UDP Flood is largely indistinguishable from benign UDP on these features.
- `Offset` is a record-position field from the capture tool; it carries a large share of
  the attribution for some classes and may reflect capture order rather than traffic behaviour.
- No baseline models were run. Evaluation is on one dataset, offline; no live deployment.

## Reproducibility note

Exact numbers can differ slightly across PyTorch versions/CPUs. Compare your `results/`
with `reference_results/` (produced by the development run) to check you are in the same range.
