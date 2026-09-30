"""
explain_attribution.py -- per-feature reconstruction-error attribution (Eq. 4).
attribution_i = (x_i - x_hat_i)^2 in scaled space; "share" = attribution_i / sum_j attribution_j.
Outputs (display seed): results/attribution_share_by_class.csv, fig_attribution_heatmap.png,
fig_attribution_examples.png. Example flow per class = the FLAGGED flow whose score is the
MEDIAN among flagged flows of that class (fixed rule, no cherry-picking).
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, numpy as np, pandas as pd, torch
import config as C
from model import DenseAutoencoder

S = C.DISPLAY_SEED
ho = pd.read_parquet(os.path.join(C.DATA_DIR, f"holdout_test_seed{S}.parquet"))
sc = joblib.load(os.path.join(C.DATA_DIR, f"scaler_seed{S}.joblib"))
m = DenseAutoencoder(len(C.FEATURE_COLUMNS)); m.load_state_dict(torch.load(os.path.join(C.RUN_DIR, f"model_seed{S}.pt"))); m.eval()
tau = json.load(open(os.path.join(C.RUN_DIR, f"threshold_seed{S}.json")))["tau"]
err = np.load(os.path.join(C.RUN_DIR, f"errors_seed{S}.npy"))
x = torch.tensor(sc.transform(ho[C.FEATURE_COLUMNS].values), dtype=torch.float32)
with torch.no_grad():
    pf = m.per_feature_error(x).numpy()
share = pf / np.clip(pf.sum(1, keepdims=True), 1e-20, None)
flag = err > tau
ben = (ho[C.LABEL_COLUMN] == C.BENIGN_LABEL).values
rows = {}
for cls in sorted(ho[C.ATTACK_TYPE_COLUMN].unique()):
    if cls == C.BENIGN_LABEL:
        mask = ben & flag; name = "Benign (false positives)"
    else:
        mask = (ho[C.ATTACK_TYPE_COLUMN] == cls).values & flag; name = cls
    if mask.sum():
        rows[f"{name} (n={int(mask.sum())})"] = share[mask].mean(0)
tab = pd.DataFrame(rows, index=C.FEATURE_COLUMNS).T
tab.to_csv(os.path.join(C.RESULTS_DIR, "attribution_share_by_class.csv"))
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(tab.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(tab.columns))); ax.set_xticklabels(tab.columns, rotation=60, ha="right")
ax.set_yticks(range(len(tab.index))); ax.set_yticklabels(tab.index)
fig.colorbar(im, label="mean share of reconstruction error")
ax.set_title(f"Which features drive the alert? (flagged flows, seed {S})"); fig.tight_layout()
fig.savefig(os.path.join(C.RESULTS_DIR, "fig_attribution_heatmap.png"), dpi=200); plt.close(fig)

classes = [c for c in sorted(ho[C.ATTACK_TYPE_COLUMN].unique()) if c != C.BENIGN_LABEL]
fig, axes = plt.subplots(2, 4, figsize=(16, 7)); ex = []
for ax, cls in zip(axes.ravel(), classes):
    idx = np.where((ho[C.ATTACK_TYPE_COLUMN] == cls).values & flag)[0]
    if len(idx) == 0:
        ax.set_title(f"{cls}: no flagged flows"); ax.axis("off"); continue
    i = idx[np.argsort(err[idx])[len(idx) // 2]]
    top = np.argsort(pf[i])[::-1][:5]
    ax.barh([C.FEATURE_COLUMNS[j] for j in top][::-1], pf[i][top][::-1], color="crimson")
    ax.set_title(f"{cls}  (row {i}, score/tau={err[i]/tau:.1f})", fontsize=9)
    ex.append(dict(attack_type=cls, holdout_row=int(i), score=float(err[i]),
                   top_feature=C.FEATURE_COLUMNS[top[0]]))
fig.suptitle(f"Median-score flagged flow per attack class: top-5 features (seed {S})")
fig.tight_layout(); fig.savefig(os.path.join(C.RESULTS_DIR, "fig_attribution_examples.png"), dpi=200)
pd.DataFrame(ex).to_csv(os.path.join(C.RESULTS_DIR, "attribution_examples.csv"), index=False)
print(tab.round(3).to_string()); print(pd.DataFrame(ex).to_string())
