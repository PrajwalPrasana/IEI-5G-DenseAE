"""
analysis_figures.py -- figures + tables that show WHERE the network works and fails.
Needs: evaluate_and_report.py already run. Outputs go to results/.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve
import config as C

S = C.DISPLAY_SEED
R = C.RESULTS_DIR

# 1. loss curves (all seeds)
fig, ax = plt.subplots(figsize=(7, 4))
for s in C.SEEDS:
    h = pd.read_csv(os.path.join(C.RUN_DIR, f"history_seed{s}.csv"))
    ax.plot(h.epoch, h.val_loss, label=f"seed {s}")
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("validation MSE (benign)")
ax.set_title("Validation loss per seed"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(R, "fig_loss_curves.png"), dpi=200); plt.close(fig)

# 2. error distributions for the display seed
ho = pd.read_parquet(os.path.join(C.DATA_DIR, f"holdout_test_seed{S}.parquet"))
err = np.load(os.path.join(C.RUN_DIR, f"errors_seed{S}.npy"))
tau = json.load(open(os.path.join(C.RUN_DIR, f"threshold_seed{S}.json")))["tau"]
ben = (ho[C.LABEL_COLUMN] == C.BENIGN_LABEL).values
bins = np.logspace(np.log10(max(err.min(), 1e-9)), np.log10(err.max()), 80)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(err[ben], bins=bins, alpha=.6, label="benign", weights=np.ones(ben.sum())/ben.sum())
ax.hist(err[~ben], bins=bins, alpha=.6, label="attack", weights=np.ones((~ben).sum())/(~ben).sum())
ax.axvline(tau, color="k", ls="--", label=f"tau = {tau:.5f}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("reconstruction error (MSE)"); ax.set_ylabel("fraction of flows per bin")
ax.set_title(f"Anomaly-score distributions (seed {S})"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(R, "fig_error_hist.png"), dpi=200); plt.close(fig)

# 3. per-class error/tau (boxplot) for the display seed
classes = sorted(ho[C.ATTACK_TYPE_COLUMN].unique())
data = [err[(ho[C.ATTACK_TYPE_COLUMN] == c).values] / tau for c in classes]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.boxplot(data, showfliers=False); ax.set_xticklabels(classes)
ax.axhline(1.0, color="k", ls="--"); ax.set_yscale("log")
ax.set_ylabel("error / tau  (above 1.0 = flagged)"); ax.tick_params(axis="x", rotation=30)
ax.set_title(f"Score relative to threshold, by traffic class (seed {S})"); fig.tight_layout()
fig.savefig(os.path.join(R, "fig_class_scores.png"), dpi=200); plt.close(fig)

# 4. ROC curves, all seeds
fig, ax = plt.subplots(figsize=(5, 5))
for s in C.SEEDS:
    h = pd.read_parquet(os.path.join(C.DATA_DIR, f"holdout_test_seed{s}.parquet"), columns=[C.LABEL_COLUMN])
    e = np.load(os.path.join(C.RUN_DIR, f"errors_seed{s}.npy"))
    f, t, _ = roc_curve((h[C.LABEL_COLUMN] != C.BENIGN_LABEL).astype(int), e)
    ax.plot(f, t, label=f"seed {s}")
ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
ax.set_title("ROC (benign vs attack)"); fig.tight_layout()
fig.savefig(os.path.join(R, "fig_roc.png"), dpi=200); plt.close(fig)

# 5. per-class recall, mean with per-seed dots
pc = pd.read_csv(os.path.join(R, "per_seed_per_class.csv"))
order = pc.groupby("attack_type").recall.mean().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, c in enumerate(order):
    v = pc[pc.attack_type == c].recall.values
    ax.bar(i, v.mean(), alpha=.5, color="tab:blue")
    ax.scatter([i] * len(v), v, color="k", s=14, zorder=3)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30)
ax.set_ylabel("recall"); ax.set_ylim(0, 1.05)
ax.set_title("Per-class recall: bar = mean, dots = individual seeds"); fig.tight_layout()
fig.savefig(os.path.join(R, "fig_perclass_recall.png"), dpi=200); plt.close(fig)

# 6. per-seed stability
ov = pd.read_csv(os.path.join(R, "per_seed_overall.csv"))
fig, ax = plt.subplots(figsize=(7, 4)); w = .25; x = np.arange(len(ov))
for j, k in enumerate(["f1", "auc", "recall"]):
    ax.bar(x + (j - 1) * w, ov[k], w, label=k)
ax.set_xticks(x); ax.set_xticklabels(ov.seed); ax.set_xlabel("seed"); ax.legend()
ax.set_title("Seed-to-seed variation"); fig.tight_layout()
fig.savefig(os.path.join(R, "fig_seed_stability.png"), dpi=200); plt.close(fig)

# 7. why UDP Flood is missed: raw feature medians, benign-UDP vs UDP Flood
raw_udp = C.PROTO_VALUES and sorted(C.PROTO_VALUES).index("udp")
sub = ho[ho.Proto == raw_udp]
cmp = pd.DataFrame({
    "benign_udp_median": sub[sub[C.LABEL_COLUMN] == C.BENIGN_LABEL][C.FEATURE_COLUMNS].median(),
    "udpflood_median": sub[sub[C.ATTACK_TYPE_COLUMN] == "UDPFlood"][C.FEATURE_COLUMNS].median(),
})
cmp.to_csv(os.path.join(R, "table_udp_vs_udpflood_feature_medians.csv"))
print("Figures and tables written to", R)
print(cmp.to_string())
