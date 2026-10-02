"""
REAL KernelSHAP (shap library) on the anomaly score, offline.
f(x) = reconstruction MSE of the trained autoencoder (scaled feature space).
Background: 100 benign training flows. Explained: up to N_PER_CLASS flagged flows per attack
class (fixed random seed). Compares KernelSHAP's top feature with the reconstruction-error
attribution's top feature. This is the slow/offline explainer, NOT the real-time path.
Outputs: results/kernelshap_agreement.csv, kernelshap_mean_abs_by_class.csv,
fig_kernelshap_vs_reconerror.png, kernelshap_timing.txt
"""
import json, os, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, numpy as np, pandas as pd, shap, torch
import config as C
from model import DenseAutoencoder

S, N_PER_CLASS, N_BG, NSAMPLES = C.DISPLAY_SEED, 10, 100, 500
rng = np.random.default_rng(0)
ho = pd.read_parquet(os.path.join(C.DATA_DIR, f"holdout_test_seed{S}.parquet"))
tr = pd.read_parquet(os.path.join(C.DATA_DIR, f"train_benign_seed{S}.parquet"))
sc = joblib.load(os.path.join(C.DATA_DIR, f"scaler_seed{S}.joblib"))
m = DenseAutoencoder(len(C.FEATURE_COLUMNS)); m.load_state_dict(torch.load(os.path.join(C.RUN_DIR, f"model_seed{S}.pt"))); m.eval()
tau = json.load(open(os.path.join(C.RUN_DIR, f"threshold_seed{S}.json")))["tau"]
err = np.load(os.path.join(C.RUN_DIR, f"errors_seed{S}.npy"))
bg = sc.transform(tr[C.FEATURE_COLUMNS].values[rng.choice(len(tr), N_BG, replace=False)])


def f(X):
    with torch.no_grad():
        return m.reconstruction_error(torch.tensor(X, dtype=torch.float32)).numpy()


expl = shap.KernelExplainer(f, bg)
rows, sv_by_cls = [], {}
for cls in sorted(ho[C.ATTACK_TYPE_COLUMN].unique()):
    if cls == C.BENIGN_LABEL:
        continue
    idx = np.where((ho[C.ATTACK_TYPE_COLUMN] == cls).values & (err > tau))[0]
    if len(idx) == 0:
        continue
    pick = rng.choice(idx, min(N_PER_CLASS, len(idx)), replace=False)
    X = sc.transform(ho[C.FEATURE_COLUMNS].values[pick])
    t0 = time.time()
    sv = np.array(expl.shap_values(X, nsamples=NSAMPLES, silent=True))
    dt = (time.time() - t0) / len(pick)
    with torch.no_grad():
        pf = m.per_feature_error(torch.tensor(X, dtype=torch.float32)).numpy()
    sv_by_cls[cls] = np.abs(sv).mean(0)
    for k in range(len(pick)):
        a, b = np.argsort(-np.abs(sv[k]))[:3], np.argsort(-pf[k])[:3]
        rows.append(dict(attack_type=cls, holdout_row=int(pick[k]), sec_per_flow=dt,
                         shap_top1=C.FEATURE_COLUMNS[a[0]], recon_top1=C.FEATURE_COLUMNS[b[0]],
                         top1_agree=int(a[0] == b[0]), top3_overlap=len(set(a) & set(b))))
    print(cls, f"{dt:.2f} s/flow", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(C.RESULTS_DIR, "kernelshap_agreement.csv"), index=False)
pd.DataFrame(sv_by_cls, index=C.FEATURE_COLUMNS).T.to_csv(os.path.join(C.RESULTS_DIR, "kernelshap_mean_abs_by_class.csv"))
summ = (f"flows explained: {len(df)}\nmean seconds per flow (KernelSHAP, nsamples={NSAMPLES}, background={N_BG}): {df.sec_per_flow.mean():.3f}\n"
        f"top-1 agreement with reconstruction-error attribution: {df.top1_agree.mean():.3f}\n"
        f"mean top-3 overlap (out of 3): {df.top3_overlap.mean():.2f}\n")
open(os.path.join(C.RESULTS_DIR, "kernelshap_timing.txt"), "w").write(summ); print(summ)
ag = df.groupby("attack_type")[["top1_agree", "top3_overlap"]].mean()
fig, ax = plt.subplots(figsize=(8, 4)); ag.top1_agree.plot.bar(ax=ax, color="tab:green")
ax.set_ylabel("top-1 agreement rate"); ax.set_ylim(0, 1.05)
ax.set_title("KernelSHAP vs reconstruction-error attribution: same top feature?"); fig.tight_layout()
fig.savefig(os.path.join(C.RESULTS_DIR, "fig_kernelshap_vs_reconerror.png"), dpi=200)
