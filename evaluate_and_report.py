"""
score the holdout set for every seed.

tau comes from validation-benign data only. FPR = FP/(FP+TN) over benign holdout flows.
F1 is the binary benign-vs-attack F1. CIs use the t-distribution (n = number of seeds).
"""
import argparse, json, os
import joblib, numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.metrics import roc_auc_score
import config as C
from model import DenseAutoencoder


def ci95(v):
    v = np.asarray(v, float); n = len(v)
    return v.mean(), stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)


def score_seed(seed):
    ho = pd.read_parquet(os.path.join(C.DATA_DIR, f"holdout_test_seed{seed}.parquet"))
    sc = joblib.load(os.path.join(C.DATA_DIR, f"scaler_seed{seed}.joblib"))
    m = DenseAutoencoder(len(C.FEATURE_COLUMNS))
    m.load_state_dict(torch.load(os.path.join(C.RUN_DIR, f"model_seed{seed}.pt")))
    m.eval()
    tau = json.load(open(os.path.join(C.RUN_DIR, f"threshold_seed{seed}.json")))["tau"]
    x = torch.tensor(sc.transform(ho[C.FEATURE_COLUMNS].values), dtype=torch.float32)
    with torch.no_grad():
        err = m.reconstruction_error(x).numpy()
    np.save(os.path.join(C.RUN_DIR, f"errors_seed{seed}.npy"), err.astype("float32"))
    ben = (ho[C.LABEL_COLUMN] == C.BENIGN_LABEL).values
    att, pos = ~ben, err > tau
    tp, fn = int((pos & att).sum()), int((~pos & att).sum())
    fp, tn = int((pos & ben).sum()), int((~pos & ben).sum())
    rec, fpr, prec = tp / (tp + fn), fp / (fp + tn), tp / (tp + fp)
    overall = dict(seed=seed, tau=tau, f1=2 * prec * rec / (prec + rec),
                   auc=roc_auc_score(att.astype(int), err), fpr=fpr, recall=rec,
                   precision=prec, tp=tp, fp=fp, fn=fn, tn=tn)
    rows = []
    for cls, g in ho[att].groupby(C.ATTACK_TYPE_COLUMN):
        idx = ho.index.get_indexer(g.index)
        rows.append(dict(seed=seed, attack_type=cls, n=len(g), recall=float((err[idx] > tau).mean())))
    return overall, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    seeds = ap.parse_args().seeds
    ov, pc = [], []
    for s in seeds:
        o, r = score_seed(s); ov.append(o); pc += r
        print(f"seed {s}: F1={o['f1']:.4f} AUC={o['auc']:.4f} FPR={o['fpr']:.4f} "
              f"Recall={o['recall']:.4f} Precision={o['precision']:.4f} tau={o['tau']:.6f}")
    ov, pc = pd.DataFrame(ov), pd.DataFrame(pc)
    ov.to_csv(os.path.join(C.RESULTS_DIR, "per_seed_overall.csv"), index=False)
    pc.to_csv(os.path.join(C.RESULTS_DIR, "per_seed_per_class.csv"), index=False)
    L = [f"Overall (mean +/- 95% CI, t-distribution, n={len(seeds)} seeds)"]
    for k in ["f1", "auc", "fpr", "recall", "precision"]:
        mu, hw = ci95(ov[k]); L.append(f"  {k:9s} {mu:.4f} +/- {hw:.4f}   (min {ov[k].min():.4f}, max {ov[k].max():.4f})")
    L.append("Per-class recall (mean +/- 95% CI; min/max over seeds)")
    for cls, g in pc.groupby("attack_type"):
        mu, hw = ci95(g["recall"])
        L.append(f"  {cls:16s} n={int(g['n'].iloc[0]):7d}  {mu:.4f} +/- {hw:.4f}  (min {g['recall'].min():.3f}, max {g['recall'].max():.3f})")
    open(os.path.join(C.RESULTS_DIR, "summary.txt"), "w").write("\n".join(L))
    print("\n".join(L))
