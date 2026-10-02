"""
single-flow CPU inference latency (1 thread), 5000 random holdout flows.
Times (a) detection forward pass, (b) forward pass + per-feature attribution.
Reports mean/p50/p99 in ms and the hardware. Numbers are specific to the machine that ran this.
"""
import os, platform, time
import joblib, numpy as np, pandas as pd, torch
import config as C
from model import DenseAutoencoder

torch.set_num_threads(1)
S = C.DISPLAY_SEED
ho = pd.read_parquet(os.path.join(C.DATA_DIR, f"holdout_test_seed{S}.parquet"))
sc = joblib.load(os.path.join(C.DATA_DIR, f"scaler_seed{S}.joblib"))
m = DenseAutoencoder(len(C.FEATURE_COLUMNS)); m.load_state_dict(torch.load(os.path.join(C.RUN_DIR, f"model_seed{S}.pt"))); m.eval()
rng = np.random.default_rng(0)
X = torch.tensor(sc.transform(ho[C.FEATURE_COLUMNS].values[rng.choice(len(ho), 5000, replace=False)]), dtype=torch.float32)


def bench(fn):
    with torch.no_grad():
        for i in range(200): fn(X[i:i + 1])
        t = []
        for i in range(len(X)):
            a = time.perf_counter_ns(); fn(X[i:i + 1]); t.append((time.perf_counter_ns() - a) / 1e6)
    t = np.array(t); return t.mean(), np.percentile(t, 50), np.percentile(t, 99)


d = bench(lambda z: m.reconstruction_error(z))
a = bench(lambda z: (m.reconstruction_error(z), m.per_feature_error(z).argmax(1)))
txt = (f"hardware: {platform.processor() or platform.machine()} | logical CPUs: {os.cpu_count()} | "
       f"torch {torch.__version__} | 1 thread\nsamples: 5000 single-flow calls (200 warm-up)\n"
       f"detection only      : mean {d[0]:.4f} ms  p50 {d[1]:.4f}  p99 {d[2]:.4f}\n"
       f"detection+attribution: mean {a[0]:.4f} ms  p50 {a[1]:.4f}  p99 {a[2]:.4f}\n")
open(os.path.join(C.RESULTS_DIR, "latency.txt"), "w").write(txt); print(txt)
