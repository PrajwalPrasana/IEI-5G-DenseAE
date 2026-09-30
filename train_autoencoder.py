"""
train_autoencoder.py -- train on BENIGN flows only, calibrate tau on validation-benign.
Saves: runs/model_seed{S}.pt, runs/threshold_seed{S}.json, runs/history_seed{S}.csv
Early stopping restores the BEST-validation weights (deep-copied).
Usage: python train_autoencoder.py --seeds 42 123 256 789 1024
"""
import argparse, copy, json, os, time
import joblib, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
import config as C
from model import DenseAutoencoder


def tensor(df, scaler):
    return torch.tensor(scaler.transform(df[C.FEATURE_COLUMNS].values), dtype=torch.float32)


def train_one(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    tr = pd.read_parquet(os.path.join(C.DATA_DIR, f"train_benign_seed{seed}.parquet"))
    va = pd.read_parquet(os.path.join(C.DATA_DIR, f"val_benign_seed{seed}.parquet"))
    sc = joblib.load(os.path.join(C.DATA_DIR, f"scaler_seed{seed}.joblib"))
    xtr, xva = tensor(tr, sc), tensor(va, sc)
    loader = DataLoader(TensorDataset(xtr), batch_size=C.BATCH_SIZE, shuffle=True)
    model = DenseAutoencoder(len(C.FEATURE_COLUMNS))
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE, betas=(0.9, 0.999))
    lossf = torch.nn.MSELoss()
    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, C.EPOCHS + 1):
        model.train(); tot, n = 0.0, 0
        for (b,) in loader:
            opt.zero_grad(); l = lossf(model(b), b); l.backward(); opt.step()
            tot += l.item() * len(b); n += len(b)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xva), xva).item()
        hist.append(dict(epoch=ep, train_loss=tot / n, val_loss=vl))
        print(f"[seed {seed}] epoch {ep:02d} train={tot/n:.6f} val={vl:.6f}", flush=True)
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= C.EARLY_STOP_PATIENCE:
                print(f"[seed {seed}] early stopping at epoch {ep}"); break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        ve = model.reconstruction_error(xva).numpy()
    tau = float(np.percentile(ve, C.THRESHOLD_PERCENTILE))
    torch.save(model.state_dict(), os.path.join(C.RUN_DIR, f"model_seed{seed}.pt"))
    json.dump(dict(seed=seed, tau=tau, best_val_loss=best, epochs_run=len(hist),
                   train_seconds=time.time() - t0),
              open(os.path.join(C.RUN_DIR, f"threshold_seed{seed}.json"), "w"), indent=2)
    pd.DataFrame(hist).to_csv(os.path.join(C.RUN_DIR, f"history_seed{seed}.csv"), index=False)
    print(f"[seed {seed}] done tau={tau:.6f} best_val={best:.6f} epochs={len(hist)} "
          f"time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    for s in ap.parse_args().seeds:
        train_one(s)
