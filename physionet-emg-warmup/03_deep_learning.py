"""Step 3 - Stage B: deep learning on RAW windows, compared fairly with Stage A.

Run (after 02_features_classical.py):  python emg_classical_dl/03_deep_learning.py
Outputs: 03_training_curves.png, 03_confusion_matrices.png, 03_noise_robustness.png, 03_results.json

Fairness rules:
* identical windows / identical time-ordered train-test split as Stage A (shared emg_common.py)
* the test set is never used for early stopping or model choice
* every model is reported with the same metrics; DL models are run with 3 seeds
"""
import json
import time
from emg_common import (HERE, FS, CLASSES, WIN, SEED, build_dataset, extract_features)  # first: sklearn workaround
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

torch.set_num_threads(4)
DEV = "cpu"
EPOCHS, BATCH, LR, SEEDS = 40, 64, 2e-3, [0, 1, 2]

# ------------------------------------------------------------------ data
ds = build_dataset()
Xtr_all, ytr_all, Xte, yte, t_tr = ds["Xtr"], ds["ytr"], ds["Xte"], ds["yte"], ds["t_tr"]

# Validation slice = last 15% of each class's TRAIN timeline; purge train windows touching it.
is_val = np.zeros(len(ytr_all), bool)
is_tr = np.ones(len(ytr_all), bool)
for k in range(3):
    m = ytr_all == k
    t0, t1 = t_tr[m].min(), t_tr[m].max()
    v0 = t1 - 0.15 * (t1 - t0)
    is_val |= m & (t_tr >= v0)
    is_tr &= ~(m & (t_tr > v0 - WIN / FS))
is_tr &= ~is_val
Xtr, ytr, Xva, yva = Xtr_all[is_tr], ytr_all[is_tr], Xtr_all[is_val], ytr_all[is_val]
print(f"DL split: train {len(ytr)} | val {len(yva)} (for early stopping) | test {len(yte)} (untouched until the end)")

SCALE = Xtr.std()          # ONE global scalar -> keeps relative amplitude information
print(f"input scaling: divide by global train std = {SCALE:.4f} mV")
tens = lambda X: torch.tensor(X / SCALE, dtype=torch.float32).unsqueeze(1)   # (N,1,800)
cw = torch.tensor(len(ytr) / (3 * np.bincount(ytr)), dtype=torch.float32)    # class weights (imbalance)


# ------------------------------------------------------------------ models
class CNN1D(nn.Module):
    """3 conv blocks learn local waveform shapes (spikes, phases); global avg+max pooling make the
    result independent of WHERE in the window a spike occurs."""
    def __init__(self):
        super().__init__()
        def blk(i, o, pool): return [nn.Conv1d(i, o, 9, padding=4), nn.BatchNorm1d(o), nn.ReLU()] + ([nn.MaxPool1d(pool)] if pool else [])
        self.f = nn.Sequential(*blk(1, 16, 4), *blk(16, 32, 4), *blk(32, 64, 0))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        h = self.f(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


class RNN(nn.Module):
    """800 raw samples -> 40 time steps of 20 samples (5 ms patches) -> GRU/LSTM -> last state."""
    def __init__(self, kind):
        super().__init__()
        cell = nn.GRU if kind == "GRU" else nn.LSTM
        self.rnn = cell(20, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(32, 3))

    def forward(self, x):
        seq = x.squeeze(1).reshape(x.shape[0], 40, 20)
        out, _ = self.rnn(seq)
        return self.head(out[:, -1])


MAKERS = {"1D-CNN": CNN1D, "GRU": lambda: RNN("GRU"), "LSTM": lambda: RNN("LSTM")}


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(tens(X)).argmax(1).numpy()


def train(name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MAKERS[name]().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Xt, yt = tens(Xtr), torch.tensor(ytr)
    Xv, yv = tens(Xva), torch.tensor(yva)
    best, best_state, hist = 1e9, None, []
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(); lossf(model(Xt[b]), yt[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tl = lossf(model(Xt), yt).item(); vo = model(Xv); vl = lossf(vo, yv).item()
        hist.append((tl, vl, (vo.argmax(1) == yv).float().mean().item()))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)          # early stopping = keep best-validation epoch
    return model, hist


# ------------------------------------------------------------------ run
res, models0, hist0 = {}, {}, {}
for name in MAKERS:
    accs, f1s = [], []
    t0 = time.time()
    for s in SEEDS:
        model, hist = train(name, SEED + s)
        pred = predict(model, Xte)
        accs.append(accuracy_score(yte, pred)); f1s.append(f1_score(yte, pred, average="macro"))
        if s == SEEDS[0]:
            models0[name], hist0[name] = model, hist
            res[name] = dict(pred=pred.tolist(), params=sum(p.numel() for p in model.parameters()))
    res[name].update(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)), accs=accs,
                     f1_mean=float(np.mean(f1s)), secs_per_run=(time.time() - t0) / len(SEEDS))
    print(f"{name:<7} params={res[name]['params']:>6}  test acc per seed={['%.3f' % a for a in accs]}  "
          f"~{res[name]['secs_per_run']:.0f}s/run")

# ------------------------------------------------------------------ side-by-side on the SAME test windows
A = json.load(open(HERE / "02_results.json"))
best = A["best_model"]
res[f"classical:{best}"] = dict(pred=A["pred_test"][best], params=None)
classical = joblib.load(HERE / "02_best_classical_model.joblib")
print(f"\n=== Same {len(yte)} held-out test windows, same metrics (seed-0 deep models) ===")
print(f"{'model':<18}{'acc':>7}{'macro-F1':>10}   per-class recall (healthy/myopathy/neuropathy)   per-class precision")
fig, axs = plt.subplots(1, 4, figsize=(18, 4.4))
for ax, (name, r) in zip(axs, [(f"classical:{best}", res[f"classical:{best}"])] + [(n, res[n]) for n in MAKERS]):
    pred = np.array(r["pred"])
    p, rec, _, _ = precision_recall_fscore_support(yte, pred, labels=range(3), zero_division=0)
    print(f"{name:<18}{accuracy_score(yte, pred):7.3f}{f1_score(yte, pred, average='macro'):10.3f}   "
          f"{np.round(rec, 3)}   {np.round(p, 3)}")
    cm = confusion_matrix(yte, pred, labels=range(3))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "k")
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(CLASSES, rotation=20); ax.set_yticklabels(CLASSES)
    ax.set_title(name); ax.set_xlabel("predicted"); ax.set_ylabel("true")
fig.tight_layout(); fig.savefig(HERE / "03_confusion_matrices.png", dpi=130); plt.close(fig)
print("3-seed test accuracy:", {n: f"{res[n]['acc_mean']:.3f}±{res[n]['acc_std']:.3f}" for n in MAKERS})

fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, n in zip(axs, MAKERS):
    h = np.array(hist0[n])
    ax.plot(h[:, 0], label="train loss"); ax.plot(h[:, 1], label="val loss")
    ax.plot(h[:, 2], label="val acc", ls="--"); ax.set_title(n); ax.set_xlabel("epoch"); ax.legend()
fig.tight_layout(); fig.savefig(HERE / "03_training_curves.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ noise stress test
# Clean test accuracy is saturated, so it cannot rank the models. Add white noise to each TEST window at a
# given per-window SNR and see who degrades first (no re-training: this is a robustness probe).
print("\n=== Noise-robustness on the test windows (mean accuracy over 10 noise draws) ===")
SNRS = [None, 20, 10, 5, 0, -5]
rng = np.random.default_rng(SEED)
curves = {n: [] for n in [f"classical:{best}", *MAKERS]}
for snr in SNRS:
    acc = {n: [] for n in curves}
    for _ in range(1 if snr is None else 10):
        if snr is None:
            Xn = Xte
        else:
            rms = np.sqrt((Xte ** 2).mean(1, keepdims=True))
            Xn = Xte + rng.standard_normal(Xte.shape) * rms / 10 ** (snr / 20)
        F, _ = extract_features(Xn)
        acc[f"classical:{best}"].append(accuracy_score(yte, classical.predict(F)))
        for n in MAKERS:
            acc[n].append(accuracy_score(yte, predict(models0[n], Xn)))
    for n in curves:
        curves[n].append(float(np.mean(acc[n])))
print(f"{'SNR (dB)':<10}" + "".join(f"{n:>20}" for n in curves))
for i, snr in enumerate(SNRS):
    print(f"{'clean' if snr is None else snr:<10}" + "".join(f"{curves[n][i]:>20.3f}" for n in curves))
fig, ax = plt.subplots(figsize=(8, 4.5))
for n, c in curves.items():
    ax.plot(range(len(SNRS)), c, "o-", label=n)
ax.set_xticks(range(len(SNRS))); ax.set_xticklabels(["clean" if s is None else f"{s} dB" for s in SNRS])
ax.set_ylabel("test accuracy"); ax.set_xlabel("per-window SNR of added white noise"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Robustness to added noise (same 75 test windows; no retraining)")
fig.tight_layout(); fig.savefig(HERE / "03_noise_robustness.png", dpi=130); plt.close(fig)

json.dump(dict(models={k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in res.items()},
               noise=dict(snr_db=[s if s is not None else "clean" for s in SNRS], curves=curves)),
          open(HERE / "03_results.json", "w"), indent=1)
print("\nsaved 03_results.json + PNGs")
