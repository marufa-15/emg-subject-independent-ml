"""Step 3 - Stage B: deep learning on raw 8-channel windows, leave-one-subject-out, plus the noise-robustness probe.

Run (after 02):  python emg_physical_action/03_deep_learning.py [--modes none subject recording] [--seeds 3] [--epochs 20]
Outputs: 03_fold_comparison.png, 03_noise_robustness.png, 03_training_curves.png, 03_results.json

PROTOCOL (same folds and same windows as Stage A):
  outer: hold out one subject. Never touched until the final evaluation.
  early stopping (inside the 3 training subjects): phase 1 trains on 2 of them and validates on the 3rd to pick the number
  of epochs; phase 2 retrains from scratch on all 3 training subjects for that many epochs. -> the DL model gets exactly
  the training subjects the classical model gets.
  input scaling: one scalar (std of the training subjects' windows) -> keeps relative amplitude information.
"""
import argparse
import json
import pickle
import time
from emg_common import (HERE, CLASS_NAMES, SUBJECTS, NORM_MODES, SEED, prepare, extract_features, loso_splits,
                        fit_model, metrics)    # first: sklearn shim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--modes", nargs="+", default=NORM_MODES)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--folds", nargs="+", type=int, default=SUBJECTS)
ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--recompute", action="store_true")
ARGS = ap.parse_args()
torch.set_num_threads(2)           # fewer native threads: avoids instability seen under heavy threading on some setups
CACHE_DIR = HERE / "_cache"; CACHE_DIR.mkdir(exist_ok=True)


def cached(key, fn):
    """Checkpoint each (mode, held-out subject, model, seed) unit to disk; a crash then only costs one unit."""
    p = CACHE_DIR / f"03_{key}.pkl"
    if p.exists() and not ARGS.recompute:
        return pickle.load(open(p, "rb"))
    v = fn()
    pickle.dump(v, open(p, "wb"))
    return v
BATCH, LR = 128, 2e-3
SNRS = [None, 20, 10, 5, 0]
MODELS = ["1D-CNN", "GRU", "LSTM"]
f1m = lambda y, p: f1_score(y, p, average="macro", labels=range(20), zero_division=0)


# ------------------------------------------------------------------ models
class CNN1D(nn.Module):
    """Input (B, 8 channels, 256 samples). Conv kernels span ALL 8 channels at once, so the first layer can learn
    cross-channel patterns (which muscles fire together). Global avg+max pooling -> position-independent."""
    def __init__(self):
        super().__init__()
        def blk(i, o, pool): return [nn.Conv1d(i, o, 9, padding=4), nn.BatchNorm1d(o), nn.ReLU()] + ([nn.MaxPool1d(pool)] if pool else [])
        self.f = nn.Sequential(*blk(8, 32, 4), *blk(32, 64, 4), *blk(64, 128, 0))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 20))

    def forward(self, x):
        h = self.f(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


class RNN(nn.Module):
    """(B, 8, 256) -> 32 time steps, each = 8 consecutive samples x 8 channels = 64 numbers -> GRU/LSTM -> last state."""
    def __init__(self, kind):
        super().__init__()
        self.rnn = (nn.GRU if kind == "GRU" else nn.LSTM)(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 20))

    def forward(self, x):
        B = x.shape[0]
        seq = x.reshape(B, 8, 32, 8).permute(0, 2, 3, 1).reshape(B, 32, 64)     # (B, steps, 8 samples * 8 ch)
        out, _ = self.rnn(seq)
        return self.head(out[:, -1])


MAKE = {"1D-CNN": CNN1D, "GRU": lambda: RNN("GRU"), "LSTM": lambda: RNN("LSTM")}


def to_t(W, scale):                     # (n, 256, 8) -> (n, 8, 256)
    return torch.tensor(np.ascontiguousarray(W.transpose(0, 2, 1)) / scale, dtype=torch.float32)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 1024]).argmax(1) for i in range(0, len(X), 1024)]).numpy()


def train(name, seed, Xt, yt, epochs, Xv=None, yv=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MAKE[name]()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    yt_t = torch.tensor(yt)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yt_t))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(); lossf(model(Xt[b]), yt_t[b]).backward(); opt.step()
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                vo = model(Xv); hist.append((lossf(vo, torch.tensor(yv)).item(), f1m(yv, vo.argmax(1).numpy())))
    return model, hist


# ------------------------------------------------------------------ classical comparison set (from 02) + noise sets
R02 = json.load(open(HERE / "02_results.json"))
t_start = time.time()
RES = {}
for mode in ARGS.modes:
    recs, d = prepare(mode)
    W, y, subj = d["X"], d["y"], d["subject"]
    RES[mode] = dict(folds={})
    print(f"\n######## normalisation = {mode} ########")
    for held, tr, te in loso_splits(subj):
        if held not in ARGS.folds:
            continue
        t0 = time.time()
        scale = float(W[tr].std())
        Xtr_all, Xte = to_t(W[tr], scale), to_t(W[te], scale)
        ytr, yte = y[tr], y[te]
        tr_subj = subj[tr]
        vs = [s for s in SUBJECTS if s != held][0]                # validation subject (a TRAINING subject)
        p1 = tr_subj != vs
        # ---- noisy copies of the test windows: SAME noisy windows for every model, seed and draw
        rng = np.random.default_rng(1000 + held)
        noisy = {}
        for snr in SNRS[1:]:
            noisy[snr] = []
            for _ in range(ARGS.draws):
                rms = np.sqrt((W[te] ** 2).mean(1, keepdims=True)) + 1e-9          # per window, per channel
                noisy[snr].append((W[te] + rng.standard_normal(W[te].shape) * rms / 10 ** (snr / 20)).astype(np.float32))
        # ---- classical model: the config picked by INNER CV in Stage A for this fold, refit on the 3 training subjects
        sel = R02["nested_selected"][mode][SUBJECTS.index(held)]
        t_, c_ = sel["cell"]

        def _classical():
            Ftr, _ = extract_features(W[tr])
            cmodel = fit_model(Ftr, ytr, t_, sel["tp"], c_, sel["cp"])
            cres = {"clean": f1m(yte, cmodel.predict(extract_features(W[te])[0]))}
            for snr in SNRS[1:]:
                cres[snr] = float(np.mean([f1m(yte, cmodel.predict(extract_features(nz)[0])) for nz in noisy[snr]]))
            return dict(cell=[t_, c_], f1_clean=cres["clean"], noise={str(k): v for k, v in cres.items() if k != "clean"})

        RES[mode]["folds"][held] = dict(classical=cached(f"{mode}_S{held}_classical_d{ARGS.draws}", _classical), dl={})
        cres = RES[mode]["folds"][held]["classical"]
        print(f"[{mode}] held-out S{held}: classical({c_}/{t_}) clean F1={cres['f1_clean']:.3f}", flush=True)
        for name in MODELS:
            per_seed = []
            for sd in range(ARGS.seeds):
                seed = SEED + 17 * sd + held

                def _one():
                    _, hist = train(name, seed, Xtr_all[p1], ytr[p1], ARGS.epochs, Xtr_all[~p1], ytr[~p1])     # phase 1: pick epochs
                    e_star = int(np.argmin([h[0] for h in hist])) + 1
                    model, _ = train(name, seed, Xtr_all, ytr, e_star)                                         # phase 2: all 3 train subjects
                    pred = predict(model, Xte)
                    m = metrics(yte, pred)
                    nz = {str(snr): float(np.mean([f1m(yte, predict(model, to_t(x, scale))) for x in noisy[snr]])) for snr in SNRS[1:]}
                    return dict(seed=seed, e_star=e_star, noise=nz, hist=hist if sd == 0 else None, **m)

                per_seed.append(cached(f"{mode}_S{held}_{name}_s{sd}_e{ARGS.epochs}_d{ARGS.draws}", _one))
                print(f"      {name} seed {sd}: F1={per_seed[-1]['f1']:.3f} epochs={per_seed[-1]['e_star']}", flush=True)
            RES[mode]["folds"][held]["dl"][name] = per_seed
            f = [s["f1"] for s in per_seed]
            print(f"   {name:<7} F1 per seed {['%.3f' % v for v in f]}  mean {np.mean(f):.3f}  (epochs {[s['e_star'] for s in per_seed]})  "
                  f"params={sum(p.numel() for p in MAKE[name]().parameters())}")
        print(f"   fold time {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)")
    json.dump(RES, open(HERE / "03_results.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ reports
print("\n\n=================== DEEP vs CLASSICAL, identical folds/windows (held-out-subject macro-F1) ===================")
for mode in ARGS.modes:
    F = RES[mode]["folds"]; hs = sorted(F)
    print(f"\n normalisation = {mode}      (chance ~0.05)")
    print(f" {'model':<28}" + "".join(f"{'S'+str(h):>8}" for h in hs) + f"{'mean':>8}{'std':>7}{'  seed-std':>11}")
    cl = [F[h]["classical"]["f1_clean"] for h in hs]
    print(f" {'classical (nested-selected)':<28}" + "".join(f"{v:8.3f}" for v in cl) + f"{np.mean(cl):8.3f}{np.std(cl):7.3f}{'  n/a':>11}")
    for name in MODELS:
        fm = [np.mean([s["f1"] for s in F[h]["dl"][name]]) for h in hs]
        ss = np.mean([np.std([s["f1"] for s in F[h]["dl"][name]]) for h in hs])
        print(f" {name:<28}" + "".join(f"{v:8.3f}" for v in fm) + f"{np.mean(fm):8.3f}{np.std(fm):7.3f}{ss:11.3f}")
    print(f" per-fold winner (mean over seeds): " + ", ".join(
        f"S{h}: " + max([("classical", F[h]['classical']['f1_clean'])] + [(n, np.mean([s['f1'] for s in F[h]['dl'][n]])) for n in MODELS], key=lambda t: t[1])[0]
        for h in hs))
    print(f" aggressive-vs-normal accuracy: " + ", ".join(f"{n} {np.mean([s['cat_acc'] for h in hs for s in F[h]['dl'][n]]):.3f}" for n in MODELS))

print("\n=================== NOISE ROBUSTNESS (no retraining; mean over folds and seeds) ===================")
print("Each cell: held-out macro-F1 [fraction of that model's own clean F1 retained]. +- = std over the 4 held-out subjects.")
SUMMARY = {}
for mode in ARGS.modes:
    F = RES[mode]["folds"]; hs = sorted(F)
    cur = {}
    cur["classical"] = {"clean": [F[h]["classical"]["f1_clean"] for h in hs],
                        **{str(s): [F[h]["classical"]["noise"][str(s)] for h in hs] for s in SNRS[1:]}}
    for n in MODELS:
        cur[n] = {"clean": [np.mean([s["f1"] for s in F[h]["dl"][n]]) for h in hs],
                  **{str(s): [np.mean([x["noise"][str(s)] for x in F[h]["dl"][n]]) for h in hs] for s in SNRS[1:]}}
    SUMMARY[mode] = cur
    print(f"\n normalisation = {mode}")
    print(f" {'model':<12}" + "".join(f"{('clean' if s is None else str(s) + ' dB'):>20}" for s in SNRS))
    for n, v in cur.items():
        cl = np.mean(v["clean"])
        print(f" {n:<12}" + "".join(f"{np.mean(v['clean' if s is None else str(s)]):7.3f}+-{np.std(v['clean' if s is None else str(s)]):.3f}[{np.mean(v['clean' if s is None else str(s)]) / cl:4.0%}]" for s in SNRS))
    print(" paired per-fold difference in retained fraction at 10 dB / 0 dB (RNN-mean minus classical), sign across folds:")
    for snr in (10, 0):
        rr = {n: np.array(cur[n][str(snr)]) / np.array(cur[n]["clean"]) for n in cur}
        rnn = (rr["GRU"] + rr["LSTM"]) / 2
        d_ = rnn - rr["classical"]
        d2 = rr["GRU"] - rr["1D-CNN"]
        print(f"   {snr:>2} dB: RNN - classical = {np.round(d_, 2)} ({int((d_ > 0).sum())}/{len(d_)} folds positive);   GRU - CNN = {np.round(d2, 2)}")

# ------------------------------------------------------------------ plots
cols = {"classical": "#444444", "1D-CNN": "#d95f02", "GRU": "#1b9e77", "LSTM": "#7570b3"}
fig, axs = plt.subplots(1, len(ARGS.modes), figsize=(5.6 * len(ARGS.modes), 4.4), sharey=True, squeeze=False)
for ax, mode in zip(axs[0], ARGS.modes):
    F = RES[mode]["folds"]; hs = sorted(F)
    series = {"classical": [F[h]["classical"]["f1_clean"] for h in hs], **{n: [np.mean([s["f1"] for s in F[h]["dl"][n]]) for h in hs] for n in MODELS}}
    for j, (n, v) in enumerate(series.items()):
        ax.bar(np.arange(len(hs)) + (j - 1.5) * 0.2, v, 0.2, color=cols[n], label=n)
        if n in MODELS:
            ax.errorbar(np.arange(len(hs)) + (j - 1.5) * 0.2, v, yerr=[np.std([s["f1"] for s in F[h]["dl"][n]]) for h in hs], fmt="none", ecolor="k", capsize=2)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([f"held out S{h}" for h in hs]); ax.set_title(f"norm: {mode}"); ax.axhline(.05, color="grey", ls=":")
axs[0, 0].set_ylabel("macro-F1 (error bar = std over seeds)"); axs[0, 0].legend(); fig.tight_layout(); fig.savefig(HERE / "03_fold_comparison.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(2, len(ARGS.modes), figsize=(5.6 * len(ARGS.modes), 7.4), squeeze=False)
xl = ["clean", "20", "10", "5", "0"]
for k, mode in enumerate(ARGS.modes):
    for n, v in SUMMARY[mode].items():
        m = np.array([np.mean(v[s]) for s in xl]); sdv = np.array([np.std(v[s]) for s in xl])
        axs[0, k].plot(range(5), m, "o-", color=cols[n], label=n); axs[0, k].fill_between(range(5), m - sdv, m + sdv, color=cols[n], alpha=.12)
        rt = np.array([np.mean(np.array(v[s]) / np.array(v["clean"])) for s in xl])
        axs[1, k].plot(range(5), rt, "o-", color=cols[n], label=n)
    for r in range(2):
        axs[r, k].set_xticks(range(5)); axs[r, k].set_xticklabels(["clean", "20 dB", "10 dB", "5 dB", "0 dB"]); axs[r, k].grid(alpha=.3)
    axs[0, k].set_title(f"norm: {mode}  - held-out macro-F1 (band = std over 4 subjects)"); axs[1, k].set_title("fraction of own clean F1 retained")
axs[0, 0].legend(); fig.tight_layout(); fig.savefig(HERE / "03_noise_robustness.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
m0 = ARGS.modes[0]; h0 = sorted(RES[m0]["folds"])[0]
for ax, n in zip(axs, MODELS):
    h = np.array(RES[m0]["folds"][h0]["dl"][n][0]["hist"])
    ax.plot(h[:, 0], label="val loss (held-in subject)"); ax.plot(h[:, 1], label="val macro-F1"); ax.set_title(f"{n}, norm={m0}, outer fold S{h0}"); ax.set_xlabel("epoch"); ax.legend()
fig.tight_layout(); fig.savefig(HERE / "03_training_curves.png", dpi=130); plt.close(fig)
json.dump(dict(runs=RES, noise_summary=SUMMARY), open(HERE / "03_results.json", "w"), indent=1, default=float)
print("\nsaved 03_results.json + PNGs")
