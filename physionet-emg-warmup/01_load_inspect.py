"""Step 1 - load, inspect and plot the three EMG recordings; justify the windowing.

Run:  python emg_classical_dl/01_load_inspect.py
Outputs (PNG, saved next to this script): 01_signals_full.png, 01_signals_zoom.png,
01_psd.png, 01_envelope.png, 01_windowing.png
"""
import matplotlib
matplotlib.use("Agg")                      # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

from emg_common import (HERE, FS, CLASSES, WIN, TRAIN_STRIDE, TEST_STRIDE, TEST_FRAC,
                        find_data_dir, read_header, load_signal, load_dat, load_txt,
                        preprocess, build_dataset)

COLORS = {"healthy": "#1b9e77", "myopathy": "#d95f02", "neuropathy": "#7570b3"}
data_dir = find_data_dir()
print("data dir:", data_dir, "\n")

# ------------------------------------------------------------------ 1. load + sanity checks
raw, sig = {}, {}
for name in CLASSES:
    h = read_header(name)
    raw[name] = load_signal(name)
    sig[name] = preprocess(raw[name])
    print(f"[{name}] fs={h['fs']} Hz  samples={h['n']}  duration={h['n']/FS:.1f} s")
    print("   header notes:", " | ".join(h["comments"]))

# Cross-check the two on-disk formats: .dat (binary, what wfdb reads) vs .txt (text export)
print("\nCross-check .dat vs .txt")
for name in CLASSES:
    d, t = load_dat(name), load_txt(name)
    ok = ~np.isnan(t)
    diff = (t - d)[ok]
    n_bad = int(np.sum(np.abs(diff) > 1e-6))
    print(f"  [{name}] corrupted .txt rows: {int((~ok).sum())} | rows where .dat != .txt: {n_bad}"
          + (f" | all differ by {np.unique(np.round(diff[np.abs(diff) > 1e-6], 4))} mV "
             f"(= 65536/gain -> int16 overflow in the .dat; .txt has true values)" if n_bad else ""))
    print(f"      .dat range [{d.min():.2f}, {d.max():.2f}] mV   repaired range "
          f"[{raw[name].min():.2f}, {raw[name].max():.2f}] mV")

# NOTE: the official `wfdb` package would read the .dat (with the overflow); it also needs
# pandas, which some locked-down environments block; parsed by hand instead to avoid that dependency.

# ------------------------------------------------------------------ 2. summary stats
print("\nSummary statistics (after DC removal + 20 Hz high-pass), mV")
print(f"{'class':<11}{'RMS':>8}{'peak':>8}{'kurtosis':>10}{'frac|x|>3*RMS':>15}")
for name in CLASSES:
    x = sig[name]
    rms = np.sqrt(np.mean(x ** 2))
    print(f"{name:<11}{rms:8.4f}{np.max(np.abs(x)):8.3f}{kurtosis(x):10.2f}"
          f"{np.mean(np.abs(x) > 3 * rms):15.4f}")

# ------------------------------------------------------------------ 3. plots
# (a) full recordings
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharey=True)
for ax, name in zip(axs, CLASSES):
    t = np.arange(len(sig[name])) / FS
    ax.plot(t, sig[name], lw=0.4, color=COLORS[name])
    ax.axvline(t[-1] * (1 - TEST_FRAC), color="k", ls="--", lw=1)
    ax.set_title(f"{name}  ({t[-1]:.1f} s) - dashed line = start of held-out test region", fontsize=10)
    ax.set_ylabel("mV")
axs[-1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(HERE / "01_signals_full.png", dpi=130); plt.close(fig)

# (b) 200 ms zoom taken from the most active part of each recording (max-RMS window)
fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharey=True)
for ax, name in zip(axs, CLASSES):
    x = sig[name]
    starts = np.arange(0, len(x) - WIN, WIN // 2)
    best = starts[np.argmax([np.sum(x[s:s + WIN] ** 2) for s in starts])]
    ax.plot(np.arange(WIN) / FS * 1000, x[best:best + WIN], lw=0.7, color=COLORS[name])
    ax.set_title(f"{name}: most active 200 ms (starts at {best/FS:.2f} s)", fontsize=10)
    ax.set_ylabel("mV")
axs[-1].set_xlabel("time within window (ms)")
fig.tight_layout(); fig.savefig(HERE / "01_signals_zoom.png", dpi=130); plt.close(fig)

# (c) power spectral density (Welch) - frequency content per class
fig, ax = plt.subplots(figsize=(9, 5))
for name in CLASSES:
    f, p = welch(sig[name], fs=FS, nperseg=1024)
    ax.semilogy(f, p, color=COLORS[name], label=name)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD (mV$^2$/Hz)"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Welch PSD of each full recording")
fig.tight_layout(); fig.savefig(HERE / "01_psd.png", dpi=130); plt.close(fig)

# (d) 200 ms RMS envelope over time -> shows contraction vs rest periods
fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharey=True)
for ax, name in zip(axs, CLASSES):
    x = sig[name]
    n = len(x) // WIN
    env = np.sqrt(np.mean(x[:n * WIN].reshape(n, WIN) ** 2, axis=1))
    ax.plot(np.arange(n) * WIN / FS, env, color=COLORS[name], marker=".", lw=0.8)
    ax.axvline(len(x) / FS * (1 - TEST_FRAC), color="k", ls="--", lw=1)
    ax.set_title(f"{name}: RMS per 200 ms block", fontsize=10); ax.set_ylabel("mV")
axs[-1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(HERE / "01_envelope.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 4. windowing + split
ds = build_dataset()
print(f"\nWindowing: {WIN} samples = {WIN/FS*1000:.0f} ms; train stride {TRAIN_STRIDE} "
      f"({100*(1-TRAIN_STRIDE/WIN):.0f}% overlap); test stride {TEST_STRIDE} (no overlap); "
      f"last {TEST_FRAC:.0%} of each recording held out")
print(f"{'class':<11}{'train windows':>15}{'test windows':>14}")
for k, name in enumerate(CLASSES):
    print(f"{name:<11}{(ds['ytr']==k).sum():>15}{(ds['yte']==k).sum():>14}")
print(f"{'TOTAL':<11}{len(ds['ytr']):>15}{len(ds['yte']):>14}")

fig, ax = plt.subplots(figsize=(11, 3.2))
for k, name in enumerate(CLASSES):
    T = len(sig[name]) / FS
    ax.barh(k, T * (1 - TEST_FRAC), color=COLORS[name], alpha=.85, label="train" if k == 0 else None)
    ax.barh(k, T * TEST_FRAC, left=T * (1 - TEST_FRAC), color="#444444", label="test" if k == 0 else None)
ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES); ax.set_xlabel("time (s)")
ax.set_title("Time-ordered within-signal split (coloured = train, grey = held-out test)")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(HERE / "01_windowing.png", dpi=130)
print("\nSaved PNGs to", HERE)
