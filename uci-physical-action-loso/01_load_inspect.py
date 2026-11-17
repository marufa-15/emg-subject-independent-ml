"""Step 1 - verify what the dataset actually is, inspect it, and fix the labelling/windowing scheme.

Run:  python emg_physical_action/01_load_inspect.py
Outputs: 01_sampling_rate.png, 01_amplitude_by_subject.png, 01_envelopes.png, 01_class_balance.png
"""
import datetime as dt
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from emg_common import (HERE, FS, CH_NAMES, CLASS_NAMES, SUBJECTS, NORMAL, AGGRESSIVE, WIN, STRIDE, EDGE_S,
                        REST_FRAC, find_root, parse_log, load_all, bandpass, build_windows, window_recording)

root = find_root()
print("archive root:", root, "\n")
recs = load_all(root)
for r in recs:
    r["xf"] = bandpass(r["x"])

# ------------------------------------------------------------------ 1. inventory
print(f"{len(recs)} recordings = {len(SUBJECTS)} subjects x {len(NORMAL)} normal + {len(AGGRESSIVE)} aggressive actions")
print(f"{'subject':<8}" + "".join(f"{'S'+str(s):>9}" for s in SUBJECTS))
print(f"{'n recs':<8}" + "".join(f"{sum(r['subject']==s for r in recs):>9}" for s in SUBJECTS))
print(f"{'min N':<8}" + "".join(f"{min(len(r['x']) for r in recs if r['subject']==s):>9}" for s in SUBJECTS))
print(f"{'max N':<8}" + "".join(f"{max(len(r['x']) for r in recs if r['subject']==s):>9}" for s in SUBJECTS))
extras = sorted(p.relative_to(root).as_posix() for p in root.glob("sub*/*/log/*.log")
                if p.stem.lower() not in NORMAL + AGGRESSIVE)
print("ignored extra recordings (not part of the 10+10 action set):", extras)

# ------------------------------------------------------------------ 2. sampling rate, empirically
# The docs give no rate. The .log header has "Recorded=... 2 10000" and "Channel k=<N> 2 3 30000 0 -4 mV".
# Test: if <N> is the per-channel sample count, N / (End-Start wall-clock seconds) = sampling rate.
N, D, labels = [], [], []
f = "%m/%d/%Y %H:%M:%S"
for s in SUBJECTS:
    for p in sorted(root.glob(f"sub{s}/*/log/*.log")):
        hdr = parse_log(p)[1]
        n = int(re.search(r"Channel 0=(\d+)", hdr).group(1))
        a = dt.datetime.strptime(re.search(r"Start=(.*)", hdr).group(1).strip(), f)
        b = dt.datetime.strptime(re.search(r"End=(.*)", hdr).group(1).strip(), f)
        N.append(n); D.append((b - a).total_seconds())
N, D = np.array(N), np.array(D)
slope = (D @ N) / (D @ D)
print(f"\nSAMPLING RATE (empirical, {len(N)} log files): header N / wall-clock duration")
print(f"  least-squares samples-per-second = {slope:.0f};  per-file ratio min/median/max = "
      f"{(N/D).min():.0f}/{np.median(N/D):.0f}/{(N/D).max():.0f}   (clock has 1 s resolution -> +-5-10% scatter)")
print(f"  typical recording: N = {np.median(N):.0f} samples, duration {np.median(D):.0f} s")
for rate in (1000, 10000, 30000):
    print(f"    if fs = {rate:>5} Hz the typical recording would last {np.median(N)/rate:7.2f} s "
          f"({'consistent' if abs(np.median(N)/rate - np.median(D)) < 1.5 else 'INCONSISTENT'} with the 9-11 s timestamps)")
print("  -> fs ~ 1 kHz. Neither '10000' nor '30000' in the header is a sampling rate. "
      f"I use FS = {FS} Hz (true value may be up to ~1.1 kHz; band edges are far from the limits, so conclusions are insensitive).")
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(D + np.random.default_rng(0).uniform(-.12, .12, len(D)), N, s=14, alpha=.6)
xs = np.array([8, 19]); ax.plot(xs, xs * 1000, "k--", label="1 kHz"); ax.plot(xs, xs * 10000, "r:", label="10 kHz")
ax.set_ylim(0, 25000); ax.set_xlabel("wall-clock duration End-Start (s)"); ax.set_ylabel("samples per channel (header N)")
ax.legend(); ax.set_title("Sampling-rate check"); fig.tight_layout(); fig.savefig(HERE / "01_sampling_rate.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 3. .log vs .txt, and what sub/*.csv really is
tot = same = 0
trunc = []
for r in recs:
    cat = r["category"]
    tp = [q for q in root.glob(f"sub{r['subject']}/{cat}/txt/*.txt") if q.stem.lower() == r["action"]][0]
    t = np.loadtxt(tp, ndmin=2)
    m = min(len(t), len(r["x"]))
    tot += 1; same += int(np.array_equal(t[:m], r["x"][:m]))
    if len(t) < len(r["x"]): trunc.append(len(r["x"]) - len(t))
print(f"\n.log vs .txt: parsed .log identical to .txt on the overlapping rows in {same}/{tot} recordings; "
      f".txt is truncated to 10000 rows (drops {sum(trunc)} samples in {len(trunc)} files, up to {max(trunc)}). -> use .log")
c = np.loadtxt(root / "sub" / "Normal" / "Walking.csv", delimiter=",", skiprows=1)
subs = [r for r in recs if r["action"] == "walking"]
m = min([len(c)] + [len(r["x"]) for r in subs])
avg = np.mean([r["x"][:m] for r in subs], axis=0)
print(f"sub/Normal/Walking.csv: max |csv - mean of the 4 subjects' raw signals| = {np.abs(c[:m, :8] - avg).max():.3f}"
      "  -> the 'sub' folder is the sample-wise AVERAGE of all 4 subjects (not a subject). It is IGNORED.")

# ------------------------------------------------------------------ 4. saturation + per-subject amplitude (the confound)
print("\nSaturation: fraction of raw samples at |x| >= 3999 (txt/log clip at 4000), per subject")
for s in SUBJECTS:
    xr = np.concatenate([r["x"] for r in recs if r["subject"] == s])
    print(f"  S{s}: " + " ".join(f"{(np.abs(xr[:, c]) >= 3999).mean():.3f}" for c in range(8)) + "   (channels " + ",".join(CH_NAMES) + ")")
rms = {s: np.array([np.sqrt((r["xf"] ** 2).mean(0)) for r in recs if r["subject"] == s]) for s in SUBJECTS}   # (20, 8)
print("\nMedian per-recording RMS (filtered, raw ADC units) by subject and channel:")
print(f"{'':>5}" + "".join(f"{c:>9}" for c in CH_NAMES))
for s in SUBJECTS:
    print(f"  S{s} " + "".join(f"{np.median(rms[s][:, c]):9.0f}" for c in range(8)))
med = np.array([np.median(rms[s], axis=0) for s in SUBJECTS])
print("  max/min ratio across subjects per channel:", np.round(med.max(0) / med.min(0), 1))
fig, ax = plt.subplots(figsize=(10, 4.2))
w = 0.2
for i, s in enumerate(SUBJECTS):
    ax.bar(np.arange(8) + (i - 1.5) * w, med[i], w, label=f"S{s}")
ax.set_xticks(range(8)); ax.set_xticklabels(CH_NAMES); ax.set_yscale("log"); ax.legend()
ax.set_ylabel("median recording RMS (ADC units, log)"); ax.set_title("Same channel, different subject: amplitude differs a lot")
fig.tight_layout(); fig.savefig(HERE / "01_amplitude_by_subject.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 4b. are the subjects equally 'comparable'? (explains the held-out-S2 collapse seen in step 2)
from scipy.signal import welch
print("\nSUBJECT/SESSION COMPARABILITY")
print(" session date in header, and samples/second (sum N / sum wall-clock seconds) per subject:")
for s in SUBJECTS:
    dates = sorted({re.search(r"Start=(\S+)", r["header"]).group(1) for r in recs if r["subject"] == s})
    n_tot = sum(len(r["x"]) for r in recs if r["subject"] == s)
    d_tot = sum((dt.datetime.strptime(re.search(r"End=(.*)", r["header"]).group(1).strip(), f)
                 - dt.datetime.strptime(re.search(r"Start=(.*)", r["header"]).group(1).strip(), f)).total_seconds()
                for r in recs if r["subject"] == s)
    print(f"   S{s}: date(s) {dates}   {n_tot / d_tot:.0f} samples/s")
print(" median spectral centroid (Hz) per channel and subject:")
print(f"       " + "".join(f"{c:>7}" for c in CH_NAMES))
for s in SUBJECTS:
    row = []
    for ch in range(8):
        v = []
        for r in recs:
            if r["subject"] == s:
                fr, p = welch(r["xf"][:, ch], fs=FS, nperseg=256); k = fr >= 20; v.append((fr[k] * p[k]).sum() / p[k].sum())
        row.append(np.median(v))
    print(f"   S{s}: " + "".join(f"{x:7.0f}" for x in row))
prof = {s: np.array([np.log(np.sqrt((r["xf"] ** 2).mean(0))) for r in recs if r["subject"] == s]) for s in SUBJECTS}   # (20 actions, 8 ch)
M = {s: (prof[s] - prof[s].mean(0)).ravel() for s in SUBJECTS}
print(" correlation between subjects of the (action x channel) log-RMS pattern (per-channel mean removed):")
print("        " + "".join(f"   S{t}" for t in SUBJECTS))
for s in SUBJECTS:
    print(f"   S{s}  " + "".join(f"{np.corrcoef(M[s], M[t])[0, 1]:5.2f}" for t in SUBJECTS))
print(" spread (std) of per-action log-RMS across the 20 actions, mean over channels: " +
      ", ".join(f"S{s} {prof[s].std(0).mean():.2f}" for s in SUBJECTS) + "  (smaller = actions look more alike)")

# ------------------------------------------------------------------ 5. envelopes -> motivates transition/rest handling
show = [("walking", 1), ("punching", 1), ("standing", 1), ("bowing", 2), ("slapping", 3), ("hugging", 4)]
fig, axs = plt.subplots(len(show), 1, figsize=(12, 10), sharex=False)
for ax, (nm, s) in zip(axs, show):
    r = next(r for r in recs if r["action"] == nm and r["subject"] == s)
    starts, act = window_recording(r["xf"])
    ax.plot(starts / FS, act, color="#1b6ca8", lw=1)
    ax.axhline(REST_FRAC * np.median(act), color="r", ls="--", lw=.8)
    ax.axvspan(0, EDGE_S, color="grey", alpha=.3); ax.axvspan(len(r["xf"]) / FS - EDGE_S, len(r["xf"]) / FS, color="grey", alpha=.3)
    ax.set_title(f"S{s} {nm}: window activity (rel. to recording median); red = rest-gate threshold", fontsize=9)
axs[-1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(HERE / "01_envelopes.png", dpi=120); plt.close(fig)

# ------------------------------------------------------------------ 6. windows, transition handling, class balance
raw = build_windows(recs, "none", gate=False)
gat = build_windows(recs, "none", gate=True)
print(f"\nWINDOWING: {WIN} samples (~{WIN/FS*1000:.0f} ms), stride {STRIDE}; first/last {EDGE_S}s of each recording dropped")
print(f"  windows: {len(raw['y'])} before rest-gating -> {len(gat['y'])} after "
      f"(dropped {100*(1-len(gat['y'])/len(raw['y'])):.1f}% with activity < {REST_FRAC:.0%} of the recording's median)")
print("TRANSITION RULE: there is no external segmentation, so each file is ONE weakly-labelled continuous recording of a"
      "\n  repeated action. Windows in the first/last 0.5 s (start/stop) and low-activity windows (pauses between repetitions,\n"
      "  which look the same for every class) are dropped. This uses no labels. It does not remove every ambiguous window.")
cnt = np.array([[np.sum((gat["subject"] == s) & (gat["y"] == k)) for k in range(20)] for s in SUBJECTS])
print("\nClass balance (windows per class) - 20-class scheme, Normal 0-9, Aggressive 10-19:")
print(f"{'class':<14}" + "".join(f"{'S'+str(s):>7}" for s in SUBJECTS) + f"{'total':>8}")
for k in range(20):
    print(f"{k:>2} {CLASS_NAMES[k]:<11}" + "".join(f"{cnt[i, k]:>7}" for i in range(4)) + f"{cnt[:, k].sum():>8}")
print(f"max/min class ratio overall: {cnt.sum(0).max()/cnt.sum(0).min():.2f};  chance accuracy = {1/20:.0%}; "
      f"majority-class share = {cnt.sum(0).max()/cnt.sum():.1%}")
fig, ax = plt.subplots(figsize=(11, 4))
bot = np.zeros(20)
for i, s in enumerate(SUBJECTS):
    ax.bar(range(20), cnt[i], bottom=bot, label=f"S{s}"); bot += cnt[i]
ax.set_xticks(range(20)); ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right"); ax.legend(); ax.set_ylabel("windows")
ax.set_title("Windows per class after gating (stacked by subject)")
fig.tight_layout(); fig.savefig(HERE / "01_class_balance.png", dpi=130); plt.close(fig)
print("\nsaved PNGs to", HERE)
