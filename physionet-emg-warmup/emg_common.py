"""Shared helpers for the 3 scripts: data loading, filtering, windowing, split.

Keeping this in one place guarantees that 02 (classical ML) and 03 (deep learning)
see EXACTLY the same windows and the same train/test split -> fair comparison.
"""
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis, skew

import sys
import types

# Compatibility shim: some locked-down environments block one compiled
# scikit-learn file (the HistGradientBoosting `_predictor` extension), and `sklearn.ensemble`
# imports it eagerly even though we never use HistGradientBoosting. If the real import fails we
# register an empty placeholder so RandomForest can still be imported. On a normal machine the
# try-block succeeds and this does nothing.
try:
    import sklearn.ensemble  # noqa: F401
except ImportError:
    for _m in [k for k in sys.modules if k.startswith("sklearn.ensemble")]:
        del sys.modules[_m]
    _stub = types.ModuleType("sklearn.ensemble._hist_gradient_boosting._predictor")
    for _n in ("_predict_from_raw_data", "_predict_from_binned_data", "_compute_partial_dependence"):
        setattr(_stub, _n, None)
    sys.modules["sklearn.ensemble._hist_gradient_boosting._predictor"] = _stub

HERE = Path(__file__).resolve().parent
FS = 4000                                   # Hz, from the .hea files
CLASSES = ["healthy", "myopathy", "neuropathy"]   # label index = position in list

# ---- windowing / split configuration (justified in 01_load_inspect.py output) ----
WIN = 800            # samples = 200 ms at 4 kHz
TRAIN_STRIDE = 200   # 75% overlap in training (more samples; overlap is OK inside train)
TEST_STRIDE = 800    # NO overlap in test (no near-duplicate windows inflating the score)
TEST_FRAC = 0.20     # last 20% of each recording's timeline is held out
SEED = 0


def find_data_dir():
    """Locate the folder holding emg_healthy.hea etc. (user said ./data/emgdb-1.0.0/,
    but this release unpacks into ./emgdb-1.0.0/examples-of-electromyograms-1.0.0/)."""
    root = HERE.parent
    for hit in list(root.rglob("emg_healthy.hea")):
        return hit.parent
    raise FileNotFoundError("emg_healthy.hea not found under " + str(root))


def read_header(name, data_dir=None):
    """Tiny manual WFDB header parser (record line + signal line)."""
    data_dir = data_dir or find_data_dir()
    lines = (data_dir / f"emg_{name}.hea").read_text().splitlines()
    rec = lines[0].split()
    sig = lines[1].split()
    gain = float(sig[2].split("/")[0])          # '10000/mV' -> 10000 ADC units per mV
    comments = [l[1:].strip() for l in lines if l.startswith("#")]
    return dict(fs=int(rec[2]), n=int(rec[3]), fmt=int(sig[1]), gain=gain,
                baseline=int(sig[4]), comments=comments)


def load_dat(name, data_dir=None):
    """Parse .hea + .dat by hand.
    WFDB format 16 = little-endian signed 16-bit integers; mV = (adc - baseline) / gain."""
    data_dir = data_dir or find_data_dir()
    h = read_header(name, data_dir)
    assert h["fmt"] == 16 and h["fs"] == FS
    adc = np.fromfile(data_dir / f"emg_{name}.dat", dtype="<i2")
    assert len(adc) == h["n"], (len(adc), h["n"])
    return (adc.astype(np.float64) - h["baseline"]) / h["gain"]


def load_txt(name, data_dir=None):
    """Parse the text export (time, mV). A few rows are corrupted -> NaN there."""
    data_dir = data_dir or find_data_dir()
    out = []
    for line in (data_dir / f"emg_{name}.txt").read_text(encoding="latin-1").splitlines():
        try:
            out.append(float(line.split()[1]))
        except (ValueError, IndexError):
            out.append(np.nan)
    return np.array(out)


def load_signal(name, data_dir=None):
    """Return the repaired EMG in millivolts.

    DATA QUIRK: the neuropathy .dat has 16-bit integer OVERFLOW (410 samples below
    -3.2768 mV wrapped around by exactly +6.5536 mV). The .txt export has the true values
    but ~2 corrupted rows per file. So: start from .dat, overwrite with .txt wherever the
    .txt parsed cleanly. (healthy / myopathy are identical in both files.)"""
    x = load_dat(name, data_dir)
    t = load_txt(name, data_dir)
    ok = ~np.isnan(t)
    x[ok] = t[ok]
    return x


def preprocess(x):
    """Remove DC offset and slow drift: 4th-order zero-phase Butterworth high-pass @ 20 Hz.
    (20 Hz is the usual lower edge of the EMG band; below that is motion artefact/drift.)
    We do NOT low-pass: a needle EMG has useful content up to ~Nyquist (2 kHz)."""
    sos = butter(4, 20, btype="highpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x - np.mean(x))


def make_windows(x, start, stop, stride, win=WIN):
    """All full windows lying entirely inside x[start:stop]."""
    idx = np.arange(start, stop - win + 1, stride)
    return np.stack([x[i:i + win] for i in idx]), idx


def build_dataset(data_dir=None):
    """Within-signal, time-ordered split.

    For each recording: first 80% of the timeline -> train windows, last 20% -> test windows.
    Windows never straddle the boundary, so train and test share NO samples.
    Returns dict with Xtr, ytr, Xte, yte (raw windows, mV) and bookkeeping."""
    Xtr, ytr, Xte, yte, ttr, tte = [], [], [], [], [], []
    for label, name in enumerate(CLASSES):
        x = preprocess(load_signal(name, data_dir))
        cut = int(len(x) * (1 - TEST_FRAC))
        a, ia = make_windows(x, 0, cut, TRAIN_STRIDE)
        b, ib = make_windows(x, cut, len(x), TEST_STRIDE)
        Xtr.append(a); ytr += [label] * len(a); ttr.append(ia / FS)
        Xte.append(b); yte += [label] * len(b); tte.append(ib / FS)
    return dict(Xtr=np.concatenate(Xtr), ytr=np.array(ytr),
                Xte=np.concatenate(Xte), yte=np.array(yte),
                t_tr=np.concatenate(ttr), t_te=np.concatenate(tte))


# ---- hand-crafted EMG features (used by Stage A, and by the noise test in Stage B) ----
BANDS = [(20, 100), (100, 300), (300, 800), (800, 2000)]
AMPLITUDE_FEATURES = ["rms", "mav", "wl", "var", "max_abs", "total_power"]   # depend on gain/electrode


def extract_features(X):
    """X: (n_windows, WIN) in mV  ->  (n_windows, n_features), names."""
    d = np.diff(X, axis=1)
    f = {}
    # ---- time domain
    f["rms"] = np.sqrt(np.mean(X ** 2, 1))                       # signal power / effort
    f["mav"] = np.mean(np.abs(X), 1)                             # mean absolute value
    f["var"] = np.var(X, 1)
    f["max_abs"] = np.max(np.abs(X), 1)
    f["wl"] = np.sum(np.abs(d), 1)                               # waveform length: amplitude+frequency+complexity
    f["zcr"] = np.mean(X[:, :-1] * X[:, 1:] < 0, 1)              # zero-crossing rate
    f["ssc"] = np.mean(d[:, 1:] * d[:, :-1] < 0, 1)              # slope-sign changes (turns)
    f["wamp"] = np.mean(np.abs(d) > 0.02, 1)                     # Willison amplitude: fraction of big steps (>20 uV)
    f["kurtosis"] = kurtosis(X, axis=1)                          # "spikiness"
    f["skewness"] = skew(X, axis=1)
    mob = np.sqrt(np.var(d, 1) / np.var(X, 1))                   # Hjorth mobility (~ mean frequency)
    mob_d = np.sqrt(np.var(np.diff(d, axis=1), 1) / np.var(d, 1))
    f["hjorth_mobility"] = mob
    f["hjorth_complexity"] = mob_d / mob                         # how un-sinusoidal the waveform is
    # ---- frequency domain (Hann-windowed periodogram, 20 Hz - Nyquist)
    P = np.abs(np.fft.rfft(X * np.hanning(X.shape[1]), axis=1)) ** 2
    fr = np.fft.rfftfreq(X.shape[1], 1 / FS)
    keep = fr >= 20
    P, fr = P[:, keep], fr[keep]
    tot = P.sum(1)
    f["total_power"] = tot
    f["mnf"] = (P * fr).sum(1) / tot                             # mean power frequency
    f["mdf"] = fr[np.argmax(np.cumsum(P, 1) >= tot[:, None] / 2, axis=1)]   # median power frequency
    f["peak_freq"] = fr[np.argmax(P, 1)]
    for lo, hi in BANDS:
        m = (fr >= lo) & (fr < hi)
        f[f"band_{lo}_{hi}"] = P[:, m].sum(1) / tot              # RELATIVE band power (scale-free)
    pn = P / tot[:, None]
    f["spec_entropy"] = -(pn * np.log(pn + 1e-12)).sum(1) / np.log(pn.shape[1])
    names = list(f)
    F = np.column_stack([f[k] for k in names])
    # amplitude-like quantities are log-normal-ish -> work in log10 so scalers/distances behave
    for k in ["rms", "mav", "var", "max_abs", "wl", "total_power"]:
        i = names.index(k)
        F[:, i] = np.log10(F[:, i] + 1e-12)
    return F, names


