"""Shared code for the UCI EMG Physical Action experiments (01, 02, 03).

Everything that must be IDENTICAL across stages lives here: parsing, filtering, normalisation,
windowing/labelling, feature extraction, and the leave-one-subject-out (LOSO) protocol.
"""
import sys
import types
from pathlib import Path

# Compatibility shim: some locked-down environments block one compiled
# scikit-learn file that `sklearn.ensemble` imports eagerly (HistGradientBoosting). If the real import
# fails, register an empty placeholder so RandomForest still imports. No-op on a normal machine.
try:
    import sklearn.ensemble  # noqa: F401
except ImportError:
    for _m in [k for k in sys.modules if k.startswith("sklearn.ensemble")]:
        del sys.modules[_m]
    _stub = types.ModuleType("sklearn.ensemble._hist_gradient_boosting._predictor")
    for _n in ("_predict_from_raw_data", "_predict_from_binned_data", "_compute_partial_dependence"):
        setattr(_stub, _n, None)
    sys.modules["sklearn.ensemble._hist_gradient_boosting._predictor"] = _stub

import re
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt
from scipy.stats import kurtosis

HERE = Path(__file__).resolve().parent
SEED = 0

# ------------------------------------------------------------------ dataset constants
FS = 1000          # Hz. NOT stated in the docs; estimated empirically in 01 (~1.0-1.1 kHz) -> nominal 1000
CH_NAMES = ["R-Bic", "R-Tri", "L-Bic", "L-Tri", "R-Thi", "R-Ham", "L-Thi", "L-Ham"]
ARM, LEG = [0, 1, 2, 3], [4, 5, 6, 7]
NORMAL = ["bowing", "clapping", "handshaking", "hugging", "jumping", "running", "seating", "standing",
          "walking", "waving"]
AGGRESSIVE = ["elbowing", "frontkicking", "hamering", "headering", "kneeing", "pulling", "punching",
              "pushing", "sidekicking", "slapping"]
# LABEL SCHEME (decided up front): ONE 20-class problem. Normal 0-9 keep their index, Aggressive 10-19.
# (The raw/csv labels restart at 0 in each category; we remap so a model never needs to be told the category.)
CLASS_NAMES = NORMAL + AGGRESSIVE
SUBJECTS = [1, 2, 3, 4]

WIN = 256          # samples ~ 256 ms  (one window sees a slice of a movement, not a whole repetition)
STRIDE = 128       # 50% overlap. Safe for any split because splits are by SUBJECT, not by window.


def find_root():
    for hit in HERE.parent.rglob("sub1"):
        if (hit / "Normal" / "log").is_dir():
            return hit.parent
    raise FileNotFoundError("could not find archive/sub1/Normal/log under " + str(HERE.parent))


# ------------------------------------------------------------------ parsing the binary .log files
def parse_log(path):
    """.log layout, reverse-engineered and verified against the .txt exports (01_load_inspect.py):
       text header ending in 'Title=Channel 8\\n', then ONE stray byte, then 8 channels x N int16
       (little-endian), stored channel-by-channel. N is the number in 'Channel 0=<N> ...'.
       Returns (N x 8 float array in raw ADC units, header text)."""
    b = Path(path).read_bytes()
    k = b.rfind(b"Title=Channel 8")
    start = b.index(b"\n", k) + 1
    header = b[:start].decode("latin-1")
    n = int(re.search(r"Channel 0=(\d+)", header).group(1))
    a = np.frombuffer(b[start + 1:start + 1 + 16 * n], dtype="<i2")
    assert a.size == 8 * n, (path, a.size, n)
    return a.astype(np.float64).reshape(8, n).T, header


def load_all(root=None):
    """All 80 recordings (4 subjects x 20 actions). Extras (sub2 Walking1v/Walking3v) are ignored."""
    root = root or find_root()
    recs = []
    for s in SUBJECTS:
        for cat, names, offset in (("Normal", NORMAL, 0), ("Aggressive", AGGRESSIVE, 10)):
            files = {p.stem.lower(): p for p in (root / f"sub{s}" / cat / "log").glob("*.log")}
            for i, nm in enumerate(names):
                x, hdr = parse_log(files[nm])
                recs.append(dict(subject=s, category=cat, action=nm, label=offset + i, x=x, header=hdr))
    return recs


# ------------------------------------------------------------------ filtering + normalisation
def bandpass(x):
    """20-450 Hz 4th-order zero-phase Butterworth band-pass (Nyquist ~500 Hz) + 50 Hz mains notch
    (UK recording site). Filtering a recording by itself uses no information from other subjects."""
    sos = butter(4, [20, 450], btype="bandpass", fs=FS, output="sos")
    y = sosfiltfilt(sos, x - x.mean(0), axis=0)
    b, a = iirnotch(50, 30, fs=FS)
    return filtfilt(b, a, y, axis=0)


NORM_MODES = ["none", "recording", "subject"]


def normalise(recs, mode):
    """Amplitude normalisation applied to the filtered signals BEFORE windowing. Label-free.
      none      : divide by one global constant (units only). Absolute amplitude is kept.
      recording : divide each channel by that recording's own RMS. Uses only the recording itself
                  (like instance-norm), but also erases real amplitude differences BETWEEN actions.
      subject   : divide each channel by the median over THAT SUBJECT's recordings of the per-recording RMS.
                  Keeps between-action amplitude differences but needs an (unlabelled) calibration set
                  from every new user: it is the only step that looks at the held-out subject's raw
                  signal, and it never uses labels. Documented, not hidden."""
    out = []
    if mode == "subject":
        ref = {}
        for s in SUBJECTS:
            rms = np.array([np.sqrt((r["xf"] ** 2).mean(0)) for r in recs if r["subject"] == s])
            ref[s] = np.median(rms, axis=0)
    for r in recs:
        if mode == "none":
            scale = 1000.0
        elif mode == "recording":
            scale = np.sqrt((r["xf"] ** 2).mean(0))
        elif mode == "subject":
            scale = ref[r["subject"]]
        else:
            raise ValueError(mode)
        out.append(r["xf"] / scale)
    return out


# ------------------------------------------------------------------ windowing + transition handling
EDGE_S = 0.5          # drop the first/last 0.5 s of every recording (start/stop of the recording)
REST_FRAC = 0.35      # drop windows whose activity is < 35% of that recording's median activity
                      #   (rest / pauses between repetitions: they look the same for every class)


def window_recording(xf, subject_unused=None):
    """Return window start indices and an activity score per window.
    activity = mean over channels of window RMS of the FILTERED, UN-normalised signal, divided by the
    per-channel recording median RMS: so it is a within-recording relative measure (no labels, no
    other recordings involved)."""
    n = len(xf)
    e = int(EDGE_S * FS)
    starts = np.arange(e, n - e - WIN + 1, STRIDE)
    W = np.stack([xf[s:s + WIN] for s in starts])             # (nw, WIN, 8)
    rms = np.sqrt((W ** 2).mean(1))                            # (nw, 8)
    ref = np.median(rms, axis=0) + 1e-9
    act = (rms / ref).mean(1)
    return starts, act


def build_windows(recs, mode, gate=True):
    """Windows from every recording. `gate`=True drops low-activity windows (rest between repetitions).
    Returns dict: X (nw,WIN,8) normalised windows, y, subject, rec_id, act."""
    xs = normalise(recs, mode)
    Xs, ys, ss, rid, acts = [], [], [], [], []
    for i, (r, xn) in enumerate(zip(recs, xs)):
        starts, act = window_recording(r["xf"])
        keep = np.ones(len(starts), bool)
        if gate:
            # activity relative to this recording's own median window-activity (which is 1 by construction)
            keep = act >= REST_FRAC * np.median(act)
        for s, a, k in zip(starts, act, keep):
            if k:
                Xs.append(xn[s:s + WIN]); ys.append(r["label"]); ss.append(r["subject"]); rid.append(i); acts.append(a)
    return dict(X=np.stack(Xs).astype(np.float32), y=np.array(ys), subject=np.array(ss),
                rec=np.array(rid), act=np.array(acts))


def prepare(mode="none", gate=True, root=None):
    recs = load_all(root)
    for r in recs:
        r["xf"] = bandpass(r["x"])
    return recs, build_windows(recs, mode, gate)


# ------------------------------------------------------------------ features (per channel + cross channel)
BANDS = [(20, 60), (60, 120), (120, 250), (250, 450)]
AMP_FEATURES = ["rms", "mav", "wl"]          # absolute-amplitude features (gain / electrode / subject dependent)


def extract_features(W):
    """W: (n, WIN, 8) -> (n, n_feat), names.
    Per channel: rms, mav, wl, zcr, ssc, kurtosis, mnf, mdf, spectral entropy, 4 relative band powers.
    Cross channel: share of total RMS per channel (scale-free 'muscle synergy'), log arm/leg and
    left/right RMS ratios, and correlation between the rectified (envelope-like) signals of every channel pair."""
    n, L, C = W.shape
    f, names = [], []

    def add(a, nm):
        f.append(a.reshape(n, -1)); names.extend(nm)

    d = np.diff(W, axis=1)
    rms = np.sqrt((W ** 2).mean(1)) + 1e-9
    add(rms, [f"rms_{c}" for c in CH_NAMES])
    add(np.abs(W).mean(1), [f"mav_{c}" for c in CH_NAMES])
    add(np.abs(d).sum(1), [f"wl_{c}" for c in CH_NAMES])
    add((W[:, :-1] * W[:, 1:] < 0).mean(1), [f"zcr_{c}" for c in CH_NAMES])
    add((d[:, 1:] * d[:, :-1] < 0).mean(1), [f"ssc_{c}" for c in CH_NAMES])
    add(kurtosis(W, axis=1), [f"kurt_{c}" for c in CH_NAMES])
    P = np.abs(np.fft.rfft(W * np.hanning(L)[None, :, None], axis=1)) ** 2
    fr = np.fft.rfftfreq(L, 1 / FS)
    k = fr >= 20
    P, fr = P[:, k], fr[k]
    tot = P.sum(1) + 1e-12
    add((P * fr[None, :, None]).sum(1) / tot, [f"mnf_{c}" for c in CH_NAMES])
    cum = np.cumsum(P, 1)
    add(fr[np.argmax(cum >= tot[:, None] / 2, axis=1)], [f"mdf_{c}" for c in CH_NAMES])
    pn = P / tot[:, None]
    add(-(pn * np.log(pn + 1e-12)).sum(1) / np.log(pn.shape[1]), [f"sentropy_{c}" for c in CH_NAMES])
    for lo, hi in BANDS:
        m = (fr >= lo) & (fr < hi)
        add(P[:, m].sum(1) / tot, [f"band{lo}_{hi}_{c}" for c in CH_NAMES])
    # --- cross-channel
    add(rms / rms.sum(1, keepdims=True), [f"share_{c}" for c in CH_NAMES])
    lr = lambda a, b: np.log(rms[:, a].mean(1) / rms[:, b].mean(1))
    add(np.column_stack([lr(ARM, LEG), lr([0, 1], [2, 3]), lr([4, 5], [6, 7])]),
        ["log_arm_leg", "log_Rarm_Larm", "log_Rleg_Lleg"])
    R = np.abs(W)
    R = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-9)
    iu = np.triu_indices(C, 1)
    corr = np.einsum("nlc,nld->ncd", R, R) / L
    add(corr[:, iu[0], iu[1]], [f"envcorr_{CH_NAMES[i]}_{CH_NAMES[j]}" for i, j in zip(*iu)])
    F = np.concatenate(f, axis=1)
    # amplitude-like quantities are log-normal-ish -> log10 (this is a fixed transform, no fitting)
    for i, nm in enumerate(names):
        if nm.split("_")[0] in ("rms", "mav", "wl"):
            F[:, i] = np.log10(np.maximum(F[:, i], 1e-9))
    return F.astype(np.float64), names


# ------------------------------------------------------------------ classical model machinery (nested-CV building blocks)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.metrics import f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLFS = {
    "SVM": lambda p: SVC(C=p["C"], gamma="scale"),
    # n_jobs=1: avoids native-threading instability seen on some setups; results are identical (seeded)
    "RF": lambda p: RandomForestClassifier(200, max_depth=p["max_depth"], n_jobs=1, random_state=SEED),
    "KNN": lambda p: KNeighborsClassifier(p["k"]),
    "LDA": lambda p: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=p["shrinkage"]),
}
GRIDS = {"SVM": [{"C": 1}, {"C": 10}], "RF": [{"max_depth": 10}, {"max_depth": None}],
         "KNN": [{"k": 5}, {"k": 15}], "LDA": [{"shrinkage": None}, {"shrinkage": "auto"}]}
# feature treatments. Each is FIT ON TRAINING DATA ONLY (inside the nested loop): scaler, PCA, MI ranking, RFE.
TREATS = {"all": [{}], "pca95": [{}], "mi": [{"k": 20}, {"k": 50}], "rfe": [{"k": 30}]}


class Prep:
    """StandardScaler -> (nothing | PCA(95% var) | top-k mutual information | RFE with a random forest).
    Everything is learned in .fit() from the data you hand it - hand it training subjects only."""
    def __init__(self, treat, tp):
        self.treat, self.tp = treat, tp

    def fit(self, F, y):
        self.sc = StandardScaler().fit(F)
        A = self.sc.transform(F)
        self.idx = None
        if self.treat == "pca95":
            self.pca = PCA(n_components=0.95, svd_solver="full").fit(A)
        elif self.treat == "mi":
            mi = mutual_info_classif(A, y, random_state=SEED, n_jobs=1)    # n_jobs>1 (loky processes) is flaky here
            self.idx = np.argsort(-mi)[:self.tp["k"]]
        elif self.treat == "rfe":
            r = RFE(RandomForestClassifier(60, n_jobs=-1, random_state=SEED),
                    n_features_to_select=self.tp["k"], step=0.2).fit(A, y)
            self.idx = np.where(r.support_)[0]
        return self

    def transform(self, F):
        A = self.sc.transform(F)
        if self.treat == "pca95":
            return self.pca.transform(A)
        return A if self.idx is None else A[:, self.idx]


class Model:
    def __init__(self, prep, clf):
        self.prep, self.clf = prep, clf

    def predict(self, F):
        return self.clf.predict(self.prep.transform(F))


def fit_model(Ftr, ytr, treat, tp, clf, cp):
    prep = Prep(treat, tp).fit(Ftr, ytr)
    return Model(prep, CLFS[clf](cp).fit(prep.transform(Ftr), ytr))


def metrics(y, pred):
    """Everything we report per held-out subject."""
    rec = recall_score(y, pred, labels=range(20), average=None, zero_division=0)
    return dict(acc=float(np.mean(y == pred)), f1=float(f1_score(y, pred, average="macro", labels=range(20), zero_division=0)),
                recall=rec.tolist(), cat_acc=float(np.mean((y >= 10) == (pred >= 10))),
                recall_normal=float(rec[:10].mean()), recall_aggr=float(rec[10:].mean()))


# ------------------------------------------------------------------ the LOSO protocol
def loso_splits(subject):
    """Outer folds: hold out each subject in turn. Yields (held_out_subject, train_idx, test_idx)."""
    for s in SUBJECTS:
        yield s, np.where(subject != s)[0], np.where(subject == s)[0]


def inner_splits(subject, train_idx):
    """Inner folds for tuning/selection, using ONLY the training subjects: leave one training subject out."""
    ts = subject[train_idx]
    for s in sorted(set(ts)):
        yield np.where(ts != s)[0], np.where(ts == s)[0]     # indices relative to train_idx
