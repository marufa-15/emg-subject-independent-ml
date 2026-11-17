# UCI EMG Physical Action Data Set: subject-independent Stage A (classical) + Stage B (deep learning)

Four subjects, so four leave-one-subject-out (LOSO) folds. Every number below is a score on a subject the models
never saw. With n = 4 subjects there are no significance tests and no confidence intervals; treat differences smaller
than the spread across subjects (std 0.12-0.15 macro-F1) as noise.

Scripts: `01_load_inspect.py` -> `02_features_classical.py` -> `03_deep_learning.py`, shared code in `emg_common.py`.
Checkpoints live in `_cache/` (see "Running it"). PNGs and `02_results.json` / `03_results.json` are in this folder.

## 0. What I checked against your description

| Your description | Verdict |
|---|---|
| 4 subjects, 8 channels (biceps/triceps and thigh/hamstring, L/R), whole-body, Theodoridis 2011, DOI 10.24432/C53W49 | Confirmed by the UCI page (also: 3 male + 1 female, age 25-30, Delsys wireless, CC BY 4.0). |
| Data in `./data/archive/` | It is in `./archive/`. |
| Sampling rate: ambiguity between "10000" and "30000" in the `.log` header | **Neither is a rate.** The UCI page gives no rate. Empirically, header `Channel k=<N>` is the per-channel sample count (N = 9,205-15,106, typically ~10,060) and the recordings last 9-11 s by the header timestamps, so fs is **~1.04-1.05 kHz** across 82 files (least squares 1045; 1042-1055 per subject; the clock has 1 s resolution). At 10 kHz a typical file would last 1 s, at 30 kHz 0.34 s. I used a nominal 1000 Hz; a 5% error moves nothing here (band edges 20-450 Hz). |
| `sub/*.csv` are pre-aggregated, pre-labelled | They are the **exact sample-wise mean of the 4 subjects** (max difference 0.000), cut to the shortest recording: a synthetic "average person". Ignored. (Labels are alphabetical per category: Normal bowing=0..waving=9, Aggressive elbowing=0..slapping=9; confirmed.) |
| sub1-sub4 `.log` and `.txt` | `.log` = text header + 1 stray byte + 8 x N int16, channel-major. `.txt` is identical on the rows it has but **truncated to 10,000 rows** (drops 23,053 samples in 37 files), so I parse the `.log`. `sub2/Normal` has two extra recordings (`Walking1v`, `Walking3v`), excluded -> 80 recordings (4 x 20). |
| (not mentioned) | **Clipping at +-4000**: up to 28% of samples on thigh channels for S1 and S3 (R-Thi 0.277/0.283) vs ~3-4% for S2 and S4. A subject-specific artefact that no feature handles explicitly here. |

The docs also mention "~15 actions/repetitions" per session; I could not verify that and did not use it.

## 1. Design decisions (made before looking at results)

- **Labels: one 20-class problem** (Normal 0-9, Aggressive 10-19). A deployed model would not be told the category, and
  one problem gives both views: category accuracy (aggressive vs normal) and within-category recall come from the same
  predictions. Balance is good: 263-320 windows per class (max/min 1.22), chance = 5%.
- **Transitions:** there is no external segmentation. Each file is one continuous, *weakly labelled* recording of a repeated
  action. I drop the first/last 0.5 s of each recording and windows with activity < 35% of the recording's median. This
  removed only 0.7% of windows: the envelopes show rhythmic movement with short dips, not rest periods, and
  windows between repetitions are legitimately part of a rhythmic class. This is a weak-label setup, not clean trials.
- **Windows:** 256 samples (~250 ms), stride 128; 5,623 windows. Splits are by subject, so overlap cannot leak.
- **Protocol:** outer = leave one subject out. Inner = leave one *training* subject out. Scaler, PCA, MI ranking, RFE,
  hyper-parameters, k, and the choice of "best" classifier all use training subjects only (`emg_common.Prep.fit`);
  DL early stopping uses one training subject as validation, then retrains on all three training subjects for that many epochs.
- **Normalisation variants:** `none`; `recording` (each channel divided by that recording's own RMS: label-free, uses only the
  sample itself); `subject` (each channel divided by the median recording RMS of that subject: label-free, but it needs
  an unlabelled calibration set from every new user and so *does* look at the held-out subject's raw signal, which is the
  one place the "touch nothing from the held-out subject" rule is relaxed, on purpose and without labels).
  **Caveat on `recording`:** dividing every channel by its own RMS also erases the *relative* channel amplitudes (which muscles
  are loud), so it is a harsh variant. A gentler one (one scalar per recording) was not run.

## 2. Stage A results (nested LOSO, macro-F1, 20 classes)

Held-out macro-F1 of the **nested-selected** classical model (cell chosen per fold by inner CV only):

| normalisation | S1 | S2 | S3 | S4 | mean +- std | accuracy | aggressive-vs-normal acc |
|---|---|---|---|---|---|---|---|
| none | 0.453 | 0.125 | 0.323 | 0.353 | 0.313 +- 0.119 | 0.347 | 0.850 |
| recording | 0.332 | 0.036 | 0.341 | 0.315 | 0.256 +- 0.128 | 0.287 | 0.766 |
| subject | 0.482 | 0.078 | 0.274 | 0.374 | 0.302 +- 0.149 | 0.329 | 0.820 |

- **Same data, different split** (fixed classifiers, normalisation none), SVM/RF/KNN/LDA: random windows 0.86/0.85/0.79/0.67;
  within-subject time split (the PhysioNet-style protocol) 0.82/0.82/0.74/0.65; **leave-one-subject-out 0.29/0.33/0.28/0.31**.
  This is the leakage gap: 0.5+ macro-F1 of "accuracy" that is recognising a subject's recording, not an action.
- **S2 is an outlier fold** (0.04-0.13). Measured differences: its session date differs in the header (10/2/2010 vs 6/2, 7/2),
  its spectral centroids are flat (68-75 Hz on every channel; the others vary by channel), its action-by-channel activity pattern correlates only
  0.76-0.78 with the others (the others correlate 0.91-0.94 with each other), and its actions are less distinct from each other
  (spread 1.27 vs 1.5-1.7). In PCA, S2 forms its own cluster while S1, S3 and held-out S4 overlap (`02_pca_subjects.png`).
  This is consistent with, not proof of, a session/electrode-placement shift. Mean over S1/S3/S4 only would be ~0.38 for
  `none`; I report all four because dropping S2 would be selecting on the outcome.
- **Per-subject normalisation did not help overall** (0.313 -> 0.302; S1 and S4 up, S2 and S3 down; all within noise).
  Per-channel per-recording normalisation was clearly worse (0.256).
- **Classifiers are indistinguishable.** Fixed-hyper-parameter "all features" LOSO means: RF 0.326, LDA 0.313, SVM 0.291, KNN 0.289; fold std ~0.13.
  The best single cell in the grid (SVM + MI-selection, subject-norm, 0.340) was found by looking at held-out scores, so it is
  optimistic; the nested-selected number above is the honest one, and the selected cell differed across folds.
- **PCA / feature selection:** 95% of the variance needs 46 of 143 components; PCA did not help (SVM 0.285 vs 0.291, RF 0.276 vs 0.326).
  MI selection (k tuned inside) gave SVM +0.03 and KNN +0.03 (3 of 4 folds each for SVM), not RF/LDA. RFE ~ no change. Small and within noise.
- **What generalises:** aggressive-vs-normal (0.77-0.85 vs 0.5 chance); whole-body-distinct actions: bowing 0.78, waving 0.69,
  walking 0.65, hugging 0.59, running 0.53 recall. **What does not:** kneeing 0.04, sidekicking 0.07, frontkicking/hamering 0.18;
  the dominant confusions are seating<->standing (static postures), clapping->hugging, slapping->headering, handshaking->waving.
  Within-category recall is far lower than category accuracy (normal 0.18-0.67, aggressive 0.15-0.28 per fold).

### Amplitude: does the PhysioNet pattern repeat?
Under LOSO (fixed hyper-parameters, normalisation none): **amplitude-only (24 features) is as good as all 143**
(SVM 0.298 vs 0.289, RF 0.316 vs 0.327, KNN 0.294 vs 0.282, LDA 0.281 vs 0.312); dropping amplitude costs 0.02-0.04; spectral-only is weakest
(0.18-0.22) and dropping spectral features does not hurt (SVM 0.303, KNN 0.314). Mutual information agrees: amplitude MI with class 0.76 vs with subject 0.17.
So **"amplitude dominates" repeats, but "and is confounded" mostly does not**: here amplitude carries real cross-subject
information (which muscles are loud for which action), and it *works on held-out subjects*, whereas on PhysioNet its
predictive power was indistinguishable from recording identity. The subject fingerprints live elsewhere: the top-5 features
by MI with subject are all cross-channel envelope correlations (MI with subject 0.33 > with class 0.24), though removing the cross-channel
group costs little (KNN 0.282 -> 0.259, others <= 0.01). The recording-normalisation result is consistent: removing between-action
channel amplitude hurts, it does not help. With 4 subjects I would not trust any single ablation difference below ~0.03.

## 3. Stage B results (3 seeds, same folds, same windows)

Mean held-out macro-F1 over seeds (std across seeds <= 0.033 everywhere, far smaller than the std across subjects):

| normalisation | classical (nested-selected) | 1D-CNN (8-ch input) | GRU | LSTM |
|---|---|---|---|---|
| none | **0.313** | 0.195 (S1 .145, S2 .095, S3 .262, S4 .280) | 0.125 | 0.121 |
| recording | **0.256** | 0.075 | 0.085 | 0.090 |
| subject | **0.302** | 0.181 | 0.156 | 0.158 |

**Deep learning does not earn its complexity here.** The classical model wins on S1, S3 and S4 in every normalisation (4/4 folds with `none`). On S2 it collapses
(0.04-0.13) and the RNNs are marginally ahead (0.08-0.12) in the `recording` and `subject` modes, but all models are near-useless there.
Early stopping chose 1-7 epochs for the CNN and 3-17 for the RNNs: the networks start fitting the training subjects' idiosyncrasies almost
immediately. Caveats that limit this conclusion: the networks are small, untuned, trained on only 3 subjects, without augmentation (e.g. random
per-channel gain, the obvious remedy for the amplitude differences), so a tuned deep model could narrow the gap. I did not test that.

### The noise-robustness lead (white noise added to held-out windows at a per-window, per-channel SNR; no retraining; 5 draws x 3 seeds x 4 subjects)

| normalisation, model | clean | 20 dB | 10 dB | 5 dB | 0 dB | retained at 0 dB |
|---|---|---|---|---|---|---|
| none, classical | 0.313 | 0.315 | 0.306 | 0.283 | 0.245 | 78% |
| none, 1D-CNN | 0.195 | 0.195 | 0.190 | 0.177 | 0.155 | 79% |
| none, GRU | 0.125 | 0.125 | 0.125 | 0.126 | 0.129 | 103% |
| none, LSTM | 0.121 | 0.122 | 0.123 | 0.127 | 0.126 | 104% |
| subject, classical / GRU | 0.302 / 0.156 | | | | 0.201 / 0.164 | 66% / 105% |
| recording, classical / GRU | 0.256 / 0.085 | 0.155 / 0.087 | 0.017 / 0.088 | 0.008 / 0.085 | 0.007 / 0.061 | 3% / 71% |

- **What replicates:** relative to their own clean score, the GRU/LSTM lose the least: RNN minus classical retained fraction is positive in **4/4 folds at 0 dB in all three
  normalisations** and 3/4 at 10 dB (none, subject; 4/4 for `recording`). The GRU also beat the CNN in retention at 0 dB in 3/4 folds (none) and 3/4 (subject).
- **What does not:** in absolute terms the classical model is still better at every noise level for `none` and `subject` (0 dB: 0.245 vs 0.129; 0.201 vs 0.164).
  The RNNs' flat, even slightly rising curves sit at a low clean score (~0.12-0.16); that looks like *insensitivity of a weak model*
  (it is using coarse cues noise barely touches), not robust accuracy. The one case where RNNs are better in absolute terms
  is `recording` normalisation at <= 10 dB, because there the classical features collapse (their score falls to 0.017); that tells you the hand-crafted features
  (zero crossings, slope changes) are fragile to white noise after per-channel rescaling, not that RNNs are good. I did not test why.
- **Verdict:** the lead is **partly confirmed as a relative effect (RNN accuracy degrades least) and killed as a practical claim** (deep models are not a better
  choice under noise on this data). One noise type, no noise augmentation for any model, 4 folds.

## 4. What this means

- **PhysioNet run vs this one:** 100% accuracy there (recognising which recording) vs macro-F1 ~0.3 (accuracy ~0.35) here, on
  20 classes with chance 0.05. The protocol-gap table reproduces the mechanism on this dataset: the same features score 0.65-0.86 when the split lets the
  subject leak and 0.28-0.33 when it does not. Same lessons that held: classifiers are not distinguishable, deep learning is not worth it with so few
  subjects. What changed: amplitude generalises (it is not just identity), the subject fingerprint moves to cross-channel correlations, and variation between
  subjects (S2) now dominates every comparison.
- **Real limitations:** only 4 subjects (3 training subjects per fold); one session per subject, so subject and session are confounded;
  weak labels from continuous recordings; clipping up to 28% of samples on some channels; sampling rate estimated (~1.05 kHz), not documented;
  small hyper-parameter grids; untuned deep models; `recording` normalisation is a harsh variant; no significance testing.
- **Next steps worth trying:** a scalar per-recording normalisation; channel-gain augmentation and noise augmentation for the deep models;
  explicit handling of clipped samples; longer windows (rhythm over several repetitions); a few labelled calibration windows from the new subject
  (adaptation); and a dataset with more subjects, ideally with several sessions per subject so session and subject can be separated.

## Running it
```
python 01_load_inspect.py            # ~10 s
python 02_features_classical.py      # ~15 min
python 03_deep_learning.py           # ~25 min (--seeds/--epochs/--modes available)
```
Requires the UCI EMG Physical Action Data Set (Theodoridis, 2011, DOI 10.24432/C53W49) extracted under `./archive/`
with the `sub1`-`sub4` subject folders; it is not bundled in this repo. `emg_common.py` parses the raw `.log` files
directly (`.txt` exports are truncated and dropped) and includes a small compatibility shim for one `scikit-learn`
import that is blocked on the machine this was developed on — harmless on a normal install. `02_results.json` and
`03_results.json` in `results/` are the exact outputs these scripts produced.
