# EMG warm-up: Stage A (classical ML) + Stage B (deep learning) on PhysioNet "Examples of Electromyograms"

**Status: a mechanics exercise, not a benchmark.** Each class is ONE recording from ONE person. Nothing here
says anything about how these methods would work on a new patient.

Run order: `01_load_inspect.py` -> `02_features_classical.py` -> `03_deep_learning.py` (shared code in `emg_common.py`).

## Setup that was actually used
- 4 kHz single-channel needle EMG, tibialis anterior. Healthy 12.7 s, myopathy 27.6 s, neuropathy 37.0 s.
- Pre-processing: remove DC, 20 Hz high-pass (zero-phase Butterworth). No low-pass.
- Windows: 800 samples = 200 ms (long enough for a few motor-unit potentials and stable spectra, short enough to
  give many samples; 150-300 ms is the usual EMG range). Train stride 200 (75% overlap), test stride 800 (no overlap).
- Split: first 80% of each recording's timeline = train, last 20% = test; no window straddles the boundary, so
  train and test share no samples. Train 1226 windows, test 75 (12 healthy / 27 myopathy / 36 neuropathy).
- Model/hyper-parameter selection used time-blocked CV (5 blocks, purged) inside train only. For deep models,
  early stopping used the last 15% of the training timeline. The test set was never used to choose anything.

## Data problems found (worth knowing)
1. **The neuropathy `.dat` is corrupted by int16 overflow**: 410 samples below -3.28 mV wrapped around by exactly
   +6.5536 mV. The true range is -7.30 to +3.26 mV. `wfdb` would read the damaged `.dat`. The `.txt` has the true
   values but 2 garbled rows per file, so `emg_common.load_signal` uses the `.dat` and overwrites it with the clean
   `.txt` values. Healthy and myopathy are identical in both files.
2. The environment blocks some compiled libraries (pandas -> wfdb unusable; one scikit-learn file). Handled by manual
   parsing and a guarded import shim in `emg_common.py`. Not a concern on a normal machine.

## What the numbers say

Clean held-out test (75 windows), all five feature sets x four classifiers:

| | SVM | RandomForest | KNN | LDA |
|---|---|---|---|---|
| all 21 features | 1.000 | 1.000 | 1.000 | 1.000 |
| PCA (95% variance, 7 comps) | 1.000 | 1.000 | 1.000 | 1.000 |
| top-6 mutual information | 0.987 | 0.987 | 0.987 | 0.987 |
| RFE-6 | 1.000 | 1.000 | 1.000 | 1.000 |
| no amplitude features (shape + frequency only) | 1.000 | 1.000 | 1.000 | 1.000 |

Deep models on raw windows (3 seeds): 1D-CNN 1.000 +/- 0.000, GRU 0.996 +/- 0.006, LSTM 0.987 +/- 0.011.
Best classical by blocked-CV (LDA, CV macro-F1 0.997 vs 0.993-0.996 for the others, a near-tie): 1.000.

**Read this as: the task is saturated.** With one recording per class, "classify the window" is effectively
"recognise which recording this window came from", and that is easy for everything. One test miss is 1.3 points,
so differences of that size are noise. The clean test cannot rank the models.

- **PCA**: 7 of 21 components give 95% variance (PC1 51%, PC2 21%); the classes separate clearly in PC1-PC2. It
  did not change accuracy (everything was already ~100%), so it is useful for visualisation/compression here, not
  for accuracy.
- **Which features mattered**: amplitude-type features dominate both rankings (max |x|, variance, RMS, total power:
  MI 0.89-0.96), then skewness and low-band (20-100 Hz) power. This matches the plots: neuropathy has large
  spikes, healthy is quiet. **But amplitude is exactly the feature most confounded by electrode position and gain
  in a single-recording-per-class design.** Dropping all six amplitude features still gave 100% test accuracy, so
  the recordings also differ in shape/frequency, but I cannot tell disease signal from recording-specific signal.
- **Noise stress test** (added white noise, no retraining; DL = one seed): all models are perfect at 20 dB. At
  10 dB LDA falls to 0.58 and the CNN to 0.67, while the GRU/LSTM stay near 0.97-0.99, and at 0 dB they hold ~0.9
  versus ~0.4 for LDA/CNN. A plausible reason is that the RNNs see 5 ms patches and the hand-crafted shape features
  (zero crossings, slope changes) are corrupted directly by white noise, but I did not test that explanation, the
  DL curves are single-seed, and the non-monotonic LDA recovery at -5 dB shows these curves should not be
  over-interpreted. Treat it as "worth looking at on a bigger dataset", not a finding.

## Is deep learning earning its complexity here?
No. A 21-feature LDA (no tuning, milliseconds to train, interpretable coefficients) matched a 24k-parameter CNN on
the clean test, and the sequence models were slightly worse and less stable across seeds. With ~1,000 heavily
overlapping training windows from 3 recordings, a deep network has little to learn beyond what a few amplitude and
spectral features already capture, and its extra capacity mostly buys the ability to memorise recording-specific
quirks. Classical ML wins on small signal datasets when (a) domain features already encode the physics, (b) samples
are few and correlated, and (c) you need to explain what the model uses. Deep models start to pay off with many
subjects/channels/classes where hand-crafted features stop being enough. The only hint in my favour of the RNNs
was noise robustness, and that is unverified. (A Transformer was not built; the same argument applies, more so.)

## The real limitation
One subject per class. Class identity is perfectly confounded with subject, electrode placement, gain, noise floor,
age (44/57/62), medication, and which phase of the task was recorded. Within the single timeline there is also a
distribution shift (neuropathy's test region is mostly its high-effort burst). Time-ordered splitting prevents
neighbouring-window leakage, but it cannot create new subjects. No generalisation claim is supported.

## What this motivated for the next dataset
Planned before looking at a multi-subject dataset, and checked against it afterwards in
[`../uci-physical-action-loso`](../uci-physical-action-loso):
- **Split by subject, never by window or repetition.** Use leave-one-subject-out or GroupKFold with subject as the
  group. Keep all of a subject's windows on one side, including overlapping windows and repeated trials.
- **Select everything inside the training subjects**: scalers, feature selection, PCA, hyper-parameters, early
  stopping (nested/grouped CV). One leak here recreated today's 100%.
- **Expect much lower, much noisier scores**, and report the per-subject spread, not just the mean.
- **Per-subject variation in amplitude/electrode placement**: try per-subject or per-recording normalisation, and
  always compare with the scale-free feature subset as in the ablation above.
- **Multi-channel input**: extend the CNN to channels-as-input-channels; for classical features, compute per
  channel and add cross-channel features.
- **Labels/transitions**: windows spanning a change of action are ambiguous; decide how to label or drop them.
  Check class balance and use macro-F1 plus per-class recall.
- **Repeat the noise/robustness probe**, but with multiple seeds and with held-out subjects. That is where a real
  deep-vs-classical difference could show up.
- Fix seeds, record versions, and keep a "test subject" untouched until the very end.

The follow-up study confirms some of this and overturns part of it — see that project's README, section "Amplitude:
does the PhysioNet pattern repeat?".
