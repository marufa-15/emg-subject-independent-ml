# EMG subject-independent evaluation

Two small studies on surface/needle EMG classification, run as a learning exercise in applying classical ML
and deep learning to physiological time series. The main point isn't either dataset's accuracy number — it's
what happens to that number when the evaluation protocol changes from "looks correct" to "is correct."

## The headline result

On the same features and the same classifiers, moving from a leaky split to a subject-held-out split changes
macro-F1 from **0.65-0.86 down to 0.28-0.33** (chance = 0.05, 20 classes) on the UCI EMG Physical Action
dataset. The accuracy didn't go away because the models got worse — it went away because the leaky version
was never measuring what it looked like it was measuring. See
[`uci-physical-action-loso/README.md`](uci-physical-action-loso/README.md), section "Same data, different
split."

## Projects

- **[`physionet-emg-warmup/`](physionet-emg-warmup/)** - PhysioNet "Examples of Electromyograms": 3 needle-EMG
  recordings (healthy / myopathy / neuropathy), one subject per class. Used to get the mechanics right
  (windowing, feature extraction, classical ML, 1D-CNN/LSTM/GRU) on a small, clean signal. Explicitly a
  warm-up: with one subject per class, "accuracy" here is partly just recognising which recording a window
  came from, not a disease signature. The report says so and explains why.

- **[`uci-physical-action-loso/`](uci-physical-action-loso/)** - UCI EMG Physical Action Data Set (Theodoridis,
  2011): 8-channel EMG, 4 subjects, 20 whole-body actions (10 normal, 10 aggressive). Evaluated with nested
  leave-one-subject-out cross-validation - every scaler, feature selector, hyperparameter, and "best model"
  choice is made using only the training subjects, never the held-out one. This is the harder, more honest
  version of the first study, and it changes some of the first study's conclusions (amplitude features turn
  out to carry real cross-subject signal here, not just recording identity) while confirming others (classical
  ML still matches or beats small deep models at this scale).

Each project folder has its own `README.md` with the full write-up: design decisions made *before* looking at
results, the numbers, what generalizes and what doesn't, and what's still unverified.

## What these are and aren't

Both are evaluation-methodology exercises on small public datasets, not benchmark results to cite and not
evidence for any clinical or prosthetic-control claim. Sample sizes are small (3 recordings; 4 subjects), and
both reports say explicitly where a result is noise-level, single-seed, or otherwise under-powered rather than
rounding it up to a finding.

## Reproducing

Each project's scripts are numbered and meant to be run in order (`01_load_inspect.py` ->
`02_features_classical.py` -> `03_deep_learning.py`); see that project's README for the exact commands and
where to get the raw data (not bundled here - see each README for the source and DOI). `results/` holds the
JSON output actually produced by these runs; `figures/` holds the generated plots.

## License

MIT - see [`LICENSE`](LICENSE).
