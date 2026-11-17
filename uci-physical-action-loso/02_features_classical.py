"""Step 2 - Stage A: classical ML with SUBJECT-LEVEL evaluation (leave-one-subject-out, nested).

Run:  python emg_physical_action/02_features_classical.py            (uses cache if present)
      python emg_physical_action/02_features_classical.py --recompute
Outputs: 02_fold_scores.png, 02_treatments.png, 02_protocol_gap.png, 02_feature_groups.png, 02_subject_vs_class_mi.png,
         02_pca_subjects.png, 02_confusion_headline.png, 02_recall_heatmap.png, 02_results.json

PROTOCOL (the rules, enforced in code):
  outer loop : hold out ONE subject (4 folds). That subject's windows are never used for fitting ANYTHING.
  inner loop : leave-one-TRAINING-subject-out inside the 3 remaining subjects -> chooses hyper-parameters,
               k for feature selection, and which classifier/treatment is 'best'. Selection score = macro-F1.
  fitted-on-training-only: StandardScaler, PCA, mutual-information ranking, RFE (see emg_common.Prep).
"""
import argparse
import json
import pickle
import time
from collections import defaultdict
from emg_common import (HERE, FS, CH_NAMES, CLASS_NAMES, SUBJECTS, NORM_MODES, CLFS, GRIDS, TREATS, SEED,
                        prepare, extract_features, loso_splits, inner_splits, Prep, fit_model, metrics)   # first: sklearn shim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser(); ap.add_argument("--recompute", action="store_true")
ARGS = ap.parse_args()
CACHE_DIR = HERE / "_cache"; CACHE_DIR.mkdir(exist_ok=True)
np.random.seed(SEED)


def cached(key, fn):
    """Checkpoint: guards against interpreter crashes mid-run (seen intermittently in some environments under heavy native-code load), so every unit of work
    is saved to disk and skipped on re-run (use --recompute to start from scratch)."""
    p = CACHE_DIR / f"02_{key}.pkl"
    if p.exists() and not ARGS.recompute:
        return pickle.load(open(p, "rb"))
    v = fn()
    pickle.dump(v, open(p, "wb"))
    return v
CLF_COL = {"SVM": "#4c72b0", "RF": "#55a868", "KNN": "#c44e52", "LDA": "#8172b2"}
f1m = lambda y, p: f1_score(y, p, average="macro", labels=range(20), zero_division=0)


def group_of(name):
    p = name.split("_")[0]
    if p in ("rms", "mav", "wl"): return "amplitude"
    if p in ("zcr", "ssc", "kurt"): return "temporal shape"
    if p in ("mnf", "mdf", "sentropy") or p.startswith("band"): return "spectral"
    return "cross-channel"


# ------------------------------------------------------------------ data for every normalisation mode
DATA = {}
for mode in NORM_MODES:
    recs, d = prepare(mode)
    F, names = extract_features(d["X"])
    DATA[mode] = dict(F=F, y=d["y"], subj=d["subject"], rec=d["rec"])
NAMES = names
GROUPS = np.array([group_of(n) for n in NAMES])
print(f"{len(NAMES)} features/window ({', '.join(f'{g}: {(GROUPS==g).sum()}' for g in dict.fromkeys(GROUPS))}); "
      f"{len(DATA['none']['y'])} windows; subjects {SUBJECTS}\n")


# ------------------------------------------------------------------ 1. nested leave-one-subject-out grid
def nested_fold(F, y, subj, held, tr, te, tag):
    cells = {}
    if True:                                                  # (kept as a block so the indentation below is unchanged)
        t0 = time.time()
        Ftr, ytr = F[tr], y[tr]
        inner = list(inner_splits(subj, tr))
        scores = defaultdict(list)
        for treat, tps in TREATS.items():
            for ti, tp in enumerate(tps):
                for itr, iva in inner:                       # everything below is fit on inner-train subjects only
                    prep = Prep(treat, tp).fit(Ftr[itr], ytr[itr])
                    A, B = prep.transform(Ftr[itr]), prep.transform(Ftr[iva])
                    for clf in CLFS:
                        for ci, cp in enumerate(GRIDS[clf]):
                            m = CLFS[clf](cp).fit(A, ytr[itr])
                            scores[(treat, ti, clf, ci)].append(f1m(ytr[iva], m.predict(B)))
        for treat in TREATS:
            for clf in CLFS:
                key, sc = max(((k, np.mean(v)) for k, v in scores.items() if k[0] == treat and k[2] == clf),
                              key=lambda kv: kv[1])
                tp, cp = TREATS[treat][key[1]], GRIDS[clf][key[3]]
                model = fit_model(Ftr, ytr, treat, tp, clf, cp)      # refit on all 3 training subjects
                pred = model.predict(F[te])
                cells[(treat, clf)] = dict(held=held, inner=float(sc), tp=tp, cp=cp, pred=pred.tolist(),
                                           y=y[te].tolist(), **metrics(y[te], pred))
        print(f"  [{tag}] held-out S{held} done in {time.time()-t0:.0f}s")
    return cells


MAIN = {}
for mode in NORM_MODES:
    Dm = DATA[mode]
    MAIN[mode] = defaultdict(list)
    for held, tr, te in loso_splits(Dm["subj"]):
        fold = cached(f"nested_{mode}_S{held}", lambda: nested_fold(Dm["F"], Dm["y"], Dm["subj"], held, tr, te, mode))
        for k, v in fold.items():
            MAIN[mode][k].append(v)
print("nested leave-one-subject-out grid complete (3 normalisations x 4 treatments x 4 classifiers x 4 held-out subjects)")

# ---- report: per held-out subject macro-F1 for every (treatment, classifier) cell
NS = {}   # nested-selected result per mode
for mode in NORM_MODES:
    print(f"\n=== Normalisation: {mode:<9} | held-out-subject macro-F1 (20 classes, chance ~0.05). Hyper-params tuned by inner LOSO ===")
    print(f"{'treatment':<8}{'clf':<5}" + "".join(f"{'S'+str(s):>8}" for s in SUBJECTS) + f"{'mean':>8}{'std':>7}{'  | acc mean':>12}")
    for treat in TREATS:
        for clf in CLFS:
            fo = MAIN[mode][(treat, clf)]
            f = [r["f1"] for r in fo]
            print(f"{treat:<8}{clf:<5}" + "".join(f"{v:8.3f}" for v in f) + f"{np.mean(f):8.3f}{np.std(f):7.3f}"
                  f"{np.mean([r['acc'] for r in fo]):12.3f}")
    # nested model selection: per outer fold pick the cell with the best INNER score
    sel = []
    for i, s in enumerate(SUBJECTS):
        best = max(((t, c) for t in TREATS for c in CLFS), key=lambda tc: MAIN[mode][tc][i]["inner"])
        r = dict(MAIN[mode][best][i]); r["cell"] = best; sel.append(r)
    NS[mode] = sel
    print(f"nested-selected (cell chosen by INNER score only): " + " | ".join(
        f"S{r['held']}: {r['cell'][1]}/{r['cell'][0]} F1={r['f1']:.3f}" for r in sel))
    print(f"   -> macro-F1 mean {np.mean([r['f1'] for r in sel]):.3f} +- {np.std([r['f1'] for r in sel]):.3f} "
          f"(min {min(r['f1'] for r in sel):.3f}, max {max(r['f1'] for r in sel):.3f}); "
          f"acc mean {np.mean([r['acc'] for r in sel]):.3f}; aggressive-vs-normal acc {np.mean([r['cat_acc'] for r in sel]):.3f}")

# headline = normalisation with best nested-selected mean macro-F1 (chosen by held-out scores, so descriptive only)
head = max(NORM_MODES, key=lambda m: np.mean([r["f1"] for r in NS[m]]))
print(f"\nHeadline mode for detailed reporting: '{head}' (note: picking it by held-out score is descriptive, not a tuned choice)")
print("Per-fold detail for the nested-selected config:")
print(f"{'held-out':<9}{'acc':>7}{'F1':>7}{'cat-acc':>9}{'recall normal':>15}{'recall aggr.':>14}   worst-3 classes (recall)")
for r in NS[head]:
    rec = np.array(r["recall"]); w = np.argsort(rec)[:3]
    print(f"S{r['held']:<8}{r['acc']:7.3f}{r['f1']:7.3f}{r['cat_acc']:9.3f}{r['recall_normal']:15.3f}{r['recall_aggr']:14.3f}   "
          + ", ".join(f"{CLASS_NAMES[i]} {rec[i]:.2f}" for i in w))
rec_all = np.array([r["recall"] for r in NS[head]])           # (4 folds, 20)
print("\nPer-class recall, mean over held-out subjects (best -> worst):")
order = np.argsort(-rec_all.mean(0))
print("  " + "; ".join(f"{CLASS_NAMES[i]} {rec_all[:, i].mean():.2f}" for i in order))
cm = sum(confusion_matrix(r["y"], r["pred"], labels=range(20)) for r in NS[head])
off = [(cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(20) for j in range(20) if i != j]
print("Most frequent confusions (count, true -> predicted):", [f"{n} {a}->{b}" for n, a, b in sorted(off, reverse=True)[:8]])

# ------------------------------------------------------------------ 2. why splitting matters: three protocols, same data
print("\n=== Protocol gap: SAME features, SAME fixed classifiers, only the split differs (norm=none) ===")
FIXED = {"SVM": {"C": 10}, "RF": {"max_depth": None}, "KNN": {"k": 5}, "LDA": {"shrinkage": "auto"}}
D = DATA["none"]; F, y, subj, rec = D["F"], D["y"], D["subj"], D["rec"]
rank = np.zeros(len(y), int); frac = np.zeros(len(y))
for r in np.unique(rec):
    ix = np.where(rec == r)[0]; rank[ix] = np.arange(len(ix)); frac[ix] = np.arange(len(ix)) / len(ix)


def run_fixed(tr, te):
    out = {}
    for clf, cp in FIXED.items():
        m = fit_model(F[tr], y[tr], "all", {}, clf, cp)
        out[clf] = f1m(y[te], m.predict(F[te]))
    return out


def _protocols():
    proto = {}
    # (A) random window K-fold: windows overlap 50% and come from the same recordings -> near-duplicates in train AND test
    fa = [run_fixed(tr, te) for tr, te in KFold(5, shuffle=True, random_state=SEED).split(F)]
    proto["A: random windows (LEAKY)"] = {c: [f[c] for f in fa] for c in FIXED}
    # (B) last run's protocol: first 70% of each recording's timeline = train, last 30% = test, 2-window gap
    trB = np.where(frac < 0.7)[0]; teB = np.where(frac >= 0.7)[0]
    trB = np.array([i for i in trB if rank[i] < (rank[rec == rec[i]].max() * 0.7 - 2)])
    proto["B: within-subject time split (PhysioNet-style)"] = {c: [v] for c, v in run_fixed(trB, teB).items()}
    # (C) leave-one-subject-out
    fc = [run_fixed(tr, te) for _, tr, te in loso_splits(subj)]
    proto["C: leave-one-subject-out (HONEST)"] = {c: [f[c] for f in fc] for c in FIXED}
    return proto


proto = cached("protocols", _protocols)
print(f"{'protocol':<48}" + "".join(f"{c:>10}" for c in FIXED))
for k, v in proto.items():
    print(f"{k:<48}" + "".join(f"{np.mean(v[c]):10.3f}" for c in FIXED))
print("(A and B pool all 4 subjects in both train and test, so they measure 'recognise this recording/subject', not generalisation.)")

# ------------------------------------------------------------------ 3. feature groups, evaluated under LOSO (fixed hyper-params, no tuning)
print("\n=== Feature-group ablation under LOSO (macro-F1 mean over held-out subjects; fixed hyper-params) ===")
subsets = {"all": np.ones(len(NAMES), bool)}
for g in dict.fromkeys(GROUPS):
    subsets[f"only {g}"] = GROUPS == g
for g in dict.fromkeys(GROUPS):
    subsets[f"all minus {g}"] = GROUPS != g
ABL = {}


def _ablate(mode, sname):
    Dm, mask = DATA[mode], subsets[sname]
    res = {c: [] for c in FIXED}
    for _, tr, te in loso_splits(Dm["subj"]):
        for clf, cp in FIXED.items():
            m = fit_model(Dm["F"][tr][:, mask], Dm["y"][tr], "all", {}, clf, cp)
            res[clf].append(f1m(Dm["y"][te], m.predict(Dm["F"][te][:, mask])))
    return res


for mode in NORM_MODES:
    Dm = DATA[mode]
    print(f"\n normalisation = {mode}")
    print(f" {'feature subset':<26}{'#':>4}" + "".join(f"{c:>8}" for c in FIXED) + f"{'  S1..S4 (mean over clf)':>30}")
    for sname, mask in subsets.items():
        res = cached(f"abl_{mode}_{sname.replace(' ', '_')}", lambda: _ablate(mode, sname))
        ABL[(mode, sname)] = res
        per_fold = np.mean([res[c] for c in FIXED], axis=0)
        print(f" {sname:<26}{mask.sum():>4}" + "".join(f"{np.mean(res[c]):8.3f}" for c in FIXED)
              + "   " + " ".join(f"{v:.2f}" for v in per_fold))

# ------------------------------------------------------------------ 4. does a feature encode the CLASS or the SUBJECT?
print("\n=== Mutual information per feature group: with the CLASS vs with the SUBJECT (descriptive, all windows) ===")
print("High subject-MI + low class-MI = a 'fingerprint' of who was recorded, not of what they did.")
MI_TAB = {}
for mode in NORM_MODES:
    Dm = DATA[mode]; A = StandardScaler().fit_transform(Dm["F"])
    mc, ms = cached(f"mi_{mode}", lambda: (mutual_info_classif(A, Dm["y"], random_state=SEED, n_jobs=1),
                                           mutual_info_classif(A, Dm["subj"], random_state=SEED, n_jobs=1)))
    MI_TAB[mode] = (mc, ms)
    print(f" normalisation = {mode}")
    print(f"   {'group':<16}{'MI(class)':>11}{'MI(subject)':>13}{'ratio class/subject':>21}")
    for g in dict.fromkeys(GROUPS):
        k = GROUPS == g
        print(f"   {g:<16}{mc[k].mean():11.3f}{ms[k].mean():13.3f}{mc[k].mean() / ms[k].mean():21.2f}")
    top = np.argsort(-ms)[:5]
    print("   top-5 features by MI with SUBJECT:", ", ".join(f"{NAMES[i]} ({ms[i]:.2f})" for i in top))
    top = np.argsort(-mc)[:5]
    print("   top-5 features by MI with CLASS  :", ", ".join(f"{NAMES[i]} ({mc[i]:.2f})" for i in top))

# ------------------------------------------------------------------ 5. PCA visual: fit on S1-S3 only, project S4
print("\n=== PCA (fit on S1-S3 only; S4 projected) ===")
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for row, mode in enumerate(["none", "subject"]):
    Dm = DATA[mode]; tr = Dm["subj"] != 4
    sc = StandardScaler().fit(Dm["F"][tr])
    from sklearn.decomposition import PCA
    p = PCA().fit(sc.transform(Dm["F"][tr])); Z = p.transform(sc.transform(Dm["F"]))
    cum = np.cumsum(p.explained_variance_ratio_)
    print(f" {mode:<8}: PC1 {p.explained_variance_ratio_[0]:.1%}, PC2 {p.explained_variance_ratio_[1]:.1%}; "
          f"components for 95%: {int(np.argmax(cum >= .95)) + 1}, 99%: {int(np.argmax(cum >= .99)) + 1} (of {len(NAMES)})")
    axs[row, 0].plot(range(1, 41), cum[:40], "k.-"); axs[row, 0].axhline(.95, color="r", ls="--", lw=.8)
    axs[row, 0].set(title=f"cumulative variance (train subjects), norm={mode}", xlabel="components")
    for s, c in zip(SUBJECTS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]):
        k = Dm["subj"] == s
        axs[row, 1].scatter(Z[k, 0], Z[k, 1], s=5, alpha=.4, color=c, label=f"S{s}" + (" (held out)" if s == 4 else ""))
    axs[row, 1].legend(markerscale=3); axs[row, 1].set(title="PC1-PC2 coloured by SUBJECT", xlabel="PC1", ylabel="PC2")
    k = Dm["y"] >= 10
    axs[row, 2].scatter(Z[~k, 0], Z[~k, 1], s=5, alpha=.4, color="#4c72b0", label="normal")
    axs[row, 2].scatter(Z[k, 0], Z[k, 1], s=5, alpha=.4, color="#c44e52", label="aggressive")
    axs[row, 2].legend(markerscale=3); axs[row, 2].set(title="PC1-PC2 coloured by CATEGORY", xlabel="PC1")
fig.tight_layout(); fig.savefig(HERE / "02_pca_subjects.png", dpi=120); plt.close(fig)

# ------------------------------------------------------------------ 6. plots
subs_lbl = [f"S{s}" for s in SUBJECTS]
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
for ax, mode in zip(axs, NORM_MODES):
    for j, clf in enumerate(CLFS):
        f = [r["f1"] for r in MAIN[mode][("all", clf)]]
        ax.bar(np.arange(4) + (j - 1.5) * 0.2, f, 0.2, color=CLF_COL[clf], label=clf)
        ax.plot(3.7, np.mean(f), "k_", ms=0)
    ax.axhline(0.05, color="grey", ls=":"); ax.set_xticks(range(4)); ax.set_xticklabels([f"held out {s}" for s in subs_lbl])
    ax.set_title(f"normalisation: {mode}  (treatment: all features)"); ax.set_ylim(0, .8)
axs[0].set_ylabel("macro-F1 on held-out subject"); axs[0].legend()
fig.tight_layout(); fig.savefig(HERE / "02_fold_scores.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(17, 4.4), sharey=True)
for ax, mode in zip(axs, NORM_MODES):
    for j, clf in enumerate(CLFS):
        m = [np.mean([r["f1"] for r in MAIN[mode][(t, clf)]]) for t in TREATS]
        e = [np.std([r["f1"] for r in MAIN[mode][(t, clf)]]) for t in TREATS]
        ax.bar(np.arange(len(TREATS)) + (j - 1.5) * 0.2, m, 0.2, yerr=e, capsize=2, color=CLF_COL[clf], label=clf)
    ax.set_xticks(range(len(TREATS))); ax.set_xticklabels(list(TREATS)); ax.set_title(f"norm: {mode} (error bar = std over 4 held-out subjects)")
axs[0].set_ylabel("mean held-out macro-F1"); axs[0].legend(); fig.tight_layout(); fig.savefig(HERE / "02_treatments.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
for j, c in enumerate(FIXED):
    ax.bar(np.arange(3) + (j - 1.5) * 0.2, [np.mean(v[c]) for v in proto.values()], 0.2, color=CLF_COL[c], label=c)
ax.set_xticks(range(3)); ax.set_xticklabels(["A: random windows\n(leaky)", "B: within-subject\ntime split", "C: leave-one-\nsubject-out"])
ax.set_ylabel("macro-F1"); ax.legend(); ax.set_title("Same data and features, different split -> very different 'accuracy'")
fig.tight_layout(); fig.savefig(HERE / "02_protocol_gap.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
for ax, mode in zip(axs, NORM_MODES):
    vals = [np.mean([np.mean(ABL[(mode, s)][c]) for c in FIXED]) for s in subsets]
    cols = ["#444"] + ["#4c72b0"] * 4 + ["#dd8452"] * 4
    ax.barh(list(subsets)[::-1], vals[::-1], color=cols[::-1]); ax.set_title(f"norm: {mode}"); ax.set_xlabel("macro-F1 (mean over 4 clf, 4 subjects)")
fig.tight_layout(); fig.savefig(HERE / "02_feature_groups.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
gl = list(dict.fromkeys(GROUPS))
for ax, mode in zip(axs, NORM_MODES):
    mc, ms = MI_TAB[mode]
    ax.bar(np.arange(4) - .2, [mc[GROUPS == g].mean() for g in gl], .4, label="MI with class", color="#4c72b0")
    ax.bar(np.arange(4) + .2, [ms[GROUPS == g].mean() for g in gl], .4, label="MI with subject", color="#c44e52")
    ax.set_xticks(range(4)); ax.set_xticklabels(gl, rotation=15); ax.set_title(f"norm: {mode}")
axs[0].legend(); axs[0].set_ylabel("mean mutual information"); fig.tight_layout(); fig.savefig(HERE / "02_subject_vs_class_mi.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 8))
cmn = cm / cm.sum(1, keepdims=True)
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.set_xticks(range(20)); ax.set_yticks(range(20))
ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=8); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.axhline(9.5, color="k", lw=.8); ax.axvline(9.5, color="k", lw=.8)
ax.set_title(f"Nested-selected classical model, norm={head}: confusion summed over 4 held-out subjects (row-normalised)")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.colorbar(im, fraction=.04); fig.tight_layout()
fig.savefig(HERE / "02_confusion_headline.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(rec_all.T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_yticks(range(20)); ax.set_yticklabels(CLASS_NAMES, fontsize=8); ax.set_xticks(range(4)); ax.set_xticklabels([f"held out S{s}" for s in SUBJECTS])
for i in range(20):
    for j in range(4):
        ax.text(j, i, f"{rec_all[j, i]:.2f}", ha="center", va="center", color="w" if rec_all[j, i] < .6 else "k", fontsize=7)
ax.set_title(f"Per-class recall by held-out subject (norm={head})"); fig.colorbar(im, fraction=.05); fig.tight_layout()
fig.savefig(HERE / "02_recall_heatmap.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ save (03 reads this for the fair comparison)
out = dict(norm_modes=NORM_MODES, headline=head, class_names=CLASS_NAMES, subjects=SUBJECTS,
           cells={m: {f"{t}|{c}": [{k: v for k, v in r.items() if k != "pred"} for r in MAIN[m][(t, c)]] for (t, c) in MAIN[m]} for m in MAIN},
           nested_selected={m: [{k: v for k, v in r.items() if k not in ("pred", "y")} for r in NS[m]] for m in NS},
           protocols={k: {c: v for c, v in d_.items()} for k, d_ in proto.items()},
           ablation={f"{m}|{s}": v for (m, s), v in ABL.items()})
json.dump(out, open(HERE / "02_results.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("\nsaved 02_results.json + PNGs")
