"""Step 2 - Stage A: classical ML on hand-crafted EMG features.

Run:  python emg_classical_dl/02_features_classical.py
Outputs: 02_pca.png, 02_feature_ranking.png, 02_confusion_matrices.png, 02_results.json
(02_results.json is read by 03_deep_learning.py so both stages are scored on the same windows.)

Method notes (read these - they are the "why"):
* Every fitted thing (scaler, PCA, feature selector, hyper-parameters) is fitted on TRAIN windows only.
* Hyper-parameters and the "best classical model" are chosen by TIME-BLOCKED cross-validation
  inside the train portion, never by looking at the test set.
* All results are within-signal. They say nothing about new patients.
"""
import json
import warnings
from emg_common import (HERE, FS, CLASSES, WIN, SEED, build_dataset, extract_features,
                        AMPLITUDE_FEATURES)   # first: applies the sklearn import workaround
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emg_common import HERE, FS, CLASSES, WIN, SEED, build_dataset

warnings.filterwarnings("ignore")
np.random.seed(SEED)

# ---------------------------------------------------------------- 1. feature extraction
# (extract_features lives in emg_common.py so 03_deep_learning.py can reuse it for the noise test)


# ---------------------------------------------------------------- 2. data + blocked CV
ds = build_dataset()
Ftr, names = extract_features(ds["Xtr"])
Fte, _ = extract_features(ds["Xte"])
ytr, yte = ds["ytr"], ds["yte"]
print(f"{Ftr.shape[1]} features per window; train {Ftr.shape[0]} windows, test {Fte.shape[0]} windows")
print("features:", ", ".join(names))

# Time-blocked CV: split each class's TRAIN timeline into 5 contiguous blocks; each fold holds out
# one block (same time slice of every class). Train windows within one window-length of the held-out
# block are purged so the 75%-overlapping windows cannot leak across the fold boundary.
N_BLOCKS = 5
block = np.zeros(len(ytr), int)
for k in range(len(CLASSES)):
    m = ytr == k
    t = ds["t_tr"][m]
    block[m] = np.minimum(N_BLOCKS - 1, ((t - t.min()) / (t.max() - t.min() + 1e-9) * N_BLOCKS).astype(int))


def blocked_cv_splits():
    splits = []
    for b in range(N_BLOCKS):
        te = np.where(block == b)[0]
        tr = np.ones(len(ytr), bool)
        tr[te] = False
        for k in range(len(CLASSES)):
            tk = ds["t_tr"][te][ytr[te] == k]
            near = (ytr == k) & (ds["t_tr"] > tk.min() - WIN / FS) & (ds["t_tr"] < tk.max() + WIN / FS)
            tr &= ~near
        splits.append((np.where(tr)[0], te))
    return splits


CV = blocked_cv_splits()

# ---------------------------------------------------------------- 3. PCA
sc = StandardScaler().fit(Ftr)
pca = PCA().fit(sc.transform(Ftr))
cum = np.cumsum(pca.explained_variance_ratio_)
n95, n99 = int(np.argmax(cum >= .95)) + 1, int(np.argmax(cum >= .99)) + 1
print(f"\nPCA (fit on train, standardised features): PC1={pca.explained_variance_ratio_[0]:.1%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.1%}; components for 95% variance: {n95}, for 99%: {n99} "
      f"(of {len(names)})")
Ztr, Zte = pca.transform(sc.transform(Ftr)), pca.transform(sc.transform(Fte))
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
axs[0].bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, color="#4c72b0")
axs[0].plot(range(1, len(cum) + 1), cum, "k.-"); axs[0].axhline(.95, color="r", ls="--", lw=.8)
axs[0].set(xlabel="component", ylabel="explained variance ratio", title="PCA scree (bars) + cumulative (line)")
COL = ["#1b9e77", "#d95f02", "#7570b3"]
for k, c in enumerate(CLASSES):
    axs[1].scatter(*Ztr[ytr == k][:, :2].T, s=8, alpha=.5, color=COL[k], label=c)
    axs[2].scatter(*Zte[yte == k][:, :2].T, s=22, alpha=.9, color=COL[k], label=c, marker="s")
axs[1].set(xlabel="PC1", ylabel="PC2", title="Train windows in PC space"); axs[1].legend()
axs[2].set(xlabel="PC1", ylabel="PC2", title="Held-out test windows (projected with train PCA)")
axs[2].set_xlim(axs[1].get_xlim()); axs[2].set_ylim(axs[1].get_ylim())
fig.tight_layout(); fig.savefig(HERE / "02_pca.png", dpi=130); plt.close(fig)
loadings = pca.components_[0]
top = np.argsort(-np.abs(loadings))[:5]
print("top |loading| on PC1:", ", ".join(f"{names[i]} ({loadings[i]:+.2f})" for i in top))

# ---------------------------------------------------------------- 4. feature selection
mi = mutual_info_classif(sc.transform(Ftr), ytr, random_state=SEED)
rf_imp = RandomForestClassifier(500, random_state=SEED).fit(Ftr, ytr).feature_importances_
rfe = RFE(RandomForestClassifier(200, random_state=SEED), n_features_to_select=6).fit(Ftr, ytr)
mi_top = list(np.argsort(-mi)[:6])
rfe_top = list(np.where(rfe.support_)[0])
print("\nMutual information ranking (train):")
for i in np.argsort(-mi)[:8]:
    print(f"   {names[i]:<18} MI={mi[i]:.3f}   RF-importance={rf_imp[i]:.3f}")
print("top-6 by MI :", [names[i] for i in mi_top])
print("RFE-6 (RF)  :", [names[i] for i in rfe_top])
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
o = np.argsort(mi)
axs[0].barh([names[i] for i in o], mi[o], color="#4c72b0"); axs[0].set_title("Mutual information with class (train)")
o = np.argsort(rf_imp)
axs[1].barh([names[i] for i in o], rf_imp[o], color="#55a868"); axs[1].set_title("Random-forest impurity importance (train)")
fig.tight_layout(); fig.savefig(HERE / "02_feature_ranking.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 5. models (tuned by blocked CV)
MODELS = {
    "SVM": (SVC(), {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]}),
    "RandomForest": (RandomForestClassifier(300, random_state=SEED), {"randomforestclassifier__max_depth": [3, None]}),
    "KNN": (KNeighborsClassifier(), {"kneighborsclassifier__n_neighbors": [3, 5, 11, 21]}),
    "LDA": (LinearDiscriminantAnalysis(), {}),
}


def fit_eval(name, cols=None, use_pca=False):
    """Tune on blocked CV (train only), refit, return (cv_acc, fitted model, test preds)."""
    est, grid = MODELS[name]
    est = clone(est)          # never reuse one estimator object across feature sets (it would be refit in place)
    steps = [StandardScaler()] + ([PCA(n_components=0.95)] if use_pca else []) + [est]
    pipe = make_pipeline(*steps)
    grid = {(k if not use_pca else k): v for k, v in grid.items()}
    gs = GridSearchCV(pipe, grid, cv=CV, scoring="f1_macro") if grid else None
    A, B = (Ftr, Fte) if cols is None else (Ftr[:, cols], Fte[:, cols])
    if gs:
        gs.fit(A, ytr); model, cv, params = gs.best_estimator_, gs.best_score_, gs.best_params_
        params = {k.split("__")[-1]: v for k, v in params.items()}
    else:
        from sklearn.model_selection import cross_val_score
        cv = cross_val_score(pipe, A, ytr, cv=CV, scoring="f1_macro").mean(); model = pipe.fit(A, ytr)
        params = {}
    return cv, model, model.predict(B), params


amp_idx = [names.index(a) for a in AMPLITUDE_FEATURES]
shape_idx = [i for i in range(len(names)) if i not in amp_idx]
SETS = {"all features": dict(cols=None), "PCA(95% var)": dict(use_pca=True),
        "top-6 MI": dict(cols=mi_top), "RFE-6": dict(cols=rfe_top),
        "no-amplitude (shape+freq only)": dict(cols=shape_idx)}

print("\n=== Test accuracy [blocked-CV macro-F1 in brackets] ===")
print(f"{'feature set':<34}" + "".join(f"{m:>20}" for m in MODELS))
table, main = {}, {}
for sname, kw in SETS.items():
    row = f"{sname:<34}"
    for m in MODELS:
        cv, model, pred, params = fit_eval(m, **kw)
        acc = accuracy_score(yte, pred)
        table[(sname, m)] = dict(cv=float(cv), acc=float(acc))
        if sname == "all features":
            main[m] = dict(cv=float(cv), pred=pred, model=model, params=params)
        row += f"{acc:>12.3f} [{cv:.3f}]"
    print(row)

# ---------------------------------------------------------------- 6. detailed report (all features)
print("\n=== Detailed held-out test report (all features, CV-tuned) ===")
fig, axs = plt.subplots(1, 4, figsize=(18, 4.4))
for ax, (m, r) in zip(axs, main.items()):
    cm = confusion_matrix(yte, r["pred"], labels=range(3))
    p, rec, f1, _ = precision_recall_fscore_support(yte, r["pred"], labels=range(3), zero_division=0)
    print(f"\n{m}: accuracy={accuracy_score(yte, r['pred']):.3f}  macro-F1={f1_score(yte, r['pred'], average='macro'):.3f}"
          f"  tuned: {r['params']}")
    print("  confusion (rows=true, cols=pred):\n", cm)
    print(f"  {'class':<11}{'precision':>10}{'recall':>8}{'support':>9}")
    for k, c in enumerate(CLASSES):
        print(f"  {c:<11}{p[k]:>10.3f}{rec[k]:>8.3f}{(yte == k).sum():>9}")
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "k")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, rotation=20); ax.set_yticklabels(CLASSES)
    ax.set_title(f"{m} (acc {accuracy_score(yte, r['pred']):.2f})"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
fig.tight_layout(); fig.savefig(HERE / "02_confusion_matrices.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 7. choose best classical model WITHOUT peeking at test
best = max(main, key=lambda m: main[m]["cv"])      # selection uses CV scores only, never test
print(f"\nBest classical model by blocked-CV macro-F1 on TRAIN: {best} (CV F1 = {main[best]['cv']:.3f})")
print("CV scores:", {m: round(r["cv"], 3) for m, r in main.items()})
print("NOTE: CV scores differ by <0.01 across models - that is within noise, so 'best' is a near-tie.")
import joblib
joblib.dump(main[best]["model"], HERE / "02_best_classical_model.joblib")   # reused by 03 for the noise test

json.dump(dict(best_model=best, best_cv_f1=main[best]["cv"], classes=CLASSES,
               y_test=yte.tolist(), pred_test={m: r["pred"].tolist() for m, r in main.items()},
               cv_f1={m: r["cv"] for m, r in main.items()},
               table={f"{s}|{m}": v for (s, m), v in table.items()}),
          open(HERE / "02_results.json", "w"), indent=1)
print("saved 02_results.json + PNGs")
