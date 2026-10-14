"""Automatic feedback on the models students build in Session 3.

``report_card`` cross-validates a model on the training data and comments on the things
an experienced reviewer would check: does it beat a dumb baseline, is it overfitting,
how noisy is the score, are the features scaled, is the metric sensible for the classes.

``HoldoutChallenge`` keeps a test set locked away and counts how many times you look at it.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, f1_score,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .data import load_dataset
from .featurize import generate_features

# Models that compare distances or coefficient sizes, so feature scale matters
_SCALE_SENSITIVE = ("KNeighbors", "SVR", "SVC", "Ridge", "Lasso", "ElasticNet", "LogisticRegression",
                    "MLP", "GaussianProcess", "KernelRidge", "SGD")
_SCALERS = ("StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "QuantileTransformer",
            "PowerTransformer", "Normalizer")
_SELECTORS = ("SelectKBest", "SelectPercentile", "SelectFromModel", "RFE", "VarianceThreshold", "PCA")

OK, WARN, NOTE = "✅", "⚠️ ", "💡"


def _steps(model):
    if isinstance(model, Pipeline):
        out = []
        for _, step in model.steps:
            out += _steps(step)
        return out
    inner = getattr(model, "estimator", None)       # GridSearchCV, RandomizedSearchCV, ...
    return [model] + (_steps(inner) if inner is not None else [])


def _names(model):
    return [type(s).__name__ for s in _steps(model)]


def report_card(model, X, y, cv=5, groups=None, units="", random_state=42, verbose=True):
    """Cross-validate ``model`` on (X, y) and print reviewer-style feedback.

    Use only training data here: the test set stays locked until the very end.
    Pass ``groups`` (e.g. formulas) to keep related rows in the same fold.
    Returns a dict with the scores.
    """
    clf = is_classifier(model)
    y = np.asarray(y)
    if groups is not None:
        splitter, split_name = GroupKFold(cv), f"{cv}-fold group CV"
    elif clf:
        splitter, split_name = StratifiedKFold(cv, shuffle=True, random_state=random_state), f"{cv}-fold stratified CV"
    else:
        splitter, split_name = KFold(cv, shuffle=True, random_state=random_state), f"{cv}-fold CV"

    if clf:
        scoring = {"bacc": "balanced_accuracy", "acc": "accuracy"}
        base = DummyClassifier(strategy="most_frequent")
    else:
        scoring = {"mae": "neg_mean_absolute_error", "r2": "r2"}
        base = DummyRegressor()
    kw = dict(cv=splitter, groups=groups, scoring=scoring)
    res = cross_validate(clone(model), X, y, return_train_score=True, n_jobs=-1, **kw)
    bres = cross_validate(base, X, y, **kw)

    names = _names(model)
    lines = []
    out = {"model": " → ".join(names), "split": split_name}
    if clf:
        cv_s, tr_s, sd = res["test_bacc"].mean(), res["train_bacc"].mean(), res["test_bacc"].std()
        b_s = bres["test_bacc"].mean()
        acc = res["test_acc"].mean()
        out.update(cv_balanced_accuracy=cv_s, sd=sd, train_balanced_accuracy=tr_s, accuracy=acc,
                   baseline_balanced_accuracy=b_s)
        head = f"Balanced accuracy ({split_name}): {cv_s:.3f} ± {sd:.3f}   |   train: {tr_s:.3f}"
        lines.append((OK if cv_s > b_s + 2 * sd else WARN) +
                     f" Baseline (always predict the most common class) scores {b_s:.3f}. "
                     + ("You beat it clearly." if cv_s > b_s + 2 * sd else "You are not clearly better than guessing."))
        minority = min(np.mean(y == c) for c in np.unique(y))
        if minority < 0.2:
            lines.append(f"{NOTE} The rarest class is only {minority:.0%} of the data. Plain accuracy is {acc:.3f} here, "
                         f"but always guessing the majority already gets {1 - minority:.3f}. Trust balanced accuracy, F1 or "
                         "recall on the rare class instead.")
        gap = tr_s - cv_s
        if gap > 0.1:
            lines.append(f"{WARN} Train score is {gap:.2f} higher than CV: the model is memorizing. Try more regularization "
                         "(fewer/shallower trees, larger min_samples_leaf, smaller C).")
        else:
            lines.append(f"{OK} Train and CV scores are close (gap {gap:.2f}): little sign of overfitting.")
    else:
        cv_s, tr_s, sd = -res["test_mae"].mean(), -res["train_mae"].mean(), res["test_mae"].std()
        b_s = -bres["test_mae"].mean()
        r2 = res["test_r2"].mean()
        u = f" {units}" if units else ""
        out.update(cv_mae=cv_s, sd=sd, train_mae=tr_s, cv_r2=r2, baseline_mae=b_s)
        head = f"MAE ({split_name}): {cv_s:.3g} ± {sd:.2g}{u}   |   train MAE: {tr_s:.3g}{u}   |   CV R²: {r2:.3f}"
        gain = 1 - cv_s / b_s
        if cv_s < b_s - 2 * sd:
            lines.append(f"{OK} Beats predicting the mean ({b_s:.3g}{u}) by {gain:.0%}.")
        else:
            lines.append(f"{WARN} Not clearly better than predicting the mean ({b_s:.3g}{u}). Check the features and the target.")
        ratio = tr_s / cv_s if cv_s > 0 else 1
        if ratio < 0.5:
            forest = any(n.startswith(("RandomForest", "ExtraTrees")) for n in names)
            lines.append(f"{WARN if not forest else NOTE} Train error is {ratio:.0%} of CV error: the model fits the training "
                         "data much better than new data. " +
                         ("That is normal for random forests (each tree memorizes); judge them on CV error only."
                          if forest else "Try a validation curve on its main complexity knob."))
        else:
            lines.append(f"{OK} Train and CV errors are similar (train/CV = {ratio:.2f}): not overfitting badly.")

    rel = sd / abs(cv_s) if cv_s else 0
    lines.append(f"{NOTE} Fold-to-fold spread is ±{sd:.2g} ({rel:.0%} of the score). Differences between models smaller "
                 "than this are probably noise.")
    if any(n.startswith(_SCALE_SENSITIVE) for n in names) and not any(n in _SCALERS for n in names):
        lines.append(f"{WARN} {[n for n in names if n.startswith(_SCALE_SENSITIVE)][0]} depends on feature scale, but there "
                     "is no scaler. Use make_pipeline(StandardScaler(), model).")
    if any(n in _SCALERS + _SELECTORS for n in names):
        lines.append(f"{OK} Preprocessing is inside the pipeline, so it is refit within every fold (no leakage).")
    if groups is None and isinstance(X, pd.DataFrame):
        conditions = [c for c in X.columns if c in ("T", "T_K", "temperature", "Temperature")]
        twins = X.drop(columns=conditions).duplicated().mean()
        if twins > 0.05:
            what = "the same composition at another temperature" if conditions else "an identical twin"
            lines.append(f"{WARN} {twins:.0%} of rows share their composition features with another row ({what}). "
                         "Random folds put twins on both sides of the split; use groups= (e.g. the formula).")
    if verbose:
        print(head)
        print("\n".join(lines))
    out["feedback"] = lines
    return out


THERMO_TASKS = {
    "thermoelectric_pf": "log10 power factor near room temperature (275-325 K)",
    "thermoelectric_ptype": "p-type (Seebeck > 0) vs n-type near room temperature",
    "thermoelectric_zt": "zT at any temperature (composition + T)",
}


def _thermo_table(name):
    """One row per composition (room-temperature tasks) or per measurement (zT), from the sysTEm dataset."""
    te = load_dataset("thermoelectric")
    if name == "thermoelectric_zt":
        df = te[te["zT"] > 0].assign(target=lambda d: d["zT"])[["formula", "T_K", "target", "source"]]
        return df.reset_index(drop=True), "regression", f"thermoelectric zT, {THERMO_TASKS[name]}", ""
    rt = te[(te["T_K"] - 300).abs() <= 25]
    if name == "thermoelectric_pf":
        rt = rt[rt["PF_uW_cmK2"] > 0].assign(target=lambda d: np.log10(d["PF_uW_cmK2"]))
        task, units = "regression", "log10 µW/(cm·K²)"
    else:
        rt = rt[rt["seebeck_uV_K"].notna()].assign(target=lambda d: d["seebeck_uV_K"])
        task, units = "classification", ""
    df = rt.groupby("formula").agg(target=("target", "mean"), source=("source", "first")).reset_index()
    if task == "classification":
        df["target"] = (df["target"] > 0).astype(int)
    return df, task, f"thermoelectric {THERMO_TASKS[name]}", units


class HoldoutChallenge:
    """A dataset with a locked test set and a counter of how often you've looked at it.

    >>> ch = HoldoutChallenge("shear_modulus")
    >>> model.fit(ch.X_train, ch.y_train)        # tune with CV on the training set only
    >>> ch.submit(model, "my random forest")      # look at the test set once, at the end

    ``task="classification"`` with ``threshold=t`` turns the target into the label ``target > t``.

    Thermoelectric tasks from the sysTEm dataset (Tang et al., IMMI 2026): ``"thermoelectric_pf"``,
    ``"thermoelectric_ptype"`` and ``"thermoelectric_zt"``. These also provide ``ch.papers_train``
    (source paper of each row) for grouped cross-validation.
    """

    def __init__(self, name, elem_prop="oliynyk", task="regression", threshold=None, test_size=0.2,
                 random_state=2026, units=None):
        self.papers_train = None
        if name in THERMO_TASKS:
            df, task, what, default_units = _thermo_table(name)
            X, y, f, _ = generate_features(df, elem_prop=elem_prop, drop_duplicates=False, verbose=False)
            if "T_K" in df:
                X["T_K"] = df["T_K"].values
            papers = df["source"].reset_index(drop=True)
            if name == "thermoelectric_zt":
                # the same composition is measured at many temperatures: keep all of its rows on one side
                tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=random_state).split(X, y, f))
            else:
                tr, te = train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_state,
                                          stratify=y if task == "classification" else None)
            self.X_train, self._X_test = X.iloc[tr], X.iloc[te]
            self.y_train, self._y_test = y.iloc[tr], y.iloc[te]
            self.formula_train, self._formula_test = f.iloc[tr], f.iloc[te]
            self.papers_train = papers.iloc[tr].values
            unit = "rows" if name == "thermoelectric_zt" else "compositions"
        else:
            df = load_dataset(name)
            X, y, f, _ = generate_features(df, elem_prop=elem_prop, verbose=False)
            if task == "classification":
                if threshold is None:
                    raise ValueError("Give a threshold for classification, e.g. threshold=0 for band gap > 0")
                y = (y > threshold).astype(int)
            strat = y if task == "classification" else None
            (self.X_train, self._X_test, self.y_train, self._y_test,
             self.formula_train, self._formula_test) = train_test_split(X, y, f, test_size=test_size,
                                                                        random_state=random_state, stratify=strat)
            what = f"label = {name} > {threshold}" if task == "classification" else name
            default_units = {"bulk_modulus": "GPa", "shear_modulus": "GPa", "bandgap": "eV"}.get(name, "")
            unit = "compounds"
        self.groups_train = self.formula_train.values
        self.name, self.task, self.threshold = name, task, threshold
        self.units = units if units is not None else default_units
        self.submissions = []
        print(f"Challenge: {what}. {len(self.X_train)} training {unit}, {self.X_train.shape[1]} features. "
              f"{len(self._y_test)} test {unit} are locked away.")
        if self.papers_train is not None:
            print(f"{NOTE} Rows come from {len(set(self.papers_train))} papers, often as doping series. Try "
                  "report_card(..., groups=ch.papers_train) to ask: does it work for a research group it hasn't seen?")
        if name == "thermoelectric_zt":
            print(f"{NOTE} Each composition appears at several temperatures (feature T_K). The test set holds whole "
                  "compositions; use report_card(..., groups=ch.groups_train) to cross-validate the same way.")

    def submit(self, model, name=None):
        """Score a fitted model on the locked test set (fits it on the training set if it isn't fitted)."""
        try:
            check_is_fitted(model)
        except NotFittedError:
            model = clone(model).fit(self.X_train, self.y_train)
        p = model.predict(self._X_test)
        n = len(self.submissions) + 1
        name = name or f"submission {n}"
        if self.task == "classification":
            row = {"name": name, "balanced acc.": balanced_accuracy_score(self._y_test, p),
                   "accuracy": accuracy_score(self._y_test, p), "F1": f1_score(self._y_test, p, zero_division=0)}
            msg = f"Test balanced accuracy: {row['balanced acc.']:.3f}  (accuracy {row['accuracy']:.3f})"
        else:
            row = {"name": name, "MAE": mean_absolute_error(self._y_test, p), "R²": r2_score(self._y_test, p)}
            base = mean_absolute_error(self._y_test, np.full(len(p), np.mean(self.y_train)))
            u = f" {self.units}" if self.units else ""
            msg = f"Test MAE: {row['MAE']:.3g}{u}   R²: {row['R²']:.3f}   (predicting the mean: {base:.3g}{u})"
        self.submissions.append(row)
        print(f"[look #{n} at the test set] {msg}")
        if n == 2:
            print(f"{WARN} Second look at the test set. If you change the model because of this number, the test set "
                  "is turning into a validation set and stops being an honest estimate.")
        elif n > 2:
            print(f"{WARN} {n} looks. Reporting the best of {n} test scores is p-hacking: some of the gain is luck. "
                  "Choose with cross-validation on the training set, and quote your *first* test score.")

    def leaderboard(self):
        """All your submissions so far."""
        return pd.DataFrame(self.submissions).set_index("name")
