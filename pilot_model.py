"""
pilot_model.py - Feasibility pilot for the sequential, capacity-aware
early-warning system on OULAD.

PURPOSE (feasibility evidence only, NOT final results):
  (1) the features predict the outcome         -> PR-AUC vs At-risk rate
  (2) calibration works                        -> Brier / ECE before & after Platt
  (3) InterpretML explanations run in time     -> ShapKernel timing (full set and
                                                  staff-action scope)
  (4) Fairlearn group metrics can be computed  -> MetricFrame on part C
  (5) LAC conformal prediction sets work        -> thresholds on part B, sets on part C

Data rules:
  * Only `fit` and `pcal` rows are used. `ccal` and `test` rows are only
    counted (to prove they are dropped) and never used or summarised.
  * HGB is trained on `fit` with default hyperparameters (no tuning).
  * `pcal` is split into three student-disjoint parts:
      part A -> fits Platt scaling only
      part B -> computes the LAC thresholds only
      part C -> all evaluation
  * Nothing is fitted on part C. No routing and no capacity values are applied.

All outputs go to pilot/.
"""

import os
import sys
import math
import time
import platform

import numpy as np
import pandas as pd
import pyarrow
import sklearn
import matplotlib
matplotlib.use("Agg")  # file output only, no GUI window
import matplotlib.pyplot as plt
import interpret
import fairlearn
import shap  # used by interpret ShapKernel; TreeExplainer used for the timing-only check

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             recall_score, precision_score)
from interpret.blackbox import ShapKernel
from fairlearn.metrics import MetricFrame, selection_rate, count

# ----------------------------------------------------------------------------
# 0. Setup
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)          # belt-and-braces for any library using the global RNG
PREPARED_DIR = "prepared"
OUT_DIR = "pilot"
CHECKPOINTS = [14, 28, 42]
N_TEST_FINAL = 9_353                  # size of the final test set (for the SHAP estimates)
N_BACKGROUND = 100
N_EXPLAIN = 20
CLIP_EPS = 1e-6
N_BINS = 10
TOP_FRACTION = 0.10                   # Fairlearn tool test: flag top 10 % of part C
SHAP_SCOPE_FRACTION = 0.15            # stand-in for staff-action cases (no routing in pilot)
N_TEST_SCOPE = int(round(SHAP_SCOPE_FRACTION * N_TEST_FINAL))   # 1,403
ALPHA = 0.10                          # LAC miscoverage level (fixed, not tuned)
SMALL_GROUP = 100
AGE_MERGE = {"0-35": "0-35", "35-55": "35 and over", "55<=": "35 and over"}

EXPECTED_ROWS = {                     # (fit, pcal) per checkpoint
    14: (6_480, 5_077),
    28: (6_421, 4_869),
    42: (6_343, 4_794),
}

os.makedirs(OUT_DIR, exist_ok=True)

versions = {
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "pyarrow": pyarrow.__version__,
    "scikit-learn": sklearn.__version__,
    "matplotlib": matplotlib.__version__,
    "interpret": interpret.__version__,
    "fairlearn": fairlearn.__version__,
    "shap (used by interpret ShapKernel)": shap.__version__,
}
print("=" * 72)
print("Library versions")
for k, v in versions.items():
    print(f"  {k:<38s} {v}")
with open(os.path.join(OUT_DIR, "environment.txt"), "w", encoding="utf-8") as f:
    f.write(f"platform: {platform.platform()}\n")
    for k, v in versions.items():
        f.write(f"{k}: {v}\n")


def fail(msg):
    """Stop the pilot when a check fails (as required by the rules)."""
    print(f"\nCHECK FAILED: {msg}\nStopping.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def ece_score(y_true, p, n_bins=N_BINS):
    """Expected Calibration Error with equal-width bins on [0, 1].
    ECE = sum over non-empty bins of (n_bin / n) * |mean(p) - mean(y)|.
    Bin i covers [i/n_bins, (i+1)/n_bins); the last bin also includes 1.0."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.sum() / len(p) * abs(p[m].mean() - y_true[m].mean())
    return ece


def reliability_points(y_true, p, n_bins=N_BINS):
    """Mean predicted probability and observed At-risk rate per non-empty bin
    (same binning as ece_score)."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    mp, obs = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            mp.append(p[m].mean())
            obs.append(y_true[m].mean())
    return np.array(mp), np.array(obs)


def logit(p):
    """log(p / (1 - p)) after clipping p to [1e-6, 1 - 1e-6]."""
    p = np.clip(np.asarray(p, dtype=float), CLIP_EPS, 1 - CLIP_EPS)
    return np.log(p / (1 - p))


def n_at_risk(y_true, y_pred):
    """Number of At-risk enrolments in the group (y_pred unused)."""
    return int(np.sum(y_true))


# ----------------------------------------------------------------------------
# 1. Load the data (once) and run the checks
# ----------------------------------------------------------------------------
data = {}          # checkpoint -> dict with fit / pcal frames and metadata
dropped_log = []   # features dropped by the all-missing-in-fit rule
for t in CHECKPOINTS:
    path = os.path.join(PREPARED_DIR, f"checkpoint_T{t}.parquet")
    df = pd.read_parquet(path)

    features = [c for c in df.columns if not c.startswith("meta_")]
    if len(features) != 31 or any(c.startswith("meta_") for c in features):
        fail(f"Day {t}: expected 31 non-meta feature columns, found {len(features)}")

    # Count (only count) the rows we are going to drop.
    n_ccal_file = int((df["meta_partition"] == "ccal").sum())
    n_test_file = int((df["meta_partition"] == "test").sum())

    # Keep only fit and pcal rows; ccal/test rows are discarded here.
    df = df[df["meta_partition"].isin(["fit", "pcal"])].copy()
    n_ccal_used = int((df["meta_partition"] == "ccal").sum())
    n_test_used = int((df["meta_partition"] == "test").sum())
    print(f"\nDay {t}: ccal rows in file = {n_ccal_file}, test rows in file = {n_test_file} "
          f"-> used after filtering: ccal = {n_ccal_used}, test = {n_test_used}")
    assert n_ccal_used == 0 and n_test_used == 0, "ccal/test rows leaked through the filter"

    fit = df[df["meta_partition"] == "fit"].copy()
    pcal = df[df["meta_partition"] == "pcal"].copy()
    exp_fit, exp_pcal = EXPECTED_ROWS[t]
    print(f"Day {t}: fit = {len(fit):,} (expected {exp_fit:,}), "
          f"pcal = {len(pcal):,} (expected {exp_pcal:,})")
    if len(fit) != exp_fit or len(pcal) != exp_pcal:
        fail(f"Day {t}: row counts do not match the expected values")

    # General rule (approved by the user): drop any feature that is ENTIRELY missing in the
    # fit rows at this checkpoint (decided from fit only). Such a column carries no
    # information and makes sklearn 1.9.1's HGB binning crash. Constant features are kept.
    all_missing = [c for c in features if fit[c].isna().all()]
    for c in all_missing:
        print(f"Day {t}: DROPPED feature '{c}' - entirely missing in fit "
              f"({len(fit):,} of {len(fit):,} rows missing)")
        dropped_log.append(dict(checkpoint=t, feature=c, n_fit=len(fit),
                                n_missing_fit=len(fit),
                                reason="entirely missing in fit rows"))
    features = [c for c in features if c not in all_missing]
    print(f"Day {t}: {len(features)} features used after the all-missing rule")

    # Convert string/object feature columns to category, categories fitted on fit rows
    # and applied identically to pcal. Values unseen in fit become NaN (missing).
    cat_cols = [c for c in features
                if pd.api.types.is_object_dtype(fit[c]) or pd.api.types.is_string_dtype(fit[c])]
    categories = {}
    unseen = {}
    for c in cat_cols:
        cats = sorted(fit[c].dropna().unique().tolist())
        categories[c] = cats
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        n_before = int(pcal[c].notna().sum())
        fit[c] = fit[c].astype(object).astype(dtype)
        pcal[c] = pcal[c].astype(object).astype(dtype)
        unseen[c] = n_before - int(pcal[c].notna().sum())
    print(f"Day {t}: categorical features ({len(cat_cols)}): {cat_cols}")
    if any(unseen.values()):
        print(f"Day {t}: pcal values unseen in fit (set to missing): "
              f"{ {k: v for k, v in unseen.items() if v} }")

    data[t] = dict(features=features, cat_cols=cat_cols, categories=categories,
                   fit=fit, pcal=pcal, unseen=unseen)

# ----------------------------------------------------------------------------
# Steps 2-9 per checkpoint
# ----------------------------------------------------------------------------
results_rows, fairness_rows, shap_rows = [], [], []
conformal_rows, scope_rows = [], []

for t in CHECKPOINTS:
    print("\n" + "=" * 72)
    print(f"CHECKPOINT Day {t}")
    print("=" * 72)
    d = data[t]
    features, cat_cols = d["features"], d["cat_cols"]
    fit, pcal = d["fit"], d["pcal"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Split pcal into part A (Platt), part B (LAC thresholds), part C (evaluation)
    #    StratifiedGroupKFold(3): the three folds' TEST index sets are taken, in the
    #    order the splitter yields them, as parts A, B and C.
    # ------------------------------------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    folds = [test_idx for _, test_idx in sgkf.split(pcal[features], pcal["meta_y"],
                                                     groups=pcal["meta_id_student"])]
    assert len(folds) == 3
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(pcal)))  # every row once
    part_a, part_b, part_c = (pcal.iloc[ix].reset_index(drop=True) for ix in folds)

    sa, sb, sc = (set(p["meta_id_student"]) for p in (part_a, part_b, part_c))
    shared = len(sa & sb) + len(sa & sc) + len(sb & sc)
    if shared:
        fail(f"Day {t}: {shared} students appear in more than one part")
    for name, p in (("A", part_a), ("B", part_b), ("C", part_c)):
        print(f"[2] part {name}: n = {len(p):,}, At-risk rate = {p['meta_y'].mean():.4f}")
    print(f"[2] students in more than one part: 0 (asserted)")

    X_fit, y_fit = fit[features], fit["meta_y"].to_numpy()
    X_a, y_a = part_a[features], part_a["meta_y"].to_numpy()
    X_b, y_b = part_b[features], part_b["meta_y"].to_numpy()
    X_c, y_c = part_c[features], part_c["meta_y"].to_numpy()

    # ------------------------------------------------------------------
    # 3. Train HGB on fit rows (default hyperparameters, no tuning)
    # ------------------------------------------------------------------
    hgb = HistGradientBoostingClassifier(categorical_features="from_dtype",
                                         random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    hgb.fit(X_fit, y_fit)
    train_seconds = time.perf_counter() - t0
    assert list(hgb.classes_) == [0, 1]
    print(f"[3] HGB trained on {len(X_fit):,} fit rows in {train_seconds:.2f} s")

    # ------------------------------------------------------------------
    # 4. Raw probabilities on part C
    # ------------------------------------------------------------------
    p_raw_c = hgb.predict_proba(X_c)[:, 1]
    at_risk_rate_c = y_c.mean()
    pr_auc = average_precision_score(y_c, p_raw_c)
    brier_raw = brier_score_loss(y_c, p_raw_c)
    ece_raw = ece_score(y_c, p_raw_c)
    print(f"[4] part C At-risk rate (random-model PR-AUC) = {at_risk_rate_c:.4f}")
    print(f"[4] PR-AUC = {pr_auc:.4f} | Brier (raw) = {brier_raw:.4f} | ECE (raw) = {ece_raw:.4f}")

    # ------------------------------------------------------------------
    # 5. Platt scaling fitted on part A ONLY
    # ------------------------------------------------------------------
    z_a = logit(hgb.predict_proba(X_a)[:, 1]).reshape(-1, 1)
    platt = LogisticRegression(random_state=RANDOM_STATE)   # default settings
    platt.fit(z_a, y_a)
    print(f"[5] Platt fitted on part A: slope = {platt.coef_[0, 0]:.4f}, "
          f"intercept = {platt.intercept_[0]:.4f}")

    def predict_calibrated(X, _hgb=hgb, _platt=platt):
        """Calibrated P(At risk) = Platt(logit(HGB raw probability))."""
        z = logit(_hgb.predict_proba(X)[:, 1]).reshape(-1, 1)
        return _platt.predict_proba(z)[:, 1]

    # ------------------------------------------------------------------
    # 6. Calibrated probabilities on part C + reliability diagram
    # ------------------------------------------------------------------
    p_cal_c = predict_calibrated(X_c)
    brier_platt = brier_score_loss(y_c, p_cal_c)
    ece_platt = ece_score(y_c, p_cal_c)
    print(f"[6] Brier: raw {brier_raw:.4f} -> Platt {brier_platt:.4f} | "
          f"ECE: raw {ece_raw:.4f} -> Platt {ece_platt:.4f}")

    mp_raw, ob_raw = reliability_points(y_c, p_raw_c)
    mp_cal, ob_cal = reliability_points(y_c, p_cal_c)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot([0, 1], [0, 1], color="#8a8a85", lw=1.2, ls=":", label="Perfect calibration")
    ax.plot(mp_raw, ob_raw, color="#eb6834", lw=2, ls="--", marker="o", ms=7,
            label=f"Raw HGB (ECE = {ece_raw:.3f})")
    ax.plot(mp_cal, ob_cal, color="#2a78d6", lw=2, marker="s", ms=7,
            label=f"Platt-calibrated (ECE = {ece_platt:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted probability of At risk (per bin)")
    ax.set_ylabel("Observed At-risk rate (per bin)")
    ax.set_title(f"Reliability diagram, Day {t} (part C, n = {len(y_c):,}, 10 bins)\n"
                 "Pilot / feasibility only", fontsize=10)
    ax.grid(color="#e5e4df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"reliability_T{t}.png"))
    plt.close(fig)
    print(f"[6] saved {OUT_DIR}/reliability_T{t}.png")

    # ------------------------------------------------------------------
    # 6b. LAC conformal prediction sets: thresholds on part B, sets on part C
    #     score s = 1 - calibrated probability of the TRUE class
    #     q_c = ceil((n_c + 1)(1 - alpha))-th smallest score among part-B rows of class c
    #     label c is in the set if its calibrated probability >= 1 - q_c
    #     (alpha = 0.10, fixed; no routing is applied)
    # ------------------------------------------------------------------
    p_cal_b = predict_calibrated(X_b)
    prob_true_b = np.where(y_b == 1, p_cal_b, 1 - p_cal_b)
    scores_b = 1 - prob_true_b
    q, n_cls = {}, {}
    for c in (1, 0):
        s_c = np.sort(scores_b[y_b == c])
        n_c = len(s_c)
        k_c = math.ceil((n_c + 1) * (1 - ALPHA))
        if k_c > n_c:
            fail(f"Day {t}: LAC rank {k_c} exceeds part-B class-{c} size {n_c}")
        q[c], n_cls[c] = s_c[k_c - 1], n_c
        print(f"[6b] class {'At risk' if c else 'Not at risk'}: n_B = {n_c}, rank k = {k_c}, "
              f"q = {q[c]:.4f}, minimum probability to include = {1 - q[c]:.4f}")

    inc_risk = p_cal_c >= 1 - q[1]                 # P(At risk)     >= 1 - q_1
    inc_not = (1 - p_cal_c) >= 1 - q[0]            # P(Not at risk) >= 1 - q_0
    set_size = inc_risk.astype(int) + inc_not.astype(int)
    only_risk = inc_risk & ~inc_not
    only_not = inc_not & ~inc_risk
    ambiguous = inc_risk & inc_not
    empty = ~inc_risk & ~inc_not
    cov_risk = inc_risk[y_c == 1].mean()
    cov_not = inc_not[y_c == 0].mean()

    # In terms of p = P(At risk): both labels are included when 1 - q_1 <= p <= q_0.
    amb_low, amb_high = 1 - q[1], q[0]
    has_amb_range = amb_low <= amb_high
    half_in_amb = bool(has_amb_range and amb_low <= 0.5 <= amb_high)
    if ambiguous.any():   # consistency check: observed ambiguous p lie inside the range
        assert p_cal_c[ambiguous].min() >= amb_low - 1e-9
        assert p_cal_c[ambiguous].max() <= amb_high + 1e-9
    print(f"[6b] part C coverage: At risk = {cov_risk:.4f}, Not at risk = {cov_not:.4f} "
          f"(target >= {1 - ALPHA:.2f})")
    print(f"[6b] sets: {{At risk}} {only_risk.mean():.4f} | {{Not at risk}} {only_not.mean():.4f} | "
          f"ambiguous {ambiguous.mean():.4f} | empty {empty.mean():.4f} | "
          f"average size {set_size.mean():.4f}")
    if has_amb_range:
        print(f"[6b] ambiguous when P(At risk) in [{amb_low:.4f}, {amb_high:.4f}]; "
              f"0.5 inside: {half_in_amb}")
    else:
        print(f"[6b] no ambiguous range; EMPTY set when P(At risk) in ({amb_high:.4f}, {amb_low:.4f})")

    conformal_rows.append(dict(
        checkpoint=t, alpha=ALPHA,
        n_partB_at_risk=n_cls[1], n_partB_not_at_risk=n_cls[0],
        q_at_risk=q[1], min_prob_at_risk=1 - q[1],
        q_not_at_risk=q[0], min_prob_not_at_risk=1 - q[0],
        n_partC=len(y_c),
        coverage_at_risk=cov_risk, coverage_not_at_risk=cov_not,
        prop_set_at_risk_only=only_risk.mean(), prop_set_not_at_risk_only=only_not.mean(),
        prop_ambiguous=ambiguous.mean(), prop_empty=empty.mean(),
        avg_set_size=set_size.mean(),
        ambiguous_p_low=amb_low if has_amb_range else np.nan,
        ambiguous_p_high=amb_high if has_amb_range else np.nan,
        half_in_ambiguous_range=half_in_amb))

    # ------------------------------------------------------------------
    # 7. InterpretML ShapKernel on the CALIBRATED model
    # ------------------------------------------------------------------
    cats = d["categories"]

    # InterpretML's ShapKernel rejects ANY missing value in its input, although HGB
    # handles them. Approved fix: placeholders ONLY inside the explainer's inputs.
    #   * missing category code (imd_band, or any categorical) -> -1  (real codes are >= 0)
    #   * missing days_since_last_activity / mean_score_available -> -999
    # The wrapper turns every placeholder back into a genuine missing value before calling
    # predict_calibrated, so the model never sees a placeholder. The original DataFrames
    # are never modified (to_codes only reads them into a new array).
    CAT_PLACEHOLDER = -1
    NUM_PLACEHOLDER = -999.0
    NUM_PLACEHOLDER_COLS = [c for c in ["days_since_last_activity", "mean_score_available"]
                            if c in features]   # mean_score_available is dropped at Day 14/28

    # Condition 3: no real value may equal a placeholder (checked on fit + pcal rows).
    for c in NUM_PLACEHOLDER_COLS:
        n_clash = int((X_fit[c] == NUM_PLACEHOLDER).sum() + (pcal[c] == NUM_PLACEHOLDER).sum())
        if n_clash:
            fail(f"Day {t}: {n_clash} real values of {c} equal the placeholder {NUM_PLACEHOLDER}")
    for c in cat_cols:
        real_codes = np.r_[X_fit[c].cat.codes.to_numpy(), pcal[c].cat.codes.to_numpy()]
        if (real_codes[real_codes != -1] == CAT_PLACEHOLDER).any():
            fail(f"Day {t}: a real category code of {c} equals the placeholder")

    def to_codes(X):
        """DataFrame -> NEW float array for the explainer (X itself is not modified).
        Categories become integer codes; missing values become placeholders."""
        out = np.empty((len(X), len(features)), dtype=float)
        for j, c in enumerate(features):
            if c in cat_cols:
                out[:, j] = X[c].cat.codes.to_numpy().astype(float)   # missing = -1
            else:
                col = X[c].to_numpy(dtype=float, na_value=np.nan)
                if c in NUM_PLACEHOLDER_COLS:
                    col = np.where(np.isnan(col), NUM_PLACEHOLDER, col)
                out[:, j] = col
        if np.isnan(out).any():
            bad = [features[j] for j in np.where(np.isnan(out).any(axis=0))[0]]
            fail(f"Day {t}: missing values in features without an approved placeholder: {bad}")
        return out

    def from_codes(A):
        """Explainer array -> DataFrame with the original column types; every
        placeholder is turned back into a genuine missing value (NaN / missing category)."""
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        cols = {}
        for j, c in enumerate(features):
            col = A[:, j]
            if c in cat_cols:
                codes = np.rint(col).astype(int)            # -1 -> missing category
                cols[c] = pd.Categorical.from_codes(codes, categories=cats[c])
            elif c in NUM_PLACEHOLDER_COLS:
                cols[c] = np.where(col == NUM_PLACEHOLDER, np.nan, col)
            else:
                cols[c] = col
        return pd.DataFrame(cols, columns=features)

    def explainer_predict(A):
        """Wrapper for ShapKernel: placeholders -> missing -> predict_calibrated.
        Returns [P(Not at risk), P(At risk)] so interpret treats it as binary
        classification and explains class 1 (At risk)."""
        p1 = predict_calibrated(from_codes(A))
        return np.column_stack([1 - p1, p1])

    # Condition 5: wrapper and predict_calibrated must agree on ALL part-C rows.
    p_wrap = explainer_predict(to_codes(X_c))[:, 1]
    n_diff = int((p_cal_c != p_wrap).sum())
    if n_diff:
        fail(f"Day {t}: wrapper differs from predict_calibrated on {n_diff} part-C rows "
             f"(max |diff| = {np.abs(p_cal_c - p_wrap).max():.3e})")
    print(f"[7] wrapper check: identical probabilities on all {len(X_c):,} part-C rows")

    def n_missing(X):
        return int(sum(X[c].isna().sum() for c in features))

    # Condition 4: same random sampling rule (100 fit rows, 20 evaluation rows, seed 42).
    background = X_fit.sample(n=N_BACKGROUND, random_state=RANDOM_STATE)
    explain_rows = X_c.sample(n=N_EXPLAIN, random_state=RANDOM_STATE)

    # Staff-action scope stand-in: top 15 % of part C by calibrated probability
    # (ties keep part-C order); a random 20 of them are explained and timed.
    k_scope = int(round(SHAP_SCOPE_FRACTION * len(p_cal_c)))
    scope_idx = np.argsort(-p_cal_c, kind="stable")[:k_scope]
    scope_rows_X = X_c.iloc[scope_idx]
    scope_explain = scope_rows_X.sample(n=N_EXPLAIN, random_state=RANDOM_STATE)
    print(f"[7] placeholders in explainer inputs: background {n_missing(background)}, "
          f"random rows {n_missing(explain_rows)}, scope rows {n_missing(scope_explain)}")

    try:
        t0 = time.perf_counter()
        explainer = ShapKernel(explainer_predict, to_codes(background),
                               feature_names=features,
                               feature_types=["continuous"] * len(features))
        shap_init_seconds = time.perf_counter() - t0
        # (a) random 20 part-C rows -> full-test-set estimate (as before)
        t0 = time.perf_counter()
        explanation = explainer.explain_local(to_codes(explain_rows))
        shap_seconds_20 = time.perf_counter() - t0
        # (b) random 20 of the top-15 % scope rows -> staff-action scope estimate
        t0 = time.perf_counter()
        explainer.explain_local(to_codes(scope_explain))
        scope_seconds_20 = time.perf_counter() - t0
    except Exception as e:  # rule: report and stop, do not switch explainer
        fail(f"Day {t}: ShapKernel failed with {type(e).__name__}: {e}")

    shap_estimate_hours = shap_seconds_20 / N_EXPLAIN * N_TEST_FINAL / 3600
    scope_per_case = scope_seconds_20 / N_EXPLAIN
    scope_estimate_seconds = scope_per_case * N_TEST_SCOPE
    print(f"[7] ShapKernel setup {shap_init_seconds:.2f} s")
    print(f"[7] random 20 part-C rows: {shap_seconds_20:.2f} s "
          f"-> full set ({N_TEST_FINAL:,}) estimate {shap_estimate_hours:.2f} h")
    print(f"[7] scope: top 15 % of part C = {k_scope} cases; 20 explained in "
          f"{scope_seconds_20:.2f} s = {scope_per_case:.3f} s/case -> "
          f"{N_TEST_SCOPE:,} test cases ~ {scope_estimate_seconds:,.0f} s "
          f"= {scope_estimate_seconds / 3600:.2f} h")

    # Optional fallback check (timing only, NOT used for explanations):
    # shap.TreeExplainer on the RAW HGB for the same 20 scope cases. It needs numeric
    # input, so categories are given as codes (missing -> NaN; trees handle NaN).
    # Additivity is checked: if SHAP values do not sum to the HGB log-odds, the tree
    # explainer is not reproducing HGB's categorical splits and its values are not faithful.
    tree_per_case, tree_err, tree_note = np.nan, np.nan, ""
    try:
        Xt = scope_explain.copy()
        for c in cat_cols:
            codes = Xt[c].cat.codes
            Xt[c] = codes.where(codes >= 0).astype(float)
        tree_expl = shap.TreeExplainer(hgb)
        t0 = time.perf_counter()
        tv = tree_expl.shap_values(Xt)
        tree_per_case = (time.perf_counter() - t0) / N_EXPLAIN
        tv = np.asarray(tv)
        if tv.ndim == 3:
            tv = tv[..., 1]
        ev = np.ravel(tree_expl.expected_value)[-1]
        tree_err = float(np.abs(tv.sum(axis=1) + ev - hgb.decision_function(scope_explain)).max())
        tree_note = ("timing only; additivity OK" if tree_err < 1e-3 else
                     "timing only; additivity FAILS (categorical splits not reproduced) - "
                     "values not faithful to HGB")
        print(f"[7] fallback shap.TreeExplainer (raw HGB): {tree_per_case:.4f} s/case; "
              f"additivity max error {tree_err:.3g} log-odds ({tree_note})")
    except Exception as e:
        tree_note = f"failed: {type(e).__name__}: {e}"
        print(f"[7] fallback shap.TreeExplainer {tree_note}")

    scope_rows.append(dict(
        checkpoint=t, n_partC=len(X_c), n_top15_partC=k_scope, n_explained=N_EXPLAIN,
        kernel_seconds_20=scope_seconds_20, kernel_seconds_per_case=scope_per_case,
        n_test_scope=N_TEST_SCOPE, kernel_scope_estimate_seconds=scope_estimate_seconds,
        kernel_scope_estimate_hours=scope_estimate_seconds / 3600,
        kernel_full_set_estimate_hours=shap_estimate_hours,
        tree_explainer="shap.TreeExplainer (raw HGB)",
        tree_seconds_per_case=tree_per_case, tree_additivity_max_error=tree_err,
        tree_note=tree_note))

    # Top-3 |SHAP| features for the first of the 20 random rows
    ex0 = explanation.data(0)
    shap_vals = np.asarray(ex0["scores"], dtype=float)
    names = list(ex0["names"])
    assert names == features
    row0 = explain_rows.iloc[0]
    order = np.argsort(-np.abs(shap_vals))[:3]
    print(f"[7] example row (part-C enrolment {part_c.loc[explain_rows.index[0], 'meta_enrolment_id']}), "
          f"calibrated P(At risk) = {predict_calibrated(explain_rows.iloc[[0]])[0]:.3f}, "
          f"base value = {ex0['extra']['scores'][0]:.3f}")
    for rank, j in enumerate(order, start=1):
        val = row0[features[j]]
        val = "missing" if pd.isna(val) else val       # condition 6: never show placeholders
        direction = "raised risk" if shap_vals[j] > 0 else "lowered risk"
        print(f"     {rank}. {features[j]} = {val}: SHAP = {shap_vals[j]:+.4f} ({direction})")
        shap_rows.append(dict(checkpoint=t, rank=rank, feature=features[j], value=val,
                              shap_value=shap_vals[j], direction=direction))

    # ------------------------------------------------------------------
    # 8. Fairlearn tool test on part C (top 10 % flagged; NOT the routing policy)
    #    Age band: 55<= merged into 35-55 as "35 and over" before MetricFrame.
    # ------------------------------------------------------------------
    k = int(round(TOP_FRACTION * len(p_cal_c)))
    order_c = np.argsort(-p_cal_c, kind="stable")        # ties: keep part-C order
    y_pred = np.zeros(len(p_cal_c), dtype=int)
    y_pred[order_c[:k]] = 1
    print(f"[8] flagged top {k} of {len(p_cal_c):,} part-C enrolments "
          f"(threshold P >= {p_cal_c[order_c[k - 1]]:.3f})")

    metrics = {
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "selection_rate": selection_rate,
        "n": count,
        "n_at_risk": n_at_risk,
    }
    for attr in ["meta_disability", "meta_age_band"]:
        sf = part_c[attr]
        if sf.isna().any():
            fail(f"Day {t}: {attr} has missing values in part C")
        sf = sf.astype(str)
        label = attr
        if attr == "meta_age_band":
            unknown = set(sf) - set(AGE_MERGE)
            if unknown:
                fail(f"Day {t}: unexpected age bands {unknown}")
            sf = sf.map(AGE_MERGE)
            label = "meta_age_band (55<= merged into 35 and over)"
        mf = MetricFrame(metrics=metrics, y_true=y_c, y_pred=y_pred,
                         sensitive_features=sf.to_numpy())
        bg = mf.by_group
        for g, r in bg.iterrows():
            fairness_rows.append(dict(
                checkpoint=t, attribute=label, group=g, n=int(r["n"]),
                n_at_risk=int(r["n_at_risk"]), recall=r["recall"],
                precision=r["precision"], selection_rate=r["selection_rate"],
                small_group_flag="small group" if r["n"] < SMALL_GROUP else ""))
        ov = mf.overall
        fairness_rows.append(dict(
            checkpoint=t, attribute=label, group="overall", n=int(ov["n"]),
            n_at_risk=int(ov["n_at_risk"]), recall=ov["recall"],
            precision=ov["precision"], selection_rate=ov["selection_rate"],
            small_group_flag=""))
        diff = mf.difference()   # max - min across groups, per metric
        fairness_rows.append(dict(
            checkpoint=t, attribute=label, group="difference (max - min)",
            n=np.nan, n_at_risk=np.nan, recall=diff["recall"], precision=np.nan,
            selection_rate=diff["selection_rate"], small_group_flag=""))
        print(f"[8] {label}:")
        print(bg[["n", "n_at_risk", "recall", "precision", "selection_rate"]]
              .to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"    overall recall = {ov['recall']:.3f}, selection rate = {ov['selection_rate']:.3f}; "
              f"largest difference: recall = {diff['recall']:.3f}, "
              f"selection rate = {diff['selection_rate']:.3f}")
        small = bg.index[bg["n"] < SMALL_GROUP].tolist()
        print(f"    small group(s) (< {SMALL_GROUP}): {small if small else 'none'}")

    results_rows.append(dict(
        checkpoint=t, n_fit=len(X_fit), n_partA=len(X_a), n_partB=len(X_b), n_partC=len(X_c),
        at_risk_rate_partC=at_risk_rate_c, pr_auc=pr_auc,
        brier_raw=brier_raw, brier_platt=brier_platt,
        ece_raw=ece_raw, ece_platt=ece_platt,
        train_seconds=train_seconds, shap_seconds_20=shap_seconds_20,
        shap_estimate_test_hours=shap_estimate_hours,
        shap_scope_seconds_per_case=scope_per_case,
        shap_scope_estimate_hours=scope_estimate_seconds / 3600,
        lac_q_at_risk=q[1], lac_q_not_at_risk=q[0],
        coverage_at_risk=cov_risk, coverage_not_at_risk=cov_not,
        prop_ambiguous=ambiguous.mean(), prop_empty=empty.mean(),
        avg_set_size=set_size.mean()))

# ----------------------------------------------------------------------------
# 9. Save results (once)
# ----------------------------------------------------------------------------
res = pd.DataFrame(results_rows)
conf = pd.DataFrame(conformal_rows)
scope = pd.DataFrame(scope_rows)
res.to_csv(os.path.join(OUT_DIR, "pilot_results.csv"), index=False)
conf.to_csv(os.path.join(OUT_DIR, "conformal_pilot.csv"), index=False)
scope.to_csv(os.path.join(OUT_DIR, "shap_scope_estimate.csv"), index=False)
pd.DataFrame(fairness_rows).to_csv(os.path.join(OUT_DIR, "fairness_pilot.csv"), index=False)
pd.DataFrame(shap_rows).to_csv(os.path.join(OUT_DIR, "shap_example.csv"), index=False)
pd.DataFrame(dropped_log, columns=["checkpoint", "feature", "n_fit", "n_missing_fit", "reason"]) \
    .to_csv(os.path.join(OUT_DIR, "dropped_features.csv"), index=False)

with open(os.path.join(OUT_DIR, "pilot_notes.txt"), "w", encoding="utf-8") as f:
    f.write(
        "Pilot notes (feasibility evidence only - not final results)\n\n"
        "Data-handling note:\n"
        "  An early exploratory check displayed label and group counts across all partitions;\n"
        "  these were not used in any modelling decision.\n\n"
        "Decisions not covered by the original rules:\n"
        "  - Features entirely missing in the fit rows at a checkpoint are dropped at that\n"
        "    checkpoint (decided from fit only; constant features kept). See dropped_features.csv.\n"
        "  - pcal split: StratifiedGroupKFold(3); the three folds' test sets, in the order the\n"
        "    splitter yields them, are parts A (Platt), B (LAC thresholds), C (evaluation).\n"
        "  - Platt scaling uses LogisticRegression with sklearn default settings (L2, C = 1.0).\n"
        "  - LAC: inclusion uses '>=' (label c in set if its probability >= 1 - q_c); the\n"
        "    ambiguous range is 1 - q_At-risk <= P(At risk) <= q_Not-at-risk.\n"
        "  - Fairlearn top 10 %: k = round(0.10 x n_partC); ties keep part-C row order.\n"
        "  - Age band: 55<= merged with 35-55 into '35 and over' for Fairlearn only.\n"
        "  - SHAP scope: top 15 % of part C (k = round(0.15 x n_partC), ties keep part-C order)\n"
        "    as a stand-in for staff-action cases; 20 of them sampled with random_state=42.\n"
        "    Test-scope estimate uses round(0.15 x 9,353) = 1,403 cases.\n"
        "  - SHAP timing: explain_local time only; explainer set-up timed separately and not\n"
        "    included in the estimates. KernelExplainer nsamples left at shap default.\n"
        "  - SHAP input (approved): ShapKernel rejects missing values, so placeholders are used\n"
        "    ONLY inside the explainer's inputs: missing category code -> -1, missing\n"
        "    days_since_last_activity / mean_score_available -> -999. The wrapper converts them\n"
        "    back to genuine missing values before predict_calibrated; no real value equals a\n"
        "    placeholder (asserted); wrapper and predict_calibrated agree exactly on all part-C\n"
        "    rows (asserted). Placeholders are reported as 'missing'. All features are declared\n"
        "    'continuous' to InterpretML (it casts everything to float64 regardless).\n"
        "  - Tree-explainer fallback (timing only): shap.TreeExplainer on the raw HGB with\n"
        "    category codes (InterpretML ShapTree is in interpret.greybox and wraps the same\n"
        "    shap TreeExplainer). Additivity against HGB log-odds is reported.\n"
        "  - pcal category values unseen in fit are set to missing.\n")

# ----------------------------------------------------------------------------
# 10. Summary
# ----------------------------------------------------------------------------
fmt = lambda x: f"{x:.4f}"
print("\n" + "=" * 72)
print("SUMMARY (feasibility evidence only - not final results)")
print("=" * 72)
print(res[["checkpoint", "at_risk_rate_partC", "pr_auc", "brier_raw", "brier_platt",
           "ece_raw", "ece_platt", "train_seconds"]].to_string(index=False, float_format=fmt))
print("\nLAC conformal sets (alpha = 0.10; thresholds on part B, evaluated on part C):")
print(conf[["checkpoint", "q_at_risk", "min_prob_at_risk", "q_not_at_risk",
            "min_prob_not_at_risk", "coverage_at_risk", "coverage_not_at_risk",
            "prop_set_at_risk_only", "prop_set_not_at_risk_only", "prop_ambiguous",
            "prop_empty", "avg_set_size", "ambiguous_p_low", "ambiguous_p_high",
            "half_in_ambiguous_range"]].T.to_string(float_format=fmt))
print("\nSHAP timing (Kernel SHAP on calibrated model):")
print(scope[["checkpoint", "kernel_seconds_per_case", "kernel_scope_estimate_hours",
             "kernel_full_set_estimate_hours", "tree_seconds_per_case",
             "tree_additivity_max_error"]].to_string(index=False, float_format=fmt))

above = (res["pr_auc"] > res["at_risk_rate_partC"]).all()
rising = res["pr_auc"].is_monotonic_increasing
print(f"\nPR-AUC above At-risk rate at every checkpoint: {above}")
print(f"PR-AUC rises monotonically Day 14 -> 28 -> 42: {rising} "
      f"(Day 14 {res.pr_auc.iloc[0]:.4f}, Day 42 {res.pr_auc.iloc[-1]:.4f})")
for _, r in res.iterrows():
    print(f"Day {int(r.checkpoint)}: Platt lowered Brier: {r.brier_platt < r.brier_raw}; "
          f"lowered ECE: {r.ece_platt < r.ece_raw}; "
          f"coverage >= 0.90 (At risk / Not at risk): "
          f"{r.coverage_at_risk >= 0.9} / {r.coverage_not_at_risk >= 0.9}; "
          f"SHAP scope estimate {r.shap_scope_estimate_hours:.2f} h "
          f"(< 2 h: {r.shap_scope_estimate_hours < 2})")
print("\nccal / test rows used: 0 at every checkpoint (asserted). "
      "Hyperparameters: sklearn defaults, not tuned. alpha fixed at 0.10. No routing applied.")
print(f"Outputs written to {OUT_DIR}/")
