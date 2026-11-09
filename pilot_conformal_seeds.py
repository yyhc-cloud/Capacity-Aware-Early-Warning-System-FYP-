"""
pilot_conformal_seeds.py - DIAGNOSTIC ONLY (feasibility pilot).

Repeats the three-way split of pcal (StratifiedGroupKFold, n_splits=3, shuffle=True)
with random_state = 0..9, keeping everything else exactly as in pilot_model.py:
  * same data preparation (fit/pcal only, all-missing-in-fit features dropped,
    categories fitted on fit),
  * same HGB (defaults, random_state=42, trained on fit -> identical for every seed),
  * part A -> Platt scaling, part B -> LAC thresholds, part C -> evaluation,
  * alpha = 0.10 (not changed, no seed is selected).

Reports, per checkpoint, the mean / min / max class-conditional coverage and
ambiguity rate across the 10 splits. Output: pilot/conformal_seed_check.csv
(per-seed rows + summary rows). No ccal or test rows are used.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

MODEL_SEED = 42                       # HGB / Platt seed, unchanged from pilot_model.py
SPLIT_SEEDS = list(range(10))         # diagnostic split seeds 0..9
ALPHA = 0.10
CLIP_EPS = 1e-6
CHECKPOINTS = [14, 28, 42]
EXPECTED_ROWS = {14: (6_480, 5_077), 28: (6_421, 4_869), 42: (6_343, 4_794)}
OUT_DIR = "pilot"


def fail(msg):
    print(f"\nCHECK FAILED: {msg}\nStopping.")
    sys.exit(1)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), CLIP_EPS, 1 - CLIP_EPS)
    return np.log(p / (1 - p))


rows = []
for t in CHECKPOINTS:
    # ---- data preparation: identical to pilot_model.py step 1 ----
    df = pd.read_parquet(os.path.join("prepared", f"checkpoint_T{t}.parquet"))
    features = [c for c in df.columns if not c.startswith("meta_")]
    if len(features) != 31:
        fail(f"Day {t}: expected 31 features")
    df = df[df["meta_partition"].isin(["fit", "pcal"])]
    assert not df["meta_partition"].isin(["ccal", "test"]).any()
    fit = df[df["meta_partition"] == "fit"].copy()
    pcal = df[df["meta_partition"] == "pcal"].copy().reset_index(drop=True)
    if (len(fit), len(pcal)) != EXPECTED_ROWS[t]:
        fail(f"Day {t}: row counts differ from expected")
    features = [c for c in features if not fit[c].isna().all()]
    cat_cols = [c for c in features
                if pd.api.types.is_object_dtype(fit[c]) or pd.api.types.is_string_dtype(fit[c])]
    for c in cat_cols:
        dtype = pd.CategoricalDtype(categories=sorted(fit[c].dropna().unique().tolist()))
        fit[c] = fit[c].astype(object).astype(dtype)
        pcal[c] = pcal[c].astype(object).astype(dtype)

    # ---- HGB: identical for every split seed (trained on fit only) ----
    hgb = HistGradientBoostingClassifier(categorical_features="from_dtype",
                                         random_state=MODEL_SEED)
    hgb.fit(fit[features], fit["meta_y"].to_numpy())
    z_pcal = logit(hgb.predict_proba(pcal[features])[:, 1]).reshape(-1, 1)
    y_pcal = pcal["meta_y"].to_numpy()

    for seed in SPLIT_SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        ia, ib, ic = [te for _, te in sgkf.split(pcal[features], y_pcal,
                                                  groups=pcal["meta_id_student"])]
        g = pcal["meta_id_student"].to_numpy()
        if (set(g[ia]) & set(g[ib])) or (set(g[ia]) & set(g[ic])) or (set(g[ib]) & set(g[ic])):
            fail(f"Day {t}, seed {seed}: a student appears in more than one part")

        # Platt on part A
        platt = LogisticRegression(random_state=MODEL_SEED).fit(z_pcal[ia], y_pcal[ia])
        p = platt.predict_proba(z_pcal)[:, 1]          # calibrated P(At risk), all pcal rows

        # LAC thresholds on part B
        q = {}
        for c in (1, 0):
            yb = y_pcal[ib]
            pb = p[ib]
            s = np.sort(1 - np.where(yb == 1, pb, 1 - pb)[yb == c])
            k = math.ceil((len(s) + 1) * (1 - ALPHA))
            if k > len(s):
                fail(f"Day {t}, seed {seed}: LAC rank exceeds class size")
            q[c] = s[k - 1]

        # Prediction sets on part C
        pc, yc = p[ic], y_pcal[ic]
        inc_risk = pc >= 1 - q[1]
        inc_not = (1 - pc) >= 1 - q[0]
        rows.append(dict(
            row_type="per_seed", checkpoint=t, split_seed=seed,
            n_partA=len(ia), n_partB=len(ib), n_partC=len(ic),
            q_at_risk=q[1], q_not_at_risk=q[0],
            coverage_at_risk=inc_risk[yc == 1].mean(),
            coverage_not_at_risk=inc_not[yc == 0].mean(),
            ambiguity_rate=(inc_risk & inc_not).mean(),
            empty_rate=(~inc_risk & ~inc_not).mean()))

per_seed = pd.DataFrame(rows)
metrics = ["coverage_at_risk", "coverage_not_at_risk", "ambiguity_rate"]
summary = []
for t, g in per_seed.groupby("checkpoint"):
    for stat in ("mean", "min", "max"):
        r = dict(row_type=f"summary_{stat}", checkpoint=t, split_seed=np.nan)
        r.update({m: getattr(g[m], stat)() for m in metrics + ["empty_rate"]})
        summary.append(r)
    r = dict(row_type="summary_n_below_0.90", checkpoint=t, split_seed=np.nan)
    r.update({m: int((g[m] < 1 - ALPHA).sum()) for m in metrics[:2]})
    summary.append(r)
out = pd.concat([per_seed, pd.DataFrame(summary)], ignore_index=True)
out.to_csv(os.path.join(OUT_DIR, "conformal_seed_check.csv"), index=False)

pd.set_option("display.width", 160)
print("Per-seed results (alpha = 0.10, diagnostic only):")
print(per_seed.drop(columns="row_type").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSummary across split seeds 0-9:")
for t, g in per_seed.groupby("checkpoint"):
    print(f"Day {t}:")
    for m in metrics:
        below = f"; splits below 0.90: {int((g[m] < 0.9).sum())}/10" if "coverage" in m else ""
        print(f"  {m:<22s} mean {g[m].mean():.4f}  min {g[m].min():.4f}  max {g[m].max():.4f}{below}")
print(f"\nSaved {OUT_DIR}/conformal_seed_check.csv (no ccal/test rows used; alpha unchanged).")
