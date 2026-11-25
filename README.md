# Capacity-Aware Early-Warning System (OULAD)

A sequential early-warning system that flags students at risk of failing or withdrawing at
**Weeks 2, 4 and 6** of a module, and decides who staff should act on when there are only a
fixed number of places to do it. It is built on the
[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset).

> **Status: feasibility pilot on development data only.** Every number in this repository comes
> from the `fit` and `pcal` partitions. The conformal-calibration (`ccal`) and test (`test`)
> partitions are assigned, then dropped before any modelling. None of the results here are the
> final evaluation.

---

## How it works

```
OULAD tables ──► leakage-safe features ──► HGB model ──► Platt ──► LAC conformal ──► capacity-aware
                 at Days 14 / 28 / 42      (+ LR         calibration  prediction sets   routing
                                           baseline)                  (α = 0.10)        (K, R, B)
```

1. **Data and features** ([prepare_data.py](prepare_data.py)). The unit is an *enrolment*
   (module, presentation, student). At each checkpoint T only records dated ≤ end of Day T are
   used. A score counts only once it would have been released (max(submitted, due) + 14 days for
   a TMA, + 7 days for a CMA); before that it is missing, never zero. 31 features covering
   assessments, VLE activity and demographics.
2. **Label.** *At risk* (`y = 1`) = final result Fail or Withdrawn; Pass/Distinction = 0.
3. **Partitions.** Whole module presentations go to one partition, and each student sits in one
   partition only (priority test > ccal > pcal > fit):

   | partition | role | presentations |
   |---|---|---|
   | `fit` | model fitting | AAA, BBB, DDD, EEE, FFF, GGG 2013J |
   | `pcal` | probability calibration (pilot evaluation) | BBB, DDD, FFF 2013B; EEE, GGG 2014B |
   | `ccal` | conformal calibration (final study) | BBB, CCC, DDD, FFF 2014B |
   | `test` | final test | AAA, BBB, CCC, DDD, EEE, FFF, GGG 2014J |

4. **Model.** `HistGradientBoostingClassifier` with scikit-learn defaults and native categorical
   and missing-value handling, trained once per checkpoint on `fit`. No tuning. Logistic
   regression is the comparison model.
5. **Calibration and conformal sets.** `pcal` is split into three student-grouped folds and
   cross-fitted: one fold fits Platt scaling, one sets the class-conditional LAC thresholds, one
   is evaluated, rotated three times. Each enrolment gets a set: {At risk}, {Not at risk},
   ambiguous (both) or empty.
6. **Routing** ([routing_engine.py](routing_engine.py)). Each enrolment gets one status per
   checkpoint, separately in each module presentation:

   | prediction set | route |
   |---|---|
   | {At risk} | Contact candidate |
   | empty {} | Review candidate |
   | {Not at risk} | Monitor |
   | ambiguous, p ≥ 0.5 | Review candidate |
   | ambiguous, p < 0.5, Week 2 or 4 | Defer (wait for more evidence) |
   | ambiguous, p < 0.5, Week 6, persistent | Review candidate |
   | ambiguous, p < 0.5, Week 6, not persistent | Monitor (uncertainty flag) |

   Candidates compete for limited places. Those who miss out go to *Contact Waiting* or
   *Review Waiting*. A student is never contacted twice. A contacted student who is still high
   risk can get a follow-up Review, but only from Review places left over.
7. **Explanation and fairness.** Kernel SHAP (InterpretML) explains individual recommendations.
   Fairlearn `MetricFrame` checks fairness of access by age band and disability.

### Capacity

Capacity is set per module presentation in whole places, in [capacity.py](capacity.py):

| symbol | meaning | default |
|---|---|---|
| K | new contacts per checkpoint | 10 |
| R | reviews per checkpoint | 5 |
| B | unique contacts across Weeks 2, 4, 6 | 25 |

See [README_capacity.md](README_capacity.md) for the validation rules, per-week K, and per-module
overrides.

---

## Pilot results (development data, not final)

About 49% of `pcal` enrolments are At risk, so being at risk is common. The question is how well
the limited places are used, not whether rare cases are found. With the default capacity, the
five `pcal` modules have **125 contact places in total, about 5% of the 2,480 at-risk
enrolments**.

| method | contacts | of which At risk | false-positive contacts | at-risk reached by Week 6 |
|---|---|---|---|---|
| Rule-based alerts | 125 | 91 | 34 | 133 |
| HGB Top-K | 125 | 108 | 17 | 163 |
| Probability-threshold deferral | 125 | 108 | 17 | 163 |
| **Proposed (conformal + persistence + Waiting)** | 125 | 108 | 17 | 152 |

Source: [pilot/v5_followup_last/summary_v5.csv](pilot/v5_followup_last/summary_v5.csv).
Model metrics (PR-AUC 0.72–0.77, Platt ECE ≤ 0.03) and conformal coverage (≈ 0.90 per class
over 100 re-splits) are in §4.3 and §5.2 of the notebook.

---

## Getting started

### Requirements

The versions used are in [pilot/environment.txt](pilot/environment.txt):

```
python 3.14.2   pandas 3.0.1   numpy 2.4.3   pyarrow 26.0.1   scikit-learn 1.9.1
scipy           matplotlib 3.11.2   interpret 0.7.8   shap 0.54.1   fairlearn 0.15.0   jupyter
```

```bash
pip install pandas numpy pyarrow scikit-learn scipy matplotlib interpret shap fairlearn jupyter
```

### Data

Download OULAD (CC BY 4.0) and put the tables in the project folder. Each table can be
`name.csv` or `name.csv.zip`, and `studentVle` may be split into `studentVle_0 … studentVle_7`.
Data files (`*.csv`, `*.zip`, `*.parquet`, `prepared/`, `processed/`) are git-ignored and stay
on your machine.

### Run the model

Open [capacity_aware_ews_model.ipynb](capacity_aware_ews_model.ipynb) in the project folder,
next to `routing_engine.py` and `capacity.py`, and run **Kernel → Restart & Run All**. It goes
from the raw tables to routing, SHAP and fairness in one pass, runs every automated check, and
stops at the first one that fails. A full run takes about 19 minutes. Lower `N_RESPLITS` for a
quick test run.

The notebook writes nothing to disk. If saved pilot outputs from the author's machine are in
`prepared/` or `pilot/`, it cross-checks against them. Otherwise those checks are marked
*skipped*, and the run still completes.

### Use the routing code on its own

```python
import routing_engine as RE
from capacity import capacity_for_all_modules

cap = capacity_for_all_modules(modules)              # defaults K=10, R=5, B=25
log, caps_log = RE.run_routing(pred, "proposed", cap)
RE.check_routing(log, caps_log, pred, "proposed", cap)
```

---

## Repository layout

| path | what it is |
|---|---|
| `capacity_aware_ews_model.ipynb` | end-to-end model notebook (main entry point) |
| `routing_engine.py` | routing code for all six methods (imported by the notebook) |
| `capacity.py` | the only place capacity numbers are set (imported by the notebook) |
| `system_flowchart.html` | system and routing diagram |
| `pilot/v5_followup_last/summary_v5.csv` | routing summary behind the results table above |
| `pilot/environment.txt`, `pilot/pilot_notes.txt` | package versions and pilot decisions |
| `prepare_data.py`, `oulad_data_preparation.py` | standalone data-preparation scripts (earlier commit) |
| `pilot_model.py`, `pilot_conformal_seeds.py` | first feasibility pilot scripts (earlier commit) |

---

## Reproducibility rules

- `random_state = 42` everywhere. No hyperparameter is tuned on the data.
- Only `fit` and `pcal` rows are used. The scripts assert that no `ccal`/`test` rows remain, and
  never print their counts or outcomes.
- Nothing is fitted on the evaluated fold.
- Decisions not covered by the original plan are listed in
  [pilot/pilot_notes.txt](pilot/pilot_notes.txt).

## Data source and licence

Kuzilek, J., Hlosta, M. and Zdrahal, Z. (2017) "Open University Learning Analytics dataset",
*Scientific Data* 4, 170171. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Author

Chia Yu Hang. Final-year project.
