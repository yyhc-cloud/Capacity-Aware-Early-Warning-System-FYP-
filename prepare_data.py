"""
prepare_data.py
===============
OULAD data cleaning and leakage-safe checkpoint datasets (Days 14, 28, 42).

This script performs data cleaning and dataset construction ONLY. It does not train, tune,
calibrate, compute conformal thresholds, route, set capacity, impute or scale.

Conventions (see project brief):
  * Unit of analysis: an enrolment = (code_module, code_presentation, id_student).
  * Label y = 1 ("At risk") for Fail / Withdrawn, y = 0 for Pass / Distinction.
    final_result and date_unregistration are kept only as meta_ columns, never as features.
  * Checkpoint T in {14, 28, 42} means the END of Day T: every table is filtered to date <= T.
    Pre-start (negative) dates are included.
  * Eligibility at T: date_registration present and <= T, and date_unregistration missing or > T.
  * Whole cohorts are assigned to partitions (fit / pcal / ccal / test). Each student may appear
    in one partition only; priority test > ccal > pcal > fit.
  * A score is visible at T only if max(date_submitted, due_date) + delay <= T
    (primary delay: 14 days for TMA, 7 days for CMA). Unavailable scores are missing, never zero.
  * Column layout: meta_* columns (not predictors) first, then feature columns (no prefix).

No randomness is used anywhere in this script; SEED is fixed only for reproducibility hygiene.

Usage:  python prepare_data.py            (reads inputs from the script's folder)
Outputs are written to ./prepared/
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'prepared')
os.makedirs(OUT, exist_ok=True)

KEY = ['code_module', 'code_presentation', 'id_student']
COHORT = ['code_module', 'code_presentation']
CHECKPOINTS = [14, 28, 42]
MAX_T = max(CHECKPOINTS)

# Marking-delay settings. 'primary' builds the main files; the others build sensitivity files.
DELAYS = {
    'primary':  {'TMA': 14, 'CMA': 7},
    'delay_7_0':  {'TMA': 7, 'CMA': 0},
    'delay_21_7': {'TMA': 21, 'CMA': 7},
}
DELAY_LABEL = {'primary': 'TMA14_CMA7 (primary)', 'delay_7_0': 'TMA7_CMA0',
               'delay_21_7': 'TMA21_CMA7'}
SCORE_COLS = ['n_submitted_mark_pending', 'n_scores_available', 'mean_score_available']

# Whole-cohort partition allocation (Section 4).
PARTITIONS = {
    'fit':  [(m, '2013J') for m in ['AAA', 'BBB', 'DDD', 'EEE', 'FFF', 'GGG']],
    'pcal': [(m, '2013B') for m in ['BBB', 'DDD', 'FFF']] + [(m, '2014B') for m in ['EEE', 'GGG']],
    'ccal': [(m, '2014B') for m in ['BBB', 'CCC', 'DDD', 'FFF']],
    'test': [(m, '2014J') for m in ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']],
}
PRIORITY = ['test', 'ccal', 'pcal', 'fit']          # highest priority first
PART_ORDER = ['fit', 'pcal', 'ccal', 'test']

# VLE activity-type grouping (Section 6). Anything not listed is reported as unmapped.
ACTIVITY_GROUPS = {
    'homepage': ['homepage'],
    'forum': ['forumng', 'oucollaborate', 'ouelluminate', 'ouwiki'],
    'quiz': ['quiz', 'externalquiz', 'questionnaire'],
    'content': ['resource', 'oucontent', 'url', 'subpage', 'page', 'glossary', 'dataplus',
                'dualpane', 'folder', 'htmlactivity', 'repeatactivity', 'sharedsubpage'],
}
TYPE_TO_GROUP = {t: g for g, ts in ACTIVITY_GROUPS.items() for t in ts}
VLE_GROUPS = list(ACTIVITY_GROUPS)

# Assessment statuses (Section 5), in the order they are assigned.
STATUSES = ['banked', 'due_before_registration', 'not_yet_due', 'submitted_early',
            'submitted_on_time', 'submitted_late', 'overdue']

# Expected values (Section 9). The script halts before writing outputs if any differ.
EXPECTED = {
    'enrolments': 32593, 'cohorts': 22,
    'eligible_T14': 28054, 'eligible_T28': 27515, 'eligible_T42': 27005,
    'left_on_or_before_day0': 3097, 'withdrawn_before_day14': 1377,
    'never_eligible': 4497, 'removed_overlap': 2410,
    'partition_fit': 6485, 'partition_pcal': 5079, 'partition_ccal': 4769, 'partition_test': 9353,
    'students_shared_after_allocation': 0,
    'identical_vle_rows_to_day42': 229745, 'banked_records': 1909,
    'missing_scores': 173, 'missing_imd_band': 1111,
    'unreachable_enrolments': 1377,
}
observed = {}


def find(name):
    """Return the path of an input file, accepting either name.csv or name.csv.zip."""
    for cand in (name, name + '.zip'):
        p = os.path.join(HERE, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


# ---------------------------------------------------------------------------------------------
# 1. Load inputs
# ---------------------------------------------------------------------------------------------
print('Loading input files ...')
info = pd.read_csv(find('studentInfo.csv'))
reg = pd.read_csv(find('studentRegistration.csv'))
sa = pd.read_csv(find('studentAssessment.csv'))
assess = pd.read_csv(find('assessments.csv'))
vle_sites = pd.read_csv(find('vle.csv'))
courses = pd.read_csv(find('courses.csv'))     # loaded for completeness; module length is NOT used

# studentVle may be one file or split into studentVle_0 ... studentVle_7. Concatenate the parts
# and drop any unnamed index column written by the splitter.
vle_parts = sorted(glob.glob(os.path.join(HERE, 'studentVle_*.csv*')))
if not vle_parts:
    vle_parts = [find('studentVle.csv')]
svle_dtypes = {'id_student': 'int32', 'id_site': 'int32', 'date': 'int16', 'sum_click': 'int32'}
frames = []
for p in vle_parts:
    f = pd.read_csv(p, dtype=svle_dtypes)
    frames.append(f.loc[:, ~f.columns.str.startswith('Unnamed')])
svle_all = pd.concat(frames, ignore_index=True)
del frames
n_vle_total = len(svle_all)
print(f'  studentVle: {len(vle_parts)} file(s), {n_vle_total:,} rows')

# Only records up to the last checkpoint can ever be used (leakage rule), so the VLE table is
# cut to date <= 42 here; all VLE integrity checks are reported "to Day 42".
svle = svle_all[svle_all['date'] <= MAX_T].copy()
del svle_all

# ---------------------------------------------------------------------------------------------
# 2. Integrity checks (reported only; nothing is deleted or altered because of them)
# ---------------------------------------------------------------------------------------------
checks = []


def check(name, value, note=''):
    checks.append({'check': name, 'count': int(value), 'note': note})


check('Duplicate enrolment keys in studentInfo', info.duplicated(KEY).sum())
check('Duplicate enrolment keys in studentRegistration', reg.duplicated(KEY).sum())
k_info = set(map(tuple, info[KEY].values))
k_reg = set(map(tuple, reg[KEY].values))
check('Enrolments in studentInfo missing from studentRegistration', len(k_info - k_reg))
check('Enrolments in studentRegistration missing from studentInfo', len(k_reg - k_info))
check('Duplicate (id_assessment, id_student) keys in studentAssessment',
      sa.duplicated(['id_assessment', 'id_student']).sum())
check('Submissions with an unknown id_assessment',
      (~sa['id_assessment'].isin(assess['id_assessment'])).sum())

site_keys = vle_sites[['id_site'] + COHORT].drop_duplicates()
svle = svle.merge(site_keys.assign(_site_ok=True), on=['id_site'] + COHORT, how='left')
check('VLE rows (to Day 42) with an unknown site', svle['_site_ok'].isna().sum(),
      'id_site + module + presentation not found in vle.csv')
svle = svle.merge(info[KEY].assign(_enr_ok=True), on=KEY, how='left')
check('VLE rows (to Day 42) with no matching enrolment', svle['_enr_ok'].isna().sum())
check('VLE rows (to Day 42) with clicks <= 0', (svle['sum_click'] <= 0).sum())
svle_cols = ['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click']
n_identical = int(svle.duplicated(svle_cols, keep='first').sum())
check('Identical VLE rows (to Day 42)', n_identical,
      'extra copies beyond the first; kept, as they may be separate sessions')
check('Missing imd_band values (studentInfo)', info['imd_band'].isna().sum())
check('Missing scores (studentAssessment)', sa['score'].isna().sum())
check('Banked records (studentAssessment, is_banked = 1)', (sa['is_banked'] == 1).sum())

enr = info.merge(reg, on=KEY, how='left', validate='one_to_one')
wd = enr['final_result'] == 'Withdrawn'
check('Withdrawn records with no unregistration date (all)',
      (wd & enr['date_unregistration'].isna()).sum())
check('Withdrawn records with no unregistration date (2014J)',
      (wd & enr['date_unregistration'].isna() & (enr['code_presentation'] == '2014J')).sum())
fl = enr['final_result'] == 'Fail'
check('Fail records with an unregistration date', (fl & enr['date_unregistration'].notna()).sum())
check('Fail records with an unregistration date on or before Day 0',
      (fl & (enr['date_unregistration'] <= 0)).sum())
check('Enrolments with missing date_registration', enr['date_registration'].isna().sum())
check('Total studentVle rows (all dates)', n_vle_total)
check('studentVle rows to Day 42', len(svle))

observed['identical_vle_rows_to_day42'] = n_identical
observed['banked_records'] = int((sa['is_banked'] == 1).sum())
observed['missing_scores'] = int(sa['score'].isna().sum())
observed['missing_imd_band'] = int(info['imd_band'].isna().sum())

# ---------------------------------------------------------------------------------------------
# 3. Enrolments, labels and eligibility status at each checkpoint
# ---------------------------------------------------------------------------------------------
enr['y'] = enr['final_result'].map({'Fail': 1, 'Withdrawn': 1, 'Pass': 0, 'Distinction': 0})
assert enr['y'].notna().all(), 'Unexpected final_result value'
enr['y'] = enr['y'].astype(int)
# Stable enrolment ID, identical in every checkpoint file.
enr['enrolment_id'] = (enr['code_module'] + '_' + enr['code_presentation'] + '_'
                       + enr['id_student'].astype(str))
enr['withdrawn_undated'] = (wd & enr['date_unregistration'].isna()).astype(int)
observed['enrolments'] = len(enr)
observed['cohorts'] = enr[COHORT].drop_duplicates().shape[0]

dr, du = enr['date_registration'], enr['date_unregistration']
status_rows = []
STATUS_ORDER = ['left_on_or_before_day0', 'withdrawn_by_checkpoint', 'not_yet_registered',
                'registration_date_missing', 'eligible']
for T in CHECKPOINTS:
    # Mutually exclusive categories, assigned in the order given in the brief.
    status = np.select([du <= 0, du <= T, dr > T, dr.isna()], STATUS_ORDER[:4], default='eligible')
    enr[f'elig_T{T}'] = status == 'eligible'
    counts = pd.Series(status).value_counts()
    for s in STATUS_ORDER:
        status_rows.append({'checkpoint': T, 'status': s, 'enrolments': int(counts.get(s, 0))})
    status_rows.append({'checkpoint': T, 'status': 'TOTAL', 'enrolments': int(counts.sum())})
    assert counts.sum() == len(enr)
    observed[f'eligible_T{T}'] = int(counts.get('eligible', 0))
A5 = pd.DataFrame(status_rows)

observed['left_on_or_before_day0'] = int((du <= 0).sum())
observed['withdrawn_before_day14'] = int(((du > 0) & (du <= 14)).sum())

# First checkpoint at which the enrolment is eligible (NaN if never eligible).
enr['first_eligible_checkpoint'] = np.nan
for T in reversed(CHECKPOINTS):
    enr.loc[enr[f'elig_T{T}'], 'first_eligible_checkpoint'] = T

# ---------------------------------------------------------------------------------------------
# 4. Partitions
# ---------------------------------------------------------------------------------------------
cohort_to_part = {c: p for p, cs in PARTITIONS.items() for c in cs}
enr['partition'] = [cohort_to_part.get((m, pr)) for m, pr in zip(enr.code_module, enr.code_presentation)]
assert enr['partition'].notna().all(), 'Some cohort is not assigned to a partition'

# Unreachable enrolments: withdraw between Day 1 and Day 14 (before the first checkpoint).
FAIR = ['disability', 'age_band']
unreach = enr[(du > 0) & (du <= 14)]
observed['unreachable_enrolments'] = len(unreach)

# Step 1: remove enrolments never eligible at any checkpoint, recording a reason for each.
enr['ever_eligible'] = enr[[f'elig_T{T}' for T in CHECKPOINTS]].any(axis=1)
observed['never_eligible'] = int((~enr['ever_eligible']).sum())
never = enr[~enr['ever_eligible']].copy()
nd_r, nd_u = never['date_registration'], never['date_unregistration']
# Reasons follow the status-table order. The brief lists three reasons; two more are needed to
# cover every never-eligible enrolment (see summary).
never['reason'] = np.select(
    [nd_u <= 0, (nd_u > 0) & (nd_u <= 14), nd_r > MAX_T, nd_r.isna()],
    ['left_on_or_before_day0', 'withdrawn_day1_to_day14', 'registered_after_day42',
     'registration_date_missing'],
    default='withdrew_after_late_registration_before_next_checkpoint')
pool = enr[enr['ever_eligible']].copy()


def overlap_table(df, label):
    """Number of students appearing in each pair of partitions."""
    sets = {p: set(df.loc[df.partition == p, 'id_student']) for p in PART_ORDER}
    rows = []
    for i, a in enumerate(PART_ORDER):
        for b in PART_ORDER[i + 1:]:
            rows.append({'stage': label, 'partition_a': a, 'partition_b': b,
                         'shared_students': len(sets[a] & sets[b])})
    return pd.DataFrame(rows)


A4 = overlap_table(pool, 'before allocation (after removing never-eligible)')

# Step 2: student-overlap rule. Each student keeps enrolments only in their highest-priority
# partition; enrolments in lower-priority partitions are removed (and saved for reinstatement).
rank = {p: i for i, p in enumerate(PRIORITY)}           # 0 = highest priority
pool['_rank'] = pool['partition'].map(rank)
best = pool.groupby('id_student')['_rank'].transform('min')
pool['removed_overlap'] = pool['_rank'] != best
pool['kept_in_partition'] = best.map({v: k for k, v in rank.items()})
removed = pool[pool['removed_overlap']]
alloc = pool[~pool['removed_overlap']]
observed['removed_overlap'] = len(removed)

# Step 3: confirm no student is shared between any two partitions.
after = overlap_table(alloc, 'after allocation')
observed['students_shared_after_allocation'] = int(after['shared_students'].sum())
A4 = pd.concat([A4, after], ignore_index=True)
for p in PART_ORDER:
    observed[f'partition_{p}'] = int((alloc.partition == p).sum())

# ---------------------------------------------------------------------------------------------
# 5. Assessments: status and score visibility per eligible enrolment per checkpoint
# ---------------------------------------------------------------------------------------------
assess_ne = assess[assess['assessment_type'] != 'Exam'].rename(columns={'date': 'due_date'})
assert assess_ne['due_date'].notna().all()
assert ((assess['assessment_type'] == 'Exam') & (assess['date'] <= MAX_T)).sum() == 0, \
    'An exam falls within the first 42 days'
sub = sa.merge(assess_ne[['id_assessment']], on='id_assessment', how='inner')
leak = {}     # per-checkpoint maxima of dates actually used, for the leakage self-checks


def visible_mask(g, T, delays):
    """Score visible at T: non-banked, submitted by T, score present, and
    max(date_submitted, due_date) + delay <= T."""
    release = np.maximum(g['date_submitted'], g['due_date']) + g['assessment_type'].map(delays)
    return g['submitted'] & g['score'].notna() & (release <= T)


def assessment_grid(T, e):
    """One row per (eligible enrolment at T) x (non-exam assessment of its cohort),
    with any matching submission attached."""
    g = e[KEY + ['date_registration']].merge(
        assess_ne[COHORT + ['id_assessment', 'assessment_type', 'due_date']], on=COHORT)
    g = g.merge(sub[['id_assessment', 'id_student', 'date_submitted', 'is_banked', 'score']],
                on=['id_assessment', 'id_student'], how='left')
    banked = g['is_banked'] == 1
    # A submission only counts if it happened by Day T (leakage rule). Banked rows are excluded
    # from all timing and score features.
    g['submitted'] = (~banked) & g['date_submitted'].notna() & (g['date_submitted'] <= T)
    s, due_by_T = g['submitted'], g['due_date'] <= T
    g['status'] = np.select(
        [banked,
         (~s) & (g['due_date'] < g['date_registration']),
         (~due_by_T) & (~s),
         s & (~due_by_T),
         due_by_T & s & (g['date_submitted'] <= g['due_date']),
         due_by_T & s & (g['date_submitted'] > g['due_date']),
         due_by_T & (~s)],
        STATUSES, default='UNASSIGNED')
    assert (g['status'] != 'UNASSIGNED').all()
    for name, d in DELAYS.items():
        g[f'vis_{name}'] = visible_mask(g, T, d)
    leak[T]['max_submission_date_used'] = g.loc[g['submitted'], 'date_submitted'].max()
    return g


def score_features(g, name):
    """n_submitted_mark_pending, n_scores_available, mean_score_available under one delay."""
    v = g[f'vis_{name}']
    out = pd.DataFrame({
        'n_submitted_mark_pending': (g['submitted'] & ~v).groupby([g[k] for k in KEY]).sum(),
        'n_scores_available': v.groupby([g[k] for k in KEY]).sum(),
    })
    out['mean_score_available'] = g[v].groupby(KEY)['score'].mean()  # missing if none
    return out


# ---------------------------------------------------------------------------------------------
# 6. VLE activity
# ---------------------------------------------------------------------------------------------
svle = svle.merge(vle_sites[['id_site'] + COHORT + ['activity_type']], on=['id_site'] + COHORT,
                  how='left')
unmapped = sorted(set(svle['activity_type'].dropna()) - set(TYPE_TO_GROUP))
svle['group'] = svle['activity_type'].map(TYPE_TO_GROUP)
n_unknown_group = int(svle['group'].isna().sum())
svle = svle[KEY + ['date', 'sum_click', 'group']]


def vle_features(T, keys):
    v = svle[svle['date'] <= T].merge(keys, on=KEY)           # only this checkpoint's enrolments
    leak[T]['max_vle_date_used'] = v['date'].max()
    out = keys.copy()
    tot = v.pivot_table(index=KEY, columns='group', values='sum_click', aggfunc='sum')
    tot = tot.reindex(columns=VLE_GROUPS).add_prefix('clicks_total_')
    l7 = v[v['date'] >= T - 6].pivot_table(index=KEY, columns='group', values='sum_click',
                                           aggfunc='sum')
    l7 = l7.reindex(columns=VLE_GROUPS).add_prefix('clicks_last7_')
    g = v.groupby(KEY)
    other = pd.DataFrame({
        'active_days': g['date'].nunique(),
        'last_date': g['date'].max(),
        'prestart_clicks': v[v['date'] < 0].groupby(KEY)['sum_click'].sum(),
    })
    out = out.merge(tot, on=KEY, how='left').merge(l7, on=KEY, how='left') \
             .merge(other, on=KEY, how='left')
    # Rule: no VLE rows up to Day T -> click counts are 0 (this is a count, not imputation).
    zero_fill = [c for c in out.columns if c.startswith('clicks_')] + ['active_days',
                                                                       'prestart_clicks']
    out[zero_fill] = out[zero_fill].fillna(0).astype('int64')
    out['days_since_last_activity'] = T - out.pop('last_date')     # missing if no activity
    return out


# ---------------------------------------------------------------------------------------------
# 7. Assemble one dataset per checkpoint
#    Features are built for every ever-eligible enrolment (including the ones later removed by
#    the overlap rule), so the overlap-reinstated files share exactly the same feature values.
# ---------------------------------------------------------------------------------------------
META = ['meta_enrolment_id', 'meta_id_student', 'meta_code_module', 'meta_code_presentation',
        'meta_partition', 'meta_checkpoint', 'meta_y', 'meta_final_result',
        'meta_date_unregistration', 'meta_first_eligible_checkpoint', 'meta_withdrawn_undated',
        'meta_disability', 'meta_age_band']
REG_F = ['days_registered_before_start', 'days_enrolled', 'num_of_prev_attempts', 'studied_credits']
DEMOG = ['age_band', 'gender', 'disability', 'highest_education', 'imd_band', 'region']

full = {name: {} for name in DELAYS}       # full[delay][T] = dataset on the whole pool
a6_rows, a7_rows = [], []
for T in CHECKPOINTS:
    leak[T] = {}
    e = pool[pool[f'elig_T{T}']].copy()
    m = pd.DataFrame({
        'meta_enrolment_id': e['enrolment_id'], 'meta_id_student': e['id_student'],
        'meta_code_module': e['code_module'], 'meta_code_presentation': e['code_presentation'],
        'meta_partition': e['partition'], 'meta_checkpoint': T, 'meta_y': e['y'],
        'meta_final_result': e['final_result'], 'meta_date_unregistration': e['date_unregistration'],
        'meta_first_eligible_checkpoint': e['first_eligible_checkpoint'].astype(int),
        'meta_withdrawn_undated': e['withdrawn_undated'],
        'meta_disability': e['disability'], 'meta_age_band': e['age_band'],
    })
    f = e[KEY].copy()
    f['days_registered_before_start'] = e['date_registration']
    f['days_enrolled'] = T - np.maximum(e['date_registration'], 0)
    for c in ['num_of_prev_attempts', 'studied_credits'] + DEMOG:
        f[c] = e[c]

    g = assessment_grid(T, e)
    st = g.pivot_table(index=KEY, columns='status', values='id_assessment', aggfunc='count')
    st = st.reindex(columns=STATUSES).fillna(0).astype(int).add_prefix('n_')

    base = pd.concat([m, f], axis=1).merge(vle_features(T, e[KEY]), on=KEY, how='left') \
                                    .merge(st.reset_index(), on=KEY, how='left')
    n_cols = [f'n_{s}' for s in STATUSES]
    base[n_cols] = base[n_cols].fillna(0).astype(int)      # cohorts with no assessments: 0 counts
    for name in DELAYS:
        d = base.merge(score_features(g, name).reset_index(), on=KEY, how='left')
        d[SCORE_COLS[:2]] = d[SCORE_COLS[:2]].fillna(0).astype(int)
        d = d.drop(columns=KEY)
        for c in DEMOG + ['meta_disability', 'meta_age_band']:
            d[c] = d[c].astype('string')                    # missing stays <NA>; no imputation
        assert len(d) == len(e) and d['meta_enrolment_id'].is_unique
        full[name][T] = d

    # Appendix A6 / A7 rows (allocated enrolments only, per partition and overall)
    ga = g.merge(e.loc[~e['removed_overlap'], KEY + ['partition']], on=KEY, how='inner')
    for p in PART_ORDER + ['ALL']:
        sg = ga if p == 'ALL' else ga[ga['partition'] == p]
        for name in DELAYS:
            a6_rows.append({'checkpoint': T, 'delay_setting': DELAY_LABEL[name], 'partition': p,
                            'visible_scores': int(sg[f'vis_{name}'].sum()),
                            'submitted_by_T_non_banked': int(sg['submitted'].sum())})
        vc = sg['status'].value_counts()
        row = {'checkpoint': T, 'partition': p}
        row.update({s: int(vc.get(s, 0)) for s in STATUSES})
        row['submitted_mark_not_yet_available'] = int((sg['submitted'] & ~sg['vis_primary']).sum())
        row['total_assessment_records'] = len(sg)
        a7_rows.append(row)

FEATURES = [c for c in full['primary'][14].columns if not c.startswith('meta_')]
removed_ids = set(removed['enrolment_id'])


def allocated(d):
    return d[~d['meta_enrolment_id'].isin(removed_ids)].reset_index(drop=True)


primary = {T: allocated(full['primary'][T]) for T in CHECKPOINTS}

# ---------------------------------------------------------------------------------------------
# 8. Leakage self-checks (Section 9)
# ---------------------------------------------------------------------------------------------
leak_rows = []


def selfcheck(name, ok, detail=''):
    leak_rows.append({'check': name, 'result': 'pass' if ok else 'fail', 'detail': detail})


# (1) Maximum VLE date and submission date used in any feature <= T.
for T in CHECKPOINTS:
    mv, ms = leak[T]['max_vle_date_used'], leak[T]['max_submission_date_used']
    selfcheck(f'T{T}: max VLE date and submission date used <= T',
              mv <= T and (pd.isna(ms) or ms <= T), f'max VLE date {mv}, max submission date {ms}')

# (2) No available score has max(date_submitted, due_date) + delay > T. Recomputed independently
#     from the raw tables and compared with n_scores_available in every file and delay setting.
raw = sa[sa['is_banked'] == 0].merge(assess_ne, on='id_assessment')
raw['enrolment_id'] = raw['code_module'] + '_' + raw['code_presentation'] + '_' + \
    raw['id_student'].astype(str)
ok2, worst = True, []
for name, dl in DELAYS.items():
    for T in CHECKPOINTS:
        rel = np.maximum(raw['date_submitted'], raw['due_date']) + raw['assessment_type'].map(dl)
        ok_rows = raw[(rel <= T) & (raw['date_submitted'] <= T) & raw['score'].notna()]
        expect = ok_rows.groupby('enrolment_id').size()
        d = full[name][T].set_index('meta_enrolment_id')['n_scores_available']
        same = d.eq(expect.reindex(d.index).fillna(0).astype(int)).all()
        ok2 &= bool(same)
        worst.append(f'{name}/T{T}: {"match" if same else "MISMATCH"}')
selfcheck('No available score has max(date_submitted, due_date) + delay > T', ok2,
          'independent recount of n_scores_available; ' + '; '.join(worst))

# (3) No eligible enrolment has date_unregistration <= T.
for T in CHECKPOINTS:
    bad = (full['primary'][T]['meta_date_unregistration'] <= T).sum()
    selfcheck(f'T{T}: no eligible enrolment has date_unregistration <= T', bad == 0,
              f'{bad} violations')

# (4) meta_enrolment_id unique per file; later enrolments appear earlier or are late registrants.
reg_day = enr.set_index('enrolment_id')['date_registration']
for T in CHECKPOINTS:
    selfcheck(f'T{T}: meta_enrolment_id unique', primary[T]['meta_enrolment_id'].is_unique)
for i, T in enumerate(CHECKPOINTS[1:], start=1):
    earlier = set().union(*[set(primary[t]['meta_enrolment_id']) for t in CHECKPOINTS[:i]])
    new = primary[T].loc[~primary[T]['meta_enrolment_id'].isin(earlier), 'meta_enrolment_id']
    prev_T = CHECKPOINTS[i - 1]
    not_late = (reg_day.loc[new] <= prev_T).sum()
    selfcheck(f'T{T}: every enrolment appears earlier or is a late registrant', not_late == 0,
              f'{len(new)} new at T{T}, all registered after Day {prev_T}' if not_late == 0
              else f'{not_late} new enrolments registered by Day {prev_T}')

# (5) No meta_ column, final_result or date_unregistration among the features.
banned = [c for c in FEATURES if c.startswith('meta_') or c in
          ('final_result', 'date_unregistration', 'module_presentation_length', 'y')]
selfcheck('No meta_, final_result, date_unregistration or module length among features',
          not banned, f'feature columns: {len(FEATURES)}; offending: {banned}')

# Extra: sensitivity and reinstated files differ from primary only where intended.
for name in ['delay_7_0', 'delay_21_7']:
    same = all(full[name][T].drop(columns=SCORE_COLS).equals(full['primary'][T].drop(columns=SCORE_COLS))
               for T in CHECKPOINTS)
    selfcheck(f'{name} files identical to primary except the 3 score columns', same)

# ---------------------------------------------------------------------------------------------
# 9. Compare with expected values; halt before writing outputs if anything fails
# ---------------------------------------------------------------------------------------------
print('\nExpected-value check:')
mismatch = False
exp_rows = []
for k, v in EXPECTED.items():
    ok = observed.get(k) == v
    mismatch |= not ok
    exp_rows.append({'item': k, 'expected': v, 'observed': observed.get(k), 'match': ok})
    print(f"  {'✓' if ok else '✗'} {k:<36} expected {v:>8,}  observed {observed.get(k):>8,}")
print('\nLeakage self-checks:')
for r in leak_rows:
    print(f"  {r['result'].upper():<5} {r['check']}  [{r['detail']}]")
    mismatch |= r['result'] != 'pass'
print(f'\nUnmapped activity types: {unmapped if unmapped else "none"}')
print(f'VLE rows with no activity type (unknown site): {n_unknown_group}')
print('\nNever-eligible reasons:\n' + never['reason'].value_counts().to_string())
pd.DataFrame(exp_rows).to_csv(os.path.join(OUT, 'expected_value_check.csv'), index=False)
pd.DataFrame(leak_rows).to_csv(os.path.join(OUT, 'leakage_self_checks.csv'), index=False)
if mismatch:
    print('\nSTOP: an expected value or self-check failed. Datasets not written.')
    sys.exit(1)

# ---------------------------------------------------------------------------------------------
# 10. Write outputs
# ---------------------------------------------------------------------------------------------
for T in CHECKPOINTS:
    primary[T].to_parquet(os.path.join(OUT, f'checkpoint_T{T}.parquet'), index=False)
    for name in ['delay_7_0', 'delay_21_7']:
        allocated(full[name][T]).to_parquet(os.path.join(OUT, f'checkpoint_T{T}_{name}.parquet'),
                                            index=False)
    # Overlap-reinstated: the removed enrolments are back in their original (cohort) partition.
    full['primary'][T].to_parquet(os.path.join(OUT, f'checkpoint_T{T}_overlap_reinstated.parquet'),
                                  index=False)

removed[['enrolment_id'] + KEY + ['partition', 'kept_in_partition', 'y', 'final_result',
                                  'date_registration', 'date_unregistration',
                                  'first_eligible_checkpoint'] + FAIR] \
    .rename(columns={'partition': 'removed_from_partition'}) \
    .to_csv(os.path.join(OUT, 'removed_overlap_enrolments.csv'), index=False)
unreach[['enrolment_id'] + KEY + ['partition', 'y', 'final_result', 'date_registration',
                                  'date_unregistration', 'withdrawn_undated'] + FAIR] \
    .rename(columns={'partition': 'cohort_partition'}) \
    .to_csv(os.path.join(OUT, 'unreachable_enrolments.csv'), index=False)
never[['enrolment_id'] + KEY + ['partition', 'reason', 'y', 'final_result', 'date_registration',
                                'date_unregistration'] + FAIR] \
    .rename(columns={'partition': 'cohort_partition'}) \
    .to_csv(os.path.join(OUT, 'never_eligible_enrolments.csv'), index=False)

# A1: allocation per partition
A1 = alloc.groupby('partition').agg(enrolments=('id_student', 'size'),
                                    students=('id_student', 'nunique'),
                                    at_risk=('y', 'sum')).reindex(PART_ORDER)
A1['not_at_risk'] = A1['enrolments'] - A1['at_risk']
A1['cohorts'] = [len(PARTITIONS[p]) for p in PART_ORDER]
A1.loc['TOTAL'] = A1.sum()
A1.loc['TOTAL', 'students'] = alloc['id_student'].nunique()
A1.reset_index().to_csv(os.path.join(OUT, 'A1_allocation.csv'), index=False)

# A2: class balance per partition and checkpoint
a2 = []
for T, d in primary.items():
    for p in PART_ORDER + ['ALL']:
        s = d if p == 'ALL' else d[d.meta_partition == p]
        a2.append({'checkpoint': T, 'partition': p, 'records': len(s),
                   'at_risk': int(s.meta_y.sum()), 'not_at_risk': int((s.meta_y == 0).sum()),
                   'at_risk_rate': round(s.meta_y.mean(), 4)})
pd.DataFrame(a2).to_csv(os.path.join(OUT, 'A2_class_balance.csv'), index=False)

# A3: module by partition (allocated, ever-eligible enrolments)
A3 = alloc.pivot_table(index='code_module', columns='partition', values='id_student',
                       aggfunc='size', fill_value=0).reindex(columns=PART_ORDER, fill_value=0)
A3['TOTAL'] = A3.sum(axis=1)
A3.loc['TOTAL'] = A3.sum()
A3.reset_index().to_csv(os.path.join(OUT, 'A3_module_by_partition.csv'), index=False)

A4.to_csv(os.path.join(OUT, 'A4_student_overlap.csv'), index=False)
A5.to_csv(os.path.join(OUT, 'A5_eligibility_status.csv'), index=False)
pd.DataFrame(a6_rows).to_csv(os.path.join(OUT, 'A6_score_visibility_sensitivity.csv'), index=False)
pd.DataFrame(a7_rows).to_csv(os.path.join(OUT, 'A7_assessment_status.csv'), index=False)
pd.DataFrame(checks).to_csv(os.path.join(OUT, 'A8_integrity_checks.csv'), index=False)

# Data dictionary (every column) and A9 (features only)
SRC_ASSESS = 'studentAssessment + assessments'
status_def = {
    'banked': 'is_banked = 1 (excluded from timing and score features)',
    'due_before_registration': 'Due before date_registration and not submitted by T (not missed)',
    'not_yet_due': 'Due after T and not submitted by T',
    'submitted_early': 'Submitted by T and due after T',
    'submitted_on_time': 'Due by T and submitted on or before the due date',
    'submitted_late': 'Due by T, submitted after the due date but by T',
    'overdue': 'Due by T and not submitted by T (even if submitted later)',
}
DEFS = {
    'meta_enrolment_id': ('derived', 'code_module_code_presentation_id_student; identical in all checkpoint files'),
    'meta_id_student': ('studentInfo', 'Student ID (linking, bootstrap by student)'),
    'meta_code_module': ('studentInfo', 'Module code'),
    'meta_code_presentation': ('studentInfo', 'Presentation code'),
    'meta_partition': ('derived', 'fit / pcal / ccal / test (whole-cohort allocation)'),
    'meta_checkpoint': ('derived', 'Checkpoint day T (14, 28, 42)'),
    'meta_y': ('studentInfo.final_result', 'Label: 1 = Fail or Withdrawn (At risk); 0 = Pass or Distinction'),
    'meta_final_result': ('studentInfo', 'Original outcome (never a feature)'),
    'meta_date_unregistration': ('studentRegistration', 'Unregistration day; for withdrawal reporting only, never a feature'),
    'meta_first_eligible_checkpoint': ('derived', 'First of 14 / 28 / 42 at which the enrolment is eligible'),
    'meta_withdrawn_undated': ('derived', '1 if final_result = Withdrawn and no unregistration date'),
    'meta_disability': ('studentInfo', 'Fairness-audit group (copy of disability)'),
    'meta_age_band': ('studentInfo', 'Fairness-audit group (copy of age_band)'),
    'days_registered_before_start': ('studentRegistration', 'date_registration (days relative to Day 0; negative = before start)'),
    'days_enrolled': ('studentRegistration', 'T - max(date_registration, 0)'),
    'num_of_prev_attempts': ('studentInfo', 'Previous attempts at this module'),
    'studied_credits': ('studentInfo', 'Credits being studied'),
    'active_days': ('studentVle', 'Distinct days with any VLE row, date <= T (incl. pre-start)'),
    'prestart_clicks': ('studentVle', 'Sum of clicks with date < 0'),
    'days_since_last_activity': ('studentVle', 'T - latest VLE date <= T; missing if no activity'),
    'n_submitted_mark_pending': (SRC_ASSESS, 'Submitted by T (non-banked) but score not visible at T; overlaps submitted statuses'),
    'n_scores_available': (SRC_ASSESS, 'Non-banked scores with max(date_submitted, due) + delay <= T (TMA 14, CMA 7)'),
    'mean_score_available': (SRC_ASSESS, 'Unweighted mean of visible scores; missing if none'),
}
for c in DEMOG:
    DEFS[c] = ('studentInfo', f'{c} (categorical; missing kept as missing)')
for grp in VLE_GROUPS:
    DEFS[f'clicks_total_{grp}'] = ('studentVle + vle', f'Sum of clicks on {grp} activities, date <= T (incl. pre-start); 0 if none')
    DEFS[f'clicks_last7_{grp}'] = ('studentVle + vle', f'Sum of clicks on {grp} activities, T-6 <= date <= T; 0 if none')
for s in STATUSES:
    DEFS[f'n_{s}'] = (SRC_ASSESS, f'Count of TMA/CMA assessments with status at T: {status_def[s]}')

cols = list(primary[14].columns)
assert set(cols) == set(DEFS), set(cols) ^ set(DEFS)


def earliest(c, cond):
    for T in CHECKPOINTS:
        if cond(primary[T][c]).any():
            return T
    return 'never'


dd = pd.DataFrame([{
    'column': c,
    'role': 'metadata (not a predictor)' if c.startswith('meta_') else 'feature',
    'feature_group': ('metadata' if c.startswith('meta_') else
                      'registration' if c in REG_F else 'demographic' if c in DEMOG else
                      'assessment' if c.startswith(('n_', 'mean_')) else 'VLE'),
    'source_table': DEFS[c][0], 'definition': DEFS[c][1],
    'dtype': str(primary[14][c].dtype),
    'earliest_checkpoint_non_missing': earliest(c, lambda s: s.notna()),
    'earliest_checkpoint_non_zero': earliest(c, lambda s: s.notna() & (s.astype(str) != '0')),
    'missing_T14': int(primary[14][c].isna().sum()),
    'missing_T28': int(primary[28][c].isna().sum()),
    'missing_T42': int(primary[42][c].isna().sum()),
} for c in cols])
dd.to_csv(os.path.join(OUT, 'data_dictionary.csv'), index=False)
dd[dd['role'] == 'feature'].drop(columns='role') \
    .to_csv(os.path.join(OUT, 'A9_feature_list.csv'), index=False)

print('\nAll expected values match and all self-checks pass. Outputs written to', OUT)
for T in CHECKPOINTS:
    print(f'  checkpoint_T{T}.parquet: {primary[T].shape[0]:,} rows x {primary[T].shape[1]} cols '
          f'({len(META)} meta + {len(FEATURES)} features); reinstated: {len(full["primary"][T]):,} rows')
