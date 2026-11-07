"""
OULAD Data Preparation: Leakage-Safe Checkpoint Datasets (Days 14, 28, 42)
Implements Section 3.1 of the methodology. Supersedes the data steps of oulad_temporal_pipeline.py.

Key Conventions:
  1. Cutoff: a checkpoint T means the END of day T, so day T itself is included everywhere (<= T).
  2. Eligibility at T: registered by T, date_unregistration absent or later than T.
  3. Label: at_risk = 1 for Fail/Withdrawn, 0 for Pass/Distinction.
  4. Each of the 22 module presentations belongs to one partition (see COHORT_ALLOCATION), and
     all of a student's rows sit in one partition. Priority on overlap is
     test > conformal_cal > prob_cal > fit, so the test cohort is never reduced.
  5. Score availability: max(date_submitted, due date) + 14 days (TMA) or + 7 days (CMA).

Outputs (written to ./processed):
  checkpoint_dataset.csv    One row per eligible enrolment per checkpoint.
  enrolment_flow.csv        One row per enrolment, with its status at each checkpoint.
  table_*.csv               Appendix tables.
"""

import os
import zipfile
import numpy as np
import pandas as pd
from typing import Dict

CHECKPOINTS = {14: 'Week 2', 28: 'Week 4', 42: 'Week 6'}
KEY = ['code_module', 'code_presentation', 'id_student']
SCORE_DELAY_DAYS = {'TMA': 14, 'CMA': 7}
OUTPUT_DIR = 'processed'

# Chosen by enumerating whole-cohort allocations without reading any 2014J outcome.
COHORT_ALLOCATION = {
    'fit': [(m, '2013J') for m in ['AAA', 'BBB', 'DDD', 'EEE', 'FFF', 'GGG']],
    'prob_cal': [(m, '2013B') for m in ['BBB', 'DDD', 'FFF']] + [(m, '2014B') for m in ['EEE', 'GGG']],
    'conformal_cal': [(m, '2014B') for m in ['BBB', 'CCC', 'DDD', 'FFF']],
    'test': [(m, '2014J') for m in ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']],
}

# Grouped because several raw types exist in one module or one presentation only,
# which would let the model recognise a module rather than a behaviour.
ACTIVITY_GROUPS = {
    'homepage': 'homepage',
    'forumng': 'forum', 'ouwiki': 'forum', 'oucollaborate': 'forum', 'ouelluminate': 'forum',
    'quiz': 'quiz', 'externalquiz': 'quiz', 'questionnaire': 'quiz',
    'oucontent': 'content', 'resource': 'content', 'subpage': 'content', 'url': 'content',
    'page': 'content', 'glossary': 'content', 'dualpane': 'content', 'dataplus': 'content',
    'htmlactivity': 'content', 'sharedsubpage': 'content', 'folder': 'content',
    'repeatactivity': 'content',
}

CATEGORICAL_FEATURES = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
NUMERIC_FEATURES = [
    'num_of_prev_attempts', 'studied_credits', 'registration_day', 'late_registered',
    'days_enrolled', 'total_clicks', 'pre_start_clicks', 'active_days', 'days_since_last_activity',
    'clicks_homepage', 'clicks_forum', 'clicks_quiz', 'clicks_content',
    'assessments_due', 'assessments_overdue', 'assessments_on_time', 'assessments_late',
    'assessments_early', 'banked_assessments', 'mean_submission_delay',
    'scores_available', 'mean_available_score',
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
IDENTIFIERS = ['id_student', 'code_module', 'code_presentation']


def load_table(name: str, base_dir: str = ".") -> pd.DataFrame:
    """Helper to load a CSV file directly or from a .zip archive."""
    csv_path = os.path.join(base_dir, f"{name}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    with zipfile.ZipFile(f"{csv_path}.zip") as z:
        return pd.read_csv(z.open(f"{name}.csv"))


def load_raw_tables(base_dir: str = ".") -> Dict[str, pd.DataFrame]:
    """Loads every table, keeping VLE rows up to the last checkpoint only."""
    tables = {name: load_table(name, base_dir) for name in
              ['studentInfo', 'studentRegistration', 'studentAssessment', 'assessments', 'vle']}

    last_day = max(CHECKPOINTS)
    columns = KEY + ['id_site', 'date', 'sum_click']
    chunks, total_rows = [], 0
    for i in range(8):
        chunk = load_table(f"studentVle_{i}", base_dir)[columns]
        total_rows += len(chunk)
        chunks.append(chunk[chunk['date'] <= last_day])
    tables['studentVle'] = pd.concat(chunks, ignore_index=True)
    assert total_rows == 10655280, f"Expected 10,655,280 VLE rows, found {total_rows}."
    return tables


def validate_raw_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Checks the raw data for structural faults and returns the results as a table."""
    info, reg = tables['studentInfo'], tables['studentRegistration']
    sa, assessments = tables['studentAssessment'], tables['assessments']
    vle, student_vle = tables['vle'], tables['studentVle']

    checks = {
        'studentInfo duplicate enrolment keys': info.duplicated(KEY).sum(),
        'studentRegistration duplicate enrolment keys': reg.duplicated(KEY).sum(),
        'studentAssessment duplicate student-assessment keys': sa.duplicated(['id_assessment', 'id_student']).sum(),
        'Enrolments in only one of studentInfo/studentRegistration': len(info.merge(reg[KEY], on=KEY, how='outer', indicator=True).query("_merge != 'both'")),
        'Submissions with an unknown id_assessment': (~sa['id_assessment'].isin(assessments['id_assessment'])).sum(),
        'VLE rows (<= Day 42) with an unknown id_site': (~student_vle['id_site'].isin(vle['id_site'])).sum(),
        'VLE rows (<= Day 42) with no enrolment': len(student_vle.merge(reg[KEY], on=KEY, how='left', indicator=True).query("_merge == 'left_only'")),
        'VLE rows (<= Day 42) with non-positive clicks': (student_vle['sum_click'] <= 0).sum(),
        'Activity types with no group': (~vle['activity_type'].isin(ACTIVITY_GROUPS)).sum(),
    }
    for check, count in checks.items():
        assert count == 0, f"Failed: {check} ({count})."

    enrol = reg.merge(info, on=KEY)
    notes = {
        'Exact duplicate VLE rows (<= Day 42), retained': student_vle.duplicated().sum(),
        'Missing imd_band, retained as missing': info['imd_band'].isna().sum(),
        'Submissions with no score, retained as missing': sa['score'].isna().sum(),
        'Banked submissions, separated from current evidence': sa['is_banked'].sum(),
        'Fail records with an unregistration date, label retained': ((enrol['final_result'] == 'Fail') & enrol['date_unregistration'].notna()).sum(),
        'Withdrawn records with no unregistration date': ((enrol['final_result'] == 'Withdrawn') & enrol['date_unregistration'].isna()).sum(),
    }
    return pd.DataFrame([{'check': k, 'count': int(v), 'type': 'must be zero'} for k, v in checks.items()] +
                        [{'check': k, 'count': int(v), 'type': 'reported'} for k, v in notes.items()])


def build_enrolments(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per enrolment with the risk label and static attributes."""
    enrol = tables['studentRegistration'].merge(tables['studentInfo'], on=KEY, validate='one_to_one')
    enrol['at_risk'] = enrol['final_result'].map({'Fail': 1, 'Withdrawn': 1, 'Pass': 0, 'Distinction': 0})
    enrol['imd_band'] = enrol['imd_band'].replace({'10-20': '10-20%'})
    enrol['registration_day'] = enrol['date_registration']
    enrol['late_registered'] = (enrol['date_registration'] > 0).astype(int)
    # Audit grouping only: age_band stays as the predictor because 55<= is too small to report alone.
    enrol['age_group_audit'] = enrol['age_band'].replace({'35-55': '35+', '55<=': '35+'})
    return enrol


def checkpoint_status(enrol: pd.DataFrame, cutoff_day: int) -> pd.Series:
    """Status at the end of cutoff_day. Earlier conditions take priority."""
    unreg, regd = enrol['date_unregistration'], enrol['date_registration']
    return pd.Series(np.select(
        [unreg <= 0, unreg <= cutoff_day, regd.isna(), regd > cutoff_day],
        ['left_before_start', 'withdrawn_by_checkpoint', 'missing_registration', 'not_yet_registered'],
        default='eligible'
    ), index=enrol.index)


def allocate_partitions(enrol: pd.DataFrame) -> pd.Series:
    """Assigns every enrolment to a partition, keeping each student in a single partition."""
    ever_eligible = pd.concat([checkpoint_status(enrol, T) == 'eligible' for T in CHECKPOINTS], axis=1).any(axis=1)
    partition = pd.Series('never_eligible', index=enrol.index)
    pool = enrol[ever_eligible]
    partition[pool.index] = 'excluded_overlap'

    cohorts = pd.Series(list(zip(pool['code_module'], pool['code_presentation'])), index=pool.index)
    taken = set()
    for name in ['test', 'conformal_cal', 'prob_cal', 'fit']:
        rows = pool[cohorts.isin(COHORT_ALLOCATION[name])]
        partition[rows.index[~rows['id_student'].isin(taken)]] = name
        taken |= set(rows['id_student'])
    return partition


def extract_vle_features(tables: Dict[str, pd.DataFrame], cutoff_day: int) -> pd.DataFrame:
    """Aggregates clicks up to and including cutoff_day."""
    sliced = tables['studentVle']
    sliced = sliced[sliced['date'] <= cutoff_day].merge(tables['vle'][['id_site', 'activity_type']], on='id_site', how='left')
    sliced['group'] = 'clicks_' + sliced['activity_type'].map(ACTIVITY_GROUPS)

    pivot = sliced.pivot_table(index=KEY, columns='group', values='sum_click', aggfunc='sum', fill_value=0)
    grouped = sliced.groupby(KEY)
    pivot['total_clicks'] = grouped['sum_click'].sum()
    pivot['pre_start_clicks'] = sliced[sliced['date'] < 0].groupby(KEY)['sum_click'].sum()
    pivot['pre_start_clicks'] = pivot['pre_start_clicks'].fillna(0)
    pivot['active_days'] = grouped['date'].nunique()
    pivot['days_since_last_activity'] = cutoff_day - grouped['date'].max()
    return pivot.reset_index()


def classify_assessments(eligible: pd.DataFrame, tables: Dict[str, pd.DataFrame], cutoff_day: int) -> pd.DataFrame:
    """One row per eligible enrolment x TMA/CMA, labelled by what was knowable at cutoff_day."""
    schedule = tables['assessments']
    schedule = schedule[schedule['assessment_type'] != 'Exam']
    grid = eligible[KEY + ['date_registration']].merge(schedule, on=['code_module', 'code_presentation'])
    grid = grid.merge(tables['studentAssessment'], on=['id_assessment', 'id_student'], how='left')

    banked = grid['is_banked'] == 1
    submitted = (grid['date_submitted'] <= cutoff_day) & ~banked
    due = grid['date'] <= cutoff_day
    grid['status'] = np.select(
        [banked,
         submitted & (grid['date_submitted'] > grid['date']),
         submitted & due,
         submitted,
         due & (grid['date'] < grid['date_registration']),
         due],
        ['banked', 'late', 'on_time', 'early', 'due_before_registration', 'overdue'],
        default='not_yet_due'
    )
    grid['score_available_day'] = (
        grid[['date_submitted', 'date']].max(axis=1) + grid['assessment_type'].map(SCORE_DELAY_DAYS)
    )
    grid['visible_score'] = grid['score'].where(submitted & (grid['score_available_day'] <= cutoff_day))
    grid['submission_delay'] = (grid['date_submitted'] - grid['date']).where(submitted)
    return grid


def extract_assessment_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Summarises assessment status per enrolment."""
    counts = pd.crosstab([grid[k] for k in KEY], grid['status']).rename(columns={
        'overdue': 'assessments_overdue', 'on_time': 'assessments_on_time', 'late': 'assessments_late',
        'early': 'assessments_early', 'banked': 'banked_assessments', 'not_yet_due': 'assessments_not_yet_due',
    })
    grouped = grid.groupby(KEY)
    counts['mean_submission_delay'] = grouped['submission_delay'].mean()
    counts['scores_available'] = grouped['visible_score'].count()
    counts['mean_available_score'] = grouped['visible_score'].mean()
    return counts.reset_index()


def build_checkpoint(enrol: pd.DataFrame, tables: Dict[str, pd.DataFrame], cutoff_day: int) -> pd.DataFrame:
    """Builds the dataset for one checkpoint."""
    eligible = enrol[checkpoint_status(enrol, cutoff_day) == 'eligible'].copy()
    eligible['checkpoint_day'] = cutoff_day
    eligible['days_enrolled'] = cutoff_day - eligible['date_registration'].clip(lower=0)

    merged = eligible.merge(extract_vle_features(tables, cutoff_day), on=KEY, how='left')
    grid = classify_assessments(eligible, tables, cutoff_day)
    merged = merged.merge(extract_assessment_features(grid), on=KEY, how='left')

    # No log rows means no recorded activity, so counts are true zeros. Means and
    # recency stay missing because they are undefined rather than zero.
    count_cols = ['total_clicks', 'pre_start_clicks', 'active_days', 'clicks_homepage', 'clicks_forum',
                  'clicks_quiz', 'clicks_content', 'assessments_overdue', 'assessments_on_time',
                  'assessments_late', 'assessments_early', 'banked_assessments',
                  'assessments_not_yet_due', 'scores_available']
    for col in count_cols:
        if col not in merged.columns:
            merged[col] = 0
    merged[count_cols] = merged[count_cols].fillna(0)
    merged['assessments_due'] = merged[['assessments_overdue', 'assessments_on_time', 'assessments_late']].sum(axis=1)
    return merged


def write_appendix_tables(enrol: pd.DataFrame, dataset: pd.DataFrame, tables: Dict[str, pd.DataFrame], out: str) -> None:
    """Writes the tables cited in the methodology appendix."""
    order = ['fit', 'prob_cal', 'conformal_cal', 'test']
    used = enrol[enrol['partition'].isin(order)]

    allocation = used.groupby('partition').agg(
        presentations=('code_presentation', lambda s: ', '.join(sorted(s.unique()))),
        enrolments=('at_risk', 'size'), students=('id_student', 'nunique'),
        at_risk=('at_risk', 'sum'), not_at_risk=('at_risk', lambda s: int((s == 0).sum()))
    ).reindex(order)
    allocation.to_csv(os.path.join(out, 'table_allocation.csv'))

    by_checkpoint = dataset.groupby(['partition', 'checkpoint_day']).agg(
        rows=('at_risk', 'size'), at_risk=('at_risk', 'sum'),
        not_at_risk=('at_risk', lambda s: int((s == 0).sum())), at_risk_rate=('at_risk', 'mean')
    ).round(3).reset_index()
    by_checkpoint.to_csv(os.path.join(out, 'table_class_balance_by_checkpoint.csv'), index=False)

    used.pivot_table(index='code_module', columns='partition', values='at_risk', aggfunc=['size', 'sum'], fill_value=0
                     ).to_csv(os.path.join(out, 'table_module_by_partition.csv'))

    def overlap_matrix(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        groups = {k: set(g['id_student']) for k, g in frame.groupby(column)}
        return pd.DataFrame({a: {b: len(groups[a] & groups[b]) for b in groups} for a in groups})

    pool = enrol[enrol['partition'] != 'never_eligible']
    overlap_matrix(pool, 'code_presentation').to_csv(os.path.join(out, 'table_overlap_before.csv'))
    overlap_matrix(used, 'partition').reindex(index=order, columns=order).to_csv(os.path.join(out, 'table_overlap_after.csv'))

    flow = enrol.melt(id_vars=['at_risk'], value_vars=[f'status_day_{T}' for T in CHECKPOINTS],
                      var_name='checkpoint', value_name='status')
    flow.pivot_table(index='status', columns='checkpoint', values='at_risk', aggfunc='size', fill_value=0
                     ).to_csv(os.path.join(out, 'table_eligibility_flow.csv'))

    rows = []
    for T in CHECKPOINTS:
        eligible = enrol[enrol[f'status_day_{T}'] == 'eligible']
        grid = classify_assessments(eligible, tables, T)
        row = grid['status'].value_counts().to_dict()
        row.update({'checkpoint_day': T, 'visible_scores': int(grid['visible_score'].count()),
                    'scores_submitted_but_hidden': int((grid['status'].isin(['on_time', 'late', 'early'])
                                                        & grid['score'].notna() & grid['visible_score'].isna()).sum())})
        rows.append(row)
    pd.DataFrame(rows).set_index('checkpoint_day').fillna(0).astype(int).to_csv(os.path.join(out, 'table_assessment_status.csv'))

    missing = dataset.groupby('checkpoint_day')[NUMERIC_FEATURES + CATEGORICAL_FEATURES].apply(lambda d: d.isna().mean()).T.round(3)
    missing[missing.sum(axis=1) > 0].to_csv(os.path.join(out, 'table_missingness.csv'))


def validate_output(enrol: pd.DataFrame, dataset: pd.DataFrame) -> None:
    """Final invariants on the processed data."""
    assert not set(FEATURES) & set(IDENTIFIERS), "Identifier used as a predictor."
    assert not dataset.duplicated(KEY + ['checkpoint_day']).any(), "Duplicate checkpoint rows."
    assert (dataset['date_registration'] <= dataset['checkpoint_day']).all(), "Row before registration."
    assert not (dataset['date_unregistration'] <= dataset['checkpoint_day']).any(), "Withdrawn student retained."
    assert dataset['at_risk'].isin([0, 1]).all(), "Invalid label."
    assert 'age_group_audit' not in FEATURES, "Audit group used as a predictor."
    assert (dataset['checkpoint_day'] >= dataset['first_eligible_checkpoint']).all(), "Row before first eligibility."
    assert dataset[[c for c in NUMERIC_FEATURES if c not in ('mean_available_score', 'mean_submission_delay', 'days_since_last_activity')]].notna().all().all()

    used = enrol[enrol['partition'].isin(['fit', 'prob_cal', 'conformal_cal', 'test'])]
    assert (used.groupby('id_student')['partition'].nunique() == 1).all(), "Student in two partitions."
    assert sum(len(c) for c in COHORT_ALLOCATION.values()) == 22, "Cohort missing or repeated."
    cohort_owners = used.groupby(['code_module', 'code_presentation'])['partition'].nunique()
    assert len(cohort_owners) == 22 and (cohort_owners == 1).all(), "Cohort in two partitions."
    last_development = used.loc[used['partition'] != 'test', 'code_presentation'].max()
    assert last_development < '2014J', "Test does not follow development."


def main(base_dir: str = ".") -> None:
    out = os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(out, exist_ok=True)

    print("Loading raw tables...")
    tables = load_raw_tables(base_dir)
    validate_raw_tables(tables).to_csv(os.path.join(out, 'table_integrity_checks.csv'), index=False)

    enrol = build_enrolments(tables)
    for T in CHECKPOINTS:
        enrol[f'status_day_{T}'] = checkpoint_status(enrol, T)
    eligible_at = pd.DataFrame({T: enrol[f'status_day_{T}'] == 'eligible' for T in CHECKPOINTS})
    enrol['first_eligible_checkpoint'] = eligible_at.idxmax(axis=1).where(eligible_at.any(axis=1))
    enrol['partition'] = allocate_partitions(enrol)

    print("Building checkpoints...")
    dataset = pd.concat([build_checkpoint(enrol, tables, T) for T in CHECKPOINTS], ignore_index=True)

    validate_output(enrol, dataset)
    write_appendix_tables(enrol, dataset, tables, out)

    columns = KEY + ['partition', 'checkpoint_day', 'first_eligible_checkpoint', 'final_result', 'at_risk',
                     'age_group_audit', 'assessments_not_yet_due'] + FEATURES
    dataset[columns].to_csv(os.path.join(out, 'checkpoint_dataset.csv'), index=False)
    flow_cols = KEY + ['partition', 'final_result', 'at_risk', 'age_band', 'age_group_audit', 'disability',
                       'date_registration', 'date_unregistration', 'first_eligible_checkpoint'] + [f'status_day_{T}' for T in CHECKPOINTS]
    enrol[flow_cols].to_csv(os.path.join(out, 'enrolment_flow.csv'), index=False)
    print(f"Done: {len(dataset)} rows written to {out}.")


if __name__ == "__main__":
    main()
