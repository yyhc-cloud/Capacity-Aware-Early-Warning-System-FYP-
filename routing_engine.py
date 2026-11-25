"""
routing_engine.py - reusable routing code (development work).

Routes every eligible enrolment at Weeks 2, 4 and 6 for six methods (run_routing), checks the
result (check_routing) and computes delay types and outcome measures. The three conformal methods
route by prediction set; the rule-based, Top-K and threshold methods use matched Review places
(the next ranked enrolments after the Contact places).

Capacity comes from a {module_id: ModuleCapacity} dictionary built with capacity.py (whole
numbers per module). At each checkpoint and module: contact places available = min(K_t, B -
contacts already made in that module); review places = R_t.

Importing this module never loads data or trains a model.
"""

from collections import defaultdict
from typing import Mapping

import numpy as np
import pandas as pd

from capacity import ModuleCapacity

# ---- Settings ------------------------------------------------------------------------------
# The defaults below are the proposed method. The other values are kept only for sensitivity
# comparisons.
#
# FOLLOWUP_REVIEW (conformal methods only): contacted enrolment with p >= CUTOFF -> follow-up
# Review candidate; p < CUTOFF -> Monitor; contacted and reviewed -> No new action. Overrides
# CONTACTED_CAN_REVIEW for the conformal methods.
FOLLOWUP_REVIEW = True
# FOLLOWUP_LAST (with FOLLOWUP_REVIEW): Review places go first to new Review candidates (never
# contacted, never reviewed); only places left over go to follow-up candidates, ranked by p.
# A follow-up candidate without a place is Monitor (never Waiting, never Unserved).
# False: follow-ups are ranked together with the new Review candidates.
FOLLOWUP_LAST = True
# CONTACTED_CAN_REVIEW: False - contacted enrolments get "No new action" (matched methods, and
# conformal methods when FOLLOWUP_REVIEW is False); True - the ordinary Review route is allowed.
CONTACTED_CAN_REVIEW = False
# COUNT_BEFORE_FIRST_SELECTION: Unserved and delay types use only checkpoints up to the first
# Contact or Review. False: every checkpoint is counted.
COUNT_BEFORE_FIRST_SELECTION = True

# ---- Constants -------------------------------------------------------------------------------
CHECKPOINTS = [14, 28, 42]
WEEK = {14: 2, 28: 4, 42: 6}
PREV = {14: None, 28: 14, 42: 28}
NEXT = {14: 28, 28: 42, 42: None}
CUTOFF = 0.5                 # "high risk" probability cutoff for ambiguous sets
SET_R, SET_N, SET_A, SET_E = "{At risk}", "{Not at risk}", "{At risk, Not at risk}", "{}"
CONTACT, REVIEW = "Contact", "Review"
CW, RW = "Contact Waiting", "Review Waiting"
DEFER, MON, MONF = "Defer", "Monitor", "Monitor (uncertainty flag)"
NNA = "No new action"
UNSERVED = "Unserved due to capacity"        # flag on Week-6 waiting rows
STATUSES = [CONTACT, REVIEW, CW, RW, DEFER, MON, MONF, NNA]
RC, RR = "Contact candidate", "Review candidate"
RF = "Follow-up candidate"                    # contacted, p >= cutoff, not yet reviewed
FOLLOWUP_NO_PLACE = "contacted earlier; still high risk; no follow-up place"
CAND_ROUTES = (RC, RR)
CONFORMAL = ("conf_no_persistence", "conf_no_waiting", "proposed")
MATCHED = ("rules", "topk", "threshold")
METHODS = MATCHED + CONFORMAL
METHOD_NAME = {
    "rules": "Rule-based alerts",
    "topk": "Sequential probability Top-K",
    "threshold": "Probability-threshold deferral",
    "conf_no_persistence": "Conformal routing without persistence",
    "conf_no_waiting": "Conformal routing with persistence, without separate Waiting states",
    "proposed": "Proposed method",
}


class RoutingCheckError(AssertionError):
    pass


def _caps_for(capacity, cohort, t):
    if cohort not in capacity:
        raise KeyError(f"no capacity given for module {cohort}")
    c = capacity[cohort]
    if not isinstance(c, ModuleCapacity):
        raise TypeError(f"capacity for {cohort} must be a ModuleCapacity")
    return c.contact_at(WEEK[t]), c.review_per_checkpoint, c.total_contact_budget


def _post_process(recs, pred, before_first=False):
    """Combine the per-module records into one log and add the Unserved and withdrawal flags.
    before_first: a Week-6 Waiting row is
    Unserved only if the enrolment had no Contact or Review place at an earlier checkpoint."""
    log = pd.concat(recs, ignore_index=True)
    log["unserved_due_to_capacity"] = (log.week == 6) & log.status.isin([CW, RW])
    if before_first:
        sel_before = set(log.loc[log.status.isin([CONTACT, REVIEW]) & (log.week < 6), "eid"])
        log["unserved_due_to_capacity"] &= ~log.eid.isin(sel_before)
    elig = {t: set(pred.loc[pred.checkpoint == t, "eid"]) for t in CHECKPOINTS}
    nxt = log.checkpoint.map(NEXT)
    log["withdrawn_after_checkpoint"] = [n == n and n is not None and e not in elig[int(n)]
                                         for e, n in zip(log.eid, nxt)]
    ever_sel = set(log.loc[log.status.isin([CONTACT, REVIEW]), "eid"])
    log["withdrawn_before_selection"] = log.withdrawn_after_checkpoint & ~log.eid.isin(ever_sel)
    log["withdrawn_from"] = np.select(
        [~log.withdrawn_before_selection, log.status.isin([DEFER, CW, RW]), log.status.isin([MON, MONF])],
        ["", "Defer/Waiting", "Monitor"], default="other")
    return log


# ============================================================================
# Conformal methods
# ============================================================================
def _run_conformal(pred, policy, capacity, contacted_can_review, cutoff=CUTOFF, followup=False,
                   before_first=False, followup_last=False):
    use_persist = policy in ("proposed", "conf_no_waiting")
    overflow_c = DEFER if policy == "conf_no_waiting" else CW
    overflow_r = DEFER if policy == "conf_no_waiting" else RW
    hist, contacts = {}, defaultdict(int)
    recs, caps = [], []
    for t in CHECKPOINTS:
        P = pred[pred.checkpoint == t]
        for cohort, g in P.groupby("cohort", sort=True):
            K_t, R_t, B = _caps_for(capacity, cohort, t)
            rows = []
            for r in g.itertuples(index=False):
                h = hist.setdefault(r.eid, dict(last_t=None, last_has_risk=False,
                                                contacted=False, reviewed=False))
                if h["last_t"] is not None and h["last_t"] != PREV[t]:
                    raise RoutingCheckError(f"{r.eid} skips a checkpoint")
                has_risk = r.set in (SET_R, SET_A)
                old_pers = bool(has_risk and h["last_t"] is not None and h["last_t"] == PREV[t]
                                and h["last_has_risk"])
                persistent = old_pers if use_persist else False
                high = r.p >= cutoff
                n_rules = int(r.clicks7 == 0) + int(r.n_overdue >= 1)
                prev_c, prev_r = h["contacted"], h["reviewed"]
                reason, nna = "", ""
                # Ordinary route from the prediction set
                if r.set == SET_R:
                    route = RC
                elif r.set == SET_E:
                    route, reason = RR, "empty"
                elif r.set == SET_N:
                    route = MON
                elif high:
                    route, reason = RR, "high risk"
                elif WEEK[t] in (2, 4):
                    route = DEFER
                elif persistent:
                    route, reason = RR, "persistent"
                else:
                    route = MONF
                # Repeat-selection rule. A previously reviewed enrolment keeps an RC route
                # (singleton {At risk}) but can never be a Review candidate again.
                if prev_c and followup:          # follow-up rule
                    if prev_r:
                        route, reason, nna = NNA, "", "contacted and reviewed earlier"
                    elif high:
                        route, reason = (RF if followup_last else RR), "follow-up after contact"
                    else:
                        route, reason = MON, ""
                elif prev_c:
                    if not (contacted_can_review and route == RR and not prev_r):
                        route, reason, nna = NNA, "", "contacted earlier"
                elif route == RR and prev_r:
                    route, reason, nna = NNA, "", "reviewed earlier"
                rows.append(dict(eid=r.eid, student=r.student, cohort=cohort, checkpoint=t,
                                 week=WEEK[t], p=r.p, set=r.set, high_risk=high,
                                 persistent=persistent, previously_contacted=prev_c,
                                 previously_reviewed=prev_r, n_rules=n_rules, clicks7=r.clicks7,
                                 route=route, review_reason=reason, no_new_action_reason=nna,
                                 rank=np.nan, status=route, y=r.y))
            cur = pd.DataFrame(rows)
            # ---- Contact capacity ----
            places_c = min(K_t, B - contacts[cohort])
            cc = cur[cur.route == RC].copy()
            if use_persist:
                cc = cc.sort_values(["persistent", "p", "eid"], ascending=[False, False, True])
            else:
                cc = cc.sort_values(["p", "eid"], ascending=[False, True])
            cur.loc[cc.index, "rank"] = np.arange(1, len(cc) + 1)
            cur.loc[cc.index, "status"] = overflow_c
            sel = cc.index[:places_c]
            cur.loc[sel, "status"] = CONTACT
            n_c = len(sel)
            contacts[cohort] += n_c
            # ---- Review capacity ----
            rc = cur[cur.route == RR].copy()
            if use_persist:
                rc = rc.sort_values(["p", "persistent", "eid"], ascending=[False, False, True])
            else:
                rc = rc.sort_values(["p", "eid"], ascending=[False, True])
            cur.loc[rc.index, "rank"] = np.arange(1, len(rc) + 1)
            cur.loc[rc.index, "status"] = overflow_r
            cur.loc[rc.index[:R_t], "status"] = REVIEW
            n_r = min(R_t, len(rc))
            # follow-up candidates only get Review places left after the new candidates
            fc = cur[cur.route == RF].sort_values(["p", "eid"], ascending=[False, True])
            if len(fc):
                cur.loc[fc.index, "rank"] = np.arange(len(rc) + 1, len(rc) + len(fc) + 1)
                cur.loc[fc.index, "status"] = MON
                cur.loc[fc.index, "review_reason"] = FOLLOWUP_NO_PLACE
                left = max(R_t - n_r, 0)
                cur.loc[fc.index[:left], "status"] = REVIEW
                cur.loc[fc.index[:left], "review_reason"] = "follow-up after contact"
                n_r += min(left, len(fc))
            caps.append(dict(cohort=cohort, checkpoint=t, n_eligible=len(cur), K_t=K_t,
                             contact_places_available=places_c, contact_used=n_c,
                             R_t=R_t, review_used=n_r, contacts_cumulative=contacts[cohort], B=B))
            for r in cur.itertuples(index=False):
                h = hist[r.eid]
                if r.status == CONTACT:
                    h["contacted"] = True
                elif r.status == REVIEW:
                    h["reviewed"] = True
                h["last_t"], h["last_has_risk"] = t, r.set in (SET_R, SET_A)
            recs.append(cur)
    return _post_process(recs, pred, before_first), pd.DataFrame(caps)


# ============================================================================
# Methods with matched Review places (rule-based, Top-K, threshold)
# ============================================================================
def _run_matched(pred, method, capacity, contacted_can_review, before_first=False):
    hist, contacts, recs, caps = {}, defaultdict(int), [], []
    for t in CHECKPOINTS:
        for cohort, g in pred[pred.checkpoint == t].groupby("cohort", sort=True):
            K_t, R_t, B = _caps_for(capacity, cohort, t)
            rows = []
            for r in g.itertuples(index=False):
                h = hist.setdefault(r.eid, dict(last_t=None, contacted=False, reviewed=False))
                if h["last_t"] is not None and h["last_t"] != PREV[t]:
                    raise RoutingCheckError(f"{r.eid} skips a checkpoint")
                n_rules = int(r.clicks7 == 0) + int(r.n_overdue >= 1)
                prev_c = h["contacted"]
                nna = ""
                # A previously contacted enrolment is only ranked (for a matched Review place,
                # never a Contact place) when CONTACTED_CAN_REVIEW is True.
                if prev_c and not contacted_can_review:
                    route, nna = NNA, "contacted earlier"
                elif method == "rules":                 # flagged = Contact candidate
                    route = RC if n_rules > 0 else MON
                elif method == "topk":                  # everyone ranked
                    route = "_pool"
                else:                                   # threshold
                    if r.p >= r.upper:
                        route = RC
                    elif r.p > r.lower:
                        route = DEFER if WEEK[t] in (2, 4) else MON
                    else:
                        route = MON
                rows.append(dict(eid=r.eid, student=r.student, cohort=cohort, checkpoint=t,
                                 week=WEEK[t], p=r.p, set=r.set, high_risk=r.p >= CUTOFF,
                                 persistent=False, previously_contacted=prev_c,
                                 previously_reviewed=h["reviewed"], n_rules=n_rules,
                                 clicks7=r.clicks7, route=route, review_reason="",
                                 no_new_action_reason=nna, rank=np.nan, status=route, y=r.y))
            cur = pd.DataFrame(rows)
            if method == "rules":
                cand = cur[cur.route == RC].sort_values(["n_rules", "clicks7", "eid"],
                                                        ascending=[False, True, True])
            elif method == "topk":
                cand = cur[cur.route == "_pool"].sort_values(["p", "eid"], ascending=[False, True])
            else:
                cand = cur[cur.route == RC].sort_values(["p", "eid"], ascending=[False, True])
            cur.loc[cand.index, "rank"] = np.arange(1, len(cand) + 1)
            if method == "topk":
                cur.loc[cand.index, ["route", "status"]] = MON
            else:
                cur.loc[cand.index, "status"] = MON if method == "rules" else CW
            places_c = min(K_t, B - contacts[cohort])
            # Contact places: the highest-ranked candidates never contacted before. A previously
            # reviewed enrolment can take one if it qualifies under the method's own rule.
            sel_c = [i for i in cand.index if not cur.at[i, "previously_contacted"]][:places_c]
            cur.loc[sel_c, "route"] = RC
            cur.loc[sel_c, "status"] = CONTACT
            # Matched Review places: the next R_t ranked, never a previously reviewed enrolment.
            chosen = set(sel_c)
            rest = [i for i in cand.index if i not in chosen and not cur.at[i, "previously_reviewed"]]
            sel_r = rest[:R_t]
            cur.loc[sel_r, ["route", "status"]] = [RR, REVIEW]
            cur.loc[sel_r, "review_reason"] = "matched Review place"
            # Previously contacted enrolments not given a Review place (CONTACTED_CAN_REVIEW only)
            left = [i for i in cur.index[cur.previously_contacted] if i not in set(sel_r)]
            cur.loc[left, ["route", "status", "no_new_action_reason"]] = [NNA, NNA, "contacted earlier"]
            contacts[cohort] += len(sel_c)
            caps.append(dict(cohort=cohort, checkpoint=t, n_eligible=len(cur), K_t=K_t,
                             contact_places_available=places_c, contact_used=len(sel_c),
                             R_t=R_t, review_used=len(sel_r), contacts_cumulative=contacts[cohort], B=B))
            for r in cur.itertuples(index=False):
                h = hist[r.eid]
                if r.status == CONTACT:
                    h["contacted"] = True
                elif r.status == REVIEW:
                    h["reviewed"] = True
                h["last_t"] = t
            recs.append(cur)
    return _post_process(recs, pred, before_first), pd.DataFrame(caps)


def run_routing(pred: pd.DataFrame, method: str, capacity: Mapping[str, ModuleCapacity],
                contacted_can_review: bool = None, followup_review: bool = None,
                count_before_first_selection: bool = None, followup_last: bool = None):
    """Route every eligible enrolment at Weeks 2, 4, 6. Returns (log, capacity_log).
    Unset arguments default to the module settings CONTACTED_CAN_REVIEW, FOLLOWUP_REVIEW,
    COUNT_BEFORE_FIRST_SELECTION and FOLLOWUP_LAST (the proposed method)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    missing = set(pred.cohort) - set(capacity)
    if missing:
        raise KeyError(f"no capacity given for modules: {sorted(missing)}")
    if contacted_can_review is None:
        contacted_can_review = CONTACTED_CAN_REVIEW
    if followup_review is None:
        followup_review = FOLLOWUP_REVIEW
    if count_before_first_selection is None:
        count_before_first_selection = COUNT_BEFORE_FIRST_SELECTION
    bf = bool(count_before_first_selection)
    if followup_last is None:
        followup_last = FOLLOWUP_LAST
    pred = pred.sort_values(["checkpoint", "cohort", "eid"]).reset_index(drop=True)
    if method in CONFORMAL:
        return _run_conformal(pred, method, capacity, bool(contacted_can_review),
                              followup=bool(followup_review), before_first=bf,
                              followup_last=bool(followup_last))
    return _run_matched(pred, method, capacity, bool(contacted_can_review), before_first=bf)


# ============================================================================
# Routing checks (capacity per module)
# ============================================================================
def review_then_contact(log):
    """Enrolments given a Review place at one checkpoint and a Contact place at a later one."""
    first_r = log[log.status == REVIEW].groupby("eid").checkpoint.min()
    first_c = log[log.status == CONTACT].groupby("eid").checkpoint.min()
    both = first_r.index.intersection(first_c.index)
    return sorted(e for e in both if first_c[e] > first_r[e])


def check_repeat_rule(log, caps, pred, method, contacted_can_review, followup=False, followup_last=False):
    """Repeat-selection rule (also with contacted_can_review, and with followup for the
    conformal methods). Returns a list of failed checks."""
    problems = []
    if not log[log.status == CONTACT].eid.is_unique:
        problems.append("an enrolment contacted twice")
    if not log[log.status == REVIEW].eid.is_unique:
        problems.append("an enrolment reviewed twice")
    if (log.previously_contacted & log.route.eq(RC)).any() or (log.previously_contacted & log.status.eq(CW)).any():
        problems.append("previously contacted enrolment was a Contact candidate")
    if (log.previously_reviewed & log.route.eq(RR)).any() or (log.previously_reviewed & log.status.eq(RW)).any():
        problems.append("previously reviewed enrolment was a Review candidate")
    if followup and method in CONFORMAL:
        problems += check_followup(log, last=followup_last)
    elif not contacted_can_review:
        if (log.previously_contacted & log.status.ne(NNA)).any():
            problems.append("contacted enrolment not 'No new action' at a later checkpoint")
        if (log.previously_contacted & log.status.eq(REVIEW)).any():
            problems.append("contacted enrolment reviewed afterwards")
    elif not log.loc[log.previously_contacted, "route"].isin([NNA, RR]).all():
        problems.append("contacted enrolment with a route other than Review candidate / No new action")
    # Review -> later Contact: the Contact row must satisfy the method's own Contact rule
    rc = log[(log.status == CONTACT) & log.previously_reviewed]
    if method in CONFORMAL:
        ok = (rc.set == SET_R).all()
    elif method == "rules":
        ok = (rc.n_rules >= 1).all()
    elif method == "threshold":
        up = pred.set_index(["eid", "checkpoint"]).upper
        ok = (rc.p.to_numpy() >= up.loc[list(zip(rc.eid, rc.checkpoint))].to_numpy()).all()
    else:                                   # topk: contacts = top-p not-yet-contacted per module
        ok = True
        for (cohort, t), s in log.groupby(["cohort", "checkpoint"]):
            n_c = int((s.status == CONTACT).sum())
            top = s[~s.previously_contacted].sort_values(["p", "eid"], ascending=[False, True]).eid[:n_c]
            ok &= set(top) == set(s.loc[s.status == CONTACT, "eid"])
    if not ok:
        problems.append("Review -> Contact case without the method's Contact condition")
    if contacted_can_review:
        problems += check_contacted_review(log, pred, method)
    return problems


def check_followup(log, last=False):
    """Conformal methods: contacted and reviewed -> No new action; contacted, not reviewed,
    p >= CUTOFF -> follow-up Review candidate; p < CUTOFF -> Monitor; every follow-up Review has
    p >= CUTOFF and was never reviewed before."""
    problems = []
    pc = log[log.previously_contacted]
    if not (pc.loc[pc.previously_reviewed, "status"] == NNA).all():
        problems.append("contacted and reviewed enrolment not 'No new action'")
    hi = pc[~pc.previously_reviewed & (pc.p >= CUTOFF)]
    if not last:
        if not (hi.route == RR).all() or not (hi.review_reason == "follow-up after contact").all():
            problems.append("contacted enrolment with p >= cutoff not a follow-up Review candidate")
    else:                       # FOLLOWUP_LAST: follow-up route; placed -> Review, otherwise Monitor + reason
        ok = (hi.route == RF).all() and hi.status.isin([REVIEW, MON]).all()
        ok &= (hi.loc[hi.status == MON, "review_reason"] == FOLLOWUP_NO_PLACE).all()
        ok &= (hi.loc[hi.status == REVIEW, "review_reason"] == "follow-up after contact").all()
        if not ok:
            problems.append("follow-up candidate not Review / Monitor with the follow-up reason")
        if (pc.status == RW).any() or pc.unserved_due_to_capacity.astype(bool).any():
            problems.append("contacted enrolment in Review Waiting or Unserved")
        for (cohort, t), g in log.groupby(["cohort", "checkpoint"]):
            f = g[g.route == RF]
            if not len(f):
                continue
            n_fu = int((f.status == REVIEW).sum())
            if n_fu and (g.route.eq(RR) & g.status.ne(REVIEW)).any():
                problems.append(f"{cohort} Week {WEEK[t]}: follow-up Review while a new Review "
                                "candidate was left unplaced")
                break
            top = f.sort_values(["p", "eid"], ascending=[False, True]).eid[:n_fu]
            if set(top) != set(f.loc[f.status == REVIEW, "eid"]):
                problems.append(f"{cohort} Week {WEEK[t]}: follow-up places not in p order")
                break
    if (log.route == RF).any() and not last:
        problems.append("follow-up route used although FOLLOWUP_LAST is off")
    lo = pc[~pc.previously_reviewed & (pc.p < CUTOFF)]
    if not (lo.status == MON).all():
        problems.append("contacted enrolment with p < cutoff not Monitor")
    fr = pc[pc.status == REVIEW]
    if not ((fr.p >= CUTOFF) & ~fr.previously_reviewed).all():
        problems.append("follow-up Review with p < cutoff or after an earlier Review")
    return problems


def check_contacted_review(log, pred, method):
    """CONTACTED_CAN_REVIEW only: every Review place given to a previously contacted enrolment
    met the method's ordinary Review criteria at that checkpoint (recomputed from p, set and the
    previous set, not from the route column). Matched methods: Review places follow the method's
    rank order. Also: a contacted enrolment is never in Contact Waiting / Unserved because of a
    new {At risk} set."""
    problems = []
    rv = log[(log.status == REVIEW) & log.previously_contacted]
    if rv.previously_reviewed.any():
        problems.append("contacted enrolment reviewed although it had been reviewed before")
    if method in CONFORMAL:
        prev_set = log.set_index(["eid", "checkpoint"]).set
        def prev_has_risk(e, t):
            k = (e, PREV[t])
            return PREV[t] is not None and k in prev_set.index and prev_set[k] in (SET_R, SET_A)
        for r in rv.itertuples(index=False):
            ok = (r.set == SET_E) or (r.set == SET_A and r.p >= CUTOFF)
            if method != "conf_no_persistence":
                ok |= (r.set == SET_A and r.week == 6 and r.p < CUTOFF and prev_has_risk(r.eid, r.checkpoint))
            if not ok:
                problems.append(f"{r.eid} Week {r.week}: Review place to a contacted enrolment "
                                "without the Review criteria")
                break
        bad = log.previously_contacted & (log.set == SET_R) & log.status.ne(NNA)
        if bad.any():
            problems.append("contacted enrolment with a new {At risk} set not 'No new action'")
    else:
        # Review places = the first R_t ranked candidates, excluding this checkpoint's contacts and
        # previously reviewed enrolments (rules: flagged; threshold: p >= upper; topk: everyone).
        up = pred.set_index(["eid", "checkpoint"]).upper if method == "threshold" else None
        for (cohort, t), s in log.groupby(["cohort", "checkpoint"]):
            if method == "rules":
                c = s[s.n_rules >= 1].sort_values(["n_rules", "clicks7", "eid"], ascending=[False, True, True])
            else:
                c = s.sort_values(["p", "eid"], ascending=[False, True])
                if method == "threshold":
                    c = c[c.p.to_numpy() >= up.loc[list(zip(c.eid, c.checkpoint))].to_numpy()]
            c = c[(c.status != CONTACT) & ~c.previously_reviewed]
            n_r = int((s.status == REVIEW).sum())
            if set(c.eid[:n_r]) != set(s.loc[s.status == REVIEW, "eid"]):
                problems.append(f"{cohort} Week {WEEK[t]}: matched Review places not in rank order")
                break
    return problems


def check_routing(log, caps, pred, method, capacity, allowed_ids=None, contacted_can_review=None,
                  followup_review=None, followup_last=None):
    """Raise RoutingCheckError listing every failed check."""
    if contacted_can_review is None:
        contacted_can_review = CONTACTED_CAN_REVIEW
    if followup_review is None:
        followup_review = FOLLOWUP_REVIEW
    if followup_last is None:
        followup_last = FOLLOWUP_LAST
    problems = []
    if log.duplicated(["eid", "checkpoint"]).any() or not log.status.isin(STATUSES).all():
        problems.append("one status per eligible enrolment per checkpoint")
    for t in CHECKPOINTS:
        if len(log[log.checkpoint == t]) != (pred.checkpoint == t).sum():
            problems.append(f"Week {WEEK[t]}: status count differs from eligible count")
    for (cohort, t), s in log.groupby(["cohort", "checkpoint"]):
        K_t, R_t, _ = _caps_for(capacity, cohort, t)
        if (s.status == CONTACT).sum() > K_t:
            problems.append(f"{cohort} Week {WEEK[t]}: K exceeded")
        if (s.status == REVIEW).sum() > R_t:
            problems.append(f"{cohort} Week {WEEK[t]}: R exceeded")
    for cohort, s in log[log.status == CONTACT].groupby("cohort"):
        if s.eid.nunique() > capacity[cohort].total_contact_budget:
            problems.append(f"{cohort}: budget B exceeded")
    if (caps.contact_used > caps.contact_places_available).any() or (caps.review_used > caps.R_t).any():
        problems.append("places used exceed places available")
    if not (log.loc[log.status == CW, "route"] == RC).all():
        problems.append("Contact Waiting case that was not a Contact candidate")
    if not (log.loc[log.status == RW, "route"] == RR).all():
        problems.append("Review Waiting case that was not a Review candidate")
    problems += check_repeat_rule(log, caps, pred, method, contacted_can_review, followup_review,
                                  followup_last)
    first = log.groupby("eid").checkpoint.transform("min")
    if log.loc[(log.week == 2) | (log.checkpoint == first), "persistent"].any():
        problems.append("persistence at Week 2 or first checkpoint")
    if allowed_ids is not None and not set(log.eid) <= set(allowed_ids):
        problems.append("rows outside the allowed enrolments (ccal/test)")
    if method in ("proposed", "conf_no_persistence"):   # candidate not selected <=> waiting
        cand_ns = log.route.isin(CAND_ROUTES) & ~log.status.isin([CONTACT, REVIEW])
        if not (cand_ns == log.status.isin([CW, RW])).all():
            problems.append("waiting does not equal 'candidate not selected'")
    if method in MATCHED:
        rv = log[log.status == REVIEW]
        if rv.previously_reviewed.any() or (rv.previously_contacted.any() and not contacted_can_review):
            problems.append("matched Review place given to a previously reviewed (or, with "
                            "CONTACTED_CAN_REVIEW False, previously contacted) enrolment")
    if problems:
        raise RoutingCheckError(f"{method}: " + "; ".join(problems))


# ============================================================================
# Delay types and outcome measures
# ============================================================================
def delay_types(log, before_first=None):
    """Delay type per At risk enrolment. before_first (default COUNT_BEFORE_FIRST_SELECTION):
    use only checkpoints up to the enrolment's first Contact or Review."""
    if before_first is None:
        before_first = COUNT_BEFORE_FIRST_SELECTION
    out = {}
    for eid, h in log[log.y == 1].groupby("eid"):
        if before_first:
            s = h.status.isin([CONTACT, REVIEW])
            if s.any():
                h = h[h.checkpoint <= h.loc[s, "checkpoint"].min()]
        cand = h.route.isin(CAND_ROUTES)
        if not cand.any():
            out[eid] = "never a candidate"
            continue
        first = h.loc[cand, "checkpoint"].min()
        u = ((h.route == DEFER) & (h.checkpoint < first)).any()
        c = (cand & ~h.status.isin([CONTACT, REVIEW])).any()
        out[eid] = "combined" if u and c else "uncertainty-related" if u else \
            "capacity-related" if c else "none"
    return pd.Series(out, dtype=object)


def outcome_measures(log, caps):
    o = {}
    ar = set(log.loc[log.y == 1, "eid"])
    sel = log[log.status.isin([CONTACT, REVIEW])]
    o["At risk enrolments"] = len(ar)
    for t in CHECKPOINTS:
        n = len(set(sel.loc[sel.checkpoint <= t, "eid"]) & ar)
        o[f"At risk selected by Week {WEEK[t]} (n)"] = n
        o[f"At risk selected by Week {WEEK[t]} (%)"] = 100 * n / len(ar)
    con, rev = log[log.status == CONTACT], log[log.status == REVIEW]
    o["Contact selections"] = len(con)
    o["Contact precision"] = con.y.mean() if len(con) else np.nan
    o["False-positive contacts"] = int((con.y == 0).sum())
    o["Review selections"] = len(rev)
    o["Review precision"] = rev.y.mean() if len(rev) else np.nan
    o["False-positive reviews"] = int((rev.y == 0).sum())
    o["Contact places available"] = int(caps.contact_places_available.sum())
    o["Contact places used"] = int(caps.contact_used.sum())
    o["Review places (R)"] = int(caps.R_t.sum())
    o["Review places used"] = int(caps.review_used.sum())
    for w in (2, 4, 6):
        o[f"Contact Waiting at Week {w}"] = int(((log.week == w) & (log.status == CW)).sum())
        o[f"Review Waiting at Week {w}"] = int(((log.week == w) & (log.status == RW)).sum())
    o["Unserved due to capacity (all)"] = int(log.unserved_due_to_capacity.sum())
    o["Unserved due to capacity (At risk)"] = int((log.unserved_due_to_capacity & (log.y == 1)).sum())
    o["At risk never selected"] = len(ar) - len(set(sel.eid) & ar)
    dt = delay_types(log).value_counts()
    for k in ["none", "uncertainty-related", "capacity-related", "combined", "never a candidate"]:
        o[f"Delay type: {k}"] = int(dt.get(k, 0))
    o["Review -> later Contact (n)"] = len(review_then_contact(log))
    pc = log[log.previously_contacted]
    o["Contacted, No new action later (rows)"] = int((pc.status == NNA).sum())
    o["Contacted, No new action later (enrolments)"] = int(pc.loc[pc.status == NNA, "eid"].nunique())
    o["Contacted, Review later (n)"] = int((pc.status == REVIEW).sum())
    return o


if __name__ == "__main__":
    print(__doc__)
    print("Import this module from capacity_aware_ews_model.ipynb; running it directly does nothing.")
