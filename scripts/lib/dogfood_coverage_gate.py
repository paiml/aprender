#!/usr/bin/env python3
"""Arithmetic half of the apr-dogfood blocking coverage gate (G2.2/G2.3/G2.4).

This module is deliberately git-free. It is handed two ledgers -- the COMPARAND,
which check_dogfood_coverage.sh extracts from protected `origin/main`, and the
CURRENT working-tree one -- and it reports where the second is worse than the
first. All git plumbing lives in the shell wrapper.

WHY THE COMPARAND IS NOT A LITERAL IN THIS REPO
-----------------------------------------------
The multi-platform dogfood gate had a floor and a universe that were both
literals in the same file, so ONE commit editing both defeated it. A gate whose
floor a PR can rewrite in the commit that breaks the floor is not a gate.

So there is no `142` and no `830` anywhere in this file. Every floor is DERIVED
from the comparand ledger at run time. To lower a floor you must land a commit on
`main`, where the required checks have already run.

WHAT IS CHECKED
---------------
  G2.2 reconciliation  no feature present in the comparand may vanish
  G2.3 floors          covered / rows / per-binary covered may not fall;
                       broken-and-ungated, UNKNOWN-hardware and
                       low-confidence-and-ungated may not rise
  G2.5 per-cluster     no cluster's gate count may fall and the zero-gate cluster
                       count may not rise (ratchet); no feature may CHANGE cluster
                       without a declared reassignment; a cluster leaves zero only
                       on an EARNED gate, never on an inherited one; every
                       cluster_label carries >= 1 earned gate (RED at
                       DOGFOOD_RELEASE=1)
  G2.6 T2 pairing      any emitted CLUSTER-level ratio must carry its FEATURE-level
                       ratio. Enforced over every surface that can emit one --
                       this gate's report, the receipt, the baseline printer and
                       the skill -- and keyed on the NUMBER (a denominator equal
                       to the cluster count) rather than on a phrase
  G2.4 waivers         every quality<=4 ungated feature has a triage entry, and
                       any feature NEWLY in that state carries an issue or a
                       written waiver. DOGFOOD_RELEASE=1 demands it of all.

Usage:
    python3 scripts/lib/dogfood_coverage_gate.py \
        --base <comparand.csv> --head <current.csv> --the44 <triage.yaml> \
        [--reassignments <log.yaml>] [--comparand-source ARMED|BOOTSTRAP] \
        [--comparand-ref <ref>] [--pair-scan <file> ...]
"""
import argparse
import collections
import csv
import os
import re
import sys

COLUMNS = ["binary", "feature", "quality_1_10", "verified_hardware",
           "top_competitor", "in_dogfood_skill", "cluster_id", "cluster_label",
           "evidence_path", "confidence"]

# The schema BEFORE the cluster columns were added. A comparand still on this
# shape is the one legitimate reason the per-cluster ratchet cannot be derived,
# and it is self-closing: it stops applying the moment the 10-column ledger is on
# main. HEAD is always held to COLUMNS -- a working tree may not drop back.
LEGACY_COLUMNS = [c for c in COLUMNS if not c.startswith("cluster_")]

# A ledger this small is not a ledger; it is a truncation. Refusing to compare
# against an empty or near-empty comparand is what stops "the file went missing"
# from reading as "every floor is satisfied".
MIN_ROWS = 100


class Fail(Exception):
    """A gate condition that must stop the run."""


def read_ledger(path, label, allow_legacy=False):
    """Read a ledger. `allow_legacy` is granted ONLY to the comparand, and only
    so a 10-column working tree can be compared against a `main` that has not
    taken the cluster columns yet. It is exhaustive over three shapes so that a
    HALF-migrated schema (one cluster column, or empty labels) can never fall
    through to "no per-cluster floor to check"."""
    if not os.path.exists(path):
        raise Fail(f"{label} ledger not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames
        if cols == COLUMNS:
            clustered = True
        elif allow_legacy and cols == LEGACY_COLUMNS:
            clustered = False
        else:
            raise Fail(f"{label} ledger has unexpected columns: {cols}\n"
                       f"        expected: {COLUMNS}"
                       + ("" if allow_legacy else
                          "\n        (the working ledger must carry the cluster "
                          "columns; dropping them would retire the per-cluster "
                          "floor in the same commit that breaks it)"))
        rows = list(rdr)
    if len(rows) < MIN_ROWS:
        raise Fail(f"{label} ledger has only {len(rows)} rows (< {MIN_ROWS}); "
                   "refusing to treat a truncated ledger as a satisfied floor")
    if clustered:
        def _unlabelled(r):
            has_id = bool((r["cluster_id"] or "").strip())  # cluster-id-guard allow (presence check, not a key)
            return not (r["cluster_label"] or "").strip() or not has_id

        blank = [f"{r['binary']}: {r['feature']}" for r in rows if _unlabelled(r)]
        if blank:
            raise Fail(f"{label} ledger has {len(blank)} row(s) with an empty "
                       "`cluster_id` or `cluster_label`. An unlabelled row belongs to "
                       "no cluster, so it is outside every per-cluster floor:\n"
                       + "\n".join(f"        - {b}" for b in blank[:10]))
    return rows, clustered


def covered(row):
    return (row["in_dogfood_skill"] or "").strip().lower() == "yes"


def quality(row):
    return int(row["quality_1_10"])


def key(row):
    """Identity of a ledger row. binary+feature, because `POST /v1/models` is a
    different row for `apr` and for `aprender-orchestrate`."""
    return (row["binary"], row["feature"])


def broken_and_ungated(rows):
    return {key(r) for r in rows if quality(r) <= 4 and not covered(r)}


def unknown_hardware(rows):
    return sum(1 for r in rows if r["verified_hardware"].strip().upper() == "UNKNOWN")


def low_and_ungated(rows):
    return sum(1 for r in rows
               if r["confidence"].strip().lower() == "low" and not covered(r))


def covered_by_binary(rows):
    out = collections.Counter()
    for r in rows:
        out[r["binary"]] += 1 if covered(r) else 0
    return out


def cluster_label(row):
    """The DURABLE cluster key. Never `cluster_id` -- see T1 below."""
    return (row["cluster_label"] or "").strip()


def by_cluster(rows):
    """-> {label: (n_features, n_gated)}, sorted by size descending then name."""
    n = collections.Counter()
    g = collections.Counter()
    for r in rows:
        lab = cluster_label(r)
        n[lab] += 1
        g[lab] += 1 if covered(r) else 0
    return collections.OrderedDict(
        (lab, (n[lab], g[lab])) for lab in sorted(n, key=lambda k: (-n[k], k)))


# --------------------------------------------------------------------------
# G2.2 -- reconciliation. A row that disappears takes its defect with it.
# --------------------------------------------------------------------------
def check_reconciliation(base, head, findings):
    gone = sorted({key(r) for r in base} - {key(r) for r in head})
    if gone:
        findings.append(
            "G2.2 reconciliation FAIL: {} feature(s) present on the comparand are "
            "absent from the working ledger. Deleting a row deletes its defect "
            "along with it; if the surface genuinely shrank, that is a finding to "
            "record, not a row to drop:\n".format(len(gone))
            + "\n".join(f"        - {b}: {f}" for b, f in gone[:20])
            + ("\n        ... and {} more".format(len(gone) - 20) if len(gone) > 20 else ""))
        return False
    print(f"  G2.2 reconciliation  PASS  no row lost ({len(base)} comparand rows all present)")
    return True


# --------------------------------------------------------------------------
# G2.3 -- floors, every one derived from the comparand.
# --------------------------------------------------------------------------
def _ratchet(findings, label, now, floor, direction, gate="G2.3 floors"):
    """direction 'up' = now must be >= floor; 'down' = now must be <= floor.

    `gate` names the gate the finding belongs to. It is a parameter and not a
    constant because the per-cluster floor reuses this helper: a G2.5 breakage
    reported as "G2.3 floors FAIL" sends the reader to the wrong gate, and a
    misattributed finding is only slightly better than no finding. Caught by
    running M5 against the REAL 830-row ledger rather than only the fixture."""
    ok = now >= floor if direction == "up" else now <= floor
    arrow = ">=" if direction == "up" else "<="
    if not ok:
        findings.append(f"{gate} FAIL: {label} is {now}, must be {arrow} {floor} "
                        f"(floor derived from the comparand, not from this tree)")
    return ok


def check_floors(base, head, findings):
    ok = True
    ok &= _ratchet(findings, "total rows (the denominator)", len(head), len(base), "up")
    ok &= _ratchet(findings, "covered rows", sum(1 for r in head if covered(r)),
                   sum(1 for r in base if covered(r)), "up")
    ok &= _ratchet(findings, "broken-and-ungated", len(broken_and_ungated(head)),
                   len(broken_and_ungated(base)), "down")
    ok &= _ratchet(findings, "UNKNOWN verified_hardware", unknown_hardware(head),
                   unknown_hardware(base), "down")
    ok &= _ratchet(findings, "low-confidence AND ungated", low_and_ungated(head),
                   low_and_ungated(base), "down")

    # Per-binary, because the aggregate hides a swap: dropping a gate on `apr`
    # while adding one elsewhere leaves the ratio untouched. Per-binary is what
    # keeps "27 of 28 binaries at zero" from being quietly traded away.
    base_cov, head_cov = covered_by_binary(base), covered_by_binary(head)
    for binary in sorted(base_cov):
        ok &= _ratchet(findings, f"covered in `{binary}`",
                       head_cov[binary], base_cov[binary], "up")
    if ok:
        n, d = sum(1 for r in head if covered(r)), len(head)
        print(f"  G2.3 floors          PASS  {n}/{d} covered "
              f"({100.0 * n / d:.1f}%), {len(base_cov)} per-binary floors held")
    return ok


# --------------------------------------------------------------------------
# G2.5 -- the PER-CLUSTER floor, and the T2 reporting rule that keeps it honest.
#
# WHY A BINARY IS THE WRONG UNIT
# ------------------------------
# `aprender-orchestrate` ships 184 features that are three unrelated subsystems:
# 95 Banco HTTP routes, a 56-feature agent stack and 17 Pacha secrets commands.
# A per-BINARY floor of ">= 1 gate" lets one gate on Pacha make all 184 look
# touched. The cluster is the unit that shares a module, a dispatch path and a
# failure mode, so a gate on one member is evidence about the cluster and a gate
# on a different subsystem is not.
#
# WHAT IS ENFORCED
#   ratchet (always)     no cluster's gate count may fall; the number of clusters
#                        with zero gates may not rise
#   release arm          every cluster_label carries >= 1 gate. RED at
#                        DOGFOOD_RELEASE=1. Nine of fourteen clusters sit at zero
#                        today, so this is red on arrival AT RELEASE and that is
#                        the finding, not a bug in the gate.
#
# T1 -- NOTHING KEYS ON `cluster_id`
# ----------------------------------
# k-means labels PERMUTE whenever the input moves. `cluster_id` is provenance;
# `cluster_label` is the durable key and is human-owned after first assignment.
# This module reads `cluster_id` ONLY to prove the id->label map is a bijection,
# never to identify a cluster, and says so with a pragma on each of those lines.
# The repo-wide ban is enforced separately by scripts/check_no_cluster_id_keys.sh,
# which is an ALLOWLIST: a standalone token is a key unless the line says why not.
#
# T2 -- CLUSTER COVERAGE IS NOT FEATURE COVERAGE
# ----------------------------------------------
# One gate in a 95-member cluster is 1%, not "covered". A receipt that reports
# "5 of 14 clusters gated" WITHOUT the underlying 142 of 830 rebuilds the vacuity
# failure one level up: a clean sweep over a proxy, looking stricter than what it
# replaced. So the pairing is MECHANICAL, not a documented convention -- and it
# is applied on every channel a cluster ratio can leave through, not only on the
# report this function prints. See the block below for what "channel" means and
# why the rule keys on the NUMBER instead of on a phrase.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# T2, MECHANICALLY, AND KEYED ON THE NUMBER
#
# The first version of this rule recognised ONE phrasing ("clusters ... N/M")
# and was applied at ONE call site -- the gate's own report. That is the same
# shape of defect it exists to prevent: a receipt states cluster coverage
# WITHOUT the feature fraction, the gate looks STRICTER than before, and it is
# measuring LESS. A pairing rule enforced over one phrasing on one channel is a
# pairing rule you can walk around by rewording, or by printing somewhere else.
#
# So the rule is now about the NUMBER. A ratio whose DENOMINATOR is the cluster
# count is a cluster-level claim whatever words surround it, and it must be
# accompanied by a ratio whose denominator is the feature count. The phrase
# forms are kept as ADDITIONAL triggers -- a union of triggers can only make the
# rule stricter -- and `N of M` counts as a ratio because prose writes it that
# way.
#
# Both counts are DERIVED from the working ledger. Neither is a literal here.
# --------------------------------------------------------------------------

# `N/M` or `N of M`. The lookarounds keep `1/2/3` and `1.5/2` out.
_RATIO = re.compile(r"(?<![\d./])(\d+)\s*(?:/|\bof\b)\s*(\d+)(?![\d./])", re.I)
# Legacy phrase triggers, retained as a union so nothing that used to be caught
# stops being caught when the ledger's cluster count happens to change.
_CLUSTER_FRAC = re.compile(r"cluster[s\-]?[^,;|]*?\b(\d+)\s*(?:/|\bof\b)\s*(\d+)", re.I)
_FEATURE_FRAC = re.compile(r"feature[s\-]?[^,;|]*?\b(\d+)\s*(?:/|\bof\b)\s*(\d+)", re.I)
# A percentage attached to the PLURAL word, in either order and within a short
# window: "35.7% of clusters gated", "clusters gated: 35.7%". Plural and adjacent
# on purpose -- "one gate in a 95-member cluster is 1%" is a statement about a
# cluster's INTERIOR, not a ratio over the cluster count, and a trigger that
# fired on it would make the rule noisy enough to be turned off.
_CLUSTER_PCT = re.compile(
    r"(\bclusters\b[^|]{0,30}?\d+(?:\.\d+)?\s*%"
    r"|\d+(?:\.\d+)?\s*%[^|]{0,30}?\bclusters\b)", re.I)
# A line may opt out only by SAYING SO, on the line, with a reason -- the same
# shape as the T1 pragma. Silence is never an opt-out.
_PAIR_PRAGMA = re.compile(r"t2-pairing[: ]*allow[ \t]*\(", re.I)
# `clusters: 14` states ONE number. There is no ratio on it to pair, and holding
# a scalar count declaration to a pairing rule would demand a second number that
# does not belong there. This is a position exemption, not a phrase one: a line
# whose entire content is `<name>: <integer>` cannot be a fraction.
_SCALAR_DECL = re.compile(r"^[\s\-]*[\w.\[\]\"'`]+\s*[:=]\s*\d+\s*,?\s*$")


def _denominators(text):
    return {int(m.group(2)) for m in _RATIO.finditer(text)}


def states_cluster_ratio(line, n_clusters):
    """Union of four triggers. The NUMBER comes first, because a phrase can be
    reworded and a denominator cannot."""
    if n_clusters and n_clusters in _denominators(line):
        return True
    # "14 clusters, 5 gated" states the ratio without writing a ratio. The
    # cluster COUNT beside the word is still the number, and it is derived.
    if n_clusters and re.search(
            r"(\b{n}\b[^|]{{0,20}}?\bclusters\b|\bclusters\b[^|]{{0,20}}?\b{n}\b)"
            .format(n=n_clusters), line, re.I):
        return True
    if _CLUSTER_FRAC.search(line):
        return True
    # A percentage next to the plural word is a cluster-level claim with the
    # fraction hidden. "35.7% of clusters gated" evades every ratio pattern.
    return bool(_CLUSTER_PCT.search(line))


def states_feature_ratio(line, n_features):
    if n_features and n_features in _denominators(line):
        return True
    return bool(_FEATURE_FRAC.search(line))


def enforce_pairing(lines, findings, n_clusters, n_features,
                    where="the per-cluster report"):
    """T2. Every emitted line stating a CLUSTER-level ratio must state the
    underlying FEATURE-level ratio too. Returns True when the text is
    admissible.

    `n_clusters` / `n_features` come from the working ledger, so the rule keys
    on the number rather than on a wording that can be edited around."""
    offenders = []
    seen = False
    for ln in lines:
        if _SCALAR_DECL.match(ln):
            continue
        if not states_cluster_ratio(ln, n_clusters):
            continue
        seen = True
        if states_feature_ratio(ln, n_features):
            continue
        if _PAIR_PRAGMA.search(ln):
            continue
        offenders.append(ln)
    if offenders:
        findings.append(
            "G2.6 T2 FAIL ({}): {} line(s) state a CLUSTER-level ratio with no "
            "FEATURE-level ratio beside it. One gate in a 95-member cluster is "
            "1%, not \"covered\"; reporting the proxy alone is how a coverage "
            "gate becomes theatre while looking stricter than before:\n".format(
                where, len(offenders))
            + "\n".join(f"        | {ln.rstrip()}" for ln in offenders[:10]))
        return False
    # Vacuity guard: a pairing rule applied to text that states no cluster ratio
    # proves nothing. Only the gate's OWN report is held to this -- a scanned
    # file may legitimately not mention clusters, and demanding that it does
    # would be a different rule wearing this one's name.
    if not seen and where == "the per-cluster report":
        findings.append(
            "G2.6 T2 FAIL: the per-cluster report emitted no cluster-level ratio "
            "at all, so the pairing rule checked nothing. An empty report is not "
            "a satisfied one.")
        return False
    return True


def _check_id_label_bijection(head, findings):
    """`cluster_id` is read HERE and nowhere else, and only to prove it agrees
    with the label. A label served by two ids -- or an id serving two labels --
    means a re-run permuted the ids and the ledger took the permutation
    halfway."""
    ids = collections.defaultdict(set)
    labs = collections.defaultdict(set)
    for r in head:
        ids[cluster_label(r)].add((r["cluster_id"] or "").strip())  # cluster-id-guard allow (bijection check, not a key)
        labs[(r["cluster_id"] or "").strip()].add(cluster_label(r))  # cluster-id-guard allow (bijection check, not a key)
    bad = ([f"label `{k}` carries {len(v)} ids: {sorted(v)}"
            for k, v in sorted(ids.items()) if len(v) != 1]
           + [f"id `{k}` carries {len(v)} labels: {sorted(v)}"
              for k, v in sorted(labs.items()) if len(v) != 1])
    if bad:
        findings.append("G2.5 FAIL: `cluster_id` and `cluster_label` disagree:\n"
                        + "\n".join(f"        - {b}" for b in bad[:10]))
        return False
    return True


# --------------------------------------------------------------------------
# MEMBERSHIP -- the ratchet the first version did not have.
#
# THE ATTACK IT CLOSES
# --------------------
# The release arm demands that every cluster_label carry at least one gate.
# Ratcheting gate COUNT per label, label PRESENCE and the zero-gate cluster
# count leaves that arm satisfiable BY RELABELLING ALONE: take a feature that
# already carries a gate, move it into a zero-gate cluster, and that cluster
# stops being at zero without one line of new evidence being written. Nine
# zeros become nine tickets only if a zero cannot be made to stop EXISTING.
# It is the same move as deleting a losing benchmark row, which this repo has
# done exactly once (d7e08043b -- 395 deletions, the only beat deletion in its
# history, and it removed the only two losing rows).
#
# THE HONEST TENSION
# ------------------
# Clusters are DERIVED. Re-running the clustering legitimately moves members,
# so a prohibition would be a lie -- it would forbid the one operation that
# keeps the ledger true. That is precisely why the answer is a DECLARATION and
# not a ban: a re-cluster must be possible; a SILENT re-cluster that retires an
# obligation must not be.
#
# SO, TWO RULES, AND THEY ARE DIFFERENT RULES
# -------------------------------------------
#   membership   a feature that changes cluster_label between the comparand and
#                the working ledger must appear in the reassignment log with its
#                exact from -> to and a stated reason. Undeclared moves FAIL.
#                Checked in BOTH directions: an entry whose `to` disagrees with
#                the working ledger describes something that did not happen, and
#                pre-authorising a move you have not made is how a declaration
#                turns into a blanket permit.
#
#   earned       a declaration makes a move LEGIBLE; it does not turn an
#                inherited gate into evidence. So the zero-gate set and the
#                release arm are computed over EARNED gates only -- a gate on a
#                feature that was already in the cluster on the comparand, or on
#                surface that is genuinely new. A gate that walked in from
#                another cluster counts for the ledger's totals and NOT for the
#                obligation it would otherwise retire. Writing a gate is the
#                only way off zero.
# --------------------------------------------------------------------------

_REASSIGN_RE = re.compile(
    r"^[ \t]*-[ \t]+binary:[ \t]*(?P<binary>.*?)[ \t]*$"
    r"(?P<body>(?:\n(?![ \t]*-[ \t]+binary:).*)*)", re.M)
_FIELD_TMPL = "^[ \t]+{}:[ \t]*(?P<v>.*?)[ \t]*$"


def _field(body, name):
    m = re.search(_FIELD_TMPL.format(name), body, re.M)
    return (m.group("v").strip().strip("'\"") if m else "")


def parse_reassignments(path):
    """-> [ {binary, feature, from, to, reason} ]. Deliberately small, and
    deliberately not a yaml library: this module has no third-party imports, and
    a gate that needs one is a gate that can fail to run."""
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    out = []
    for m in _REASSIGN_RE.finditer(text):
        body = m.group("body")
        out.append({
            "binary": m.group("binary").strip().strip("'\""),
            "feature": _field(body, "feature"),
            "from": _field(body, "from"),
            "to": _field(body, "to"),
            "reason": _field(body, "reason"),
        })
    return out


def check_membership(base, head, declared, findings):
    """No feature changes cluster silently, and no declaration describes a move
    that did not happen."""
    base_lab = {key(r): cluster_label(r) for r in base}
    head_lab = {key(r): cluster_label(r) for r in head}
    moved = {k: (base_lab[k], head_lab[k]) for k in base_lab
             if k in head_lab and base_lab[k] != head_lab[k]}
    declared_by_key = {(d["binary"], d["feature"]): d for d in declared}

    ok = True
    undeclared = sorted(k for k in moved if k not in declared_by_key)
    if undeclared:
        findings.append(
            "G2.5 membership FAIL: {} feature(s) changed cluster_label with no "
            "entry in the reassignment log. A cluster is DERIVED, so re-running "
            "the clustering may legitimately move members -- but a SILENT move "
            "retires an obligation: a gated feature walked into a zero-gate "
            "cluster is a zero that stopped existing without a gate being "
            "written. Declare the move (binary/feature/from/to/reason):\n"
            .format(len(undeclared))
            + "\n".join("        - {}: {}   {} -> {}".format(
                b, f, moved[(b, f)][0], moved[(b, f)][1])
                for b, f in undeclared[:20]))
        ok = False

    wrong = []
    for (b, f), d in sorted(declared_by_key.items()):
        k = (b, f)
        if k not in head_lab:
            wrong.append(f"{b}: {f}   declared, but absent from the working ledger")
        elif head_lab[k] != d["to"]:
            wrong.append("{}: {}   declared `to: {}`, ledger says `{}`".format(
                b, f, d["to"] or "<empty>", head_lab[k]))
        elif k in moved and moved[k][0] != d["from"]:
            wrong.append("{}: {}   declared `from: {}`, comparand says `{}`".format(
                b, f, d["from"] or "<empty>", moved[k][0]))
        elif not d["reason"]:
            wrong.append(f"{b}: {f}   declared with no stated reason")
    if wrong:
        findings.append(
            "G2.5 membership FAIL: {} reassignment entr(y/ies) do not describe "
            "the working ledger. A declaration that does not match reality is a "
            "blanket permit -- pre-authorising a move you have not made lets the "
            "next one happen silently:\n".format(len(wrong))
            + "\n".join(f"        - {w}" for w in wrong[:20]))
        ok = False

    if ok:
        print("  G2.5 membership      PASS  {} feature(s) changed cluster, every "
              "one declared ({} log entries)".format(len(moved), len(declared)))
    return ok


def earned_by_cluster(base, head):
    """-> {label: (n_features, gates, earned, inherited)}, largest cluster first.

    EARNED excludes a gate that arrived by moving an already-gated feature in
    from another cluster. It is the number the zero-gate obligation is measured
    on, because an inherited gate is evidence about the cluster it came FROM."""
    base_lab = {key(r): cluster_label(r) for r in base}
    n = collections.Counter()
    g = collections.Counter()
    e = collections.Counter()
    for r in head:
        lab = cluster_label(r)
        n[lab] += 1
        if not covered(r):
            continue
        g[lab] += 1
        prior = base_lab.get(key(r))
        if prior is None or prior == lab:
            e[lab] += 1
    return collections.OrderedDict(
        (lab, (n[lab], g[lab], e[lab], g[lab] - e[lab]))
        for lab in sorted(n, key=lambda k: (-n[k], k)))


def check_cluster_floors(base, head, base_clustered, findings, release,
                         declared, armed_note):
    """The third floor, beside overall and per-binary."""
    ok = _check_id_label_bijection(head, findings)
    head_c = by_cluster(head)
    feat_n = sum(1 for r in head if covered(r))
    feat_d = len(head)

    if base_clustered:
        ok &= check_membership(base, head, declared, findings)
        earned_c = earned_by_cluster(base, head)
        base_c = by_cluster(base)
        # Ratchet, derived from the comparand -- never from a literal here.
        for lab in sorted(base_c):
            ok &= _ratchet(findings, f"gates in cluster `{lab}`",
                           head_c.get(lab, (0, 0))[1], base_c[lab][1], "up",
                           gate="G2.5 per-cluster")
        # A cluster that vanishes takes its whole floor with it.
        gone = sorted(set(base_c) - set(head_c))
        if gone:
            findings.append(
                "G2.5 FAIL: {} cluster_label(s) on the comparand are absent from "
                "the working ledger. Renaming a label retires every gate "
                "obligation that cited it; relabel in a commit that says so:\n"
                .format(len(gone))
                + "\n".join(f"        - {g}" for g in gone[:10]))
            ok = False
        # Zero counted on EARNED gates: an inherited gate may not lift a cluster
        # off zero, or the ratchet is satisfiable by relabelling alone.
        ok &= _ratchet(findings, "clusters with ZERO earned gates",
                       sum(1 for v in earned_c.values() if v[2] == 0),
                       sum(1 for _, g in base_c.values() if g == 0), "down",
                       gate="G2.5 per-cluster")
    else:
        earned_c = collections.OrderedDict(
            (lab, (n, g, g, 0)) for lab, (n, g) in head_c.items())
        print("  SCHEMA UPGRADE: the comparand ledger has no cluster columns, so "
              "the\n                  per-cluster RATCHET and the membership "
              "ratchet have no floor\n                  to derive and are skipped "
              "for this run only. The release arm\n                  below still "
              "applies. Once the 10-column ledger is on main this\n"
              "                  branch is unreachable.")

    zero = [lab for lab, v in earned_c.items() if v[2] == 0]
    gated_clusters = len(earned_c) - len(zero)
    if release and zero:
        findings.append(
            "G2.5 FAIL (DOGFOOD_RELEASE=1): {} of {} clusters carry no EARNED "
            "gate at all. Nine clusters at zero is nine clusters with NO "
            "EVIDENCE -- that is the gap, not the {} uncovered rows. A gate on "
            "one member is evidence about the cluster; zero members is silence, "
            "and a gate that walked in from another cluster is evidence about "
            "where it came from:\n".format(len(zero), len(earned_c),
                                           feat_d - feat_n)
            + "\n".join("        - {:<26} {:>4} features, 0 earned gates"
                        "{}".format(lab, earned_c[lab][0],
                                    f" ({earned_c[lab][3]} inherited)"
                                    if earned_c[lab][3] else "")
                        for lab in zero))
        ok = False

    # ---- the report. Built as text FIRST so T2 can read it back. ----
    lines = ["  G2.5 per-cluster     {}  clusters gated {}/{} ({:.1f}%), "
             "features gated {}/{} ({:.1f}%)".format(
                 "PASS " if ok else "FAIL ",
                 gated_clusters, len(earned_c),
                 100.0 * gated_clusters / len(earned_c) if earned_c else 0.0,
                 feat_n, feat_d, 100.0 * feat_n / feat_d if feat_d else 0.0)]
    tot_gates = sum(v[1] for v in earned_c.values())
    for lab, (n, g, e, inh) in earned_c.items():
        lines.append(
            "        {:<26} features {:>3}/{:<3} ({:>5.1f}%)  "
            "share of gate effort {:>5.1f}%{}".format(
                lab, g, n, 100.0 * g / n if n else 0.0,
                100.0 * g / tot_gates if tot_gates else 0.0,
                f"  [{inh} inherited, {e} earned]" if inh else ""))
    lines.append(f"        [{armed_note}]")

    ok = enforce_pairing(lines, findings, len(earned_c), feat_d) and ok
    for ln in lines:
        print(ln)
    return ok

# --------------------------------------------------------------------------
# G2.4 -- the 44. A known-broken ungated feature needs an issue or a waiver.
# --------------------------------------------------------------------------
ENTRY_RE = re.compile(
    r"^- feature: (?P<feature>.*?)$"
    r"(?P<body>(?:\n(?!- feature:).*)*)", re.M)


def parse_triage(path):
    """-> {feature: (has_issue, has_waiver)}. A deliberately small parser: the
    file is generated by scripts/dogfood_the44.py and its shape is fixed."""
    if not os.path.exists(path):
        raise Fail(f"triage file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    out = {}
    for m in ENTRY_RE.finditer(text):
        feature = m.group("feature").strip().strip("'\"")
        body = m.group("body")
        has_issue = bool(re.search(r"^\s+issue:\s*(?!null\s*$)\S", body, re.M))
        has_waiver = bool(re.search(r"^\s+waiver:\s*(?!null\s*$)\S", body, re.M))
        out[feature] = (has_issue, has_waiver)
    return out


def _bullets(features):
    return "\n".join(f"        - {f}" for f in features[:20])


def _check_membership(broken_now, triage, findings):
    """Both directions -- an untracked defect and a stale entry are both ways
    for the triage file to stop describing reality."""
    ok = True
    missing = sorted({f for _, f in broken_now} - set(triage))
    if missing:
        findings.append("G2.4 waivers FAIL: quality<=4 ungated feature(s) with no "
                        "triage entry:\n" + _bullets(missing))
        ok = False
    stale = sorted(set(triage) - {f for _, f in broken_now})
    if stale:
        findings.append("G2.4 waivers FAIL: triage entries for feature(s) no longer "
                        "broken-and-ungated:\n" + _bullets(stale))
        ok = False
    return ok


def _check_new_breakage(broken_now, broken_before, triage, findings):
    """A feature NEWLY quality<=4-and-ungated needs an issue or a waiver at
    once. The pre-existing set is grandfathered by the comparand, which is what
    makes this a ratchet rather than a wall that is red on arrival."""
    newly = sorted({f for _, f in broken_now} - {f for _, f in broken_before})
    untriaged = [f for f in newly if not any(triage.get(f, (False, False)))]
    if untriaged:
        findings.append(
            "G2.4 waivers FAIL: feature(s) newly quality<=4 and ungated, with "
            "neither an issue nor a written waiver:\n" + _bullets(untriaged))
    return not untriaged, newly


def _check_release_arm(triage, findings):
    """Off mid-cycle so the gate is usable; the release checklist sets
    DOGFOOD_RELEASE=1 and then every entry must be triaged."""
    untriaged = sorted(f for f, v in triage.items() if not any(v))
    if untriaged:
        findings.append(
            f"G2.4 waivers FAIL (DOGFOOD_RELEASE=1): {len(untriaged)} of "
            f"{len(triage)} triage entries carry neither an issue nor a waiver. "
            "A waiver is a sentence saying why shipping this ungated is "
            "acceptable; \"low priority\" is not one:\n" + _bullets(untriaged))
    return not untriaged


def check_waivers(base, head, triage_path, findings, release):
    triage = parse_triage(triage_path)
    broken_now = broken_and_ungated(head)
    broken_before = broken_and_ungated(base)

    ok = _check_membership(broken_now, triage, findings)
    new_ok, newly = _check_new_breakage(broken_now, broken_before, triage, findings)
    ok = new_ok and ok
    if release:
        ok = _check_release_arm(triage, findings) and ok
    if ok:
        arm = "release arm ON" if release else "release arm off"
        print(f"  G2.4 waivers         PASS  {len(broken_now)} broken-and-ungated, "
              f"{len(triage)} triaged, {len(newly)} new ({arm})")
    return ok


def scan_files_for_pairing(paths, n_clusters, n_features, findings):
    """T2 over the OTHER channels. The gate's own report is only one place a
    cluster ratio can be emitted; the receipt, the baseline printer's output and
    the skill's allocation table are three more, and a rule enforced on one
    channel is a rule with three ways around it."""
    ok = True
    for path in paths:
        if not os.path.exists(path):
            findings.append(
                f"G2.6 T2 FAIL: {path} is named as a surface that can emit a "
                "cluster ratio, but it does not exist. A scan target that is "
                "missing is a channel that is unchecked.")
            ok = False
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        ok = enforce_pairing(lines, findings, n_clusters, n_features,
                             where=path) and ok
    return ok


def run(args):
    head, _ = read_ledger(args.head, "working-tree")
    n_clusters = len(by_cluster(head))
    n_features = len(head)

    # --pair-scan alone is the T2-over-other-channels mode. It needs the working
    # ledger (to derive the two denominators) and nothing else.
    if args.pair_scan and not args.base:
        findings = []
        ok = scan_files_for_pairing(args.pair_scan, n_clusters, n_features, findings)
        if ok:
            print("  G2.6 T2 pairing      PASS  {} surface(s) scanned; every "
                  "cluster-level ratio (denominator {}) carries its feature-level "
                  "ratio (denominator {})".format(
                      len(args.pair_scan), n_clusters, n_features))
        for f in findings:
            print(f"  {f}")
        return 0 if ok else 1

    base, base_clustered = read_ledger(args.base, "comparand", allow_legacy=True)
    declared = parse_reassignments(args.reassignments)
    release = bool(os.environ.get("DOGFOOD_RELEASE"))

    # The banner must name the path that RAN. An earlier draft described
    # `read_ledger(..., allow_legacy=True)` as the active branch while
    # resolve_base_ref was in fact taking BOOTSTRAP first -- so the comparand was
    # this branch's own HEAD commit, carrying the 10-column schema, and the
    # legacy branch never executed at all. A window that is honest about being
    # open is fine; a window whose description does not match its code is not.
    schema = "10-column (cluster columns present)" if base_clustered \
        else "8-column legacy (allow_legacy ENGAGED)"
    armed_note = "ratchet {} -- comparand {}, {} schema".format(
        "armed from the comparand" if base_clustered
        else "NOT armed: the comparand predates the cluster columns",
        args.comparand_source or "unknown", schema)

    findings = []
    ok = check_reconciliation(base, head, findings)
    ok = check_floors(base, head, findings) and ok
    ok = check_cluster_floors(base, head, base_clustered, findings, release,
                              declared, armed_note) and ok
    if args.pair_scan:
        pair_ok = scan_files_for_pairing(args.pair_scan, n_clusters, n_features,
                                         findings)
        if pair_ok:
            print("  G2.6 T2 pairing      PASS  {} further surface(s) scanned "
                  "(receipt/skill/baseline); a cluster-level ratio is a ratio "
                  "over {} and it must carry one over {}".format(
                      len(args.pair_scan), n_clusters, n_features))
        ok = pair_ok and ok
    ok = check_waivers(base, head, args.the44, findings, release) and ok
    for f in findings:
        print(f"  {f}")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", help="comparand ledger (from origin/main)")
    p.add_argument("--head", required=True, help="working-tree ledger")
    p.add_argument("--the44", help="triage yaml")
    p.add_argument("--reassignments", default=None,
                   help="declared cluster reassignment log (yaml)")
    p.add_argument("--comparand-source", default=None,
                   help="ARMED|BOOTSTRAP -- how the shell wrapper resolved the "
                        "comparand, so the banner can name the path that ran")
    p.add_argument("--pair-scan", action="append", default=[], metavar="FILE",
                   help="additional surface to hold to the T2 pairing rule "
                        "(receipt, skill, baseline output). Repeatable.")
    args = p.parse_args(argv)
    if args.base and not args.the44:
        p.error("--the44 is required alongside --base")
    if not args.base and not args.pair_scan:
        p.error("give --base (full gate) or --pair-scan (T2-only mode)")
    try:
        return run(args)
    except Fail as exc:
        print(f"  GATE ABORTED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
