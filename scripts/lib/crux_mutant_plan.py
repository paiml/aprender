"""crux_mutant_plan.py: which F6 mutants scripts/check_crux_inference_judge.sh runs (#4098).

Each F6 mutant deletes one rule from a copy of the judge and re-runs the WHOLE case table, so the full set costs
~630 of the table's 655 s (lambda, 2026-09-23), and guard-tree runs every cargo-free guard under one 30-minute job
limit (the 0.69.1 freeze head's guard-tree was cancelled at 30.7 min). So:

  all      every mutant. The default wherever no CI event says otherwise (a local run), on `schedule` and
           `workflow_dispatch`, and in guards-nightly.yml, which asks for it by name (CRUX_MUTANTS=all).
  sample   on `pull_request`, `merge_group` and `push`: every mutant when the diff touches a file the mutants
           test (the judge, its imports, this table, this planner) — a change to a rule re-proves every rule —
           otherwise a ROTATING slice of `--sample` mutants, its offset taken from the head sha, so successive
           heads cover different rules and none is skipped for ever.
  none     CRUX_MUTANTS=none: the table's recursion into itself for each mutant, never a CI mode.

A sample is refused below its floor rather than run vacuously: `--sample` under `--floor` (or a selection that came
out smaller than the floor while more mutants exist) exits 2 naming why. CRUX_MUTANTS overrides the event; an
unknown value exits 2. An unreadable diff never passes as "untouched": it is named in the reason, and the sample
still runs (the nightly runs the rest).

  crux_mutant_plan.py --labels a,b,c [--event E] [--mode M] [--changed FILE|-] [--head SHA] [--sample N] [--floor N]
  prints one JSON object: {"mode", "reason", "selected": [...], "total"}
"""
import argparse
import json
import sys

# A change to any of these can change what a mutant tests, so it re-runs every mutant.
WATCHED = (
    "scripts/lib/crux_inference_judge.py",
    "scripts/lib/crux_oracles.py",
    "scripts/lib/crux_serve_routes.py",
    "scripts/lib/crux_prompt_certify.py",
    "scripts/check_crux_inference_judge.sh",
    "scripts/lib/crux_mutant_plan.py",
    "scripts/lib/crux_mutant_plan.sh",
    "scripts/lib/resolve_base.sh",
)
FULL_EVENTS = ("schedule", "workflow_dispatch")
MIN_FLOOR = 6  # the floor may be RAISED by --floor, never lowered: 0 made a 0-of-24 sample a pass (round 4, lane 2)
SAMPLED_EVENTS = ("pull_request", "merge_group", "push")


def plan(labels, event, mode, changed, head, sample, floor):
    total = len(labels)
    if mode:
        if mode not in ("all", "sample", "none"):
            raise SystemExit("crux_mutant_plan: CRUX_MUTANTS=%r is not all, sample or none" % mode)
        why = "CRUX_MUTANTS=%s" % mode
    elif event in FULL_EVENTS or not event:
        mode, why = "all", ("event %s runs every mutant" % event) if event else "no CI event (a local run): every mutant"
    elif event in SAMPLED_EVENTS:
        mode, why = "sample", "event %s" % event
    else:
        raise SystemExit("crux_mutant_plan: event %r is neither a full nor a sampled event; name it in this planner"
                         % event)
    if mode == "none":
        return {"mode": "none", "reason": why, "selected": [], "total": total}
    if mode == "all":
        return {"mode": "all", "reason": why, "selected": list(labels), "total": total}
    if floor < MIN_FLOOR:
        raise SystemExit("crux_mutant_plan: a floor of %d is under the minimum of %d: the floor can be raised, never "
                         "lowered" % (floor, MIN_FLOOR))
    if sample < floor:
        raise SystemExit("crux_mutant_plan: a sample of %d is under the floor of %d: a smaller sample would pass "
                         "vacuously" % (sample, floor))
    if changed is None:
        why += "; the diff could not be read, so nothing counts as touched (the nightly runs the rest)"
    else:
        hit = sorted(set(changed) & set(WATCHED))
        if hit:
            return {"mode": "all", "reason": why + "; the diff touches %s: every mutant" % ", ".join(hit),
                    "selected": list(labels), "total": total}
        why += "; the diff touches none of the mutants' files"
    if total <= sample:
        return {"mode": "sample", "reason": why + "; %d mutants, all run" % total, "selected": list(labels),
                "total": total}
    try:
        off = int((head or "")[:8], 16) % total
    except ValueError:
        raise SystemExit("crux_mutant_plan: head %r is not a sha: the rotation needs one" % head)
    sel = [labels[(off + i) % total] for i in range(sample)]
    if len(sel) < floor:
        raise SystemExit("crux_mutant_plan: selected %d, under the floor of %d" % (len(sel), floor))
    return {"mode": "sample", "reason": why + "; rotating slice from %d of %d (head %s)" % (off, total, head[:9]),
            "selected": sel, "total": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--event", default="")
    p.add_argument("--mode", default="")
    p.add_argument("--changed", help="a file of changed paths, one per line; absent = the diff was unreadable")
    p.add_argument("--head", default="")
    p.add_argument("--sample", type=int, default=6)
    p.add_argument("--floor", type=int, default=6)
    a = p.parse_args()
    labels = [x for x in a.labels.split(",") if x]
    if not labels:
        raise SystemExit("crux_mutant_plan: no mutant labels: an empty set is never a pass")
    changed = None
    if a.changed:
        try:
            changed = [l.strip() for l in open(a.changed) if l.strip()]
        except OSError:
            changed = None
    print(json.dumps(plan(labels, a.event, a.mode, changed, a.head, a.sample, a.floor)))


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if isinstance(e.code, str):
            sys.stderr.write(e.code + "\n")
            sys.exit(2)
        raise
