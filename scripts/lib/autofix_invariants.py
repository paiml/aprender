"""autofix_invariants.py -- what a BOOKKEEPING auto-fix may change, and nothing more (#4045 M8).

The release cop, relaying the operator's shift-left order: census/README regen, the claim-literal baseline re-pointed
on PURE LINE DRIFT ("move only, never add"), the complexity baseline on PURE SHRINKS, contract count bumps. Each
auto-fix is a COMMIT the watch proposes, never a silent pass -- and a regeneration that would ADMIT something new is
refused here, because re-deriving a baseline is exactly how a guard gets quieter in the direction nobody reviewed.

  claim_move_only(old, new)       old/new: [(path, line_text)] -- the new baseline's literals, per path, are a
                                  sub-multiset of the old ones: a literal may MOVE (line drift) or LEAVE, never ARRIVE
  complexity_shrink_only(old, new) old/new: {"<path>::<fn>": (cyclomatic, cognitive)} -- no new function, and no
                                  function's numbers go UP (a baseline may only shrink)
Both return [why] -- empty when the change is admissible.
"""

from collections import Counter


def claim_move_only(old, new):
    out = []
    by_old, by_new = {}, {}
    for p, t in old:
        by_old.setdefault(p, Counter())[t.strip()] += 1
    for p, t in new:
        by_new.setdefault(p, Counter())[t.strip()] += 1
    for p, c in sorted(by_new.items()):
        extra = c - by_old.get(p, Counter())
        for t, n in sorted(extra.items()):
            out.append("claim literal ADDED in %s: %r (x%d) -- a re-point may move or drop a literal, never admit one" % (p, t[:80], n))
    return out


def parse_complexity(text):
    """complexity_baseline.txt lines '<path>::<fn> <cyc> <cog>' -> {key: (cyc, cog)}; comments/blank skipped."""
    out = {}
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            out[parts[0]] = (int(parts[1]), int(parts[2]))
    return out


def complexity_shrink_only(old, new):
    out = []
    for k, (c, g) in sorted(new.items()):
        if k not in old:
            out.append("complexity baseline ADDS %s (%d/%d) -- a baseline may only shrink" % (k, c, g))
        else:
            oc, og = old[k]
            if c > oc or g > og:
                out.append("complexity baseline RAISES %s %d/%d -> %d/%d -- a baseline may only shrink" % (k, oc, og, c, g))
    return out
