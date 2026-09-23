"""ladder_equiv.py -- may a receipt measured at commit A stand for the cut commit B? (#3957 F2, #3710 ruling 3)

check_model_ladder.sh binds every receipt to the cut by its `apr_sha` (#3957 F2). Two ways a receipt
measured at A stands for B, and nothing else:

  evidence   B's tree equals A's outside evidence/ (committing receipts is itself a commit; R7).
  hotfix     the operator-ruled scoped hotfix (#3710 ruling 3, #4022): A is the contract's pinned
             `hotfix_scope.receipts_at`, EVERY path the diff touches matches a `hotfix_scope.paths`
             glob, and Cargo.lock changes no dependency -- only versions of the workspace's own
             (source-less) packages may move. Anything else in the diff, an unparseable lockfile, or
             a different A is NOT equivalent.

`classify` is pure: the caller runs git and hands over the changed paths and both lockfiles, so the
case table drives every branch without a repository. It returns (kind or None, proof), and the
proof is printed by the gate: a receipt that binds by hotfix states which files the hotfix touched.
"""

import fnmatch
import tomllib


def _lock_shape(text):
    """-> {name: [(version-or-None, source, checksum, deps), ...]} with workspace versions blanked."""
    doc = tomllib.loads(text)
    shape = {}
    for p in doc.get("package") or []:
        src = p.get("source")
        deps = tuple(sorted(d.split(" ")[0] for d in (p.get("dependencies") or [])))
        shape.setdefault(p["name"], []).append((p.get("version") if src else None, src, p.get("checksum"), deps))
    return {k: sorted(v, key=repr) for k, v in shape.items()}


def lock_dep_change(before, after):
    """-> None when the two Cargo.lock texts differ only in workspace package versions, else a reason."""
    try:
        a, b = _lock_shape(before), _lock_shape(after)
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        return "Cargo.lock unparseable: %s" % exc
    added, removed = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    if added or removed:
        return "Cargo.lock adds %s / removes %s" % (added, removed)
    changed = sorted(k for k in a if a[k] != b[k])
    return ("Cargo.lock changes dependency data of %s" % changed) if changed else None


def classify(receipt_sha, cut_sha, paths, scope, lock_before=None, lock_after=None):
    """-> (kind, proof): kind is "evidence", "hotfix" or None (not equivalent)."""
    if receipt_sha == cut_sha:
        return "same", "the receipt was measured at the cut"
    outside = [p for p in paths if not p.startswith("evidence/")]
    if not outside:
        return "evidence", "the trees differ only under evidence/"
    scope = scope or {}
    if receipt_sha != scope.get("receipts_at"):
        return None, "differs outside evidence/ in %d path(s) and is not the hotfix scope's pinned receipts_at" % len(outside)
    globs = list(scope.get("paths") or [])
    stray = [p for p in outside if not any(fnmatch.fnmatchcase(p, g) for g in globs)]
    if stray:
        return None, "the hotfix diff touches path(s) outside its scope: %s" % stray
    if "Cargo.lock" in outside:
        if lock_before is None or lock_after is None:
            return None, "Cargo.lock changed and could not be read at both commits"
        why = lock_dep_change(lock_before, lock_after)
        if why:
            return None, "the hotfix changes dependencies -- %s" % why
    return "hotfix", "scoped hotfix (%s): %s" % (scope.get("ruling") or "ruling", ", ".join(outside))
