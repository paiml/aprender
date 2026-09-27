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
import hashlib
import json
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


def lock_delta(before, after):
    """The CANONICAL dependency delta between two Cargo.lock texts -> (delta dict, sha256 of its JSON).

    Per package name present in either lock: the dependency edges added and removed, and any change to
    an external package's version/source/checksum. Workspace (source-less) version bumps are not in it.
    Canonical JSON (sorted keys, sorted lists), so the same delta always hashes the same and ANY other
    change to the dependency graph hashes differently."""
    a, b = _lock_shape(before), _lock_shape(after)
    delta = {"packages_added": sorted(set(b) - set(a)), "packages_removed": sorted(set(a) - set(b)), "changed": {}}
    for name in sorted(set(a) & set(b)):
        if a[name] == b[name]:
            continue
        ea = {d for entry in a[name] for d in entry[3]}
        eb = {d for entry in b[name] for d in entry[3]}
        attrs_a = sorted((e[0], e[1], e[2]) for e in a[name])
        attrs_b = sorted((e[0], e[1], e[2]) for e in b[name])
        delta["changed"][name] = {"edges_added": sorted(eb - ea), "edges_removed": sorted(ea - eb),
                                  "attrs": None if attrs_a == attrs_b else [attrs_a, attrs_b]}
    blob = json.dumps(delta, sort_keys=True, separators=(",", ":")).encode()
    return delta, hashlib.sha256(blob).hexdigest()


def _override_for(scope, delta_sha):
    """#4022: an operator override pinned to EXACTLY one lock delta, recorded with its ruling."""
    for o in scope.get("lock_overrides") or []:
        if isinstance(o, dict) and o.get("delta_sha256") == delta_sha \
                and all(str(o.get(k) or "").strip() for k in ("ticket", "ruling", "date")):
            return o
    return None


def classify(receipt_sha, cut_sha, paths, scope, lock_before=None, lock_after=None):
    """-> (kind, proof): kind is "same", "evidence", "hotfix", "override" or None (not equivalent)."""
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
            try:
                delta, dsha = lock_delta(lock_before, lock_after)
            except (tomllib.TOMLDecodeError, KeyError, TypeError):
                return None, "the hotfix changes dependencies -- %s" % why
            o = _override_for(scope, dsha)
            if o is None:
                return None, "the hotfix changes dependencies -- %s (lock delta sha256 %s, no operator override pinned to it)" % (why, dsha[:16])
            return "override", "OPERATOR OVERRIDE (%s lock delta) sha256 %s, %s, operator: %s -- files: %s" % (
                o["ticket"], dsha[:16], o["date"], o["ruling"], ", ".join(outside))
    return "hotfix", "scoped hotfix (%s): %s" % (scope.get("ruling") or "ruling", ", ".join(outside))
