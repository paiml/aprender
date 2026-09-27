#!/usr/bin/env python3
"""lookahead.py -- the look-ahead slot machinery of APR-LOOKAHEAD-001.

One tool, one verb per spec row, all reading the same state file
(cop-state/lookahead.json; interim /mnt/nvme-raid0/cop-inbox/lookahead.json).
The cop is the single writer of that file; workers write heartbeat files.

  validate STATE                      LA-00  schema + I1 (coverage) + I2 (uniqueness)

Exit codes, shared by every verb:
  0  the property holds (or the action was done)
  1  a violation / a refusal -- printed one per line as `VIOLATION <id>: ...`
  2  cannot judge (file missing, not JSON, schema missing) -- never a silent pass
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = os.environ.get(
    "LOOKAHEAD_SCHEMA", os.path.join(ROOT, "docs", "lookahead", "lookahead-state.schema.json")
)
SLOTS = ("L1", "L2", "L3")


class CannotJudge(Exception):
    """Unknown is not a pass: every path that cannot decide exits 2."""


def load_json(path, dup_sink=None):
    """Parse JSON, recording duplicate keys: `{"L1": .., "L1": ..}` is two workers
    in one slot, and a plain json.load would silently keep the last one."""

    def hook(pairs):
        seen = {}
        for k, v in pairs:
            if k in seen and dup_sink is not None:
                dup_sink.append(k)
            seen[k] = v
        return seen

    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh, object_pairs_hook=hook)
    except FileNotFoundError as exc:
        raise CannotJudge(f"{path}: not found") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CannotJudge(f"{path}: not JSON ({exc})") from exc


# ---- a subset of JSON Schema, enough for the schema file above ------------------
def _resolve(schema, node):
    ref = node.get("$ref")
    if ref is None:
        return node
    if not ref.startswith("#/"):
        raise CannotJudge(f"schema: unsupported $ref {ref}")
    target = schema
    for part in ref[2:].split("/"):
        if not isinstance(target, dict) or part not in target:
            raise CannotJudge(f"schema: $ref {ref} does not resolve")
        target = target[part]
    return target


def schema_errors(schema, node, value, path="$"):
    node = _resolve(schema, node)
    errs = []
    want = node.get("type")
    kinds = {"object": dict, "array": list, "string": str}
    if want == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return [f"{path}: expected integer"]
    elif want in kinds and not isinstance(value, kinds[want]):
        return [f"{path}: expected {want}"]
    if "pattern" in node and isinstance(value, str) and not re.search(node["pattern"], value):
        errs.append(f"{path}: {value!r} does not match {node['pattern']}")
    if isinstance(value, dict):
        props = node.get("properties", {})
        for key in node.get("required", []):
            if key not in value:
                errs.append(f"{path}: missing required {key!r}")
        for key, sub in value.items():
            if key in props:
                errs.extend(schema_errors(schema, props[key], sub, f"{path}.{key}"))
            elif node.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {key!r}")
    if isinstance(value, list) and "items" in node:
        for i, item in enumerate(value):
            errs.extend(schema_errors(schema, node["items"], item, f"{path}[{i}]"))
    return errs


# ---- trains ------------------------------------------------------------------
def train_add(train, k):
    major, minor = train.split(".")
    return f"{major}.{int(minor) + k}"


def state_violations(state, dups):
    """Every reason `state` is not a legal look-ahead state, as (id, message)."""
    out = [("I2", f"duplicate key {k!r}: two entries for one slot") for k in dups]
    schema = load_json(SCHEMA)
    out += [("SCHEMA", e) for e in schema_errors(schema, schema, state)]
    if out:
        return out
    cur = state["current_train"]
    for k, slot in enumerate(SLOTS, start=1):
        want = train_add(cur, k)
        got = state["slots"][slot]["train"]
        if got != want:
            out.append(("I1", f"{slot} holds train {got}, want {want} (current {cur} + {k})"))
        want_handoff = f"docs/lookahead/{got}.md"
        if state["slots"][slot]["handoff"] != want_handoff:
            out.append(("I1", f"{slot} handoff {state['slots'][slot]['handoff']} is not {want_handoff}"))
    owner = {}
    for slot in SLOTS:
        w = state["slots"][slot]["worker"]
        if w in owner:
            out.append(("I2", f"worker {w!r} holds both {owner[w]} and {slot}"))
        owner.setdefault(w, slot)
    return out


def cmd_validate(args):
    if len(args) != 1:
        raise CannotJudge("usage: validate STATE")
    dups = []
    state = load_json(args[0], dups)
    bad = state_violations(state, dups)
    for vid, msg in bad:
        print(f"VIOLATION {vid}: {msg}")
    if bad:
        return 1
    print(f"OK {args[0]}: current {state['current_train']}, slots "
          + ", ".join(f"{s}={state['slots'][s]['train']}:{state['slots'][s]['worker']}" for s in SLOTS))
    return 0


VERBS = {"validate": cmd_validate}


def main(argv):
    if len(argv) < 2 or argv[1] not in VERBS:
        print("usage: lookahead.py {" + "|".join(VERBS) + "} ...", file=sys.stderr)
        return 2
    try:
        return VERBS[argv[1]](argv[2:])
    except CannotJudge as exc:
        print(f"CANNOT-JUDGE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
