#!/usr/bin/env python3
"""roadmap_diff.py — a PR's roadmap.yaml diff is ADDITIVE (PMAT-980, G-6, #2874).

WHY THIS EXISTS
---------------
`pmat work add` (and `pmat work complete`) re-serialise ALL of
docs/roadmaps/roadmap.yaml on every call, not just the entry that changed. One
17-line ticket arrived with 2,531 unrelated lines rewritten: long strings were
re-folded onto one line, and `phases: []` / `subtasks: []` /
`estimated_effort: null` / `labels: []` were materialised onto every entry that
had previously omitted them. Two in-flight PRs editing the roadmap in the same
window then conflict by construction, and a reviewer cannot see the one entry
that actually changed inside the noise.

The PP-066 driver's rule: a PR's roadmap.yaml diff may only

  * add new top-level entries (the id set grows), or
  * change a ticket's own LIFECYCLE fields (status, updated, notes,
    github_issue, labels, assigned_to, priority).

Anything else — a deleted entry, an edited title/spec/acceptance_criteria, or
a wholesale re-render that touches no semantic content at all — is a
violation. This module is both a library (`check`, `trim`) and the CLI that
scripts/check_roadmap_diff_additive.sh drives.

ENTRY MODEL
-----------
`docs/roadmaps/roadmap.yaml` is a pmat work-contract file: a `roadmap:` key
holding a YAML sequence, each item beginning at column 0 with `- id: <ID>`.
Entries are split at that exact byte offset — BYTE-EXACT blocks, not
re-serialised — so an unrelated entry that pmat did not touch compares equal
byte-for-byte and never needs YAML at all. Only entries that DO differ in
bytes are parsed, to decide whether the difference is content or noise.

Usage:
    python3 scripts/lib/roadmap_diff.py check --base <path-or-ref> \
        --head <path-or-ref> [--file docs/roadmaps/roadmap.yaml]
    python3 scripts/lib/roadmap_diff.py trim --base <path-or-ref> \
        [--file docs/roadmaps/roadmap.yaml] [--write]

A `<path-or-ref>` that is a file on disk is read directly; anything else is
treated as a git ref and resolved via `git show <ref>:<file>`.

EXIT CODES (check): 0 no violation, 1 a rule was violated, 2 a usage/read
error (missing file, unresolvable ref, unparsable PyYAML-less environment).
EXIT CODES (trim): 0 always, unless a usage/read error (2).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import OrderedDict

try:
    import yaml
except ImportError:  # pragma: no cover - environment defect, not a code path
    sys.stderr.write(
        "FATAL PyYAML is not importable. An unparsed roadmap is not\n"
        "      'no unproven diff' — install it (uv pip install pyyaml).\n"
    )
    sys.exit(2)

# Fields a ticket may change about ITSELF without that being a content edit.
LIFECYCLE_KEYS = {
    "status",
    "updated",
    "notes",
    "github_issue",
    "labels",
    "assigned_to",
    "priority",
}

# Keys `pmat work add` materialises onto every entry even when the entry never
# carried them. Absent and present-with-this-default are the SAME content.
NORMALIZE_DEFAULTS = {
    "phases": [],
    "subtasks": [],
    "estimated_effort": None,
    "labels": [],
}

ENTRY_RE = re.compile(r"^- id:[ \t]*(.*)$", re.MULTILINE)

# A YAML anchor definition (`&id001`) or alias reference (`*id001`) inside an
# entry block. Used only by `render_trim`, to refuse collapsing an alias's
# entry back to base bytes when its anchor's entry did not also collapse —
# see the comment at the one call site (Pass 2).
ANCHOR_DEF_RE = re.compile(r"&([A-Za-z0-9_-]+)")
ANCHOR_ALIAS_RE = re.compile(r"\*([A-Za-z0-9_-]+)")


def _inject_anchor(base_block: str, head_block: str, name: str):
    """Copy the exact `key: &name value` LINE from `base_block` onto the
    matching `key:` line of `head_block`, so an alias elsewhere resolves
    against head's rendering of the definer. Only called when the definer's
    CORE content is known unchanged (`core_equal`), so the value being
    transplanted is the same one head already carries — this rewrites
    syntax, not content. Returns None if no such line, or no matching key in
    head_block, is found (the caller then falls back to forcing the alias's
    entry to head bytes instead)."""
    m = re.search(rf"^([^\n]*&{re.escape(name)}\b[^\n]*)$", base_block, re.MULTILINE)
    if not m:
        return None
    base_line = m.group(1)
    key_m = re.match(r"^(\s*[A-Za-z0-9_-]+):", base_line)
    if not key_m:
        return None
    key_prefix = re.escape(key_m.group(1))
    patched, count = re.subn(
        rf"^{key_prefix}:.*$",
        lambda _mm: base_line.replace("\\", "\\\\"),
        head_block,
        count=1,
        flags=re.MULTILINE,
    )
    return patched if count == 1 else None


class UsageError(Exception):
    """A read/resolve failure — exit 2, never a rule violation."""


# --------------------------------------------------------------------------
# entry splitting
# --------------------------------------------------------------------------
def parse_id_value(raw: str) -> str:
    """Undo the one or two YAML scalar quoting shapes an id can carry."""
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return v[1:-1].replace("''", "'")
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return v[1:-1].replace('\\"', '"')
    return v


def split_entries(text: str):
    """-> (preamble, [(id, block_bytes), ...]) in file order. No dedup here —
    duplicate ids are a caller-level violation, not a parse error."""
    matches = list(ENTRY_RE.finditer(text))
    if not matches:
        return text, []
    preamble = text[: matches[0].start()]
    entries = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        entries.append((parse_id_value(m.group(1)), block))
    return preamble, entries


def id_counts(entries):
    counts: "OrderedDict[str, int]" = OrderedDict()
    for eid, _ in entries:
        counts[eid] = counts.get(eid, 0) + 1
    return counts


def parse_block(block: str):
    """A block is `- id: X\\n  k: v\\n...` — a valid one-item YAML sequence.
    This CANNOT resolve a YAML anchor/alias that spans two entries (real
    data: docs/roadmaps/roadmap.yaml carries `created: *id001` referencing an
    anchor defined on a different entry) — see `positional_roadmap_items`,
    which is tried first and does not have this limitation."""
    doc = yaml.safe_load(block)
    if not isinstance(doc, list) or len(doc) != 1 or not isinstance(doc[0], dict):
        raise ValueError("entry block did not parse as a single mapping")
    return doc[0]


def positional_roadmap_items(full_doc, entries):
    """-> list of dicts aligned 1:1 by POSITION with `entries` (the byte-split
    blocks), or None if the whole-document parse disagrees with the byte
    split on entry count (a signal to fall back to parsing each block in
    isolation, which cannot resolve anchors but needs no such alignment).
    Parsing the WHOLE document once, rather than block-by-block, is what lets
    a `created: *id001` alias resolve against an anchor defined on an
    earlier, unrelated entry."""
    if not isinstance(full_doc, dict):
        return None
    items = full_doc.get("roadmap")
    if not isinstance(items, list) or len(items) != len(entries):
        return None
    return items


def resolve_entry_dict(idx: int, block: str, items):
    """The parsed mapping for entry number `idx`: the whole-document parse
    when it lines up positionally, else a best-effort parse of the isolated
    block (raises YAMLError/ValueError on an anchor/alias it cannot see)."""
    if items is not None and idx < len(items) and isinstance(items[idx], dict):
        return items[idx]
    return parse_block(block)


def normalize(d: dict) -> dict:
    out = dict(d)
    for k, default in NORMALIZE_DEFAULTS.items():
        if k not in out:
            out[k] = default
    return out


def strip_lifecycle(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in LIFECYCLE_KEYS}


def lifecycle_only(d: dict) -> dict:
    return {k: v for k, v in d.items() if k in LIFECYCLE_KEYS}


# --------------------------------------------------------------------------
# source resolution
# --------------------------------------------------------------------------
def _repo_root() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except OSError:
        pass
    return os.getcwd()


def _read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as e:
        raise UsageError(f"cannot read file {path}: {e}") from e


def _git_show(ref: str, file_arg: str) -> str:
    root = _repo_root()
    try:
        proc = subprocess.run(
            ["git", "-C", root, "show", f"{ref}:{file_arg}"],
            capture_output=True, text=True, timeout=60,
        )
    except OSError as e:
        raise UsageError(f"git show {ref}:{file_arg} failed to run: {e}") from e
    if proc.returncode != 0:
        raise UsageError(f"git show {ref}:{file_arg} failed: {proc.stderr.strip()}")
    return proc.stdout


def read_source(value: str, file_arg: str) -> str:
    """value is a path (read directly) or a git ref (resolved against file_arg
    via `git show <ref>:<file>`)."""
    if os.path.isfile(value):
        return _read_file(value)
    return _git_show(value, file_arg)


def _parse_full(text: str):
    """-> (document-or-None, first error line or None)."""
    try:
        return yaml.safe_load(text), None
    except yaml.YAMLError as e:
        msg = str(e).splitlines()[0] if str(e) else "unknown error"
        return None, msg


def _index_by_id(entries):
    """id -> (block, position). First occurrence wins; a duplicate is
    reported separately and does not need a second, confusing report."""
    by_id: "OrderedDict[str, tuple]" = OrderedDict()
    for idx, (i, block) in enumerate(entries):
        by_id.setdefault(i, (block, idx))
    return by_id


def _duplicate_lines(base_entries, head_entries):
    """-> (violations, notes). A duplicate id already present in BASE is
    baselined (known, not this PR's doing — main carried PMAT-966 twice on
    2026-09-05, minted by two sessions); it is reported as a note so the
    dedup PR that removes the extra copy is the remedy, and it stays visible.
    A duplicate that GROWS at head (count above base's) is a violation."""
    violations, notes = [], []
    base_n, head_n = id_counts(base_entries), id_counts(head_entries)
    for i, n in base_n.items():
        if n > 1:
            notes.append(f"known duplicate-id: id={i} appears {n} times in base (pre-existing; remedy = a dedup PR that re-mints the later copy)")
    for i, n in head_n.items():
        if n > 1 and n > base_n.get(i, 0):
            violations.append(f"VIOLATION duplicate-id: id={i} appears {n} times in head (base had {base_n.get(i, 0)})")
    return violations, notes


def _top_level_lines(base_full, head_full):
    base_keys = set(base_full.keys()) if isinstance(base_full, dict) else set()
    head_keys = set(head_full.keys()) if isinstance(head_full, dict) else set()
    if base_keys == head_keys:
        return []
    return [
        "VIOLATION top-level-keys-changed: "
        f"base={sorted(base_keys)} head={sorted(head_keys)}"
    ]


def _changed_keys(core_base: dict, core_head: dict) -> str:
    changed = sorted(set(core_base) ^ set(core_head)) or sorted(
        k for k in core_base if core_base.get(k) != core_head.get(k)
    )
    return ", ".join(changed) or "?"


def _resolve_pair(i, base_pair, head_pair, base_items, head_items):
    """-> (base_dict, head_dict) normalised, or raises YAMLError/ValueError."""
    base_block, base_idx = base_pair
    head_block, head_idx = head_pair
    base_d = resolve_entry_dict(base_idx, base_block, base_items)
    head_d = resolve_entry_dict(head_idx, head_block, head_items)
    return normalize(base_d), normalize(head_d)


def classify_pair(i, base_pair, head_pair, base_items, head_items):
    """-> (kind, detail). kind in {same, lifecycle, reserialised, changed,
    unparsable}. Both sides are normalised (materialised empty keys, incl.
    `labels`) BEFORE the core/lifecycle split — otherwise a materialised
    `labels: []` on one side alone reads as a lifecycle CHANGE even though it
    changed nothing, and a pure re-serialisation is misclassified as an
    allowed lifecycle edit."""
    if head_pair[0] == base_pair[0]:
        return "same", ""
    try:
        base_dn, head_dn = _resolve_pair(i, base_pair, head_pair, base_items, head_items)
    except (yaml.YAMLError, ValueError) as e:
        return "unparsable", f"id={i}: {e}"
    core_base, core_head = strip_lifecycle(base_dn), strip_lifecycle(head_dn)
    if core_base != core_head:
        return "changed", _changed_keys(core_base, core_head)
    if lifecycle_only(base_dn) != lifecycle_only(head_dn):
        return "lifecycle", ""
    return "reserialised", ""


_VIOLATION_TEXT = {
    "unparsable": "VIOLATION unparsable-entry: {detail}",
    "changed": "VIOLATION reserialised: id={i} (non-lifecycle field(s) changed: {detail})",
    "reserialised": "VIOLATION reserialised: id={i} (bytes differ, no field actually changed)",
}


def _classify_common(base_by_id, head_by_id, base_items, head_items):
    """-> (lines, lifecycle_ids, reserialised_ids) over the ids on both sides."""
    lines, lifecycle_ids, reserialised_ids = [], [], []
    common = [i for i in base_by_id if i in head_by_id]
    for i in common:
        kind, detail = classify_pair(i, base_by_id[i], head_by_id[i], base_items, head_items)
        if kind == "lifecycle":
            lifecycle_ids.append(i)
        elif kind in _VIOLATION_TEXT:
            lines.append(_VIOLATION_TEXT[kind].format(i=i, detail=detail))
            reserialised_ids.append(i)
    return lines, lifecycle_ids, reserialised_ids


def _summary_lines(base_entries, head_entries, added, lifecycle, reserialised, deleted):
    lines = []
    if deleted:
        lines.append(
            f"VIOLATION deleted: {len(deleted)} base id(s) missing at head: "
            + ", ".join(deleted[:3]) + (" ..." if len(deleted) > 3 else "")
        )
    if reserialised:
        lines.append(f"reserialised={len(reserialised)} first_ids=" + ", ".join(reserialised[:3]))
    lines.append(
        "roadmap-diff: "
        f"base={len(base_entries)} head={len(head_entries)} "
        f"added={len(added)} lifecycle={len(lifecycle)} "
        f"reserialised={len(reserialised)} deleted={len(deleted)}"
    )
    return lines


def _head_lines(base_full, head_full, head_err):
    """R4: the head must parse and keep the top-level keys."""
    if head_err is not None:
        return [f"VIOLATION head-unparsable: {head_err}"]
    return _top_level_lines(base_full, head_full)


def run_check(base_text: str, head_text: str) -> "tuple[int, list[str]]":
    """-> (exit_code, lines to print, in order). R4a: base must parse (a
    usage error — base is already-merged content); head must parse (a rule
    violation — the PR broke the file)."""
    base_full, base_err = _parse_full(base_text)
    if base_err is not None:
        raise UsageError(f"base does not parse as YAML: {base_err}")
    head_full, head_err = _parse_full(head_text)
    _, base_entries = split_entries(base_text)
    _, head_entries = split_entries(head_text)
    base_by_id, head_by_id = _index_by_id(base_entries), _index_by_id(head_entries)
    base_items = positional_roadmap_items(base_full, base_entries)
    head_items = positional_roadmap_items(head_full, head_entries)
    deleted = [i for i in base_by_id if i not in head_by_id]
    added = [i for i in head_by_id if i not in base_by_id]
    pair_lines, lifecycle, reserialised = _classify_common(base_by_id, head_by_id, base_items, head_items)
    dup_violations, dup_notes = _duplicate_lines(base_entries, head_entries)
    lines = _head_lines(base_full, head_full, head_err) + dup_violations + pair_lines
    violated = bool(lines) or bool(deleted)
    lines = dup_notes + lines + _summary_lines(base_entries, head_entries, added, lifecycle, reserialised, deleted)
    return (1 if violated else 0), lines


def _decide_entries(base_by_id, head_by_id, base_items, head_items):
    """Pass 1: per-entry base/head decision, ignoring anchors.
    -> (decision id->"base"|"head", core_equal id->bool)."""
    decision: "OrderedDict[str, str]" = OrderedDict()
    core_equal: dict[str, bool] = {}
    for i, base_pair in base_by_id.items():
        if i not in head_by_id:
            continue  # restored verbatim later; no decision needed
        kind, _detail = classify_pair(i, base_pair, head_by_id[i], base_items, head_items)
        decision[i] = "base" if kind in ("same", "reserialised") else "head"
        core_equal[i] = kind in ("same", "reserialised", "lifecycle")
    return decision, core_equal


def _anchor_definers(base_by_id):
    definer_of: dict[str, str] = {}  # anchor name -> defining entry id
    for i, (base_block, _idx) in base_by_id.items():
        for m in ANCHOR_DEF_RE.finditer(base_block):
            definer_of.setdefault(m.group(1), i)
    return definer_of


def _alias_needs_head(name, definer, base_by_id, head_by_id, decision, core_equal, block_override, patched_anchor):
    """One alias of one entry: True iff the alias-user must fall back to head
    bytes (its definer cannot be kept verbatim or transplanted)."""
    if definer is None or decision.get(definer) == "base":
        return False  # anchor already present verbatim
    if definer in head_by_id and name not in patched_anchor:
        _try_transplant(name, definer, base_by_id, head_by_id, core_equal, block_override)
        patched_anchor.add(name)
    return definer not in block_override


def _resolve_anchors(base_by_id, head_by_id, decision, core_equal):
    """Pass 2. A YAML anchor (`&id001`) is resolved against the WHOLE
    document, so an alias (`*id001`) can only collapse to base bytes if the
    entry DEFINING that anchor is ALSO present, by anchor name, in the
    assembled output. Real data: docs/roadmaps/roadmap.yaml once carried one
    `&id001` shared by 25 entries, one of which (the definer) had a genuine
    lifecycle-only edit and so could not collapse whole; its CORE content was
    unchanged (`core_equal`), so the anchor line is transplanted onto head's
    rendering of that entry. Only when that fails is an alias-user forced to
    head bytes, which is content-safe on its own: head's rendering never
    carries anchors (pmat's serializer expands them to literal values).
    -> block_override id->patched block; `decision` is updated in place."""
    definer_of = _anchor_definers(base_by_id)
    block_override: dict[str, str] = {}
    patched_anchor: set = set()
    for i, (base_block, _idx) in base_by_id.items():
        if decision.get(i) != "base":
            continue
        aliases = [m.group(1) for m in ANCHOR_ALIAS_RE.finditer(base_block)]
        if any(_alias_needs_head(n, definer_of.get(n), base_by_id, head_by_id, decision, core_equal, block_override, patched_anchor) for n in aliases):
            decision[i] = "head"
    return block_override


def _try_transplant(name, definer, base_by_id, head_by_id, core_equal, block_override):
    if not core_equal.get(definer):
        return
    patched = _inject_anchor(base_by_id[definer][0], head_by_id[definer][0], name)
    if patched is not None:
        block_override[definer] = patched


def _base_side_blocks(base_by_id, head_by_id, decision, block_override):
    out: list[str] = []
    for i, (base_block, _idx) in base_by_id.items():
        keep_base = i not in head_by_id or decision[i] == "base"  # never delete
        out.append(base_block if keep_base else block_override.get(i, head_by_id[i][0]))
    return out


def _assemble(base_by_id, head_by_id, decision, block_override):
    new_blocks = [block for i, (block, _idx) in head_by_id.items() if i not in base_by_id]
    return _base_side_blocks(base_by_id, head_by_id, decision, block_override) + new_blocks


def render_trim(base_text: str, head_text: str) -> str:
    """Rebuild `head_text`'s roadmap as base's bytes for every entry that is
    unchanged or only re-serialised, head's rendering for entries that are new
    or lifecycle-changed, restoring a base entry deleted at head (this repo
    never deletes roadmap entries), then appending new entries at the end in
    head order."""
    base_preamble, base_entries = split_entries(base_text)
    head_preamble, head_entries = split_entries(head_text)
    base_full, _ = _parse_full(base_text)
    head_full, _ = _parse_full(head_text)
    base_items = positional_roadmap_items(base_full, base_entries)
    head_items = positional_roadmap_items(head_full, head_entries)
    base_by_id, head_by_id = _index_by_id(base_entries), _index_by_id(head_entries)
    decision, core_equal = _decide_entries(base_by_id, head_by_id, base_items, head_items)
    block_override = _resolve_anchors(base_by_id, head_by_id, decision, core_equal)
    preamble = head_preamble if head_preamble else base_preamble
    return preamble + "".join(_assemble(base_by_id, head_by_id, decision, block_override))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _cmd_check(args) -> int:
    base_text = read_source(args.base, args.file)
    head_text = read_source(args.head, args.file)
    code, lines = run_check(base_text, head_text)
    for line in lines:
        print(line)
    return code


def _cmd_trim(args) -> int:
    base_text = read_source(args.base, args.file)
    if not os.path.isfile(args.file):
        raise UsageError(f"trim's head is the working file, not found: {args.file}")
    with open(args.file, encoding="utf-8") as fh:
        head_text = fh.read()
    result = render_trim(base_text, head_text)
    if args.write:
        with open(args.file, "w", encoding="utf-8") as fh:
            fh.write(result)
    else:
        sys.stdout.write(result)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadmap_diff.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--base", required=True)
    p_check.add_argument("--head", required=True)
    p_check.add_argument("--file", default="docs/roadmaps/roadmap.yaml")
    p_check.set_defaults(func=_cmd_check)

    p_trim = sub.add_parser("trim")
    p_trim.add_argument("--base", required=True)
    p_trim.add_argument("--file", default="docs/roadmaps/roadmap.yaml")
    p_trim.add_argument("--write", action="store_true")
    p_trim.set_defaults(func=_cmd_trim)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except UsageError as e:
        sys.stderr.write(f"USAGE ERROR: {e}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
