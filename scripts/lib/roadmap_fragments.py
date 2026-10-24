#!/usr/bin/env python3
"""roadmap_fragments.py — docs/roadmaps/roadmap.yaml as a GENERATED artifact.

THE DEFECT THIS EXISTS FOR (#3296, five-whys #3294). Every PR in this repo
writes docs/roadmaps/roadmap.yaml, so the Amdahl serial fraction on the merge
path is 1: N pull requests contend on one file however disjoint their code is.
`.gitattributes` declares `merge=roadmap`, but that driver is CUSTOM and is
registered per-invocation by scripts/ci_resolve_dirty.sh, so GitHub's
merge-queue server -- which has no such config -- never runs it and falls back
to a plain text merge. Measured on the nine-PR batch #3295: the textual merge
placed PMAT-3226 after PMAT-3229 and check_roadmap_sorted.sh went RED, while
all nine inputs were individually sorted and green.

`union` is built-in and therefore DOES apply server-side, but it is correct
only for line-delimited order-insensitive data (impl-estimates.jsonl, #3256).
roadmap.yaml needs SORTED insertion. Two files, two policies.

THE SHAPE. A ticket writes exactly one file:

    docs/roadmaps/entries/<ID>.yaml     <- one entry block, verbatim
    docs/roadmaps/roadmap.yaml          <- base + entries, GENERATED

Unique filename by construction => pull requests are pairwise disjoint on the
roadmap => the conflict rate is 0 by proof rather than by luck.

ORDERING IS THE AGGREGATOR'S JOB, not the merge's. An entry is inserted at the
slot check_roadmap_sorted.sh would demand: within its own id-prefix, numerals
ascending, in the slots that prefix already occupies. Legacy free-form ids are
exempt and keep their position -- the same rule roadmap_merge.py enforces, and
it is imported from there rather than restated, so the two cannot drift.

    python3 scripts/lib/roadmap_fragments.py aggregate [--write]
    python3 scripts/lib/roadmap_fragments.py --selftest

Exit: 0 ok - 1 a violation (or a failing selftest row) - 2 usage/vacuity.
"""

import argparse
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from scripts.lib.roadmap_diff import split_entries  # noqa: E402
from scripts.lib.roadmap_merge import parse_id  # noqa: E402

ROADMAP = os.path.join(REPO_ROOT, "docs", "roadmaps", "roadmap.yaml")
ENTRIES = os.path.join(REPO_ROOT, "docs", "roadmaps", "entries")


def fragment_path(eid, entries_dir=ENTRIES):
    return os.path.join(entries_dir, eid + ".yaml")


def read_fragments(entries_dir=ENTRIES):
    """-> [(id, block)] sorted by filename. Absent dir is EMPTY, not an error:
    before the first fragment lands the aggregate is just the base."""
    if not os.path.isdir(entries_dir):
        return []
    out = []
    for name in sorted(os.listdir(entries_dir)):
        if not name.endswith(".yaml"):
            continue
        eid = name[: -len(".yaml")]
        with open(os.path.join(entries_dir, name), encoding="utf-8") as fh:
            text = fh.read()
        if text and not text.endswith("\n"):
            text += "\n"  # a last line without one (a comment) would swallow the next entry
        out.append((eid, text))
    return out


def insertion_index(base_entries, eid):
    """The slot check_roadmap_sorted.sh would demand for `eid`.

    Within the id's own prefix: before the first entry of that prefix whose
    numeral is greater. A prefix that does not occur yet, or a legacy id with
    no (prefix, numeral) shape, appends -- appending is always sorted when
    nothing of that prefix precedes it."""
    parsed = parse_id(eid)
    if parsed is None:
        return len(base_entries)
    prefix, numeral = parsed
    for i, (other, _) in enumerate(base_entries):
        p = parse_id(other)
        if p is None or p[0] != prefix:
            continue
        if p[1] > numeral:
            return i
    return len(base_entries)


def aggregate(base_text, fragments):
    """base + fragments, each at its sorted slot.

    IDEMPOTENT BY CONSTRUCTION, and that is not a nicety: `make
    roadmap-aggregate` runs post-merge on main, so an aggregator that is not a
    pure function of (base, fragments) makes main churn a commit on every
    merge. The first cut took the LIVE roadmap.yaml as the base and raised
    `duplicate id` the moment its own output was fed back -- it failed on run
    one, against a tree that already carried the fragment it was inserting.

    So a fragment SUPERSEDES any base entry with the same id rather than
    colliding with it: drop, then insert at the sorted slot. aggregate(X) and
    aggregate(aggregate(X)) are then the same bytes, proved by the selftest.

    A duplicate WITHIN the fragment set is still an error -- two files cannot
    claim one id, and the filesystem guarantees they do not."""
    preamble, entries = split_entries(base_text)
    seen = set()
    for eid, _ in fragments:
        if eid in seen:
            raise ValueError("duplicate id among fragments: %s" % eid)
        seen.add(eid)
    entries = [(e, b) for e, b in entries if e not in seen]
    for eid, block in fragments:
        entries.insert(insertion_index(entries, eid), (eid, block))
    return preamble + "".join(block for _, block in entries)


FILENAME_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,110}$")


def census(text):
    """-> {(safe, shape): [ids]}. Which entries CAN be fragments at all.

    A filename is the whole mechanism: unique name => disjoint pull requests.
    So an id that cannot be a filename cannot be a fragment, and the gate must
    not assume a shape the data does not have. Real ids in this file include
    `Push completed work to origin/main (5 commits)` -- prose, with a path
    separator in it."""
    out = {}
    for eid, _ in split_entries(text)[1]:
        key = ("safe" if FILENAME_SAFE.match(eid) else "unsafe",
               "prefixN" if parse_id(eid) else "legacy")
        out.setdefault(key, []).append(eid)
    return out


# ------------------------------------------------------------------- split

ANCHOR_DEF_RE = re.compile(r"&(id\d+)\s+")
ANCHOR_USE_RE = re.compile(r"\*(id\d+)\b")


def collect_anchors(text):
    """-> {name: literal}. A roadmap anchor is a scalar on its own key line,
    so the literal is the rest of that line after the anchor name."""
    out = {}
    for line in text.splitlines():
        m = ANCHOR_DEF_RE.search(line)
        if m:
            out[m.group(1)] = line[m.end():].rstrip()
    return out


def self_contain(block, anchors):
    """A fragment must parse ALONE. roadmap.yaml defines `created: &id001 ...`
    on one entry and aliases it from 17 others (roadmap_diff.py's own docstring
    names this as real data its block parser cannot resolve), so a fragment
    carrying `*id001` would be unparseable YAML. Inline the literal and drop
    the anchor; the VALUE is unchanged, which split() then proves per entry."""
    block = ANCHOR_DEF_RE.sub("", block)
    return ANCHOR_USE_RE.sub(lambda m: anchors.get(m.group(1), m.group(0)), block)


def split(text, entries_dir=ENTRIES, write=False):
    """roadmap.yaml -> one fragment per ticket. Returns [(id, block)].

    Every fragment is proved to parse alone AND to carry the same mapping the
    monolith did, before anything is written."""
    import yaml
    preamble, entries = split_entries(text)
    anchors = collect_anchors(text)
    whole = yaml.safe_load(text)
    items = whole["roadmap"]
    if len(items) != len(entries):
        raise ValueError("byte split (%d) disagrees with the parse (%d)"
                         % (len(entries), len(items)))
    frags = [(eid, _proved_fragment(eid, block, want, anchors))
             for (eid, block), want in zip(entries, items)]
    _refuse_duplicates([e for e, _ in frags])
    if write:
        _write_fragments(frags, preamble, entries_dir)
    return frags


def _proved_fragment(eid, block, want, anchors):
    """The fragment for one entry, refused unless it parses ALONE and carries
    exactly the mapping the monolith carried."""
    import yaml
    frag = self_contain(block, anchors)
    got = yaml.safe_load(frag)
    if not (isinstance(got, list) and len(got) == 1 and got[0] == want):
        raise ValueError("fragment for %s does not carry the same mapping" % eid)
    return frag


def _refuse_duplicates(ids):
    if len(set(ids)) == len(ids):
        return
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    raise ValueError("duplicate id(s) in the monolith: %s" % ", ".join(dupes))


def _write_fragments(frags, preamble, entries_dir):
    os.makedirs(entries_dir, exist_ok=True)
    for eid, frag in frags:
        with open(fragment_path(eid, entries_dir), "w", encoding="utf-8") as fh:
            fh.write(frag)
    with open(os.path.join(entries_dir, "..", "_preamble.yaml"), "w",
              encoding="utf-8") as fh:
        fh.write(preamble)



# ------------------------------------------------------------------- changed

def entry_map(text):
    """-> (preamble, {id: block bytes}). Duplicate ids CONCATENATE rather than
    overwrite: uniqueness is check_roadmap_ids_unique.sh's rule, and a reader
    that silently dropped one side of a duplicate would compare the wrong
    bytes."""
    pre, entries = split_entries(text)
    out = {}
    for eid, block in entries:
        out[eid] = out.get(eid, "") + block
    return pre, out


def changed_entries(base_text, head_text):
    """-> (preamble_changed, [(verdict, id)]) over BYTE-EXACT entry blocks.

    A re-serialisation that changes no field reads as CHANGED here, and that is
    deliberate: check_roadmap_diff_additive.sh already refuses that shape
    ("bytes differ, no field actually changed"), so the fragment gate must not
    be the one place it reads as "nothing happened"."""
    bpre, b = entry_map(base_text)
    hpre, h = entry_map(head_text)
    verdicts = [("ADDED", e) if e not in b else ("CHANGED", e)
                for e in h if e not in b or h[e] != b[e]]
    verdicts += [("REMOVED", e) for e in b if e not in h]
    verdicts.sort(key=lambda v: (v[1], v[0]))
    return bpre != hpre, verdicts


# --------------------------------------------------------------------- adopt

def _refuse_borrowed_anchor(eid, block, text):
    """An entry that DEFINES an anchor other entries alias cannot be adopted on
    its own: self_contain() strips the definition, and every `*idNNN` left in
    the base would then be unresolvable YAML. Refuse rather than write a
    roadmap.yaml that does not parse."""
    defined = set(ANCHOR_DEF_RE.findall(block))
    if not defined:
        return
    rest = text.replace(block, "", 1)
    borrowed = sorted(d for d in defined if ("*" + d) in rest)
    if borrowed:
        raise ValueError(
            "%s defines anchor(s) %s that other entries alias; adopting it alone "
            "would leave them unresolvable. Split the whole file instead."
            % (eid, ", ".join(borrowed)))


def adopt(ids, roadmap_path=ROADMAP, entries_dir=ENTRIES):
    """Move entries out of the aggregate's BASE and into docs/roadmaps/entries/.

    This exists because `pmat work add` writes the monolith and knows nothing
    about fragments (MEASURED 2026-09-16 on a clean worktree: one 16-line entry
    appended to docs/roadmaps/roadmap.yaml and nothing else -- appended to the
    TAIL, which check_roadmap_sorted.sh also refuses). Rather than block the
    ticket path, the gate names a two-step:

        pmat work add "..." --github-issue N
        python3 scripts/lib/roadmap_fragments.py adopt <ID>

    The second step writes the fragment and regenerates the aggregate, which
    also moves the entry to its sorted slot. Every fragment is proved to parse
    ALONE and to carry the mapping the monolith carried before anything is
    written -- the same proof split() applies."""
    import yaml
    with open(roadmap_path, encoding="utf-8") as fh:
        text = fh.read()
    _preamble, entries = split_entries(text)
    anchors = collect_anchors(text)
    items = yaml.safe_load(text)["roadmap"]
    if len(items) != len(entries):
        raise ValueError("byte split (%d) disagrees with the parse (%d)"
                         % (len(entries), len(items)))
    at = {}
    for i, (eid, _) in enumerate(entries):
        at.setdefault(eid, []).append(i)
    os.makedirs(entries_dir, exist_ok=True)
    written = []
    for eid in ids:
        where = at.get(eid, [])
        if not where:
            raise ValueError("%s is not a top-level entry of %s" % (eid, roadmap_path))
        if len(where) > 1:
            raise ValueError("%s appears %d times in %s -- fix the duplicate first "
                             "(check_roadmap_ids_unique.sh)" % (eid, len(where), roadmap_path))
        block = entries[where[0]][1]
        _refuse_borrowed_anchor(eid, block, text)
        frag = _proved_fragment(eid, block, items[where[0]], anchors)
        with open(fragment_path(eid, entries_dir), "w", encoding="utf-8") as fh:
            fh.write(frag)
        written.append(eid)
    return written


# ---------------------------------------------------------------- self-test

BASE = (
    "roadmap_version: '1.0'\nroadmap:\n"
    "- id: PMAT-100\n  title: a\n"
    "- id: PMAT-300\n  title: c\n"
    "- id: LEGACY-THING\n  title: legacy\n"
)
ROWS = []


def row(name, ok, detail=""):
    ROWS.append((name, ok, detail))
    print("%-4s %-52s %s" % ("ok" if ok else "FAIL", name, detail))


def _ids(text):
    return [eid for eid, _ in split_entries(text)[1]]


def _census_rows():
    """RMFR-F-004 over the REAL file. An obligation discharged by a test that
    does not test it is theater, so the census runs here rather than being
    asserted in the contract alone."""
    if not os.path.exists(ROADMAP):
        row("census SKIPPED: no roadmap.yaml here", True, "[U] not measured")
        return
    with open(ROADMAP, encoding="utf-8") as fh:
        c = census(fh.read())
    n = {k: len(v) for k, v in c.items()}
    total = sum(n.values())
    unsafe = n.get(("unsafe", "prefixN"), 0) + n.get(("unsafe", "legacy"), 0)
    row("census: every entry is accounted for by id shape", total > 0,
        "fragmentable=%d safe-legacy=%d NOT-filename-safe=%d total=%d"
        % (n.get(("safe", "prefixN"), 0), n.get(("safe", "legacy"), 0), unsafe, total))
    row("census: a real id contains a path separator (the regex must reject it)",
        any("/" in e for v in c.values() for e in v),
        "filename-safety is load-bearing, not defensive")


def _shape_rc(path):
    import types
    err, sys.stderr = sys.stderr, open(os.devnull, "w")
    try:
        return _cmd_shape(types.SimpleNamespace(file=path))
    finally:
        sys.stderr = err


def _shape_rows():
    # PMAT-6270: a fragment is one entry, and a last line without a newline cannot eat the next.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ed = os.path.join(d, "entries")
        os.makedirs(ed)
        def put(name, text):
            with open(os.path.join(ed, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        put("PMAT-500.yaml", "- id: PMAT-500\n  title: a\n  # tail, no newline")
        put("PMAT-501.yaml", "- id: PMAT-501\n  title: b\n")
        ids = [e for e, _ in split_entries(aggregate(BASE, read_fragments(ed)))[1]]
        row("shape: a fragment ending in a comment without a newline keeps the next entry",
            "PMAT-500" in ids and "PMAT-501" in ids, "ids=%s" % ids)
        put("PMAT-502.yaml", '- id: PMAT-502\n  title: "line one\n- id: PMAT-6270\n  line two"\n')
        why = fragment_shape(os.path.join(ed, "PMAT-502.yaml"))
        row("shape: a '- id:' line inside a quoted scalar is refused (the splitter would cut there)",
            "lines opening" in why, why)
        put("PMAT-503.yaml", "- id: PMAT-503\n  title: x\n...\n")
        put("PMAT-504.yaml", "---\n- id: PMAT-504\n  title: x\n")
        row("shape: a fragment ending in '...' or opening with '---' is refused",
            all("document marker" in fragment_shape(os.path.join(ed, n))
                for n in ("PMAT-503.yaml", "PMAT-504.yaml")))
        put("PMAT-505.yaml", "- id: PMAT-505\n  created: &id001 2026-10-24\n")
        put("PMAT-506.yaml", "- id: PMAT-506\n  created: *id001\n")
        row("shape: an anchor or an alias is refused (anchors are shared across the aggregate)",
            all("anchor" in fragment_shape(os.path.join(ed, n)) for n in ("PMAT-505.yaml", "PMAT-506.yaml")))
        with open(os.path.join(ed, "PMAT-507.yaml"), "w", encoding="utf-8") as fh:
            fh.write("\ufeff---\n- id: PMAT-507\n")
        row("shape: a byte order mark before '---' is refused",
            "byte order mark" in fragment_shape(os.path.join(ed, "PMAT-507.yaml")))
        put("PMAT-508.yaml", "- id: 'PMAT-508'  # quoted\n  title: x\n")
        put("PMAT-509.yml", "- id: PMAT-509\n")
        row("shape: the first line is exactly '- id: <stem>', and the name ends in .yaml",
            "does not start" in fragment_shape(os.path.join(ed, "PMAT-508.yaml"))
            and "<id>.yaml" in fragment_shape(os.path.join(ed, "PMAT-509.yml")))
        put("PMAT-510.yaml", "- id: PMAT-510\n  ? [a]\n  : 1\n")
        row("shape: an unhashable key is a refusal, not a crash",
            "unparseable" in fragment_shape(os.path.join(ed, "PMAT-510.yaml")))
        os.makedirs(os.path.join(ed, "PMAT-512.yaml"))
        row("shape: a missing or unreadable path exits 2 (not measured), never 1",
            _shape_rc(os.path.join(ed, "PMAT-511.yaml")) == 2 and _shape_rc(os.path.join(ed, "PMAT-512.yaml")) == 2)
        row("shape: a plain one-entry fragment passes",
            fragment_shape(os.path.join(ed, "PMAT-501.yaml")) == "")


def _base_rows():
    # 9. THE FROZEN BASE (#6185). roadmap-base.yaml beside roadmap.yaml is what the
    #    aggregate grows from; roadmap.yaml is only the thing being judged.
    import tempfile
    import types
    extra = BASE + "- id: PMAT-400\n  title: only in base\n"
    with tempfile.TemporaryDirectory() as d:
        (rm, bs) = (os.path.join(d, "roadmap.yaml"), os.path.join(d, "roadmap-base.yaml"))
        os.makedirs(os.path.join(d, "entries"))
        ns = types.SimpleNamespace(check=True, write=False)
        quiet = open(os.devnull, "w")
        err, sys.stderr = sys.stderr, quiet
        try:
            def verdict():
                return _cmd_aggregate(ns, rm, os.path.join(d, "entries"))
            open(rm, "w").write(BASE)
            no_base = verdict()                 # no roadmap-base.yaml: roadmap is its own base
            open(bs, "w").write(extra)
            stale = verdict()                   # base has PMAT-400, roadmap does not
            open(rm, "w").write(extra)
            fresh = verdict()                   # roadmap == aggregate(base)
        finally:
            sys.stderr = err
            quiet.close()
    row("no roadmap-base.yaml: roadmap.yaml is its own base (scratch-dir callers)", no_base == 0)
    row("roadmap-base.yaml entry missing from roadmap.yaml is DRIFT", stale == 1, "rc=%d" % stale)
    row("roadmap.yaml == aggregate(roadmap-base.yaml, entries/) passes", fresh == 0, "rc=%d" % fresh)

    # 9m. MUTATION: if base_path ignored base.yaml, the stale row must go RED.
    saved = globals()["base_path"]
    globals()["base_path"] = lambda roadmap: os.path.join(os.devnull, "nope")
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "entries"))
        open(os.path.join(d, "roadmap.yaml"), "w").write(BASE)
        open(os.path.join(d, "roadmap-base.yaml"), "w").write(extra)
        err, sys.stderr = sys.stderr, open(os.devnull, "w")
        try:
            diverges = _cmd_aggregate(types.SimpleNamespace(check=True, write=False),
                                      os.path.join(d, "roadmap.yaml"),
                                      os.path.join(d, "entries")) == 0
        finally:
            sys.stderr.close()
            sys.stderr = err
            globals()["base_path"] = saved
    row("mutation: ignoring roadmap-base.yaml is CAUGHT", diverges,
        "" if diverges else "mutant survived -- row 9 proves nothing")


def selftest():
    # 1. THE LOSSLESS ROW: no fragments must reproduce the base byte for byte.
    out = aggregate(BASE, [])
    row("no fragments -> base is byte-identical", out == BASE,
        "" if out == BASE else "differs")

    # 2-4. placement
    out = aggregate(BASE, [("PMAT-200", "- id: PMAT-200\n  title: b\n")])
    row("numeral between two -> sorted slot",
        _ids(out) == ["PMAT-100", "PMAT-200", "PMAT-300", "LEGACY-THING"], str(_ids(out)))
    out = aggregate(BASE, [("PMAT-400", "- id: PMAT-400\n  title: d\n")])
    row("numeral after all of its prefix -> appended",
        _ids(out)[-1] == "PMAT-400", str(_ids(out)))
    out = aggregate(BASE, [("APEX-1", "- id: APEX-1\n  title: new prefix\n")])
    row("unseen prefix -> appended", _ids(out)[-1] == "APEX-1", str(_ids(out)))
    out = aggregate(BASE, [("FREEFORM", "- id: FREEFORM\n  title: legacy\n")])
    row("legacy id (no numeral) -> appended", _ids(out)[-1] == "FREEFORM", str(_ids(out)))

    # 5. numeric, not lexical: 100 < 90 as strings, 90 < 100 as numbers.
    out = aggregate(BASE, [("PMAT-90", "- id: PMAT-90\n  title: ninety\n")])
    row("numeral compared NUMERICALLY (PMAT-90 before PMAT-100)",
        _ids(out)[0] == "PMAT-90", str(_ids(out)))

    # 6. a fragment SUPERSEDES a base entry of the same id, in place
    out = aggregate(BASE, [("PMAT-100", "- id: PMAT-100\n  title: superseded\n")])
    row("fragment supersedes a base entry of the same id",
        _ids(out) == ["PMAT-100", "PMAT-300", "LEGACY-THING"] and "superseded" in out,
        str(_ids(out)))

    # 6b. two fragments CANNOT claim one id (the filesystem already prevents it)
    try:
        aggregate(BASE, [("PMAT-9", "- id: PMAT-9\n  t: a\n"),
                         ("PMAT-9", "- id: PMAT-9\n  t: b\n")])
        row("duplicate id AMONG fragments refused", False, "no error raised")
    except ValueError as e:
        row("duplicate id AMONG fragments refused",
            "duplicate id among fragments" in str(e), str(e)[:44])

    # 6c. IDEMPOTENCE. `make roadmap-aggregate` runs post-merge on main, so an
    #     aggregator that is not a pure function churns a commit every merge.
    frag = [("PMAT-200", "- id: PMAT-200\n  title: b\n")]
    once = aggregate(BASE, frag)
    twice = aggregate(once, frag)
    row("IDEMPOTENT: aggregate(aggregate(x)) == aggregate(x)", once == twice,
        "" if once == twice else "second pass changed %d bytes" % abs(len(twice) - len(once)))

    # 6d. DETERMINISM: fragment order on disk must not change the output.
    two = [("PMAT-150", "- id: PMAT-150\n  t: x\n"), ("PMAT-250", "- id: PMAT-250\n  t: y\n")]
    row("DETERMINISTIC: fragment input order does not change the bytes",
        aggregate(BASE, two) == aggregate(BASE, list(reversed(two))))

    # 7. several fragments at once stay mutually sorted
    out = aggregate(BASE, [
        ("PMAT-150", "- id: PMAT-150\n  title: x\n"),
        ("PMAT-250", "- id: PMAT-250\n  title: y\n"),
    ])
    row("two fragments -> both at their own slots",
        _ids(out) == ["PMAT-100", "PMAT-150", "PMAT-250", "PMAT-300", "LEGACY-THING"],
        str(_ids(out)))

    # 8. MUTATION: if placement were append-only, row 2 must go RED. A guard
    #    whose failure mode is untested is decoration (#3294).
    saved = globals()["insertion_index"]
    globals()["insertion_index"] = lambda entries, eid: len(entries)
    try:
        mutated = aggregate(BASE, [("PMAT-200", "- id: PMAT-200\n  title: b\n")])
        caught = _ids(mutated) != ["PMAT-100", "PMAT-200", "PMAT-300", "LEGACY-THING"]
    finally:
        globals()["insertion_index"] = saved
    row("mutation: append-only placement is CAUGHT", caught,
        "" if caught else "mutant survived -- row 2 proves nothing")

    _base_rows()
    _shape_rows()
    _census_rows()

    bad = sum(1 for _, ok, _ in ROWS if not ok)
    print("\n%d row(s), %d red" % (len(ROWS), bad))
    return 1 if bad else 0


def _check(base, out, frags):
    """roadmap.yaml is what the aggregator produces, and re-aggregating is a
    no-op. The second half is not decoration: `make roadmap-aggregate` runs
    post-merge on main, so a non-idempotent generator churns a commit on every
    merge. Proved here on real data, not only in the selftest."""
    if aggregate(out, frags) != out:
        sys.stderr.write("FAIL aggregate is not idempotent on this input\n")
        return 1
    if out != base:
        sys.stderr.write(
            "FAIL docs/roadmaps/roadmap.yaml is not what the aggregator produces "
            "from docs/roadmaps/entries/ -- run `make roadmap-aggregate`\n")
        return 1
    sys.stderr.write("ok  roadmap.yaml == aggregate(%d fragment(s)), idempotent\n"
                     % len(frags))
    return 0


def _emit(a, current, out, frags, roadmap, base):
    """The three terminal arms of `aggregate`: verify, write, or print.
    `current` is the roadmap.yaml as it stands; `base` is what `out` grew from."""
    if a.check:
        return _check(current, out, frags)
    if a.write:
        with open(roadmap, "w", encoding="utf-8") as fh:
            fh.write(out)
        sys.stderr.write("aggregate: %d base + %d fragment(s) -> %s\n"
                         % (len(split_entries(base)[1]), len(frags), roadmap))
        return 0
    sys.stdout.write(out)
    return 0


def _paths(a):
    """(roadmap, entries). --roadmap moves BOTH by default: a caller judging a
    tree that is not this checkout (check_roadmap_fragment_required.sh extracts
    base and head into a scratch dir) must never silently read this repo's live
    docs/roadmaps/entries/ as the other tree's fragments."""
    roadmap = a.roadmap or ROADMAP
    if a.entries:
        return roadmap, a.entries
    if a.roadmap:
        return roadmap, os.path.join(os.path.dirname(os.path.abspath(roadmap)), "entries")
    return roadmap, ENTRIES


def _read(path, what):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as e:
        sys.stderr.write("FAIL %s is unreadable (%s) -- this box cannot judge\n" % (what, e))
        return None


def base_path(roadmap):
    """The frozen roadmap-base.yaml next to `roadmap` (#6185). Absent, roadmap.yaml is its own
    base -- the fixed point every scratch-dir caller (base/head extracted for a
    guard) and the selftest still rely on."""
    return os.path.join(os.path.dirname(os.path.abspath(roadmap)), "roadmap-base.yaml")


def _cmd_aggregate(a, roadmap, entries_dir):
    current = _read(roadmap, roadmap)
    if current is None:
        return 2
    base = current
    if os.path.exists(base_path(roadmap)):
        base = _read(base_path(roadmap), base_path(roadmap))
        if base is None:
            return 2
    frags = read_fragments(entries_dir)
    try:
        out = aggregate(base, frags)
    except ValueError as e:
        sys.stderr.write("FAIL %s\n" % e)
        return 1
    return _emit(a, current, out, frags, roadmap, base)


def _cmd_changed(a):
    """`changed --base A --head B` -> one `VERDICT<TAB>ID` line per entry whose
    BYTES differ, plus `PREAMBLE` if the header did. The consumer decides
    policy; this only reports. rc 2 (never 0) if either side is unreadable."""
    if not a.base or not a.head:
        sys.stderr.write("changed: --base and --head are both required\n")
        return 2
    base = _read(a.base, a.base)
    head = _read(a.head, a.head)
    if base is None or head is None:
        return 2
    pre_changed, verdicts = changed_entries(base, head)
    if pre_changed:
        sys.stdout.write("PREAMBLE\t(the header outside every entry)\n")
    for verdict, eid in verdicts:
        sys.stdout.write("%s\t%s\n" % (verdict, eid))
    return 0


def _unique_loader():
    """A SafeLoader that refuses a repeated key in one mapping (PyYAML keeps the last)."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        seen = set()
        for knode, _ in node.value:
            key = loader.construct_object(knode, deep=True)
            if key in seen:
                raise ValueError("repeated key %r" % (key,))
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return Loader


def _text_problem(text, stem):
    """The first text-level defect of a fragment, or ''. The splitter cuts on every `- id:` line
    (even inside a quoted scalar), a `---` / `...` line or a BOM breaks the aggregate's parse, and
    the first line is read raw by every text consumer, so it must be exactly `- id: <stem>`."""
    lines = text.splitlines()
    if "\ufeff" in text:
        return "carries a byte order mark"
    if any(re.match(r"(---|\.\.\.)(\s|$)", line) for line in lines):
        return "carries a '---' or '...' document marker line"
    cuts = sum(1 for line in lines if line.startswith("- id:"))
    if cuts != 1:
        return "has %d lines opening '- id:', want exactly 1" % cuts
    return "" if lines[0] == "- id: " + stem else "does not start with '- id: %s'" % stem


def _anchor_problem(text):
    """Anchors are scoped to the whole aggregate document: one fragment's `&a` collides with the
    base's or another fragment's, so a fragment carries none and aliases none."""
    import yaml
    try:
        events = list(yaml.parse(text))
    except yaml.YAMLError:
        return ""  # the loader reports it
    if any(getattr(e, "anchor", None) or isinstance(e, yaml.AliasEvent) for e in events):
        return "carries a YAML anchor or alias (anchors are shared across the aggregate)"
    return ""


def _load_unique(text):
    """-> (doc, reason): the parsed fragment, or why it would not parse."""
    import yaml
    try:
        return yaml.load(text, Loader=_unique_loader()), ""  # noqa: S506 - a SafeLoader subclass
    except (TypeError, ValueError, yaml.YAMLError) as e:
        return None, "unparseable: %s" % str(e).splitlines()[0]


def fragment_shape(path):
    """'' when the file is ONE mapping whose id is its stem; else the reason (PMAT-6270).
    Raises OSError when the file cannot be read: that is not-measured, never a verdict."""
    base = os.path.basename(path)
    if not base.endswith(".yaml"):
        return "is not named <id>.yaml"
    stem = base[:-len(".yaml")]
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    why = _text_problem(text, stem) or _anchor_problem(text)
    if why:
        return why
    doc, why = _load_unique(text)
    items = doc if isinstance(doc, list) else []
    if why or len(items) != 1:
        return why or "holds %d top-level items, want exactly 1" % len(items)
    return "" if isinstance(items[0], dict) and items[0].get("id") == stem else "its item is not a mapping with id %s" % stem


def _cmd_shape(a):
    """`shape --file F` -> rc 0 if F is one entry filed under its own id, else rc 1 and the reason."""
    if not a.file:
        sys.stderr.write("shape: --file is required\n")
        return 2
    try:
        import yaml  # noqa: F401
    except ImportError:
        sys.stderr.write("shape: PyYAML is not installed; not measured\n")
        return 2
    try:
        why = fragment_shape(a.file)
    except OSError as e:
        sys.stderr.write("shape: cannot read %s (%s); not measured\n" % (a.file, e.strerror or e))
        return 2
    if why:
        sys.stdout.write(why + "\n")
        return 1
    return 0


def _cmd_adopt(a, roadmap, entries_dir):
    if not a.ids:
        sys.stderr.write("adopt: name at least one entry id\n")
        return 2
    try:
        written = adopt(a.ids, roadmap, entries_dir)
        base = _read(roadmap, roadmap)
        if base is None:
            return 2
        frags = read_fragments(entries_dir)
        out = aggregate(base, frags)
    except (ValueError, KeyError) as e:
        sys.stderr.write("FAIL %s\n" % e)
        return 1
    with open(roadmap, "w", encoding="utf-8") as fh:
        fh.write(out)
    sys.stderr.write("adopt: %s -> %s/ ; %s regenerated from %d fragment(s)\n"
                     % (", ".join(written), entries_dir, roadmap, len(frags)))
    return 0


def _parser():
    ap = argparse.ArgumentParser(prog="roadmap_fragments.py")
    ap.add_argument("cmd", nargs="?", choices=["aggregate", "changed", "adopt", "shape"])
    ap.add_argument("ids", nargs="*", help="adopt: the entry id(s) to move into entries/")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="fail if roadmap.yaml is not what the aggregator produces")
    ap.add_argument("--roadmap", help="judge this roadmap.yaml instead of the repo's")
    ap.add_argument("--entries", help="read fragments from here instead of docs/roadmaps/entries/")
    ap.add_argument("--base", help="changed: the roadmap.yaml to compare FROM")
    ap.add_argument("--head", help="changed: the roadmap.yaml to compare TO")
    ap.add_argument("--file", help="shape: the one fragment to judge")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main(argv=None):
    ap = _parser()
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    roadmap, entries_dir = _paths(a)
    if a.cmd == "changed":
        return _cmd_changed(a)
    if a.cmd == "shape":
        return _cmd_shape(a)
    if a.cmd == "adopt":
        return _cmd_adopt(a, roadmap, entries_dir)
    if a.cmd != "aggregate":
        ap.print_usage(sys.stderr)
        return 2
    return _cmd_aggregate(a, roadmap, entries_dir)


if __name__ == "__main__":
    sys.exit(main())
