#!/usr/bin/env python3
"""Deterministic ids for anonymous proof obligations.

An obligation with no `id:` cannot be cited — not by a kani harness, not by a
test, not by a receipt, not by a commit. 3,622 of them were anonymous, and
`pv validate` reported `0 error(s)` for every one (#3314).

THE RULE
    <PREFIX>-<TYPE>-<NNN>
      NNN    the obligation's 1-based position in proof_obligations
      TYPE   from the obligation's own `type:` field
      PREFIX the prefix this file's kani harnesses ALREADY cite, if any;
             otherwise the initials of the contract's name parts.

The initials rule is not invented: it reproduces both prefixes a human chose in
this tree (gated-delta-net-v1 -> GDN, qwen35-hybrid-forward-v1 -> QHF).

COLLISIONS. 111 initials-prefixes are claimed by more than one contract. That is
cosmetic only while nothing cites across files — and it stops being cosmetic the
first time something does. So the id space is made global on day one: when a
base prefix is claimed by more than one file, every claimant that is not already
committed to it by its own kani harnesses takes a 4-hex suffix derived from its
own path. Keying the suffix to the file's OWN path (not to the set of
colliders) is what makes it stable: adding a contract later can never renumber
one that already exists.

NEVER RENAMES. A file whose obligations all carry ids is skipped entirely — its
convention is its own (52 such files use REG-OB-001, PO-HEH-001,
OBLIG-DATA-QUALITY-007-..., none of which this rule would reproduce). Within a
partially-named file, existing ids are untouched and a computed id that would
collide with one is bumped.

IDEMPOTENT. Running twice is a no-op: the second pass sees every file fully
named and skips it. `--check` asserts that and exits non-zero if not.
"""
import argparse, collections, hashlib, pathlib, re, sys

try:
    import yaml
except ImportError:  # pragma: no cover - environment, not logic
    print("ERROR: PyYAML is required", file=sys.stderr)
    raise SystemExit(2)

TYPE_CODE = {
    "invariant": "INV", "equivalence": "EQ", "bound": "BND", "bounds": "BND",
    "postcondition": "POST", "precondition": "PRE", "monotonicity": "MON",
    "soundness": "SND", "completeness": "CMP", "determinism": "DET",
    "roundtrip": "RT", "conservation": "CON", "safety": "SAFE",
    "idempotency": "IDEM", "idempotence": "IDEM", "classification": "CLS",
    "ordering": "ORD", "frame": "FRM", "state_machine": "SM",
    "termination": "TERM", "liveness": "LIVE", "linearity": "LIN",
    "loop_invariant": "LINV", "loop_variant": "LVAR", "associativity": "ASSOC",
    "independence": "IND", "old_state": "OLD", "subcontract": "SUB",
    "symmetry": "SYM", "equality": "EQ", "purity": "PURE", "totality": "TOT",
}
UNTYPED = "OBL"          # 79 obligations carry no `type:` at all

# Handled by #3315/#3316, which is in flight against the same two files.
# Once that lands they are fully named and this generator skips them anyway,
# so the exclusion decays to a no-op and idempotence is unaffected.
EXCLUDE = {"contracts/qwen35-hybrid-forward-v1.yaml",
           "contracts/qwen35-e2e-verification-v1.yaml"}

ID_RE = re.compile(r"^[A-Z][A-Z0-9]*-[A-Z]+-\d{3}$")


def type_code(raw):
    t = str(raw).strip().lower() if raw is not None else ""
    if not t:
        return UNTYPED
    if t in TYPE_CODE:
        return TYPE_CODE[t]
    letters = re.sub(r"[^A-Z]", "", t.upper())
    return letters[:4] or UNTYPED


def initials(stem):
    parts = [p for p in re.split(r"[-_.]", stem) if p and not re.fullmatch(r"v\d+", p)]
    out = "".join(p[0] for p in parts if p[0].isalnum()).upper()
    out = re.sub(r"^[^A-Z]+", "", out)
    return out or "OBL"


def cited_prefix(doc):
    """The prefix this file's own kani harnesses already commit to, if any."""
    seen = collections.Counter()
    for k in (doc.get("kani_harnesses") or []):
        if not isinstance(k, dict):
            continue
        ob = k.get("obligation")
        if isinstance(ob, str) and ID_RE.match(ob.strip()):
            seen[ob.strip().split("-")[0]] += 1
    return seen.most_common(1)[0][0] if seen else None


def load(path):
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    obs = doc.get("proof_obligations")
    if not isinstance(obs, list) or not obs:
        return None
    if not all(isinstance(o, dict) for o in obs):
        return None
    return doc, obs


def collect(root):
    """-> {relpath: (path, doc, obs, cited)} for every candidate file."""
    out = {}
    for f in sorted(root.rglob("*.yaml")):
        rel = f.as_posix()
        if rel in EXCLUDE:
            continue
        got = load(f)
        if got is None:
            continue
        doc, obs = got
        if all(o.get("id") for o in obs):        # fully named: never touched
            continue
        out[rel] = (f, doc, obs, cited_prefix(doc))
    return out


def _id_prefix(value):
    """The prefix an existing id claims, or None."""
    if not isinstance(value, str):
        return None
    m = re.match(r"^([A-Z][A-Z0-9]*)-", value.strip())
    return m.group(1) if m else None


def _collect_claims(cands, root):
    """(fixed, movable) prefix -> {relpath}, over the WHOLE corpus.

    `fixed` are prefixes claimed by ids that ALREADY EXIST: those cannot be
    renamed, so such a file owns its prefix outright. `movable` are prefixes
    this generator would mint and can still relocate.
    """
    fixed = collections.defaultdict(set)
    movable = collections.defaultdict(set)
    for f in sorted(root.rglob("*.yaml")):
        got = load(f)
        if got is None:
            continue
        _, obs = got
        rel = f.as_posix()
        for o in obs:
            pre = _id_prefix(o.get("id"))
            if pre:
                fixed[pre].add(rel)
        if rel in cands:
            movable[cands[rel][3] or initials(f.stem)].add(rel)
    return fixed, movable


def _canonical(cands):
    """relpath -> the lexicographically first candidate with IDENTICAL bytes.

    56 stems exist twice under contracts/ (`contracts/X.yaml` and
    `contracts/aprender/X.yaml`), and 8 of those pairs were byte-identical. They
    are ONE logical contract in two places; the duplication is a pre-existing
    defect that `check_duplicate_stems` already tracks against a baseline. A
    first cut treated each copy as a separate claimant, handed the second one a
    path-keyed suffix, and turned 8 identical pairs into 8 DIVERGENT ones --
    `real_corpus_census_names_every_baselined_stem` went 48 -> 55. Identical
    copies get identical ids, so the generator never widens that drift.
    """
    by_bytes = {}
    for rel, (path, _, _, _) in cands.items():
        by_bytes.setdefault(path.read_bytes(), []).append(rel)
    canon = {}
    for rels in by_bytes.values():
        first = min(rels)
        for r in rels:
            canon[r] = first
    return canon


def assign_prefixes(cands, root):
    """Base prefix per file, with a path-keyed suffix when another file owns it.

    Ownership is resolved against the whole corpus, including the 52 files this
    generator never touches: their existing ids claim prefixes too (PO- on 22
    files, OBLIG- on 16), and a computed prefix duplicating one would mint a
    genuine cross-file collision. A first cut counted only computed prefixes and
    produced exactly that -- AL-BND-001 in two files.

    A file whose ids ALREADY EXIST wins its prefix outright, whatever the sort
    order: those ids cannot be renamed, so ownership is not theirs to lose. A
    second cut used a plain lexicographic min and handed ZZ to `zebra-zoo`
    over `zeta-zulu`, which already had ZZ-INV-001 -- caught by the selftest,
    not by the corpus, where the orderings happened not to bite.

    Byte-identical candidates are ONE claimant (see `_canonical`): every copy
    takes the decision made for its canonical path, so identical files end up
    with identical ids rather than one bare and one suffixed.

    Among movable claimants the owner is the lexicographically first, so
    ownership does not move when an unrelated contract is added. Everyone else
    takes sha256(own relative path)[:4] -- keyed to the file's OWN path, never
    to the set of colliders, which is what keeps an existing id stable.
    """
    fixed, movable = _collect_claims(cands, root)
    canon = _canonical(cands)
    owner = {}
    for pre in set(fixed) | set(movable):
        pool = fixed[pre] if fixed.get(pre) else movable[pre]
        owner[pre] = min(canon.get(r, r) for r in pool)
    final = {}
    for rel, (path, _, _, cited) in cands.items():
        base = cited or initials(path.stem)
        me = canon.get(rel, rel)
        final[rel] = base if owner.get(base) == me else \
            f"{base}{hashlib.sha256(me.encode()).hexdigest()[:4].upper()}"
    return final


def plan_file(obs, prefix):
    """-> list of (index, id) for the anonymous rows only."""
    taken = {o["id"] for o in obs if o.get("id")}
    out = []
    for i, o in enumerate(obs, 1):
        if o.get("id"):
            continue
        cand = f"{prefix}-{type_code(o.get('type'))}-{i:03d}"
        n = i
        while cand in taken:                     # never shadow an existing id
            n += 1
            cand = f"{prefix}-{type_code(o.get('type'))}-{n:03d}"
        taken.add(cand)
        out.append((i, cand))
    return out


def _block_bounds(text):
    """(body_start, body_end) of the proof_obligations list body."""
    start = text.index("\nproof_obligations:\n") + 1
    body = start + len("proof_obligations:\n")
    m = re.search(r"\n(?=[A-Za-z_][A-Za-z0-9_]*:)", text[body:])
    return body, body + (m.start() + 1 if m else len(text) - body)


def _group_items(block, item_re):
    """The block's lines grouped one list-item per entry."""
    items = []
    for line in block.splitlines(keepends=True):
        if item_re.match(line) or not items:
            items.append([line])
        else:
            items[-1].append(line)
    return items


def rewrite(text, plan):
    """Insert `- id: X` as the first key of each anonymous obligation.

    A targeted text edit, not a YAML round-trip: re-serialising would reformat
    every contract in the corpus and bury the change.

    The list may be indented. 83 of 823 contracts write `  - type:` rather than
    `- type:`, and a first cut matching only column 0 silently no-oped on every
    one -- it reported them named and changed nothing, which --check then caught
    as 410 obligations still anonymous. The indent is read from the file's own
    first item rather than assumed.
    """
    if not plan:
        return text
    body, end = _block_bounds(text)
    block, tail = text[body:end], text[end:]
    im = re.search(r"^([ \t]*)- ", block, re.M)
    if im is None:
        return text
    indent = im.group(1)
    item_re = re.compile(r"^" + re.escape(indent) + r"- ")

    by_pos, pos = dict(plan), 0
    items = _group_items(block, item_re)
    for it in items:
        if not item_re.match(it[0]):
            continue
        pos += 1
        if pos in by_pos:
            it[0] = f"{indent}- id: {by_pos[pos]}\n{indent}  " + it[0][len(indent) + 2:]
    return text[:body] + "".join("".join(i) for i in items) + tail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default="contracts")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any anonymous obligation remains (idempotence gate)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    root = pathlib.Path(a.root)
    cands = collect(root)
    if a.check:
        if cands:
            print(f"FAIL: {len(cands)} file(s) still carry an anonymous obligation")
            for rel in sorted(cands)[:10]:
                print(f"  {rel}")
            return 1
        print("OK: every proof obligation carries an id")
        return 0

    prefixes = assign_prefixes(cands, root)
    files = named = 0
    for rel in sorted(cands):
        path, _, obs, _ = cands[rel]
        plan = plan_file(obs, prefixes[rel])
        if not plan:
            continue
        path.write_text(rewrite(path.read_text(encoding="utf-8"), plan), encoding="utf-8")
        files += 1
        named += len(plan)
    print(f"named {named} obligation(s) across {files} file(s)")
    return 0


# name, contract body, expected ids for the ANONYMOUS rows ([] + skip=True
# means the file must not be touched at all).
CASES = (
    ("initials + type codes",
     "proof_obligations:\n- type: invariant\n  property: one\n- type: bound\n  property: two\n",
     ["ABC-INV-001", "ABC-BND-002"], False),
    ("untyped obligation",
     "proof_obligations:\n- property: no type here\n",
     ["ABC-OBL-001"], False),
    ("kani-cited prefix wins over initials",
     "proof_obligations:\n- type: invariant\n  property: one\n"
     "kani_harnesses:\n- id: K1\n  obligation: ZZZ-INV-001\n",
     ["ZZZ-INV-001"], False),
    ("existing id is never shadowed",
     "proof_obligations:\n- id: ABC-INV-001\n  type: invariant\n  property: kept\n"
     "- type: invariant\n  property: anonymous\n",
     ["ABC-INV-002"], False),
    ("fully named file is skipped",
     "proof_obligations:\n- id: REG-OB-001\n  type: invariant\n  property: kept\n",
     [], True),
    ("INDENTED list items are named too",
     "proof_obligations:\n  - type: invariant\n    property: one\n"
     "  - type: bound\n    property: two\n",
     ["ABC-INV-001", "ABC-BND-002"], False),
)

# source, plan, expected output -- rewrite() must place the id first and leave
# every other byte alone, in both the flush and the indented shape.
REWRITES = (
    ("flush list",
     "kind: K\nproof_obligations:\n- type: invariant\n  property: one\nnext_key: v\n",
     [(1, "ABC-INV-001")],
     "kind: K\nproof_obligations:\n- id: ABC-INV-001\n  type: invariant\n  property: one\nnext_key: v\n"),
    ("indented list",
     "kind: K\nproof_obligations:\n  - type: invariant\n    property: one\nnext_key: v\n",
     [(1, "ABC-INV-001")],
     "kind: K\nproof_obligations:\n  - id: ABC-INV-001\n    type: invariant\n    property: one\nnext_key: v\n"),
    ("no-op when nothing to name",
     "kind: K\nproof_obligations:\n- id: ABC-INV-001\n  type: invariant\n  property: one\nnext_key: v\n",
     [],
     "kind: K\nproof_obligations:\n- id: ABC-INV-001\n  type: invariant\n  property: one\nnext_key: v\n"),
)


def _run_case(name, body, want, want_skip):
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "a-b-c-v1.yaml").write_text(body)
    cands = collect(d)
    if want_skip:
        return (True, "") if not cands else (False, "expected the file to be SKIPPED")
    if not cands:
        return False, "file was skipped but should have been named"
    rel = next(iter(cands))
    _, _, obs, _ = cands[rel]
    got = [i for _, i in plan_file(obs, assign_prefixes(cands, d)[rel])]
    return (got == want), f"got {got} want {want}"


def _report(ok, label, detail=""):
    """Print one case-table row; return 0 on pass, 1 on fail."""
    if ok:
        print(f"PASS {label}")
        return 0
    print(f"FAIL {label}: {detail}")
    return 1


def _selftest_cases():
    rc = 0
    for name, body, want, skip in CASES:
        ok, why = _run_case(name, body, want, skip)
        rc |= _report(ok, name, why)
    return rc


def _selftest_collisions():
    """Two files that compute the SAME base prefix must not both get it.

    This is the second bug the corpus caught and review did not: collision
    detection counted only COMPUTED prefixes, so it could not see that a file
    this generator never touches already owned one, and it minted AL-BND-001 in
    two files. Both halves are asserted here -- a collision between two
    candidates, and a collision with an already-named file that is skipped.
    """
    import tempfile
    rc = 0
    d = pathlib.Path(tempfile.mkdtemp())
    # two candidates whose initials BOTH reduce to "AB" -- they must collide,
    # or this row proves nothing (an earlier fixture used a-b / a-b-other-thing,
    # which give AB and ABOT: no collision, and all three rows passed with the
    # bug restored).
    (d / "alpha-beta-v1.yaml").write_text(
        "proof_obligations:\n- type: invariant\n  property: one\n")
    (d / "apple-banana-v1.yaml").write_text(
        "proof_obligations:\n- type: invariant\n  property: two\n")
    # a file that is fully named and therefore SKIPPED, but still owns "ZZ"
    (d / "zeta-zulu-v1.yaml").write_text(
        "proof_obligations:\n- id: ZZ-INV-001\n  type: invariant\n  property: owned\n")
    # a candidate whose initials are ALSO ZZ -- it must not take the owned prefix
    (d / "zebra-zoo-v1.yaml").write_text(
        "proof_obligations:\n- type: invariant\n  property: three\n")

    cands = collect(d)
    pre = assign_prefixes(cands, d)
    got = {pathlib.Path(k).name: v for k, v in pre.items()}

    ab = [v for k, v in got.items() if k in ("alpha-beta-v1.yaml", "apple-banana-v1.yaml")]
    rc |= _report(len(set(ab)) == len(ab), "collision/two candidates get distinct prefixes",
                  f"got {ab}")
    zz = got.get("zebra-zoo-v1.yaml", "")
    rc |= _report(zz != "ZZ", "collision/a skipped file still owns its prefix",
                  f"candidate took {zz!r}, which the named file owns")

    # byte-identical copies of one contract in two dirs must get IDENTICAL ids
    (d / "sub").mkdir()
    same = "proof_obligations:\n- type: invariant\n  property: twin\n"
    (d / "twin-v1.yaml").write_text(same)
    (d / "sub" / "twin-v1.yaml").write_text(same)
    cands2 = collect(d)
    pre2 = assign_prefixes(cands2, d)
    twins = {pathlib.Path(k).as_posix(): v for k, v in pre2.items() if k.endswith("twin-v1.yaml")}
    rc |= _report(len(set(twins.values())) == 1,
                  "collision/byte-identical duplicate-stem copies share one prefix",
                  f"got {twins}")

    # and the whole point: no id is ever minted twice (over the FIRST fixture set)
    ids = []
    for rel, (_, _, obs, _) in cands.items():
        ids += [i for _, i in plan_file(obs, pre[rel])]
    rc |= _report(len(ids) == len(set(ids)), "collision/no id is minted twice", f"got {ids}")
    return rc


def _selftest_rewrites():
    rc = 0
    for name, src, plan, want in REWRITES:
        got = rewrite(src, plan)
        rc |= _report(got == want, f"rewrite/{name}", repr(got))
    return rc


def selftest():
    rc = _selftest_cases() | _selftest_collisions() | _selftest_rewrites()
    print("selftest: " + ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
