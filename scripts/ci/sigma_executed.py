#!/usr/bin/env python3
"""Σ-executed invariant (#4433): the tests CI ran are exactly the tests it owes.

Operator ruling 2026-09-25: "Same test set as today (Σ executed unchanged)"; the
cop's condition on the ci.yml PR: "The test-id set across the 5 jobs must equal
today's set (by id, not by count), and a mutation that drops one test must turn
it RED."

Every mode compares two sets of test IDS and fails on any difference, in either
direction. The UNIVERSE is listed with the filter written here, in the Σ step --
never read back from the step that ran the tests -- so a narrowed run step (an
added --exclude, a --partition, a dropped fragment) cannot narrow its own yardstick.

  nextest  --junit J --list-json L    ids = "<binary-id> <test>"
  cargo    --log F --step S --list L --ignored I
                                      a `cargo test` step's "test X ... ok|FAILED"
                                      lines vs `-- --list` minus `-- --list --ignored`
  explicit --log F --step S --list L  the explicit-command runner's
                                      "::group::[i/N] <cmd>" lines vs its --list
  --self-test                         the case table: each mode must go RED on a
                                      dropped id, an extra id and an empty universe

--log/--step read the fat driver's section log (FAT_SECTION_LOG), sliced to the
step whose "##[step N] <name>" marker contains S.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def compare(kind: str, executed: set, universe: set) -> int:
    missing = sorted(universe - executed)
    extra = sorted(executed - universe)
    print(f"Σ {kind}: universe={len(universe)} executed={len(executed)} "
          f"missing={len(missing)} extra={len(extra)}")
    if not universe:
        print(f"::error::Σ {kind}: the universe is EMPTY -- zero owed tests is broken wiring, not a pass")
        return 1
    for m in missing[:50]:
        print(f"::error::Σ {kind}: owed but NOT executed: {m}")
    for x in extra[:50]:
        print(f"::error::Σ {kind}: executed but not in the universe: {x}")
    if missing or extra:
        more = max(len(missing), len(extra)) - 50
        if more > 0:
            print(f"  ... and {more} more")
        return 1
    print(f"Σ {kind}: OK -- {len(universe)} ids, executed set == universe")
    return 0


def junit_ids(path: Path) -> set:
    ids = set()
    for tc in ET.parse(path).getroot().iter("testcase"):
        if tc.find("skipped") is not None:
            continue
        ids.add(f"{tc.get('classname')} {tc.get('name')}")
    return ids


def nextest_list_ids(path: Path) -> set:
    doc = json.loads(Path(path).read_text())
    ids = set()
    for bid, suite in (doc.get("rust-suites") or {}).items():
        for name, case in (suite.get("testcases") or {}).items():
            fm = (case.get("filter-match") or {}).get("status")
            if fm == "matches":
                ids.add(f"{bid} {name}")
    return ids


def step_slice(log: str, step: str) -> str:
    """The text between the step's marker and the next step marker."""
    out, on = [], False
    for line in log.splitlines():
        m = re.match(r"^##\[step \d+\] (.*)$", line)
        if m:
            label = m.group(1)
            if on and not re.match(r"^(success|failure) \(", label):
                break
            if step in label and not re.match(r"^(success|failure) \(", label):
                on = True
                continue
        if on:
            out.append(line)
    if not on:
        raise SystemExit(f"Σ: no step marker containing {step!r} in the section log")
    return "\n".join(out)


def cargo_ids(text: str) -> set:
    # libtest suffixes a #[should_panic] test: "test X - should panic ... ok".
    return {m.group(1) for m in re.finditer(
        r"^test (\S+)(?: - should panic)? \.\.\. (?:ok|FAILED)\b", text, re.M)}


def cargo_list_ids(text: str) -> set:
    return {m.group(1) for m in re.finditer(r"^(\S+): test$", text, re.M)}


def explicit_ids(text: str) -> set:
    return {m.group(1).strip() for m in re.finditer(r"^::group::\[\d+/\d+\] (.*)$", text, re.M)}


def run(argv) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--self-test":
        return self_test()
    mode, rest = argv[0], argv[1:]
    # --junit and --log repeat: a sharded run passes one per shard and the
    # executed set is their UNION, compared with the one universe.
    opts, many = {}, {"--junit": [], "--log": []}
    for k, v in zip(rest[::2], rest[1::2]):
        if k in many:
            many[k].append(v)
        else:
            opts[k] = v
    if mode == "nextest":
        executed = set().union(*(junit_ids(Path(j)) for j in many["--junit"]))
        return compare(opts.get("--kind", "nextest"), executed,
                       nextest_list_ids(Path(opts["--list-json"])))
    slices = [step_slice(Path(f).read_text(errors="replace"), opts["--step"]) for f in many["--log"]]
    if mode == "cargo":
        uni = cargo_list_ids(Path(opts["--list"]).read_text()) - \
            cargo_list_ids(Path(opts["--ignored"]).read_text())
        return compare(opts.get("--kind", "cargo"), set().union(*map(cargo_ids, slices)), uni)
    if mode == "explicit":
        uni = {l.strip() for l in Path(opts["--list"]).read_text().splitlines() if l.strip()}
        return compare(opts.get("--kind", "explicit"), set().union(*map(explicit_ids, slices)), uni)
    print(f"Σ: unknown mode {mode!r}", file=sys.stderr)
    return 2


# --------------------------------------------------------------------------
JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="s">
{cases}</testsuite></testsuites>"""


def self_test() -> int:
    rows = []
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        def nextest_row(label, executed, universe, want):
            cases = "\n".join(f'<testcase classname="{b}" name="{n}"/>' for b, n in executed)
            (d / "j.xml").write_text(JUNIT.format(cases=cases))
            suites = {}
            for b, n in universe:
                suites.setdefault(b, {"testcases": {}})["testcases"][n] = {
                    "filter-match": {"status": "matches"}}
            suites.setdefault("aprender-core", {"testcases": {}})["testcases"]["t::skipped_by_filter"] = {
                "filter-match": {"status": "mismatch", "reason": "ignored"}}
            (d / "l.json").write_text(json.dumps({"rust-suites": suites}))
            rc = run(["nextest", "--junit", str(d / "j.xml"), "--list-json", str(d / "l.json")])
            rows.append((label, want, rc))

        u = [("aprender-core", "a::one"), ("aprender-core", "a::two"), ("apr-cli", "c::three")]
        nextest_row("nextest: executed == universe", u, u, 0)
        nextest_row("nextest: MUTANT drops one test", u[:-1], u, 1)
        nextest_row("nextest: an extra test", u + [("x", "y")], u, 1)
        nextest_row("nextest: empty universe", [], [], 1)
        # a skipped testcase is not executed
        cases = u[:-1]
        (d / "j.xml").write_text(JUNIT.format(cases="\n".join(
            [f'<testcase classname="{b}" name="{n}"/>' for b, n in cases]
            + ['<testcase classname="apr-cli" name="c::three"><skipped/></testcase>'])))
        rc = run(["nextest", "--junit", str(d / "j.xml"), "--list-json", str(d / "l.json")])
        rows.append(("nextest: a <skipped/> case is not executed", 1, rc))

        def log_with(step_body):
            return ("##[step 3] Earlier step\ntest z::other ... ok\n##[step 3] success (1s)\n"
                    f"##[step 4] Compute tests (x)\n{step_body}\n##[step 4] success (2s)\n"
                    "##[step 5] Later\ntest z::later ... ok\n")

        (d / "list.txt").write_text("b::one: test\nb::two: test\nb::ign: test\n\n3 tests\n")
        (d / "ign.txt").write_text("b::ign: test\n")
        for label, body, want in (
            ("cargo: executed == universe", "test b::one ... ok\ntest b::two ... FAILED\ntest b::ign ... ignored", 0),
            ("cargo: a should_panic test counts as executed",
             "test b::one ... ok\ntest b::two - should panic ... ok", 0),
            ("cargo: MUTANT drops one test", "test b::one ... ok", 1),
            ("cargo: other steps' tests do not count", "test b::one ... ok\n", 1),
        ):
            (d / "sec.log").write_text(log_with(body))
            rc = run(["cargo", "--log", str(d / "sec.log"), "--step", "Compute tests",
                      "--list", str(d / "list.txt"), "--ignored", str(d / "ign.txt")])
            rows.append((label, want, rc))

        (d / "ex.txt").write_text("cargo test -p a --test one\ncargo test -p b --test two\n")
        for label, body, want in (
            ("explicit: every fragment ran", "::group::[1/2] cargo test -p a --test one\n::endgroup::\n"
             "::group::[2/2] cargo test -p b --test two\n::endgroup::", 0),
            ("explicit: MUTANT drops one fragment", "::group::[1/2] cargo test -p a --test one\n::endgroup::", 1),
        ):
            (d / "sec.log").write_text(log_with(body).replace("Compute tests", "Integration tests"))
            rc = run(["explicit", "--log", str(d / "sec.log"), "--step", "Integration tests",
                      "--list", str(d / "ex.txt")])
            rows.append((label, want, rc))
        # Sharded: one junit / one log per shard, the executed set is the union.
        def junit(name, cases):
            (d / name).write_text(JUNIT.format(cases="\n".join(
                f'<testcase classname="{b}" name="{n}"/>' for b, n in cases)))
            return str(d / name)
        suites = {}
        for b, n in u:
            suites.setdefault(b, {"testcases": {}})["testcases"][n] = {"filter-match": {"status": "matches"}}
        (d / "l.json").write_text(json.dumps({"rust-suites": suites}))
        for label, parts, want in (
            ("nextest shards: union of three partitions == universe", [u[:1], u[1:2], u[2:]], 0),
            ("nextest shards: MUTANT a shard's partition lost", [u[:1], u[1:2]], 1),
            ("nextest shards: no junit at all", [], 1),
        ):
            args = ["nextest", "--list-json", str(d / "l.json")]
            for i, part in enumerate(parts):
                args += ["--junit", junit(f"s{i}.xml", part)]
            rows.append((label, want, run(args)))
        g1 = "::group::[1/1] cargo test -p a --test one\n::endgroup::"
        g2 = "::group::[1/1] cargo test -p b --test two\n::endgroup::"
        for label, bodies, want in (
            ("explicit shards: union of the shard logs == list", [g1, g2], 0),
            ("explicit shards: MUTANT a shard's log lost", [g1], 1),
        ):
            args = ["explicit", "--step", "Integration tests", "--list", str(d / "ex.txt")]
            for i, body in enumerate(bodies):
                (d / f"s{i}.log").write_text(log_with(body).replace("Compute tests", "Integration tests"))
                args += ["--log", str(d / f"s{i}.log")]
            rows.append((label, want, run(args)))
        try:
            (d / "sec.log").write_text(log_with(""))
            run(["explicit", "--log", str(d / "sec.log"), "--step", "No such step", "--list", str(d / "ex.txt")])
            rows.append(("a missing step marker refuses", 1, 0))
        except SystemExit:
            rows.append(("a missing step marker refuses", 1, 1))
    bad = 0
    print("\n--- Σ-executed case table ---")
    for label, want, got in rows:
        ok = (want == 0) == (got == 0)
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} want={'PASS' if want == 0 else 'RED '} got_rc={got}  {label}")
    print(f"Σ self-test: {len(rows) - bad}/{len(rows)} rows as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
