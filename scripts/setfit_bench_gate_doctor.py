#!/usr/bin/env python3
"""Doctor a SCRATCH copy of the benchmark directory into a verifier spot-check.

TWO MODES, one per attack the probe replays:

``escape``
    Spot-check E. Moves the committed lock record out of the tree and repoints
    the row's ``lock_record_path`` at it. Prints the escape path.

``selection-hash-zeros``
    Spot-check F. Doctors the row's ``selection_manifest_hash`` - THE PAIRING
    KEY - to 64 zeros and leaves the committed selection manifest untouched, so
    the only disagreement is between the row's claim and the manifest the gate
    recomputes. Prints the doctored key.

``f-avg-to-0-99``
    Spot-check D. Doctors ``quality.f_avg`` to 0.99 AND its ``f_avg_bits``
    sibling to match, leaving the row's own ``confusion_matrix`` untouched - so
    the row is internally consistent in every way EXCEPT against the counts it
    records itself, which is precisely what the closed-form cross-check exists to
    find. Verification measured this tree returning rc=0 with the published mean
    moving 0.4746 -> 0.5278. Prints the doctored headline.

Both repair the row's own ``semantic_hash``, the manifest's ``row_sha256`` for
that cell, and the manifest's envelope digest, for the reason in step 3 below.

Called only by ``scripts/setfit_bench_gate_door_probe.sh``. It is a separate file
rather than an inline heredoc because ``bashrs`` — the shell linter this repo uses
instead of shellcheck — does not skip a quoted heredoc body, and reports a dozen
phantom shell parse errors against the Python inside one. A probe whose own lint
gate is red teaches a reader to ignore lint output.

WHAT IT DOES, and why each step is load-bearing:

1. Re-derives the COMMITTED digests first and refuses to continue if the scheme
   does not reproduce. Without this the script could write wrong digests, the
   gate would refuse at the digest step, and the probe would report success
   having proven nothing about path resolution.
2. MOVES the committed lock record out of the benchmark directory, so the escape
   target holds the very bytes the row attests and nothing legitimate inside the
   tree can satisfy the row. A target holding OTHER bytes would be refused as a
   provenance mismatch — still red, but red for the wrong reason, proving the
   escape was DETECTED rather than that it SUCCEEDED.
3. Repairs the row's ``semantic_hash``, the manifest's ``row_sha256`` for that
   cell, and the manifest's envelope digest. ``lock_record_path`` sits INSIDE the
   hashed payload, so an unrepaired edit is refused before provenance is reached.

The digest scheme is ``sha256`` over the payload's COMPACT, SERDE-ORDER JSON —
key order as the file carries it, not sorted. That was measured against the
committed tree, not assumed: sorting the keys reproduces neither the row nor the
manifest digest.

This is fixture doctoring of throwaway JSON inside a scratch directory, the same
method ``05-VERIFICATION.md`` used to produce spot-checks B through G. It is not
the ML-stack substitution that ``crates/aprender-train/CLAUDE.md``'s Python
prohibition targets, and it never touches the checkout.

Usage: setfit_bench_gate_doctor.py <mode> <scratch-bench-dir> <outside-dir> <cell>
Prints the mode's own handle (escape path, or doctored key) on stdout.
"""

import hashlib
import json
import os
import shutil
import struct
import sys


def digest(payload):
    """The committed scheme: sha256 over the payload's compact serde-order JSON."""
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_pretty(path, value):
    """Write pretty JSON with the trailing newline the Rust writer emits.

    The FILE's whitespace is free: the digest is over the payload's own canonical
    compact bytes, so rows stay reviewable in diffs without weakening anything.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


MODES = ("escape", "selection-hash-zeros", "f-avg-to-0-99")

# The committed tree this script must never doctor, anchored on THIS FILE's own
# location rather than on the caller's cwd.
#
# CR-01. The guard used to read:
#
#     os.path.realpath(bench_dir).startswith(os.path.realpath("benchmarks"))
#
# and was wrong twice over. `realpath("benchmarks")` is CWD-RELATIVE and does not
# require the path to exist, so from `crates/` it resolved to
# `<repo>/crates/benchmarks`, matched nothing, and let the script proceed to
# `shutil.move` the committed lock record out of the real tree. Measured: the
# guard refused from the repo root and did NOT refuse from `crates/`. And
# `str.startswith` is a STRING-PREFIX test on a path — the exact containment bug
# plan 05-15 spent a round removing from `resolve_committed_evidence_path`, which
# is component-wise `Path::starts_with` over canonicalized paths for this reason.
# A sibling named `benchmarks-evil` string-prefixes `benchmarks` while being no
# part of it.
#
# Anchored on `__file__`, the base is the same from every working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED_BENCHMARKS = os.path.join(REPO_ROOT, "benchmarks")


def is_inside_committed_benchmarks(path):
    """True when `path` is the committed benchmarks tree or lives inside it.

    Component-wise via ``os.path.commonpath``, never ``str.startswith``: a
    sibling whose name merely begins with ``benchmarks`` is NOT inside it.
    Both sides go through ``realpath`` first, so a symlink into the committed
    tree is caught by where it LANDS rather than by how it is spelled.
    """
    target = os.path.realpath(path)
    base = os.path.realpath(COMMITTED_BENCHMARKS)
    try:
        return os.path.commonpath([target, base]) == base
    except ValueError:
        # Different drives (Windows) — cannot be inside. Fail closed is not
        # available here: `commonpath` raising means the paths share no root,
        # which is positive evidence of NON-containment, not an unknown.
        return False


def self_test():
    """Must-match / must-not-match case table for the containment guard.

    CLAUDE.md Verification Discipline rule 7 — a guard ships a case table, and
    the table is re-run rather than the predicate re-read. Rule 4 — the table is
    exercised from more than one working directory, because CWD-dependence is
    the defect it exists to catch.

    Pure predicate only: it never doctors anything, so it is safe to run against
    the real committed paths.
    """
    committed = COMMITTED_BENCHMARKS
    cases = [
        # (path, must_refuse, why)
        (committed, True, "the committed tree itself"),
        (os.path.join(committed, "tweeteval-stance"), True, "a cell dir inside it"),
        (os.path.join(committed, "tweeteval-stance", "rows"), True, "deeper inside it"),
        (committed + "-evil", False, "SIBLING that string-prefixes it — startswith goes red here"),
        (os.path.join(REPO_ROOT, "benchmarksomething"), False, "another prefix sibling"),
        ("/tmp/scratch-bench", False, "a scratch dir outside the repo"),
        (os.path.join(REPO_ROOT, "crates"), False, "a repo dir that is not benchmarks"),
        (os.path.join(REPO_ROOT, "benchmarks", "..", "crates"), False, "climbs back out"),
    ]
    cwds = [REPO_ROOT, os.path.join(REPO_ROOT, "crates"), "/tmp"]

    failures = []
    for cwd in cwds:
        if not os.path.isdir(cwd):
            continue
        os.chdir(cwd)
        for path, must_refuse, why in cases:
            got = is_inside_committed_benchmarks(path)
            ok = got == must_refuse
            print(
                "[guard] cwd={0:<28} refuse={1:<5} expected={2:<5} {3} ({4})".format(
                    os.path.basename(cwd) or cwd, str(got), str(must_refuse),
                    "ok" if ok else "FAIL", why,
                )
            )
            if not ok:
                failures.append((cwd, path, must_refuse, got))

    if failures:
        print("\n{0} case(s) FAILED".format(len(failures)))
        for cwd, path, want, got in failures:
            print("  cwd={0} path={1} expected refuse={2} got={3}".format(cwd, path, want, got))
        return 1
    print("\nall {0} cases pass from {1} working directories".format(
        len(cases) * len([c for c in cwds if os.path.isdir(c)]),
        len([c for c in cwds if os.path.isdir(c)]),
    ))
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) != 5 or argv[1] not in MODES:
        raise SystemExit(
            "usage: setfit_bench_gate_doctor.py <{0}> "
            "<scratch-bench-dir> <outside-dir> <cell>".format("|".join(MODES))
        )
    mode, bench_dir, outside_dir, cell = argv[1], argv[2], argv[3], argv[4]

    row_path = os.path.join(bench_dir, "rows", cell + ".json")
    lock_path = os.path.join(bench_dir, "locks", cell + ".lock.json")
    manifest_path = os.path.join(bench_dir, "run-manifest.json")
    escape_path = os.path.join(outside_dir, "anywhere.json")

    # Refuse to doctor the committed tree, from any working directory a caller
    # runs this from. See `is_inside_committed_benchmarks` for why the previous
    # cwd-relative `startswith` form could not hold (CR-01), and `--self-test`
    # for the case table that keeps it honest.
    if is_inside_committed_benchmarks(bench_dir):
        raise SystemExit(
            "refusing to doctor the committed benchmark directory: {0} is inside {1}".format(
                os.path.realpath(bench_dir), COMMITTED_BENCHMARKS
            )
        )

    row = load(row_path)
    manifest = load(manifest_path)

    # PROVE THE REPAIR SCHEME BEFORE USING IT.
    if digest(row["payload"]) != row["semantic_hash"]:
        raise SystemExit("the row digest scheme no longer reproduces the committed digest")
    if digest(manifest["payload"]) != manifest["semantic_hash"]:
        raise SystemExit("the manifest digest scheme no longer reproduces the committed digest")

    if mode == "escape":
        shutil.move(lock_path, escape_path)
        row["payload"]["evidence"]["setfit"]["lock"]["lock_record_path"] = escape_path
        handle = escape_path
    elif mode == "f-avg-to-0-99":
        # THE CONFUSION MATRIX IS LEFT ALONE. Doctoring it too would produce a row
        # that is internally consistent and would be ACCEPTED - which is the
        # residual the report discloses, not the attack this case replays. Only
        # the published headline moves, and its bits sibling moves with it so the
        # refusal cannot come from the row contradicting itself in two encodings.
        quality = row["payload"]["quality"]
        handle = "0.99"
        if quality["f_avg"] == 0.99:
            raise SystemExit("the committed row already publishes 0.99; this is vacuous")
        quality["f_avg"] = 0.99
        quality["f_avg_bits"] = struct.unpack("<Q", struct.pack("<d", 0.99))[0]
    else:
        # THE COMMITTED SELECTION MANIFEST IS LEFT ALONE. Only the row's claim
        # about it moves, so the single disagreement is the one the gate is
        # supposed to find by recomputing from the manifest's own bytes. Editing
        # the manifest instead would produce a refusal that proves the manifest
        # was read, but not that the ROW's claim was ever compared against it.
        selections = os.path.join(bench_dir, "selections")
        if not os.path.isdir(selections):
            raise SystemExit("the scratch copy carries no selections/ directory")
        handle = "0" * 64
        if row["payload"]["selection_manifest_hash"] == handle:
            raise SystemExit("the committed row already claims the doctored key; this is vacuous")
        row["payload"]["selection_manifest_hash"] = handle

    row["semantic_hash"] = digest(row["payload"])
    write_pretty(row_path, row)

    matched = 0
    for entry in manifest["payload"]["cells"]:
        rendered = "{0}-s{1}-seed{2}".format(entry["method"], entry["shots"], entry["seed"])
        if rendered == cell:
            entry["row_sha256"] = row["semantic_hash"]
            matched += 1
    if matched != 1:
        raise SystemExit(
            "expected exactly one manifest entry for {0}, found {1}".format(cell, matched)
        )
    manifest["semantic_hash"] = digest(manifest["payload"])
    write_pretty(manifest_path, manifest)

    print(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
