#!/usr/bin/env python3
"""Split aprender-serve's lib tests into per-module shards for `make coverage` (#4023).

Why: one process running all ~16k aprender-serve lib tests builds up memory across tests
(#4028): 30 GB peak single-threaded and 45 GB at 22 threads, measured on gx10, while each
top-level module alone stays small. On yoga's 28 GB box earlyoom SIGTERMed it under llvm-cov
(coverage-nightly run 35868368976), so the nightly measured nothing for the largest crate.
Running the suite as several processes, one module group each, keeps every process small;
the `--no-report` runs' profiles are merged by one final `cargo llvm-cov report`.

Usage: coverage_serve_shards.py <test-list> <skips-file> <out-dir>
  <test-list>   the binary's `--list` output (`path: test` lines; anything else is ignored)
  <skips-file>  scripts/coverage-skips.txt (exact paths; #-comments and blanks ignored)
  <out-dir>     receives shard-NN.txt, one exact test path per line

Modules with >= ALONE tests get a shard of their own; the rest are packed, largest first,
into shards of at most PACK tests. A module is never split, so each process holds one
module's state. The partition is CHECKED before writing: every listed test not skipped is
in exactly one shard. A partition that loses or duplicates a test exits 1 and writes nothing.
"""
import collections
import pathlib
import sys

ALONE = 1000
PACK = 2000


def main(argv):
    if len(argv) != 4:
        sys.exit(__doc__)
    listed = [l[: -len(": test")] for l in pathlib.Path(argv[1]).read_text().splitlines() if l.endswith(": test")]
    skips = {
        l.strip()
        for l in pathlib.Path(argv[2]).read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    }
    wanted = [t for t in listed if t not in skips]
    if not wanted:
        sys.exit("coverage_serve_shards: the test list is empty; refusing to write zero shards")

    by_module = collections.OrderedDict()
    for t in sorted(wanted):
        by_module.setdefault(t.split("::", 1)[0], []).append(t)

    shards, packing = [], []
    for mod, tests in sorted(by_module.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(tests) >= ALONE:
            shards.append(tests)
        elif packing and len(packing) + len(tests) > PACK:
            shards.append(packing)
            packing = list(tests)
        else:
            packing.extend(tests)
    if packing:
        shards.append(packing)

    flat = [t for s in shards for t in s]
    if len(flat) != len(set(flat)) or set(flat) != set(wanted):
        missing, extra = set(wanted) - set(flat), len(flat) - len(set(flat))
        sys.exit(f"coverage_serve_shards: partition is wrong ({len(missing)} missing, {extra} duplicated)")

    out = pathlib.Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("shard-*.txt"):
        old.unlink()
    for i, s in enumerate(shards):
        (out / f"shard-{i:02d}.txt").write_text("\n".join(s) + "\n")
    print(
        f"coverage_serve_shards: {len(wanted)} tests ({len(listed) - len(wanted)} skipped) "
        f"in {len(shards)} shards: " + ", ".join(str(len(s)) for s in shards)
    )


if __name__ == "__main__":
    main(sys.argv)
