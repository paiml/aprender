#!/usr/bin/env python3
"""decode_floor.py -- an rc is not published while it decodes slower than the previous release line.

WHY. #4273 (the decode fix) shipped in v0.69.5-rc.1 and never reached the 0.70.0 car. The car
decoded ~20x slower and nothing on the release path measured decode at all. PARITY-004 keeps
timing out of REQUIRED PR checks (scripts/check_no_timing_in_required.sh), so this floor runs
at release time instead: rc_fleet_stage.sh runs it on gx10 after the rc is installed there
and before the draft is flipped visible.

WHAT. On one host, under one gpu-q ticket, the rc's `apr bench <model> --json` and the
previous line's `apr bench` run INTERLEAVED (rc, prev, rc, prev, ...). The verdict compares
the median tokens_per_second of each side:

    pass        rc/prev >= release_gates.decode_floor.value (scripts/perf-matrix.yaml, PP-33)
    fail        rc/prev below the floor, or the rc did not take the CUDA path, or the rc's
                receipts name a commit that is not the tag's. Never waivable.
    unmeasured  no usable pair: the GPU stayed busy, the prev asset is missing, the prev side
                did not take the CUDA path, a run crashed. rc_fleet_stage maps this to
                `unreachable`, which only a dated waiver row `decode-floor` can cover.

A side that fell back to CPU is not a measurement of the thing being gated, so it is never
compared (CLAUDE.md verification rule 2: prove the mechanism engaged).

    decode_floor.py verdict --commit SHA < lines     # lines: `rc|prev<TAB><one-line bench JSON>`
    decode_floor.py prev-tag vX.Y.Z-rc.N < tags      # the tag to measure against, or exit 1
    decode_floor.py get model|n|max_tokens|gpu_wait  # a release_gates.decode_floor setting
    decode_floor.py --self-test
EXIT 0 ok · 1 no answer (prev-tag) · 2 usage/env
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
sys.path.insert(0, HERE)
import bench_receipt  # noqa: E402  (the one matrix reader, PP-33)
from carry_forward_gate import pick_prev_tag, tag_key  # noqa: E402


def setting(name):
    return bench_receipt.matrix_number("release_gates", "decode_floor", name)


def floor():
    return float(setting("floor")["value"])


def prev_tag(tags, tag):
    """The previous release LINE's newest tag -- the same line the carry-forward gate audits.
    Not merely the tag just below: an rc.2 measured against a regressed rc.1 would ratchet down."""
    k = tag_key(tag)
    if not k:
        return None
    return pick_prev_tag(tags, "%d.%d.%d" % k[:3])


def _side(runs, name):
    """(median tok/s, None) or (None, why)."""
    if not runs:
        return None, "%s: no runs" % name
    vals = []
    for r in runs:
        cls = (r.get("provenance") or {}).get("compute_class")
        if cls != "cuda":
            return None, "%s: compute_class %r, not cuda" % (name, cls)
        tps = r.get("tokens_per_second")
        if not isinstance(tps, (int, float)) or tps <= 0:
            return None, "%s: tokens_per_second %r" % (name, tps)
        vals.append(float(tps))
    return statistics.median(vals), None


def verdict(pairs, commit, fl):
    """pairs: list of (side, parsed JSON or None). Returns (state, detail)."""
    rc = [j for s, j in pairs if s == "rc"]
    prev = [j for s, j in pairs if s == "prev"]
    if any(j is None for _, j in pairs):
        return "unmeasured", "a bench run produced no JSON"
    for j in rc:
        bc = (j.get("provenance") or {}).get("build_commit", "")
        if not bc or bc == "unknown" or not commit.startswith(bc[:9]) or len(bc) < 7:
            return "fail", "rc receipt build_commit %r is not the tag's %s" % (bc, commit[:9])
    r_med, why = _side(rc, "rc")
    if why:
        # the rc not taking the CUDA path on a CUDA host is the rc's defect, not the harness's
        return ("fail" if rc else "unmeasured"), why
    p_med, why = _side(prev, "prev")
    if why:
        return "unmeasured", why
    ratio = r_med / p_med
    detail = "rc %.1f tok/s vs prev %.1f tok/s = %.3fx (floor %.2f, n=%d+%d)" % (
        r_med, p_med, ratio, fl, len(rc), len(prev))
    return ("pass" if ratio >= fl else "fail"), detail


def parse_lines(text):
    pairs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        side, _, body = line.partition("\t")
        if side not in ("rc", "prev"):
            continue
        try:
            pairs.append((side, json.loads(body)))
        except ValueError:
            pairs.append((side, None))
    return pairs


# ---- self-test ------------------------------------------------------------------------------
SHA = "1ac56f258" + "0" * 31


def _run(tps, cls="cuda", bc=SHA[:9]):
    return {"tokens_per_second": tps, "provenance": {"compute_class": cls, "build_commit": bc}}


CASES = [
    # (label, pairs, want state)
    ("equal speed passes", [("rc", _run(100)), ("prev", _run(100))] * 3, "pass"),
    ("20x decode regression fails (#4273)", [("rc", _run(5)), ("prev", _run(100))] * 3, "fail"),
    ("just under the floor fails", [("rc", _run(89)), ("prev", _run(100))] * 3, "fail"),
    ("at the floor passes", [("rc", _run(90)), ("prev", _run(100))] * 3, "pass"),
    ("median, not mean: one slow outlier rc run does not fail",
     [("rc", _run(100)), ("prev", _run(100)), ("rc", _run(10)), ("prev", _run(100)),
      ("rc", _run(101)), ("prev", _run(100))], "pass"),
    ("rc on CPU is a fail, never compared", [("rc", _run(500, "cpu")), ("prev", _run(100))], "fail"),
    ("prev on CPU is unmeasured (the rc would win on a lie)", [("rc", _run(100)), ("prev", _run(5, "cpu"))], "unmeasured"),
    ("rc receipt from another commit fails", [("rc", _run(100, bc="deadbeef1")), ("prev", _run(100))], "fail"),
    ("rc receipt with unknown commit fails", [("rc", _run(100, bc="unknown")), ("prev", _run(100))], "fail"),
    ("no runs at all is unmeasured", [], "unmeasured"),
    ("no prev runs is unmeasured", [("rc", _run(100))], "unmeasured"),
    ("a crashed run is unmeasured", [("rc", None), ("prev", _run(100))], "unmeasured"),
    ("zero tok/s rc fails", [("rc", _run(0)), ("prev", _run(100))], "fail"),
]

TAGS = ["v0.68.2", "v0.69.3", "v0.69.5-rc.1", "v0.69.3-rc.2", "v0.70.0-rc.1", "nightly"]
TAG_CASES = [
    ("0.70 rc measures against the newest 0.69 tag", "v0.70.0-rc.2", "v0.69.5-rc.1"),
    ("never the previous rc of its own line", "v0.70.0-rc.1", "v0.69.5-rc.1"),
    ("0.69 rc measures against 0.68", "v0.69.6-rc.1", "v0.68.2"),
    ("nothing older -> none", "v0.1.0-rc.1", None),
    ("a non-version tag -> none", "nightly", None),
]


def self_test():
    fail = 0
    fl = 0.90  # the fixtures' floor; the real one is read from the matrix below
    for label, pairs, want in CASES:
        got, detail = verdict(pairs, SHA, fl)
        ok = got == want
        fail |= not ok
        print("  %s %s -> %s%s" % ("ok  " if ok else "FAIL", label, got, "" if ok else " (want %s: %s)" % (want, detail)))
    for label, tag, want in TAG_CASES:
        got = prev_tag(TAGS, tag)
        ok = got == want
        fail |= not ok
        print("  %s %s -> %s%s" % ("ok  " if ok else "FAIL", label, got, "" if ok else " (want %s)" % want))
    try:
        real = floor()
        for k in ("model", "n", "max_tokens", "gpu_wait"):
            setting(k)
        ok = 0 < real <= 1.0
        print("  %s matrix release_gates.decode_floor present, floor=%s" % ("ok  " if ok else "FAIL", real))
        fail |= not ok
    except (KeyError, TypeError, ValueError) as exc:
        print("  FAIL matrix: %s" % exc)
        fail = 1
    # the parser keeps a crashed line as None instead of dropping it (a dropped crash reads as fewer runs)
    got = parse_lines("rc\t{\"a\":1}\nprev\tnot json\nnoise\n")
    ok = got == [("rc", {"a": 1}), ("prev", None)]
    fail |= not ok
    print("  %s a non-JSON bench line is kept as a crash, not dropped" % ("ok  " if ok else "FAIL"))
    fail |= _mutants()
    return 1 if fail else 0


MUTANTS = [
    # (label, old, new, the case that must turn red)
    ("the floor comparison is load-bearing", "\"pass\" if ratio >= fl else \"fail\"", "\"pass\"",
     "20x decode regression fails (#4273)"),
    ("the rc cuda check is load-bearing", "if cls != \"cuda\":", "if False:",
     "rc on CPU is a fail, never compared"),
    ("the build_commit check is load-bearing", "not commit.startswith(bc[:9])", "False",
     "rc receipt from another commit fails"),
    ("median, not mean", "statistics.median(vals)", "statistics.mean(vals)",
     "median, not mean: one slow outlier rc run does not fail"),
]


def _mutants():
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    body, cut, tail = src.partition("\nCASES = [")  # anchors are matched in the code, not in these tables
    fail = 0
    for label, old, new, case in MUTANTS:
        if body.count(old) != 1:
            print("  FAIL mutant '%s': anchor matches %d times" % (label, body.count(old)))
            fail = 1
            continue
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "decode_floor.py")
            with open(p, "w", encoding="utf-8") as h:
                h.write((body.replace(old, new) + cut + tail).replace("HERE = os.path.dirname(os.path.abspath(__file__))",
                                                      "HERE = %r" % HERE))
            code = ("import sys; sys.argv=['x']; import importlib.util as u; s=u.spec_from_file_location('m', %r); "
                    "m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "c=[c for c in m.CASES if c[0]==%r][0]; print(m.verdict(c[1], m.SHA, 0.90)[0]==c[2])" % (p, case))
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        ok = out == "False"
        fail |= not ok
        print("  %s mutant: %s (case '%s' %s)" % ("ok  " if ok else "FAIL", label, case,
                                                   "turns red" if ok else "STAYS GREEN: " + out))
    return fail


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    if argv[:1] == ["verdict"] and len(argv) == 3 and argv[1] == "--commit":
        state, detail = verdict(parse_lines(sys.stdin.read()), argv[2], floor())
        print("%s\t%s" % (state, detail))
        return 0
    if argv[:1] == ["prev-tag"] and len(argv) == 2:
        t = prev_tag(sys.stdin.read().split(), argv[1])
        if not t:
            return 1
        print(t)
        return 0
    if argv[:1] == ["get"] and len(argv) == 2:
        v = setting(argv[1])
        print(v["value"] if isinstance(v, dict) else v)
        return 0
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
