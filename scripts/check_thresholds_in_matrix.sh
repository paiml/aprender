#!/usr/bin/env bash
# check_thresholds_in_matrix.sh — PP-33. No gate reader may carry a threshold.
#
# WHY THIS EXISTS
# ---------------
# perf-matrix.yaml is declared to be the ONE place a number a gate compares
# against lives, and nothing checked it. Measured on 2026-09-02 the tree carried
# FOUR independent encodings of the same release thresholds:
#
#   scripts/lib/parity_block.py   FLOOR = 0.80, STRETCH = 1.50, CEILING = 1.50
#   scripts/lib/bench_receipt.py  lane.get("floor", 0.80), lane.get("ceiling", 1.50)
#   scripts/perf_gate.sh          `if de>=1.0 and ag<b1`, `if util<0.5`
#   scripts/perf-matrix.yaml      arms.B1.floor, arms.B2.floor
#
# STRETCH and CEILING were the same number and STRETCH gated nothing; the two
# bench_receipt defaults applied whenever a lane omitted its floor, so a lane
# could be scored against a literal no matrix edit could reach. The spec named
# `STRETCH = 1.50` as PP-33's must-fire and nothing in the tree could turn red
# on it -- marking that rule anything but OPEN was theater.
#
# WHAT IS SCANNED, AND WHAT IS NOT
# --------------------------------
#   * a FLOAT literal adjacent to a comparison operator   (`< 0.5`, `>= 1.00`)
#   * a module-level CONSTANT bound to a float            (`STRETCH = 1.50`)
#
# and nothing else. A definitional comparison against an INTEGER -- `timeouts !=
# 0`, `c == 1`, `n < 2`, `days < 0` -- is not a threshold and is not matched: a
# rule that flagged those would be hand-exempted within a week, and a guard full
# of exemptions is a guard nobody reads.
#
# WHOLE-LINE COMMENTS ARE DROPPED; trailing ones are NOT. That direction is
# deliberate: a float comparison quoted in a trailing comment is a loud false
# alarm, while stripping from the first `#` would silently drop a real one that
# happens to sit after a `#` inside a string.
#
# THE ALLOWLIST is three entries and each says why. It is keyed to file + exact
# text, so an allowlisted line that CHANGES stops being allowlisted.
#
#   bash scripts/check_thresholds_in_matrix.sh              # check
#   bash scripts/check_thresholds_in_matrix.sh --selftest   # case table
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_check() { # run_check <root>
  python3 - "$1" <<'PY'
import os, re, sys

root = sys.argv[1]
TARGETS = (
    os.path.join("scripts", "perf_gate.sh"),
    os.path.join("scripts", "lib", "parity_block.py"),
    os.path.join("scripts", "lib", "perf_receipt.py"),
    os.path.join("scripts", "lib", "bench_receipt.py"),
)

# A float beside a comparison operator, either order.
CMP = re.compile(r"(?:[<>]=?|[=!]=)\s*-?\d+\.\d+"
                 r"|(?<![\w.])-?\d+\.\d+\s*(?:[<>]=?|[=!]=)")
# A module-level constant bound to a float literal: the parity_block.py shape.
CONST = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*-?\d+\.\d+\s*$")

# ALLOWLIST -- file, exact stripped text, and the reason it is not a threshold.
# Every entry is a value used to BUILD a fixture or to name a stable identity,
# never one a verdict is compared against.
ALLOWLIST = {
    (os.path.join("scripts", "lib", "perf_receipt.py"), "FIXTURE_RATIO = 1.05"):
        "a synthetic ratio used to construct the P2 selftest fixture. Nothing "
        "is compared against it; it is chosen to sit inside the bounds the "
        "matrix declares so the fixture's own verdict is PASS.",
}

findings = []
allow_used = set()
for rel in TARGETS:
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        findings.append((rel, 0, "the file this guard scans is absent -- the "
                                 "scan is broken, not clean"))
        continue
    with open(path, encoding="utf-8") as fh:
        for number, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            key = (rel, stripped)
            if key in ALLOWLIST:
                allow_used.add(key)
                continue
            if CMP.search(line):
                findings.append((rel, number, "compares against the float "
                                              "literal in `%s`" % stripped[:96]))
            elif CONST.match(stripped):
                findings.append((rel, number, "binds a module-level float "
                                              "constant `%s`" % stripped[:96]))

# A stale allowlist entry makes the list look more considered than it is, and is
# the same rotting-claim shape the schema map refuses one file over.
for key in sorted(set(ALLOWLIST) - allow_used):
    findings.append((key[0], 0, "is allowlisted for the text %r, which is no "
                                "longer in the file" % key[1]))

if findings:
    print("FAIL  a gate reader carries a threshold (PP-33)")
    for rel, number, why in findings:
        print("      %s:%s %s" % (rel, number or "-", why))
    print("      Every number a gate compares against lives in "
          "scripts/perf-matrix.yaml with a threshold_class and an author.")
    sys.exit(1)
print("ok    %d reader(s) scanned, no threshold literal outside "
      "scripts/perf-matrix.yaml" % len(TARGETS))
PY
}

# ------------------------------------------------------------- case table ---
selftest() {
  local pass=0 fail=0 td
  td="$(mktemp -d)" || return 2
  case "$td" in
    /tmp/*|/var/folders/*) : ;;
    *) printf 'refusing to rm -rf %s\n' "${td:-<empty>}" >&2; return 2 ;;
  esac
  trap 'rm -rf "${td:?}"' RETURN

  _fixture() { # name -> a copy of the four scanned readers at $td/<name>
    rm -rf "${td:?}/$1"
    mkdir -p "$td/$1/scripts/lib"
    cp "$ROOT/scripts/perf_gate.sh" "$td/$1/scripts/"
    cp "$ROOT/scripts/lib/parity_block.py" "$td/$1/scripts/lib/"
    cp "$ROOT/scripts/lib/perf_receipt.py" "$td/$1/scripts/lib/"
    cp "$ROOT/scripts/lib/bench_receipt.py" "$td/$1/scripts/lib/"
  }

  _expect() { # name, root, expected(pass|fail)
    local got
    if run_check "$2" >/dev/null 2>&1; then got=pass; else got=fail; fi
    if [ "$got" = "$3" ]; then
      printf '  ok    %-38s expect=%s\n' "$1" "$3"; pass=$((pass + 1))
    else
      printf '  BROKE %-38s expected %s got %s\n' "$1" "$3" "$got"; fail=$((fail + 1))
    fi
  }

  # THE TREE AS IT STANDS. If this is not green the rest proves nothing.
  _expect threshold_in_matrix "$ROOT" pass

  # THE SPEC'S OWN MUTATION, verbatim: the constant that shipped in
  # parity_block.py for the life of the P2 chain.
  _fixture threshold_outside_matrix
  printf 'STRETCH = 1.50\n' >> "$td/threshold_outside_matrix/scripts/lib/parity_block.py"
  _expect threshold_outside_matrix "$td/threshold_outside_matrix" fail

  # ... and the other half of the same defect: not a constant, a COMPARISON.
  # Deleting STRETCH alone would satisfy the named mutation while leaving the
  # live threshold in place, so the comparison site has its own row.
  _fixture threshold_comparison_literal
  printf 'if 1 < 0.80:\n    pass\n' >> "$td/threshold_comparison_literal/scripts/lib/bench_receipt.py"
  _expect threshold_comparison_literal "$td/threshold_comparison_literal" fail

  # DISCRIMINATION -- an INTEGER comparison is definitional, not a threshold.
  # Without this row the guard could be keying on any comparison at all, and the
  # first `if timeouts != 0` would be hand-exempted.
  _fixture integer_comparison_is_definitional
  printf 'if 1 != 0:\n    pass\n' >> "$td/integer_comparison_is_definitional/scripts/lib/bench_receipt.py"
  _expect integer_comparison_is_definitional "$td/integer_comparison_is_definitional" pass

  # DISCRIMINATION -- a float in a WHOLE-LINE comment is documentation. The
  # headers of all four readers quote the numbers they used to carry.
  _fixture float_in_a_comment_is_prose
  printf '# the old floor was < 0.80 and it lived here\n' \
    >> "$td/float_in_a_comment_is_prose/scripts/lib/bench_receipt.py"
  _expect float_in_a_comment_is_prose "$td/float_in_a_comment_is_prose" pass

  # A SCAN THAT FOUND NOTHING BECAUSE THERE WAS NOTHING TO SCAN is not a pass.
  _fixture scanned_file_absent
  rm -f "$td/scanned_file_absent/scripts/lib/parity_block.py"
  _expect scanned_file_absent "$td/scanned_file_absent" fail

  printf '  %d passed, %d broken\n' "$pass" "$fail"
  [ "$fail" = 0 ]
}

case "${1:-}" in
  --selftest|--self-test) selftest ;;
  "") run_check "$ROOT" ;;
  *) printf 'usage: %s [--selftest]\n' "$(basename "$0")" >&2; exit 2 ;;
esac
