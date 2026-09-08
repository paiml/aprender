#!/usr/bin/env bash
# baseline_tool_version_test.sh — falsifier for
# lib_baseline_ratchet.sh's baseline_require_tool_version (BSE-10a second
# half, PMAT-1066).
#
# A ratchet compares (tree, instrument). check_hardcoded_paths.sh /
# PMAT-1059 measured this the hard way: 277 vs 317 on an UNCHANGED tree, once
# the fleet moved pmat 3.31.0 -> 3.37.0 -- two verdicts from two instruments,
# read as "the tree grew". baseline_require_tool_version generalises that
# discipline: every ratchet baseline carries a leading
#
#     # tool_version=<tool> <version>
#
# header (or the literal `none` with a reason, for a baseline produced by
# grep/git/cargo-metadata alone). Rows:
#
#   (a) every baseline scripts/check_baseline_ratchets.sh classifies as
#       `ratchet` (its own coverage backstop -- the true universe, not a
#       grep of caller scripts) carries the header, except
#       hardcoded_path_shipped_baseline.txt, which already carries a richer,
#       three-state (count/pmat_version/basis) instrument stamp of its own
#       (PMAT-1059) that this ticket does not supersede -- see
#       docs/roadmaps/roadmap.yaml's PMAT-1066 note. 0 files is a FAIL: an
#       enumeration that finds nothing looks like a pass for the wrong
#       reason.
#   (b) a fixture baseline recorded under `pmat 0.0.1`, a mock `pmat` on PATH
#       answering `pmat 0.0.2` -> exits 4, exactly one line.
#   (c) same fixture, mock answers `pmat 0.0.1` -> proceeds (rc 0).
#   (d) tool=none -> no version probe, proceeds (rc 0) even with no such
#       binary on PATH.
#   (e) header absent -> exits 1.
#   (f) MUTANT: the compare arm (TOOLVER-CMP-BEGIN/END markers) deleted ->
#       the mutant must go GREEN (rc 0) on the exact input row (b) went RED
#       on. If it does not, this falsifier does not discriminate.
#
#   bash scripts/tests/baseline_tool_version_test.sh
#
# Refs: PMAT-1066, BSE-10a, PMAT-1059.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$ROOT/scripts/lib_baseline_ratchet.sh"
BACKSTOP="$ROOT/scripts/check_baseline_ratchets.sh"

if [ ! -f "$LIB" ]; then
    printf 'ENV - %s is missing (the box cannot answer)\n' "$LIB" >&2
    exit 2
fi

# shellcheck source=scripts/lib_baseline_ratchet.sh
. "$LIB" || { printf 'ENV - could not source %s\n' "$LIB" >&2; exit 2; }

n=0
red=0

row() { # row WANT_RC LABEL MUST_MATCH -- CMD...
    local want=$1 label=$2 pat=$3 rc=0 out
    shift 3
    n=$((n + 1))
    out=$("$@" 2>&1); rc=$?
    if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -qE -- "$pat"; then
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    else
        printf 'FAIL  row %-2s rc=%s (wanted %s, must match /%s/)  %s\n' "$n" "$rc" "$want" "$pat" "$label"
        printf '%s\n' "$out" | sed 's/^/        /'
        red=1
    fi
}

BASH_BIN=$(command -v bash) || { printf 'ENV - no bash on PATH\n' >&2; exit 2; }

TD=$(mktemp -d "${TMPDIR:-/tmp}/bl-toolver-test.XXXXXX") || exit 2
cleanup() {
    local victim=${TD:-}
    case "$victim" in
        *bl-toolver-test.*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
        *) return 0 ;;
    esac
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# (a) Universe: ask the coverage backstop, not a grep of caller scripts --
# it is the file that already resolves "every baseline the lib reads"
# (find UNION git ls-files, classify(), HARD FAIL on the unclassified). Its
# own PASS line prints "N ratcheted"; we re-derive the file LIST from its
# per-row `ok    ratchet  scripts/<file>` lines rather than trusting the
# count alone.
# ---------------------------------------------------------------------------
EXEMPT_FROM_HEADER="scripts/hardcoded_path_shipped_baseline.txt"

n=$((n + 1))
if [ ! -f "$BACKSTOP" ]; then
    printf 'FAIL  row %-2s  %s is missing\n' "$n" "$BACKSTOP"
    red=1
else
    backstop_out=$(bash "$BACKSTOP" 2>&1)
    backstop_rc=$?
    universe=$(printf '%s\n' "$backstop_out" | sed -nE 's/^ok +ratchet +(scripts\/[A-Za-z0-9_./-]+\.txt) .*/\1/p' | LC_ALL=C sort -u)
    ucount=$(printf '%s\n' "$universe" | grep -c . || true)
    if [ "$backstop_rc" -ne 0 ] || [ "$ucount" -eq 0 ]; then
        printf 'FAIL  row %-2s  %s enumerated %s ratcheted baseline(s) (rc=%s) -- 0 is a FAIL, not a vacuous pass\n' \
            "$n" "$BACKSTOP" "$ucount" "$backstop_rc"
        red=1
    else
        missing=""
        checked=0
        while IFS= read -r f; do
            [ -n "$f" ] || continue
            [ "$f" != "$EXEMPT_FROM_HEADER" ] || continue
            checked=$((checked + 1))
            if ! grep -qE '^#[[:space:]]*tool_version=' "$ROOT/$f" 2>/dev/null; then
                missing="${missing}${f}\n"
            fi
        done <<< "$universe"
        if [ -n "$missing" ]; then
            printf 'FAIL  row %-2s  %s ratcheted baseline(s) checked, missing the header on:\n' "$n" "$checked"
            printf "$missing" | sed 's/^/        /'
            red=1
        else
            printf 'ok    row %-2s rc=0  %s ratcheted baseline(s) (of %s total) all carry "# tool_version=" (1 exempt: %s, PMAT-1059)\n' \
                "$n" "$checked" "$ucount" "$EXEMPT_FROM_HEADER"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Fixtures for (b)-(f): a mock `pmat` whose reported version is an argument,
# never a hardcoded string, so the SAME mock drives both the match and the
# mismatch row.
# ---------------------------------------------------------------------------
mkmock() { # mkmock <dir> <reported-version>
    mkdir -p "$1"
    cat > "$1/pmat" <<EOF
#!/bin/sh
printf 'pmat %s\ncommit: mock\nworktree: mock\n' '$2'
EOF
    chmod +x "$1/pmat"
}

printf '# tool_version=pmat 0.0.1\n1\n' > "$TD/versioned.txt"
printf '1\n' > "$TD/noheader.txt"
printf '# tool_version=none (measured by grep; not a versioned analyser)\n1\n' > "$TD/none.txt"

# (b) mismatch -> rc 4, exactly one line.
mkmock "$TD/mock-mismatch" 0.0.2
b_out=$(PATH="$TD/mock-mismatch:$PATH" baseline_require_tool_version "$TD/versioned.txt" 2>&1)
b_rc=$?
b_lines=$(printf '%s\n' "$b_out" | wc -l | tr -d ' ')
n=$((n + 1))
if [ "$b_rc" = 4 ] && [ "$b_lines" = 1 ] && printf '%s' "$b_out" | grep -qE 'recorded under pmat 0\.0\.1.*runner has pmat 0\.0\.2'; then
    printf 'ok    row %-2s rc=4  mismatch (recorded 0.0.1, runner 0.0.2): exactly one line\n' "$n"
else
    printf 'FAIL  row %-2s rc=%s lines=%s  mismatch did not exit 4 with exactly one line\n' "$n" "$b_rc" "$b_lines"
    printf '%s\n' "$b_out" | sed 's/^/        /'
    red=1
fi

# (c) match -> rc 0.
mkmock "$TD/mock-match" 0.0.1
row 0 "match (recorded 0.0.1, runner 0.0.1): proceeds" '^$|.*' \
    env PATH="$TD/mock-match:$PATH" "$BASH_BIN" -c ". '$LIB' && baseline_require_tool_version '$TD/versioned.txt'"

# (d) tool=none -> rc 0, no probe (no such binary anywhere on PATH).
row 0 "tool=none: no version probe, proceeds" '^$|.*' \
    env PATH="/nonexistent:/usr/bin:/bin" "$BASH_BIN" -c ". '$LIB' && baseline_require_tool_version '$TD/none.txt'"

# (e) header absent -> rc 1.
row 1 "header absent: FAIL" 'no "# tool_version=' \
    "$BASH_BIN" -c ". '$LIB' && baseline_require_tool_version '$TD/noheader.txt'"

# (f) MUTANT: delete the compare arm and re-run (b)'s exact input. The
# mutant must go GREEN (rc 0) where the real function went RED (rc 4); if it
# does not, this falsifier proves nothing (capability-gated-tests-are-vacuous
# lesson).
MUTANT="$TD/lib_baseline_ratchet.mutant.sh"
sed '/# --- TOOLVER-CMP-BEGIN ---/,/# --- TOOLVER-CMP-END ---/d' "$LIB" > "$MUTANT"
n=$((n + 1))
if diff -q "$LIB" "$MUTANT" > /dev/null 2>&1; then
    printf 'FAIL  row %-2s  sed found no TOOLVER-CMP-BEGIN/END markers to delete -- mutant is byte-identical\n' "$n"
    red=1
else
    mut_out=$(PATH="$TD/mock-mismatch:$PATH" bash -c ". '$MUTANT' && baseline_require_tool_version '$TD/versioned.txt'" 2>&1)
    mut_rc=$?
    if [ "$mut_rc" = 0 ]; then
        printf 'ok    row %-2s rc=0  mutant (compare arm deleted) is GREEN on row (b)'"'"'s mismatch input -- falsifier discriminates\n' "$n"
    else
        printf 'FAIL  row %-2s rc=%s  mutant should have gone GREEN (rc=0) but did not -- the fixture proves nothing\n' "$n" "$mut_rc"
        printf '%s\n' "$mut_out" | sed 's/^/        /'
        red=1
    fi
fi

printf '%s/%s checks, %s failed\n' "$((n - red))" "$n" "$red"
[ "$red" = 0 ]
