#!/usr/bin/env bash
# check_pmat_pinned.sh - every execution surface resolves the analyser through
# scripts/pmat_bin.sh, never PATH (PMAT-1059, DAG row G-10, #2999).
#
# WHY. scripts/hardcoded_path_shipped_baseline.txt held "277", measured by an
# instrument nobody named. The fleet's forjar pin moved 3.31.0 -> 3.37.0
# (paiml/infra machines/intel/forjar.yaml, PMAT-231) and the same tree counted
# 317: every PR went red for a defect no PR introduced. A gate's number is a
# property of (tree, instrument); the instrument is pinned in ONE place,
# scripts/pmat_bin.sh, and every caller takes "$PMAT" from it.
#
# THE ASSERTION (operator, 2026-09-06, verbatim):
#     grep -rEn '(^|[^_/])pmat ' scripts/ .github/workflows/ | grep -v pmat_bin   == 0 lines
# It counts prose too: a comment that says "run the analyser" in the old spelling
# is the line a reader copies. Lines that name pmat_bin are the resolver's own.
#
# SHRINK-ONLY (G-10b, PMAT-1063, #3013): the count is ratcheted against
# scripts/pmat_unpinned_baseline.txt (kind `count` in check_baseline_ratchets.sh),
# measured — never typed — by this guard's own scan at the commit named in the
# file. The sweep to 0 is G-10c (PMAT-1064). A count above the baseline is RED
# naming every line; below it is an improvement to record with --update; a
# missing baseline is ENV (exit 2), never a pass.
#
#   bash scripts/check_pmat_pinned.sh              # 0 count <= baseline . 1 grew (listed) . 2 no baseline
#   bash scripts/check_pmat_pinned.sh --update     # record an improvement (count < baseline)
#   bash scripts/check_pmat_pinned.sh --self-test  # case table, the resolver's three polarities, the ratchet's both polarities
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_pmat_pinned
RE='(^|[^_/])pmat '   # the only sanctioned spelling is "$PMAT", resolved by pmat_bin.sh
SCAN_ROOT="${PIN_SCAN_ROOT:-$ROOT}"                      # PIN_SCAN_ROOT: a fixture tree, for --self-test
BASELINE="${PIN_BASELINE:-$ROOT/scripts/pmat_unpinned_baseline.txt}"

scan() { # scan <dir>... -> offending lines (file:line:text)
    ( cd "$SCAN_ROOT" && grep -rEn "$RE" "$@" 2>/dev/null | grep -v 'pmat_bin' ) || true
}
baseline_count() { # -> the number on the baseline's first non-comment line, or "" when the file is missing/unparseable
    [ -f "$BASELINE" ] || return 0
    sed -nE 's/^[[:space:]]*([0-9]+)[[:space:]]*(#.*)?$/\1/p' "$BASELINE" | head -n1
}

if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/pmatpin.XXXXXX"); trap 'rm -rf "${TD:?}"' EXIT
    n=0; red=0
    row() { local want=$1 label=$2 line=$3 got; n=$((n + 1))
        if printf '%s\n' "$line" | grep -Eq "$RE" && ! printf '%s\n' "$line" | grep -q 'pmat_bin'; then got=match; else got=clean; fi
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s %-5s %s\n' "$n" "$got" "$label"; else printf 'FAIL  row %-2s %s (wanted %s) %s\n' "$n" "$got" "$want" "$label"; red=1; fi; }
    p='pmat'   # assembled so this file does not carry the unpinned spelling itself
    row match "bare invocation in command position"        "    $p comply check --failures-only"
    row match "workflow run: line"                         "        run: $p analyze complexity --format json"
    row match "inside a substitution"                      "VER=\$($p --version | head -1)"
    row match "a comment that teaches the old spelling"    "# run $p analyze satd before pushing"
    row match "after a pipe"                               "cat x | $p query init"
    row clean "the sanctioned spelling"                    "    \"\$PMAT\" comply check --failures-only"
    row clean "the resolver named on the line"             "    . scripts/${p}_bin.sh || exit 1"
    row clean "a path segment, not a command"              "    cargo run -p ${p}-agent-mail"
    row clean "an identifier with an underscore"           "    ${p}_ver=\$(\"\$PMAT\" --version)"
    row clean "install line with the crate name LAST"      "    cargo install --version \"\$PMAT_PIN\" --locked $p"
    row clean "possessive prose"                           "# ${p}'s hardcoded-paths analysis is the detector"
    # the resolver's three polarities (scripts/pmat_bin.sh)
    fake() { printf '#!/usr/bin/env bash\ncase "$1" in --version) echo "%s %s"; exit 0;; esac\necho ok\n' "$p" "$1" > "$2"; chmod +x "$2"; }
    pin=$(sed -nE 's/^PMAT_PIN="([0-9.]+)"$/\1/p' "$ROOT/scripts/${p}_bin.sh" | head -1)
    fake "$pin" "$TD/at-pin"; fake "3.0.0" "$TD/off-pin"
    n=$((n + 1)); if out=$(PMAT_BIN_OVERRIDE="$TD/at-pin" bash -c ". '$ROOT/scripts/${p}_bin.sh' && printf '%s' \"\$PMAT\"") && [ "$out" = "$TD/at-pin" ]; then printf 'ok    row %-2s resolver: a binary at the pin (%s) resolves to it\n' "$n" "$pin"; else printf 'FAIL  row %-2s resolver did not resolve the pinned binary: %s\n' "$n" "$out"; red=1; fi
    n=$((n + 1)); if PMAT_BIN_OVERRIDE="$TD/off-pin" PMAT_BIN_NO_FALLBACK=1 bash -c ". '$ROOT/scripts/${p}_bin.sh'" >/dev/null 2>&1; then printf 'FAIL  row %-2s resolver accepted a binary off the pin\n' "$n"; red=1; else printf 'ok    row %-2s resolver refuses a binary off the pin (3.0.0 != %s)\n' "$n" "$pin"; fi
    n=$((n + 1)); if PMAT_BIN_OVERRIDE="$TD/absent" PMAT_BIN_NO_FALLBACK=1 bash -c ". '$ROOT/scripts/${p}_bin.sh'" >/dev/null 2>&1; then printf 'FAIL  row %-2s resolver passed with no binary at all\n' "$n"; red=1; else printf 'ok    row %-2s resolver refuses when no binary exists (ENV, not PASS)\n' "$n"; fi
    # the resolver sets no shell options in the caller (sourced-lib rule)
    n=$((n + 1)); if [ "$(PMAT_BIN_OVERRIDE="$TD/at-pin" bash -c "set +e; . '$ROOT/scripts/${p}_bin.sh'; set -o | grep -E '^(errexit|nounset)' | grep -c on")" = 0 ]; then printf 'ok    row %-2s resolver is option-neutral\n' "$n"; else printf 'FAIL  row %-2s resolver leaked shell options into the caller\n' "$n"; red=1; fi
    # the ratchet's both polarities on a fixture tree (PIN_SCAN_ROOT / PIN_BASELINE)
    F="$TD/tree"; mkdir -p "$F/scripts" "$F/.github/workflows"
    printf '#!/usr/bin/env bash\n%s analyze satd\n"$PMAT" query x\n' "$p" > "$F/scripts/a.sh"
    printf 'jobs:\n  x:\n    steps:\n      - run: %s comply check\n' "$p" > "$F/.github/workflows/w.yml"
    ratchet_row() { local want=$1 label=$2 bl=$3 rc=0 out; n=$((n + 1))
        if [ "$bl" = MISSING ]; then rm -f "$TD/bl.txt"; else printf '%s\n' "$bl" > "$TD/bl.txt"; fi
        out=$(PIN_SCAN_ROOT="$F" PIN_BASELINE="$TD/bl.txt" bash "${BASH_SOURCE[0]}" 2>&1) || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"; else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; printf '%s\n' "$out" | tail -4 | sed 's|^|        |'; red=1; fi; }
    ratchet_row 0 "ratchet: 2 unpinned lines, baseline 2: PASS"                       "2   # fixture"
    ratchet_row 1 "ratchet: 2 unpinned lines, baseline 1: RED naming the lines (the registered mutation)" "1"
    ratchet_row 0 "ratchet: 2 unpinned lines, baseline 3: PASS and an improvement to record" "3"
    ratchet_row 2 "ratchet: no baseline file: ENV (exit 2), never a pass"             MISSING
    ratchet_row 2 "ratchet: an unparseable baseline (INVALID): ENV, never a number"    "INVALID"
    printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi

hits=$(scan scripts/ .github/workflows/)
count=$(printf '%s' "$hits" | grep -c . || true)
if [ "${1:-}" = "--update" ]; then
    printf '%s   # unpinned analyser references under scripts/ and .github/workflows/; basis: bash scripts/check_pmat_pinned.sh (grep -rEn "(^|[^_/])pmat " scripts/ .github/workflows/ | grep -v pmat_bin | wc -l) at %s; shrink-only (check_baseline_ratchets.sh kind count); G-10b PMAT-1063, sweep to 0 is G-10c\n' "$count" "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)" > "$BASELINE"
    printf 'baseline set to %s\n' "$count"; exit 0
fi
bl=$(baseline_count)
if [ -z "$bl" ]; then
    printf '%s: ENV - no parseable count in %s (a missing or INVALID baseline is not a number and never a pass)\n' "$PROG" "${BASELINE#"$ROOT"/}" >&2; exit 2
fi
if [ "$count" -gt "$bl" ]; then
    printf 'FAIL  %s: unpinned=%s baseline=%s — %s new line(s) invoke or teach the unpinned analyser spelling (the resolver is scripts/pmat_bin.sh):\n' "$PROG" "$count" "$bl" "$((count - bl))"
    printf '%s\n' "$hits" | head -60 | sed 's|^|  |'
    [ "$count" -le 60 ] || printf '  ... and %s more\n' "$((count - 60))"
    exit 1
fi
if [ "$count" -lt "$bl" ]; then printf 'Improved: %s -> %s. Run --update to record it.\n' "$bl" "$count"; fi
if [ "$count" -eq 0 ]; then printf 'PASS  %s: unpinned=0 — every analyser reference under scripts/ and .github/workflows/ is resolved by scripts/pmat_bin.sh\n' "$PROG"; else printf 'PASS  %s: unpinned=%s baseline=%s (shrink-only; the sweep to 0 is G-10c)\n' "$PROG" "$count" "$bl"; fi
exit 0
