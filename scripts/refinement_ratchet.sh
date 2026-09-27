#!/usr/bin/env bash
# refinement_ratchet.sh — ONT-3b (#4073): the ONLY writer of formalization.yaml's `unrefined_baseline`.
#
# `pv lint --gate refinement` reads `unrefined_baseline` and never writes it. It is RED when the
# measured unrefined count differs from the baseline in either direction. Before this script existed,
# the number could only move by hand edit, which R-6 forbids. This script is `make refinement-ratchet`,
# never in CI:
#
#   measured <  recorded  -> lowered to the measured count
#   measured == recorded  -> untouched (byte-identical)
#   measured >  recorded  -> REFUSED, exit 1. A ratchet that rises on request is not a ratchet.
#   not recorded          -> recorded (the first baseline)
#
# The one exception to the refused rise is a RE-MEASUREMENT: the recorded number was taken over a stale
# discharge summary, so it never measured the tree. That is a rule interpretation, not a mechanic, so the
# script does not decide it. `--remeasure RECEIPT` raises the number only when RECEIPT is an existing
# file under evidence/ (the quorum verdict that ruled it a re-measurement). The receipt path and both
# numbers go into a comment directly above the key, so the file names what moved it.
#
#   bash scripts/refinement_ratchet.sh                         # measure with the pinned pv, rewrite downward
#   bash scripts/refinement_ratchet.sh --remeasure evidence/…  # a quorum-ruled re-measurement may rise
#   bash scripts/refinement_ratchet.sh --self-test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMALIZATION="${FORMALIZATION:-$REPO_ROOT/crates/aprender-contracts-staging/lean/formalization.yaml}"
KEY="unrefined_baseline"
NOTE="# unrefined_baseline re-measured"

# key_value FILE -> the recorded integer, or nothing when the key is absent. A top-level key only.
key_value() {
    { grep -E "^$KEY:[[:space:]]*[0-9]+[[:space:]]*$" "$1" || true; } | head -1 \
        | sed -E 's/^[^:]*:[[:space:]]*//; s/[[:space:]]*$//'
}

# set_value FILE N [NOTE_LINE]: replace the key's line in place (or append it), and when NOTE_LINE is
# given put it directly above the key, replacing an earlier re-measure note there.
set_value() {
    local file="$1" n="$2" note="${3:-}" tmp
    tmp="$(mktemp "$file.XXXXXX")"
    if grep -qE "^$KEY:" "$file"; then
        awk -v k="$KEY" -v n="$n" -v note="$note" -v pfx="$NOTE" '
            { lines[NR] = $0 }
            END {
                for (i = 1; i <= NR; i++) {
                    if (index(lines[i], k ":") == 1) {
                        if (note != "") print note
                        print k ": " n
                        continue
                    }
                    if (note != "" && index(lines[i], pfx) == 1 && i < NR && index(lines[i + 1], k ":") == 1) continue
                    print lines[i]
                }
            }' "$file" > "$tmp"
    else
        cat "$file" > "$tmp"
        [ -n "$note" ] && printf '%s\n' "$note" >> "$tmp"
        printf '%s: %s\n' "$KEY" "$n" >> "$tmp"
    fi
    mv "$tmp" "$file"
}

# ratchet FILE MEASURED [RECEIPT] -> 0 ok (maybe rewritten), 1 refused rise, 2 bad receipt
ratchet() {
    local file="$1" now="$2" receipt="${3:-}" was
    was="$(key_value "$file")"
    if [ -z "$was" ]; then
        set_value "$file" "$now"
        printf '  recorded  %s %s\n' "$KEY" "$now"
    elif [ "$now" -lt "$was" ]; then
        set_value "$file" "$now"
        printf '  lowered   %s %s -> %s\n' "$KEY" "$was" "$now"
    elif [ "$now" -eq "$was" ]; then
        printf '  unchanged %s %s\n' "$KEY" "$now"
    elif [ -z "$receipt" ]; then
        printf '  REFUSED   %s %s -> %s: the ratchet only turns down (a quorum-ruled re-measurement passes --remeasure)\n' "$KEY" "$was" "$now"
        return 1
    else
        case "$receipt" in
            evidence/*) : ;;
            *) printf 'NO-GO: --remeasure %s is not under evidence/\n' "$receipt" >&2; return 2 ;;
        esac
        [ -f "$REPO_ROOT/$receipt" ] || { printf 'NO-GO: --remeasure %s does not exist\n' "$receipt" >&2; return 2; }
        set_value "$file" "$now" "$NOTE $was -> $now: $receipt (make refinement-ratchet)"
        printf '  RE-MEASURED %s %s -> %s per %s\n' "$KEY" "$was" "$now" "$receipt"
    fi
}

# measure -> the unrefined count the gate reports. A decline prints no count, and that is a refusal
# here, not a zero.
measure() {
    local out rc n
    set +e
    out="$("$PV" lint "$REPO_ROOT/contracts" --gate refinement 2>/dev/null)"
    rc=$?
    set -e
    n="$(printf '%s' "$out" | jq -r '.unrefined // empty' 2>/dev/null || true)"
    case "$n" in
        ''|*[!0-9]*)
            printf 'NO-GO: `pv lint --gate refinement` (exit %s) reported no unrefined count; nothing was measured\n' "$rc" >&2
            return 2 ;;
    esac
    printf '%s\n' "$n"
}

self_test() {
    local t pass=0 fail=0 rc
    t="$(mktemp -d)"
    case "$t" in /tmp/*|/var/tmp/*) : ;; *) printf 'NO-GO: odd mktemp path %s\n' "$t" >&2; return 2 ;; esac
    row() { # row NAME GOT WANT
        if [ "$2" = "$3" ]; then pass=$((pass+1)); printf '  ok    %-56s %s\n' "$1" "$3"
        else fail=$((fail+1)); printf '  FAIL  %-56s want=%s got=%s\n' "$1" "$3" "$2"; fi
    }
    printf 'refinement_ratchet self-test\n'
    printf 'models:\n  - module: A.lean\n    unrefined_baseline: 9\n# keep me\nunrefined_baseline: 76\n' > "$t/f.yaml"
    row "the top-level key is read, not the nested one"          "$(key_value "$t/f.yaml")" 76
    ratchet "$t/f.yaml" 70 >/dev/null
    row "a fall is written"                                      "$(key_value "$t/f.yaml")" 70
    row "the nested key is untouched"                            "$(grep -c '^    unrefined_baseline: 9$' "$t/f.yaml")" 1
    set +e; ratchet "$t/f.yaml" 131 >/dev/null; rc=$?; set -e
    row "a rise without a receipt is refused"                    "$rc" 1
    row "a refused rise leaves the number"                       "$(key_value "$t/f.yaml")" 70
    cp "$t/f.yaml" "$t/g.yaml"; ratchet "$t/g.yaml" 70 >/dev/null
    row "an equal count leaves the file byte-identical"          "$(md5sum < "$t/g.yaml" | cut -c1-8)" "$(md5sum < "$t/f.yaml" | cut -c1-8)"
    local root="$REPO_ROOT"
    REPO_ROOT="$t"; mkdir -p "$t/evidence/q"; : > "$t/evidence/q/verdict.json"
    set +e; ratchet "$t/f.yaml" 131 notes/verdict.json >/dev/null 2>&1; rc=$?; set -e
    row "a receipt outside evidence/ is refused"                 "$rc" 2
    set +e; ratchet "$t/f.yaml" 131 evidence/q/missing.json >/dev/null 2>&1; rc=$?; set -e
    row "a receipt that does not exist is refused"               "$rc" 2
    row "a refused receipt leaves the number"                    "$(key_value "$t/f.yaml")" 70
    ratchet "$t/f.yaml" 131 evidence/q/verdict.json >/dev/null
    row "a receipted re-measurement rises"                       "$(key_value "$t/f.yaml")" 131
    row "the note names both numbers and the receipt"            "$(grep -c "^$NOTE 70 -> 131: evidence/q/verdict.json" "$t/f.yaml")" 1
    row "the note sits directly above the key"                   "$(grep -A1 "^$NOTE" "$t/f.yaml" | tail -1)" "unrefined_baseline: 131"
    ratchet "$t/f.yaml" 140 evidence/q/verdict.json >/dev/null
    row "a second re-measurement replaces the note, not stacks"  "$(grep -c "^$NOTE" "$t/f.yaml")" 1
    row "an unrelated comment survives"                          "$(grep -c '^# keep me$' "$t/f.yaml")" 1
    REPO_ROOT="$root"
    printf 'models: []\n' > "$t/none.yaml"
    ratchet "$t/none.yaml" 5 >/dev/null
    row "an absent key is recorded"                              "$(key_value "$t/none.yaml")" 5
    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ -n "$t" ] && [ -d "$t" ] && rm -rf "${t:?}"
    [ "$fail" -eq 0 ]
}

main() {
    local receipt=""
    case "${1:-}" in
        --self-test) self_test; return $? ;;
        --remeasure) receipt="${2:?--remeasure needs a receipt path under evidence/}" ;;
        '') ;;
        *) printf 'usage: %s [--remeasure evidence/<receipt>] | --self-test\n' "$(basename "$0")" >&2; return 2 ;;
    esac
    [ -f "$FORMALIZATION" ] || { printf 'NO-GO: %s does not exist\n' "$FORMALIZATION" >&2; return 2; }
    # shellcheck source=scripts/pv_bin.sh
    . "$REPO_ROOT/scripts/pv_bin.sh" || return 2
    local n
    n="$(measure)" || return 2
    printf '== refinement ratchet (ONT-3b) -> %s ==\n' "${FORMALIZATION#"$REPO_ROOT"/}"
    ratchet "$FORMALIZATION" "$n" "$receipt"
}

main "$@"
