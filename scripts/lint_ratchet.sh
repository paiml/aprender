#!/usr/bin/env bash
# lint_ratchet.sh — PVL-001 EV-11: move the two `pv lint` ratchets DOWN, never up.
#
# `pv lint --gate theorem-pairing` and `--gate depends-on-present` read their
# baselines from contracts/lint-baseline.json and NEVER write it: at or below
# the baseline they pass and report the count. This script is the only thing
# that changes those numbers (`make lint-ratchet`, never in CI):
#
#   measured <  recorded  -> lowered to the measured count
#   measured == recorded  -> untouched
#   measured >  recorded  -> REFUSED, exit 1 (the gate is already red; a
#                            ratchet that could be raised here is not one)
#   not recorded          -> recorded (the first baseline)
#
# It also records `"command": "make lint-ratchet"`, so the file names the
# command that moves it.
#
# The file is edited LINE BY LINE, never re-serialized: check_ont_ratchet.sh
# requires `armed_gates` / `armed_shapes` on one line, and `jq .` would spread
# them over several and make `make ont-ratchet` refuse the file.
#
#   bash scripts/lint_ratchet.sh              # measure with the pinned pv, rewrite downward
#   bash scripts/lint_ratchet.sh --self-test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${BASELINE:-$REPO_ROOT/contracts/lint-baseline.json}"
COMMAND_VALUE="make lint-ratchet"

# A top-level key is one at EXACTLY two spaces of indent — the layout this script and
# check_ont_ratchet.sh write — so a nested key of the same name is never read or rewritten.
# key_value FILE KEY -> the recorded integer, or nothing when the key is absent.
key_value() {
    { grep -E "^  \"$2\"[[:space:]]*:" "$1" || true; } | head -1 \
        | sed -E 's/.*:[[:space:]]*//; s/[^0-9].*$//'
}

# set_line FILE KEY JSON_VALUE: replace the key's line in place, or insert it
# before the `"ont"` object (after armed_* and the other top-level scalars).
set_line() {
    local file="$1" key="$2" value="$3" tmp
    tmp="$(mktemp "$file.XXXXXX")"
    if grep -qE "^  \"$key\"[[:space:]]*:" "$file"; then
        awk -v k="\"$key\"" -v v="$value" '
            !done && index($0, k) && $0 ~ "^  " k "[[:space:]]*:" {
                comma = ($0 ~ /,[[:space:]]*$/) ? "," : ""
                print "  " k ": " v comma; done = 1; next
            }
            { print }' "$file" > "$tmp"
    elif grep -qE '^[[:space:]]*"ont"[[:space:]]*:' "$file"; then
        awk -v k="\"$key\"" -v v="$value" '
            !done && $0 ~ /^[[:space:]]*"ont"[[:space:]]*:/ { print "  " k ": " v ","; done = 1 }
            { print }' "$file" > "$tmp"
    else
        rm -f "$tmp"
        printf 'NO-GO: %s has neither "%s" nor an "ont" object to insert before; refusing to guess\n' "$file" "$key" >&2
        return 2
    fi
    mv "$tmp" "$file"
}

# ratchet_one FILE KEY MEASURED -> 0 ok (maybe rewritten), 1 refused rise
ratchet_one() {
    local file="$1" key="$2" now="$3" was
    was="$(key_value "$file" "$key")"
    if [ -z "$was" ]; then
        set_line "$file" "$key" "$now" || return 2
        printf '  recorded  %-30s %s\n' "$key" "$now"
    elif [ "$now" -lt "$was" ]; then
        set_line "$file" "$key" "$now" || return 2
        printf '  lowered   %-30s %s -> %s\n' "$key" "$was" "$now"
    elif [ "$now" -eq "$was" ]; then
        printf '  unchanged %-30s %s\n' "$key" "$now"
    else
        printf '  REFUSED   %-30s %s -> %s: the ratchet only turns down\n' "$key" "$was" "$now"
        return 1
    fi
}

# measure GATE FIELD -> the count the gate reports. A report is printed on
# exit 0 (at/below the baseline), 1 (above it) and 2 with no baseline recorded;
# a decline prints no report, and that is a refusal here, not a zero.
measure() {
    local gate="$1" field="$2" out rc n
    set +e
    out="$("$PV" lint "$REPO_ROOT/contracts" --gate "$gate" 2>/dev/null)"
    rc=$?
    set -e
    n="$(printf '%s' "$out" | jq -r --arg f "$field" '.[$f] // empty' 2>/dev/null || true)"
    case "$n" in
        ''|*[!0-9]*)
            printf 'NO-GO: `pv lint --gate %s` (exit %s) reported no %s; nothing was measured\n' "$gate" "$rc" "$field" >&2
            return 2 ;;
    esac
    printf '%s\n' "$n"
}

self_test() {
    local t pass=0 fail=0
    t="$(mktemp -d)"
    case "$t" in /tmp/*|/var/tmp/*) : ;; *) printf 'NO-GO: odd mktemp path %s\n' "$t" >&2; return 2 ;; esac
    row() { # row NAME GOT WANT
        if [ "$2" = "$3" ]; then pass=$((pass+1)); printf '  ok    %-52s %s\n' "$1" "$3"
        else fail=$((fail+1)); printf '  FAIL  %-52s want=%s got=%s\n' "$1" "$3" "$2"; fi
    }
    local rc
    printf 'lint_ratchet self-test\n'
    printf '{\n  "armed_gates": ["validate", "sigma"],\n  "unpaired_theorem_modules": 130,\n  "ont": {\n    "x": 1\n  }\n}\n' > "$t/b.json"
    ratchet_one "$t/b.json" unpaired_theorem_modules 129 >/dev/null
    row "a fall is written"                              "$(key_value "$t/b.json" unpaired_theorem_modules)" 129
    set +e; ratchet_one "$t/b.json" unpaired_theorem_modules 131 >/dev/null; rc=$?; set -e
    row "a rise is refused"                              "$rc" 1
    row "a refused rise leaves the number"               "$(key_value "$t/b.json" unpaired_theorem_modules)" 129
    ratchet_one "$t/b.json" contracts_without_depends_on 278 >/dev/null
    row "an absent key is recorded"                      "$(key_value "$t/b.json" contracts_without_depends_on)" 278
    set_line "$t/b.json" command "\"$COMMAND_VALUE\""
    row "the command is recorded"                        "$(grep -c "\"command\": \"$COMMAND_VALUE\"," "$t/b.json")" 1
    set_line "$t/b.json" command "\"$COMMAND_VALUE\""
    row "re-recording the command does not duplicate it" "$(grep -c '"command"' "$t/b.json")" 1
    row "armed_gates stays on one line"                  "$(grep -c '"armed_gates": \["validate", "sigma"\],' "$t/b.json")" 1
    if command -v jq >/dev/null 2>&1; then
        jq -e '(.unpaired_theorem_modules|numbers) and (.contracts_without_depends_on|numbers) and (.command|strings) and .ont.x == 1' \
            "$t/b.json" >/dev/null 2>&1 && row "the file is valid JSON with the probe's keys" ok ok \
            || row "the file is valid JSON with the probe's keys" bad ok
    fi
    ratchet_one "$t/b.json" unpaired_theorem_modules 129 >/dev/null
    row "an equal count leaves the file byte-identical"  "$(md5sum < "$t/b.json" | cut -c1-8)" "$(cp "$t/b.json" "$t/c.json"; ratchet_one "$t/c.json" unpaired_theorem_modules 129 >/dev/null; md5sum < "$t/c.json" | cut -c1-8)"
    printf '{\n  "command": "x",\n  "ont": {\n    "unpaired_theorem_modules": 5\n  }\n}\n' > "$t/nest.json"
    row "a nested key of the same name is not the recorded one" "$(key_value "$t/nest.json" unpaired_theorem_modules)" ""
    ratchet_one "$t/nest.json" unpaired_theorem_modules 7 >/dev/null
    row "recording beside a nested key leaves the nested one" "$(grep -c '^    "unpaired_theorem_modules": 5$' "$t/nest.json")" 1
    printf '{\n  "armed_gates": []\n}\n' > "$t/noont.json"
    set +e; set_line "$t/noont.json" command '"x"' 2>/dev/null; rc=$?; set -e
    row "no ont object and no key is refused, not guessed" "$rc" 2
    printf 'self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ -n "$t" ] && [ -d "$t" ] && rm -rf "$t"
    [ "$fail" -eq 0 ]
}

main() {
    case "${1:-}" in
        --self-test) self_test; return $? ;;
        '') ;;
        *) printf 'usage: %s [--self-test]\n' "$(basename "$0")" >&2; return 2 ;;
    esac
    [ -f "$BASELINE" ] || { printf 'NO-GO: %s does not exist\n' "$BASELINE" >&2; return 2; }
    # shellcheck source=scripts/pv_bin.sh
    . "$REPO_ROOT/scripts/pv_bin.sh" || return 2
    local pairing depends rc=0
    pairing="$(measure theorem-pairing unpaired_theorem_modules)" || return 2
    depends="$(measure depends-on-present contracts_without_depends_on)" || return 2
    printf '== lint ratchet (PVL-001 EV-11) -> %s ==\n' "${BASELINE#"$REPO_ROOT"/}"
    set_line "$BASELINE" command "\"$COMMAND_VALUE\"" || return 2
    ratchet_one "$BASELINE" unpaired_theorem_modules "$pairing" || rc=$?
    ratchet_one "$BASELINE" contracts_without_depends_on "$depends" || rc=$?
    return "$rc"
}

main "$@"
