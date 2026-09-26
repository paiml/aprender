#!/usr/bin/env bash
# check_cb200_tdg_grade.sh -- CB-200 (pmat comply's TDG grade gate) on EVERY PR (#4101).
#
# WHY THIS EXISTS
# ---------------
# 0.69.1 shipped with CB-200 at 618 definitions below grade B against a baseline
# of 599. The first place anything measured it was the pre-publish dogfood,
# hours after the code merged. check_complexity_ratchet.sh ratchets the baseline
# NUMBER (it cannot be raised), but nothing per-PR measured the COUNT against it,
# so 22 new below-B definitions merged one at a time and nobody saw them go by.
# This guard asks pmat for exactly that one rule, `--checks CB-200`, on the PR's
# tree. It is not a grep over the full report.
#
# A COLD INDEX, ALWAYS
# --------------------
# pmat keeps its comply index OUTSIDE the tree (~/.cache/.../comply/index/
# <dir>-<hash>/context.db). Measured once on #4101: after the tree was
# refactored and committed, `pmat comply check` still read 615, and the index
# file's mtime was the FIRST run's, 3 hours earlier. Deleting the index read
# 600. The trigger is not pinned down: on a small fixture pmat DOES re-read a
# changed tree with a warm cache (the stale-index case below passes with the
# override removed; measured). A self-hosted runner reuses $HOME between jobs,
# so every run points XDG_CACHE_HOME at a fresh temporary directory rather than
# trust an index it cannot see the age of. The stale-index case proves the
# guard reads the grown tree; it does NOT prove the override is necessary.
#
# Exit: 0 CB-200 at or under the [tdg] baseline (pmat reports Warn/Pass)
#       1 CB-200 Fail (over the baseline), or pmat returned no CB-200 verdict
#       2 environment: pmat or python3 missing -- refuses to pass, never 0
#
#   bash scripts/check_cb200_tdg_grade.sh             # the tree
#   bash scripts/check_cb200_tdg_grade.sh --self-test  # both polarities + the stale-index case
set -euo pipefail

# Every temporary file and directory lives under ONE root, removed on any exit.
TMPROOT=$(mktemp -d)
trap 'rm -rf "${TMPROOT:?}"' EXIT

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

# cb200_verdict <dir> -> prints "<status><TAB><first line of the message>".
# pmat's exit status is not the verdict: the verdict is the CB-200 row of the JSON report.
cb200_verdict() {
    local dir="$1" cache json
    cache=$(mktemp -d -p "$TMPROOT")
    json=$(mktemp -p "$TMPROOT")
    XDG_CACHE_HOME="$cache" pmat comply check --path "$dir" --checks CB-200 --format json > "$json" 2> "$json.err" || true
    python3 - "$json" <<'PY'
import json, sys
try:
    doc = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError) as exc:
    print("NONE\tpmat printed no JSON report: %s" % exc)
    sys.exit(0)
rows = [c for c in doc.get("checks") or [] if str(c.get("name", "")).startswith("CB-200")]
if len(rows) != 1:
    print("NONE\tthe report carries %d CB-200 rows, not 1" % len(rows))
else:
    print("%s\t%s" % (rows[0].get("status"), str(rows[0].get("message", "")).splitlines()[0]))
PY
    rm -rf "${cache:?}" "$json" "$json.err"
}

judge() { # <status> -> 0 held, 1 red
    case "$1" in
        Pass | Warn) return 0 ;;
        *) return 1 ;;
    esac
}

need_tools() {
    command -v pmat > /dev/null 2>&1 || {
        printf 'ENV: pmat is not on PATH; the fleet pin installs it (tools.toml; CI never installs tools).\n' >&2
        printf 'Without the analyser this guard cannot decide, so it refuses to pass (exit 2, never 0).\n' >&2
        exit 2
    }
    command -v python3 > /dev/null 2>&1 || {
        printf 'ENV: python3 is required to read the pmat JSON report.\n' >&2
        exit 2
    }
}

# tangle <name> -> one Python function with enough branches that pmat grades it below B.
tangle() {
    local k
    printf 'def %s(a, b, c, d, e):\n    total = 0\n    for x in range(a):\n' "$1"
    for k in $(seq 1 24); do
        printf '        if x %% %s == b and c > %s or d == %s and e:\n            total += %s\n' "$((k + 1))" "$k" "$k" "$k"
        printf '        elif x > %s and not e:\n            total -= %s\n' "$k" "$k"
    done
    printf '    return total\n\n\n'
}

# fixture <dir> <baseline> <n> -> a git repository whose gates file holds <baseline>, with <n> below-B functions.
fixture() {
    local d="$1" i
    mkdir -p "$d/src"
    printf '[tdg]\nbaseline = %s\nmin_grade = "B"\n' "$2" > "$d/.pmat-gates.toml"
    : > "$d/src/tangle.py"
    for i in $(seq 1 "$3"); do
        tangle "tangle_$i" >> "$d/src/tangle.py"
    done
    (cd "$d" && git init -q && git add -A && git -c core.hooksPath=/dev/null -c user.email=t@t -c user.name=t commit -qm fixture)
}

case_row() { # <name> <want held|red> <verdict> -> prints the row; returns 1 when the verdict is the wrong polarity
    local s="${3%%$'\t'*}" got=red
    if judge "$s"; then got=held; fi
    if [ "$got" = "$2" ]; then
        printf '  ok    %-14s CB-200 %s -> %s\n' "$1" "$s" "$got"
        return 0
    fi
    printf '  BROKE %-14s CB-200 %s -> %s, want %s: %s\n' "$1" "$s" "$got" "$2" "${3#*$'\t'}"
    return 1
}

self_test() {
    need_tools
    local td bad=0
    td=$(mktemp -d -p "$TMPROOT")
    # 1. over the baseline -> RED, the polarity this guard exists for
    fixture "$td/over" 0 1
    case_row over-baseline red "$(cb200_verdict "$td/over")" || bad=$((bad + 1))
    # 2. at the baseline -> held: a guard red for every input is a constant, not a gate
    fixture "$td/held" 1 1
    case_row at-baseline held "$(cb200_verdict "$td/held")" || bad=$((bad + 1))
    # 3. the SAME directory, grown past its baseline, measured again: the second run must read the
    #    grown tree. (With XDG_CACHE_HOME not overridden this case also passes on a fixture this small;
    #    see A COLD INDEX above -- it pins the behaviour the guard needs, not the reason for the override.)
    tangle tangle_2 >> "$td/held/src/tangle.py"
    (cd "$td/held" && git add -A && git -c core.hooksPath=/dev/null -c user.email=t@t -c user.name=t commit -qm grow)
    case_row stale-index red "$(cb200_verdict "$td/held")" || bad=$((bad + 1))
    # 4. no verdict is RED, never a pass
    case_row no-verdict red "$(printf 'NONE\tno CB-200 row')" || bad=$((bad + 1))
    # 5-7. the one-time re-baseline receipt: a count the sha does not measure is RED, the true
    #      count is held, and a receipt already on origin/main is spent and never re-measured.
    local head
    fixture "$td/rb" 1 1
    head=$(git -C "$td/rb" rev-parse HEAD)
    receipt_row() { # <name> <want held|red> <measured>
        local got=red
        printf 'sha: %s\nmeasured: %s\ntool_version: x\n' "$head" "$3" > "$td/rb/$RECEIPT_REL"
        if receipt_check "$td/rb" > /dev/null 2>&1; then got=held; fi
        if [ "$got" = "$2" ]; then printf '  ok    %-14s receipt -> %s\n' "$1" "$got"; return 0; fi
        printf '  BROKE %-14s receipt -> %s, want %s\n' "$1" "$got" "$2"
        return 1
    }
    mkdir -p "$td/rb/scripts"
    receipt_row receipt-true held 1 || bad=$((bad + 1))
    receipt_row receipt-false red 2 || bad=$((bad + 1))
    (cd "$td/rb" && git add -A && git -c core.hooksPath=/dev/null -c user.email=t@t -c user.name=t commit -qm receipt &&
        git update-ref refs/remotes/origin/main HEAD)
    receipt_row receipt-spent held 2 || bad=$((bad + 1))
    rm -rf "${td:?}"
    printf 'check_cb200_tdg_grade self-test: %s broken\n' "$bad"
    [ "$bad" -eq 0 ]
}

# RECEIPT <- a ONE-TIME re-baseline (car #4429, cop ruling B). scripts/lib_baseline_ratchet.sh
# admits a rise of scripts/cb200_baseline.txt only up to the count a receipt says was MEASURED at
# a main sha, and only while the receipt is new. The number is a claim until something measures
# it, so while the receipt is new this guard measures the receipt's sha the same way (a cold index
# over `git archive <sha>`) and requires the count it records. Once the receipt is on origin/main
# it is spent and this is skipped: the next pull request pays for one measurement, not two.
RECEIPT_REL='scripts/cb200_baseline.rebaseline'

cb200_count() { # <dir> -> the number of definitions below min_grade, from the CB-200 message
    local v
    v=$(cb200_verdict "$1")
    printf '%s\n' "${v#*$'\t'}" | sed -nE 's/^([0-9]+) definition.*/\1/p' | grep -m1 .
}

receipt_check() { # <root> -> 0 no receipt / spent / proven, 1 the recorded count is not what the sha measures
    local root="$1" sha measured got tree
    [ -f "$root/$RECEIPT_REL" ] || return 0
    if git -C "$root" cat-file -e "origin/main:$RECEIPT_REL" 2>/dev/null; then
        printf 'ok    %s is on origin/main: spent; the baseline is shrink-only again.\n' "$RECEIPT_REL"
        return 0
    fi
    sha=$(sed -nE 's/^sha:[[:space:]]*([0-9a-f]{40})[[:space:]]*$/\1/p' "$root/$RECEIPT_REL" | head -1)
    measured=$(sed -nE 's/^measured:[[:space:]]*([0-9]+)[[:space:]]*$/\1/p' "$root/$RECEIPT_REL" | head -1)
    if [ -z "$sha" ] || [ -z "$measured" ]; then
        printf 'RED   %s carries no 40-hex sha: and integer measured: to re-measure.\n' "$RECEIPT_REL"
        return 1
    fi
    git -C "$root" cat-file -e "${sha}^{commit}" 2>/dev/null ||
        git -C "$root" fetch -q --no-tags --depth=1 origin "$sha" 2>/dev/null || {
        printf 'RED   the receipt sha %s cannot be fetched, so its count is UNMEASURED.\n' "${sha:0:12}"
        return 1
    }
    tree=$(mktemp -d -p "$TMPROOT")
    git -C "$root" archive "$sha" | tar -x -C "$tree"
    got=$(cb200_count "$tree") || got=""
    if [ "$got" != "$measured" ]; then
        printf 'RED   %s records %s at %s, but a cold index there measures <%s>.\n' "$RECEIPT_REL" "$measured" "${sha:0:12}" "$got"
        return 1
    fi
    printf 'ok    %s: %s below-B definitions at %s, re-measured (one-time re-baseline).\n' "$RECEIPT_REL" "$got" "${sha:0:12}"
    return 0
}

case "${1:-}" in
    -h | --help) usage; exit 0 ;;
    --self-test | --selftest) if self_test; then exit 0; else exit 1; fi ;;
    "") ;;
    *) printf 'unknown argument %s (try --help)\n' "$1" >&2; exit 2 ;;
esac

need_tools
root=$(git rev-parse --show-toplevel)
printf '=== CB-200: definitions below the [tdg] min_grade may not exceed the baseline (check_cb200_tdg_grade.sh) ===\n'
# PROVE THE MECHANISM ENGAGED: name the analyser that produced the number.
printf 'pmat: %s (%s)\n' "$(pmat --version 2>/dev/null | head -1)" "$(command -v pmat)"
v=$(cb200_verdict "$root")
s=${v%%$'\t'*}
printf 'CB-200 %s: %s\n' "$s" "${v#*$'\t'}"
receipt_check "$root" || exit 1
if judge "$s"; then
    printf 'ok    CB-200 held (a cold index over this tree; the baseline is .pmat-gates.toml [tdg])\n'
    exit 0
fi
printf 'RED   CB-200 is over its baseline. Fix the new below-B definitions; raising [tdg] baseline is not the fix (#4101).\n'
exit 1
