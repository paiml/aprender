#!/usr/bin/env bash
# crux_nightly_smoke.sh — release day reads the night's CRUX, it never runs CRUX (#4702)
#
#   bash scripts/release/crux_nightly_smoke.sh --nightly DIR --head SHA --hosts A,B --thinking off,on [--cert FILE]
#   bash scripts/release/crux_nightly_smoke.sh --self-test
#
# WHY. A release used to run the CRUX smoke sweep on release day, on the release binary, on
# every named host. That is ~85 min on the critical path, for a result the nightly full lane
# already measures on main's head. Release day now judges the night's full-lane receipts for H
# (the head the release ships) with the same smoke rule, and runs nothing.
#
# THE RULE (the smoke rule of scripts/lib/crux_smoke_scope.py, read here from the nightly):
#   - every receipt under DIR is a crux-inference-receipt/v1 measured by the binary built from H,
#     either a full apr.sha / apr_sha equal to H, or an `apr X (short)` version line whose short
#     sha is a prefix of H and that every cell's engines.apr.version repeats; and none DECLINEd;
#   - every host in --hosts has a receipt;
#   - for every certified model (prompt-certification.json, admitted_by_sha_thinking) and every
#     --thinking mode admitted for it, each host has at least one cell, all of them GREEN, and a
#     GREEN positive control.
# A missing nightly, receipt, host or smoke cell is a REFUSAL: an unmeasured cell is not a pass,
# and release day runs no CRUX to fill it. A cell outside the smoke matrix (a mode not admitted
# for its model) is not judged.
#
# EXIT 0 PASS · 1 REFUSE · 2 usage. Bash and jq only.
set -uo pipefail
PROG=crux_nightly_smoke
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

refuse() { printf 'REFUSE %s\n' "$*"; FAILED=1; }

# judge NIGHTLY HEAD HOSTS THINKING CERT -> prints one line per cell and a verdict; returns 0|1.
judge() {
    local nightly=$1 head=$2 hosts_csv=$3 modes_csv=$4 cert=$5
    local f base sha line short idx models m h t got red ctl
    local -a hosts modes files admitted model_list
    FAILED=0
    IFS=, read -r -a hosts <<< "$hosts_csv"
    IFS=, read -r -a modes <<< "$modes_csv"
    if ! [[ $head =~ ^[0-9a-f]{40}$ ]]; then
        refuse "H '$head' is not a full 40-hex sha -- the night's receipts bind to the head the release ships"
        return 1
    fi
    if [ ! -d "$nightly" ]; then
        refuse "no nightly full-lane result for H ${head:0:12} at '$nightly' -- release day runs no CRUX to make one"
        return 1
    fi
    files=()
    for f in "$nightly"/*.json; do
        [ -e "$f" ] || continue
        case $(basename "$f") in prompt-certification*) continue ;; esac
        files+=("$f")
    done
    if [ "${#files[@]}" -eq 0 ]; then
        refuse "the nightly at '$nightly' holds no CRUX receipt for H ${head:0:12} -- release day runs no CRUX to make one"
        return 1
    fi
    if ! models=$(jq -er '.admitted_by_sha_thinking | if type == "object" then keys[] else error("absent") end' "$cert" 2> /dev/null) || [ -z "$models" ]; then
        refuse "no admitted_by_sha_thinking in the certification '$cert' -- the smoke matrix (model x admitted mode) is unknown"
        return 1
    fi
    mapfile -t model_list <<< "$models"
    idx=$(mktemp) || return 1
    for f in "${files[@]}"; do
        base=$(basename "$f")
        if ! jq -e '.schema == "crux-inference-receipt/v1"' "$f" > /dev/null 2>&1; then
            refuse "$base is not a readable crux-inference-receipt/v1"
            continue
        fi
        sha=$(jq -r '(.apr | objects | .sha) // .apr_sha // empty' "$f")
        if [ -z "$sha" ]; then
            line=$(jq -r '.apr | objects | .version_line // empty' "$f")
            short=""
            if [[ $line =~ ^apr\ [^\ ]+\ \(([0-9a-f]{7,40})\)$ ]]; then short=${BASH_REMATCH[1]}; fi
            if [ -n "$short" ] && [ "${head#"$short"}" != "$head" ] &&
                jq -e --arg l "$line" '[.cells[]? | .engines.apr.version? // empty | select(. != $l)] | length == 0' "$f" > /dev/null; then
                sha=$head
            else
                sha="unbound:${line:-none}"
            fi
        fi
        if [ "$sha" != "$head" ]; then
            refuse "$base was measured by apr '$sha', not the binary built from H ${head:0:12} -- the night tested something else"
            continue
        fi
        if [ "$(jq -r '.summary.verdict // empty' "$f")" = DECLINE ]; then
            refuse "$base DECLINED ($(jq -r '.summary.declined_because // "no reason"' "$f"))"
            continue
        fi
        jq -r '.host as $h | .cells[]? | [(.key.host // $h), .key.model_sha256, .key.thinking, .verdict, (.positive_control == true)] | @tsv' "$f" >> "$idx"
        jq -r '"HOST\t" + (.host // "")' "$f" >> "$idx"
    done
    for h in "${hosts[@]}"; do
        if ! grep -qxF "HOST	$h" "$idx"; then
            refuse "host $h has no nightly CRUX receipt for H ${head:0:12} -- the smoke judge needs every named host"
            continue
        fi
        for m in "${model_list[@]}"; do
            mapfile -t admitted < <(jq -r --arg m "$m" '.admitted_by_sha_thinking[$m] | objects | to_entries[] | select(.value == true) | .key' "$cert")
            local any=0
            for t in "${modes[@]}"; do
                printf '%s\n' "${admitted[@]}" | grep -qxF "$t" || continue
                any=1
                got=$(awk -F'\t' -v h="$h" -v m="$m" -v t="$t" '$1 == h && $2 == m && $3 == t' "$idx" | wc -l)
                red=$(awk -F'\t' -v h="$h" -v m="$m" -v t="$t" '$1 == h && $2 == m && $3 == t && $4 != "GREEN"' "$idx" | wc -l)
                ctl=$(awk -F'\t' -v h="$h" -v m="$m" -v t="$t" '$1 == h && $2 == m && $3 == t && $4 == "GREEN" && $5 == "true"' "$idx" | wc -l)
                if [ "$got" -eq 0 ]; then
                    refuse "$h model ${m:0:12} thinking=$t: no smoke cell in the nightly -- not measured, and release day runs no CRUX"
                elif [ "$red" -gt 0 ]; then
                    refuse "$h model ${m:0:12} thinking=$t: $red of $got cell(s) not GREEN"
                elif [ "$ctl" -eq 0 ]; then
                    refuse "$h model ${m:0:12} thinking=$t: the positive control is missing or not GREEN"
                else
                    printf 'ok     %s model %s thinking=%s: %s cell(s) GREEN, control GREEN\n' "$h" "${m:0:12}" "$t" "$got"
                fi
            done
            if [ "$any" -eq 0 ]; then
                refuse "$h model ${m:0:12}: no --thinking mode is admitted for it -- nothing about it can be smoke-proven"
            fi
        done
    done
    rm -f -- "${idx:?}"
    return "$FAILED"
}

# One case: a copy of the green fixture, an edit, the expected exit and a phrase the output must hold.
H_FIX=1234567890abcdef1234567890abcdef12345678
case_row() {
    local script=$1 name=$2 want_rc=$3 want=$4 edit=$5 args=${6:-}
    local d out rc
    d=$(mktemp -d) || return 1
    cp "$HERE/crux_nightly_cases/green/"*.json "$d/"
    ( cd "$d" && eval "$edit" ) > /dev/null 2>&1
    # shellcheck disable=SC2086 # $args is a fixed word list from the case table
    out=$(bash "$script" --nightly "$d" --head "$H_FIX" --hosts host-a,host-b --thinking off,on $args 2>&1)
    rc=$?
    rm -rf -- "${d:?}"
    if [ "$rc" -eq "$want_rc" ] && printf '%s\n' "$out" | grep -qF -- "$want"; then
        return 0
    fi
    printf '  row %s: rc=%s want %s; output lacks "%s"\n' "$name" "$rc" "$want_rc" "$want" >&2
    return 1
}

# set FILE JQ -> rewrite FILE in place through jq (used by the case table's edits).
jset() { local t; t=$(mktemp) && jq "$2" "$1" > "$t" && mv -- "$t" "$1"; }
export -f jset

# The case table. Prints the number of rows that did not hold for SCRIPT.
case_table() {
    local s=$1 bad=0 A B
    A=$(printf 'a%.0s' $(seq 64))
    B=$(printf 'b%.0s' $(seq 64))
    case_row "$s" green 0 "PASS" ':' || bad=$((bad + 1))
    case_row "$s" green-ignores-unadmitted-red 0 "ok     host-b model bbbbbbbbbbbb thinking=off" ':' || bad=$((bad + 1))
    case_row "$s" no-nightly-dir 1 "no nightly full-lane result" ':' "--nightly /nonexistent/crux-nightly" || bad=$((bad + 1))
    case_row "$s" no-receipts 1 "holds no CRUX receipt" 'rm -f host-a-gpu.json host-b-gpu.json' || bad=$((bad + 1))
    case_row "$s" missing-host 1 "host host-b has no nightly CRUX receipt" 'rm -f host-b-gpu.json' || bad=$((bad + 1))
    case_row "$s" missing-smoke-cell 1 "host-a model aaaaaaaaaaaa thinking=on: no smoke cell" \
        "jset host-a-gpu.json 'del(.cells[] | select(.key.model_sha256 == \"$A\" and .key.thinking == \"on\"))'" || bad=$((bad + 1))
    case_row "$s" red-cell 1 "1 of 2 cell(s) not GREEN" \
        "jset host-b-gpu.json '(.cells[] | select(.key.model_sha256 == \"$B\" and .key.prompt_id == \"p1\" and .key.thinking == \"off\") | .verdict) = \"RED\"'" || bad=$((bad + 1))
    case_row "$s" control-missing 1 "positive control is missing" \
        "jset host-b-gpu.json '(.cells[] | select(.key.model_sha256 == \"$B\" and .key.thinking == \"off\") | .positive_control) = false'" || bad=$((bad + 1))
    case_row "$s" other-head 1 "not the binary built from H" \
        "jset host-a-gpu.json '.apr.sha = \"ffffffffffffffffffffffffffffffffffffffff\"'" || bad=$((bad + 1))
    case_row "$s" declined 1 "host-b-gpu.json DECLINED (gpu absent)" \
        "jset host-b-gpu.json '.summary = {verdict: \"DECLINE\", declined_because: \"gpu absent\"}'" || bad=$((bad + 1))
    case_row "$s" short-sha-of-h 0 "PASS" \
        "jset host-a-gpu.json 'del(.apr.sha) | .apr.version_line = \"apr 0.70.2 (1234567890a)\"'" || bad=$((bad + 1))
    case_row "$s" short-sha-not-h 1 "not the binary built from H" \
        "jset host-a-gpu.json 'del(.apr.sha) | .apr.version_line = \"apr 0.70.2 (234567890ab)\"'" || bad=$((bad + 1))
    case_row "$s" dirty-version-line 1 "not the binary built from H" \
        "jset host-a-gpu.json 'del(.apr.sha) | .apr.version_line = \"apr 0.70.2 (1234567890a-dirty)\"'" || bad=$((bad + 1))
    case_row "$s" cell-names-other-binary 1 "not the binary built from H" \
        "jset host-a-gpu.json 'del(.apr.sha) | .apr.version_line = \"apr 0.70.2 (1234567890a)\" | .cells[0].engines.apr.version = \"apr 0.70.2 (ffffffffff)\"'" || bad=$((bad + 1))
    case_row "$s" not-a-receipt 1 "is not a readable crux-inference-receipt/v1" 'printf "{" > host-a-gpu.json' || bad=$((bad + 1))
    case_row "$s" no-certification 1 "no admitted_by_sha_thinking" 'rm -f prompt-certification.json' || bad=$((bad + 1))
    case_row "$s" model-without-mode 1 "model bbbbbbbbbbbb: no --thinking mode is admitted" \
        "jset prompt-certification.json '.admitted_by_sha_thinking[\"$B\"].off = false'" || bad=$((bad + 1))
    case_row "$s" short-head 1 "is not a full 40-hex sha" ':' "--head 1234567890ab" || bad=$((bad + 1))
    printf '%s\n' "$bad"
}

self_test() {
    local self="$HERE/$PROG.sh" bad mut n=0 fails=0 stub td
    printf '%s --self-test\n' "$PROG"
    bad=$(case_table "$self")
    if [ "$bad" -eq 0 ]; then echo "PASS case table: 18 rows hold on the real judge"; else echo "FAIL case table: $bad row(s) do not hold"; fails=$((fails + 1)); fi

    # Release day runs no CRUX: with apr and the CRUX drivers stubbed on PATH to leave a mark, a
    # PASS run must leave none.
    td=$(mktemp -d) || return 1
    for stub in apr crux_sweep_shards.sh crux_inference_dogfood.sh; do
        printf '#!/bin/sh\ntouch "%s/ran-%s"\n' "$td" "$stub" > "$td/$stub"
        chmod +x "$td/$stub"
    done
    if PATH="$td:$PATH" bash "$self" --nightly "$HERE/crux_nightly_cases/green" --head "$H_FIX" --hosts host-a,host-b --thinking off,on > /dev/null 2>&1 &&
        ! compgen -G "$td/ran-*" > /dev/null; then
        echo "PASS a PASS judgement invokes no apr and no CRUX driver"
    else
        echo "FAIL the judge did not pass on the green fixture, or invoked apr / a CRUX driver"
        fails=$((fails + 1))
    fi
    rm -rf -- "${td:?}"

    # Mutants: each weakens one rule in a copy; the case table must go red on every one.
    local -a muts=(
        's/if \[ "\$got" -eq 0 \]; then/if false; then/'
        's/elif \[ "\$red" -gt 0 \]; then/elif false; then/'
        's/elif \[ "\$ctl" -eq 0 \]; then/elif false; then/'
        's/if \[ "\$sha" != "\$head" \]; then/if false; then/'
        's/= DECLINE \]; then/= NEVER \]; then/'
        's/if ! grep -qxF "HOST\t\$h" "\$idx"; then/if false; then/'
        's/if \[ "\$any" -eq 0 \]; then/if false; then/'
        's/ != "\$head" \] \&\&/ = "\$head" ] || true \&\&/'
        's/select(. != \$l)/select(false)/'
        's/    if \[ ! -d "\$nightly" \]; then/    if false; then/'
        's/grep -qxF "\$t" || continue/true/'
    )
    td=$(mktemp -d) || return 1
    for mut in "${muts[@]}"; do
        n=$((n + 1))
        sed "$mut" "$self" > "$td/m.sh"
        mkdir -p "$td/crux_nightly_cases" && cp -r "$HERE/crux_nightly_cases/green" "$td/crux_nightly_cases/"
        if cmp -s "$self" "$td/m.sh"; then
            echo "FAIL mutant $n did not apply ($mut) -- a mutant that changes nothing proves nothing"
            fails=$((fails + 1))
            continue
        fi
        bad=$(case_table "$td/m.sh" 2> /dev/null)
        if [ "$bad" -gt 0 ]; then
            echo "PASS mutant $n killed by $bad row(s): $mut"
        else
            echo "FAIL mutant $n SURVIVED: $mut"
            fails=$((fails + 1))
        fi
    done
    rm -rf -- "${td:?}"
    if [ "$fails" -eq 0 ]; then echo "$PROG self-test: PASS"; return 0; fi
    echo "$PROG self-test: FAIL ($fails)"
    return 1
}

main() {
    local nightly="" head="" hosts="" modes="" cert=""
    if ! command -v jq > /dev/null; then echo "$PROG: jq is required" >&2; return 2; fi
    if [ "${1:-}" = --self-test ]; then self_test; return; fi
    while [ $# -gt 0 ]; do
        case $1 in
            --nightly) nightly=${2:-}; shift 2 ;;
            --head) head=${2:-}; shift 2 ;;
            --hosts) hosts=${2:-}; shift 2 ;;
            --thinking) modes=${2:-}; shift 2 ;;
            --cert) cert=${2:-}; shift 2 ;;
            *) echo "$PROG: unknown argument '$1'" >&2; return 2 ;;
        esac
    done
    if [ -z "$nightly" ] || [ -z "$head" ] || [ -z "$hosts" ] || [ -z "$modes" ]; then
        echo "usage: $PROG --nightly DIR --head SHA --hosts A,B --thinking off,on [--cert FILE] | --self-test" >&2
        return 2
    fi
    cert=${cert:-$nightly/prompt-certification.json}
    printf '%s: judging the nightly full-lane CRUX for H %s (hosts %s, thinking %s); release day runs no CRUX\n' "$PROG" "$head" "$hosts" "$modes"
    if judge "$nightly" "$head" "$hosts" "$modes" "$cert"; then
        echo "CRUX-NIGHTLY-SMOKE: PASS"
        return 0
    fi
    echo "CRUX-NIGHTLY-SMOKE: REFUSE"
    return 1
}

main "$@"
