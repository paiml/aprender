#!/usr/bin/env bash
# findings_ledger.sh -- write and check a findings ledger under contracts/findings-ledger-v1.yaml (#4455).
#
# pv IS THE VALIDATOR. This script never judges a line itself. It copies the real contract into a scratch
# corpus, points an `entity:` at the ledger, and runs
#     "$PV" lint <scratch>/contracts --gate shapes --shape findings-ledger-v1
# which arms the shape whatever contracts/lint-baseline.json says. The exit code is pv's:
#     0 every line conforms   1 a line is refused   2 a line pv cannot read (torn), so not judged
#     3 not a ledger: absent, empty, or a declaration error   64 a usage error (this script's own)
# An empty ledger is not a pass, and neither is a line pv cannot read.
#
# NO PR OR RELEASE STEP RUNS THIS. The contract binds lines written after it landed; the lines that
# predate it in docs/findings/ are not rewritten, so `check` on one of those files is expected to refuse.
#
# Usage:
#   bash scripts/findings_ledger.sh check LEDGER
#   bash scripts/findings_ledger.sh add (--session S | --ledger PATH) --id FND-<yyyymmdd>-<slug> \
#        --title T --evidence E --repro R --epic '#<n>'|unknown --severity P0|P1|P2|P3 [--sha SHA40]
#
# `add` writes the line from its flags, has pv judge that line alone, and appends it to the ledger
# (docs/findings/<S>.jsonl, or PATH) only on exit 0; a refused line leaves the ledger byte-identical.
# A flag left out leaves its key out, and pv refuses the line. --sha defaults to `git rev-parse HEAD`.
#
# pv: $PV when set, else the pv built from this tree (scripts/pv_bin.sh). Never a pv found on PATH.
# Case table: crates/aprender-contracts-cli/tests/ont_findings_ledger.rs over tests/fixtures/ont/findings-ledger/.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SHAPE=findings-ledger-v1
CONTRACT="$ROOT/contracts/$SHAPE.yaml"
TMPD=""  # judge's scratch corpus
CAND=""  # add's candidate line
trap 'if [ -n "$TMPD" ]; then rm -rf "${TMPD:?}"; fi; if [ -n "$CAND" ]; then rm -rf "${CAND:?}"; fi' EXIT

usage() { printf 'findings_ledger: %s\n' "$*" >&2; exit 64; }

# json_str VALUE -> VALUE as a JSON string literal. rc 1 on a control character other than \n \r \t.
json_str() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    case $s in *[[:cntrl:]]*) return 1 ;; esac
    printf '"%s"' "$s"
}

# judge LEDGER -> pv's report on stdout; returns pv's exit code.
judge() {
    local ledger=$1 dir rc=0
    [ -f "$ledger" ] || { printf 'findings_ledger: no ledger at %s; nothing judged\n' "$ledger" >&2; return 3; }
    [ -f "$CONTRACT" ] || { printf 'findings_ledger: no contract at %s\n' "$CONTRACT" >&2; return 3; }
    if grep -q '^entity:' "$CONTRACT"; then
        printf 'findings_ledger: %s already declares an entity; this script would add a second\n' "$CONTRACT" >&2
        return 3
    fi
    dir=$(mktemp -d)
    TMPD=$dir
    mkdir "$dir/contracts" "$dir/data"
    cp "$CONTRACT" "$dir/contracts/$SHAPE.yaml"
    printf '\nentity:\n  type: json\n  ref: data/findings.jsonl\n' >>"$dir/contracts/$SHAPE.yaml"
    cp "$ledger" "$dir/data/findings.jsonl"
    "$PV" lint "$dir/contracts" --gate shapes --shape "$SHAPE" || rc=$?
    rm -rf "${dir:?}"
    TMPD=""
    return "$rc"
}

cmd_check() {
    [ $# -eq 1 ] || usage "check takes one LEDGER"
    judge "$1"
}

cmd_add() {
    local session="" ledger="" sha="" line key val lit rc=0
    local -a keys=(id title evidence repro suspected_epic severity found_at_sha)
    local -A v=()
    while [ $# -gt 0 ]; do
        [ $# -ge 2 ] || usage "$1 needs a value"
        case $1 in
            --session) session=$2 ;;
            --ledger) ledger=$2 ;;
            --id) v[id]=$2 ;;
            --title) v[title]=$2 ;;
            --evidence) v[evidence]=$2 ;;
            --repro) v[repro]=$2 ;;
            --epic) v[suspected_epic]=$2 ;;
            --severity) v[severity]=$2 ;;
            --sha) sha=$2 ;;
            *) usage "unknown flag $1" ;;
        esac
        shift 2
    done
    if [ -z "$ledger" ]; then
        [[ $session =~ ^[a-z0-9][a-z0-9.-]*$ ]] || usage "--session must match ^[a-z0-9][a-z0-9.-]*\$, or pass --ledger"
        ledger="$ROOT/docs/findings/$session.jsonl"
    fi
    [ -n "$sha" ] || sha=$(git -C "$ROOT" rev-parse HEAD)
    v[found_at_sha]=$sha

    line=""
    for key in "${keys[@]}"; do
        [ -n "${v[$key]+set}" ] || continue
        val=${v[$key]}
        lit=$(json_str "$val") || usage "--$key holds a control character"
        line+="${line:+,}\"$key\":$lit"
    done
    line="{$line}"

    CAND=$(mktemp -d)
    printf '%s\n' "$line" >"$CAND/candidate.jsonl"
    judge "$CAND/candidate.jsonl" || rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'findings_ledger: pv did not pass the line (exit %s); %s is untouched\n' "$rc" "$ledger" >&2
        return "$rc"
    fi
    # A ledger whose last line has no newline would glue the new line onto it: one torn line.
    if [ -s "$ledger" ] && [ -n "$(tail -c 1 "$ledger")" ]; then
        printf 'findings_ledger: %s does not end in a newline; appending would tear its last line\n' "$ledger" >&2
        return 3
    fi
    mkdir -p "$(dirname "$ledger")"
    printf '%s\n' "$line" >>"$ledger"
    printf 'findings_ledger: appended %s to %s\n' "${v[id]-}" "$ledger"
}

[ $# -ge 1 ] || usage "usage: findings_ledger.sh check LEDGER | add ..."
if [ -z "${PV:-}" ]; then
    . "$ROOT/scripts/pv_bin.sh" || exit 3
fi
sub=$1
shift
case $sub in
    check) cmd_check "$@" ;;
    add) cmd_add "$@" ;;
    *) usage "unknown subcommand $sub" ;;
esac
