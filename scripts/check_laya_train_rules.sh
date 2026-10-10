#!/usr/bin/env bash
# check_laya_train_rules.sh - the rule checker of APR-LAYA-TRAIN-001 (§9, row LT-0 g).
#
# Prints one line per hard rule of the spec that is a command (H1, H2, H4, H5,
# H10, H11, H12, H15), for the diff `<base>...<head>`. Each line carries the
# rule's verdict and its anti-vacuity arm, the fact that shows the rule looked
# at something. The cop runs it on every PR of this spec before the merge queue
# is armed and posts the output on the PR.
#
#   scripts/check_laya_train_rules.sh [--repo DIR] [--base REF] [--head REF]
#   scripts/check_laya_train_rules.sh --self-test
#
# Exit 0 = every rule PASS (or n/a). Exit 1 = at least one RED. Exit 2 = usage.
#
# --self-test builds a throwaway repository under mktemp, plants each rule's
# mutant on its own branch and requires the checker to name it, and requires a
# clean branch to pass. It never runs git in this checkout.
#
# Needles that would match this file's own text are assembled from pieces, so
# a PR that adds this file does not trip H1 or H15 on itself.

set -euo pipefail

PY_WORD="pyth""on"
PIP_WORD="pi""p "
UV_WORD="uv"" run"
DIRTY_FLAG="--allow""-dirty"
SSH_WORD="ss""h "
# ssh only in command position (line start, or after ; & | ( or a backtick, optionally
# behind sudo/exec/command/time), so prose that names it is not a hit.
SSH_RE='(^|[;&|(`])[[:space:]]*((sudo|exec|command|time)[[:space:]]+)?'"ss""h"'[[:space:]]'
PUBLISH_RE='(cargo|make)[[:space:]]+'"publi""sh"

LT1_RECEIPT="evidence/laya-train/lt-1/receipt.json"
# The ONE path H4 exempts (Q5 ruling, handoff/quorum-a3-laya-q5-lt1-order.md): the LT-1
# proxy-layer probe is measurement, not model code, and must merge before its receipt.
LT1_PROBE="crates/aprender-train/examples/laya_engine_probe.rs"
LT9_DECL="evidence/laya-train/lt-9/declaration.json"
LT10_DIR="evidence/laya-train/lt-10"
NEW_CRATE="crates/aprender-decide-train"
SIZE_CAP=1048576

REPO=""
BASE="origin/main"
HEAD_REF="HEAD"
RED=0

usage() {
    printf 'usage: %s [--repo DIR] [--base REF] [--head REF] | --self-test\n' "$0" >&2
    exit 2
}

g() { git -C "$REPO" "$@"; }

line() {
    # line <rule> <PASS|RED|n/a> <detail> <arm>
    printf '%-4s %-4s | %s | arm: %s\n' "$1" "$2" "$3" "$4"
    if [ "$2" = "RED" ]; then RED=1; fi
}

changed_names() { g diff --name-only "$BASE...$HEAD_REF"; }
added_names() { g diff --name-only --diff-filter=A "$BASE...$HEAD_REF"; }

# Paths whose added lines count as "a Makefile, a justfile, a script or a workflow".
runner_paths() {
    changed_names | awk '/(^|\/)Makefile[^\/]*$|\.mk$|(^|\/)justfile$|\.sh$|\.bash$|^scripts\/|^\.github\/workflows\//'
}

# Added lines (without the leading '+') of runner paths, as "path:text".
added_runner_lines() {
    local p
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        g diff -U0 "$BASE...$HEAD_REF" -- "$p" \
            | awk -v p="$p" '/^\+\+\+/ {next} /^\+/ {print p ":" substr($0, 2)}'
    done < <(runner_paths)
}

rule_h1() {
    local files lines hits
    files=$(added_names | awk '/\.py$|(^|\/)pyproject\.toml$|(^|\/)uv\.lock$/')
    lines=$(added_runner_lines | awk -v a="$PY_WORD" -v b="$UV_WORD" -v c="$PIP_WORD" \
        'index($0, a) || index($0, b) || index($0, c)')
    hits=$(printf '%s\n%s\n' "$files" "$lines" | awk 'NF' | head -n 5 | paste -sd ';' -)
    if [ -n "$hits" ]; then
        line H1 RED "Python added: $hits" "files and runner lines scanned"
    else
        line H1 PASS "no .py/pyproject/uv.lock added, no runner line invokes it" \
            "$(runner_paths | awk 'NF' | wc -l) runner paths, $(added_names | awk 'NF' | wc -l) added files scanned"
    fi
}

rule_h2() {
    local hits
    hits=$(changed_names | awk '/^crates\/aprender-decide\/|^crates\/aprender-core\/src\/models\/modernbert\/|^scripts\/laya_train\/numeric_cases\.json$|\/fixtures\/(modernbert_tiny|laya_tiny)\//' \
        | head -n 5 | paste -sd ';' -)
    if [ -n "$hits" ]; then
        line H2 RED "judge path changed: $hits" "name-only diff"
    else
        line H2 PASS "no judge path changed" "$(changed_names | awk 'NF' | wc -l) changed paths scanned"
    fi
}

rule_h4() {
    local on_base=absent on_head=absent touches
    g cat-file -e "$BASE:$LT1_RECEIPT" 2>/dev/null && on_base=present
    g cat-file -e "$HEAD_REF:$LT1_RECEIPT" 2>/dev/null && on_head=present
    local exempt
    touches=$(changed_names | awk -v p="$LT1_PROBE" '/^crates\// && $0 != p' | wc -l)
    exempt=$(changed_names | awk -v p="$LT1_PROBE" '$0 == p' | wc -l)
    local arm="$LT1_RECEIPT base=$on_base head=$on_head; crates/ paths=$touches; exempt $LT1_PROBE=$exempt"
    if [ "$touches" -gt 0 ] && [ "$on_base" = absent ] && [ "$on_head" = absent ]; then
        line H4 RED "touches crates/ before LT-1's receipt" "$arm"
    else
        line H4 PASS "measure-first holds" "$arm"
    fi
}

rule_h5() {
    local engine other files imports_ok second_tensor
    if ! g cat-file -e "$HEAD_REF:$NEW_CRATE/Cargo.toml" 2>/dev/null; then
        line H5 n/a "$NEW_CRATE absent on head" "crate looked up at $HEAD_REF"
        return
    fi
    engine=$(g show "$BASE:$LT1_RECEIPT" 2>/dev/null \
        | awk -F'"' '/"engine"[[:space:]]*:/ {print $4; exit}')
    case "$engine" in
        core) other='aprender_train::autograd|entrenar::autograd|^[[:space:]]*aprender-train[[:space:]]*=|package[[:space:]]*=[[:space:]]*"aprender-train"' ;;
        train) other='aprender::autograd|aprender_core::autograd' ;;
        *)
            line H5 RED "no chosen engine in $BASE:$LT1_RECEIPT" "engine='${engine:-none}'"
            return ;;
    esac
    files=$(g ls-tree -r --name-only "$HEAD_REF" -- "$NEW_CRATE" | awk '/\.rs$|Cargo\.toml$/')
    local hits="" n_ok=0 f body
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        body=$(g show "$HEAD_REF:$f")
        if printf '%s\n' "$body" | grep -Eq "$other"; then hits="$hits $f"; fi
        if [ "$engine" = core ]; then
            printf '%s\n' "$body" | grep -Eq 'aprender::autograd' && n_ok=$((n_ok + 1))
        else
            printf '%s\n' "$body" | grep -Eq 'aprender_train::autograd|entrenar::autograd' && n_ok=$((n_ok + 1))
        fi
    done <<<"$files"
    second_tensor=$(g diff -U0 "$BASE...$HEAD_REF" -- '*.rs' | awk '
        /^\+\+\+ / {f=$2; next}
        /^\+[[:space:]]*(pub(\([a-z]+\))?[[:space:]]+)?struct[[:space:]]+Tensor([^A-Za-z0-9_]|$)/ {t[f]=1}
        /^\+.*fn[[:space:]]+backward[[:space:]]*\(/ {b[f]=1}
        END {for (k in t) if (k in b) print k}' | paste -sd ';' -)
    if [ -n "$hits" ] || [ -n "$second_tensor" ]; then
        line H5 RED "other engine named:${hits:- none}; second Tensor with backward: ${second_tensor:-none}" \
            "engine=$engine"
    else
        line H5 PASS "only the chosen engine" "engine=$engine; its import found in $n_ok file(s)"
    fi
}

utc_commit_time() {
    # utc_commit_time <ref> <path>: commit time of the commit that added <path>
    TZ=UTC g log --diff-filter=A --date=format-local:%Y-%m-%dT%H:%M:%SZ --format='%cd %h' \
        "$1" -- "$2" | tail -n 1
}

rule_h10() {
    local decl outcomes n bad="" f started
    if ! g cat-file -e "$BASE:$LT9_DECL" 2>/dev/null; then
        if g cat-file -e "$HEAD_REF:$LT10_DIR" 2>/dev/null \
            && g ls-tree --name-only "$HEAD_REF" "$LT10_DIR/" | awk '/outcome/' | grep -q .; then
            line H10 RED "outcome present but no declaration on $BASE" "$LT9_DECL absent on base"
        else
            line H10 n/a "no declaration on $BASE" "$LT9_DECL absent on base"
        fi
        return
    fi
    decl=$(utc_commit_time "$BASE" "$LT9_DECL")
    outcomes=$(g ls-tree -r --name-only "$HEAD_REF" -- "$LT10_DIR" | awk '/outcome[^\/]*\.json$/')
    n=$(printf '%s\n' "$outcomes" | awk 'NF' | wc -l)
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        started=$(g show "$HEAD_REF:$f" | awk -F'"' '/"run_started_utc"[[:space:]]*:/ {print $4; exit}')
        if [ -z "$started" ] || [[ "$started" < "${decl%% *}" ]]; then bad="$bad $f(${started:-no run_started_utc})"; fi
    done <<<"$outcomes"
    if [ "$n" -gt 1 ] || [ -n "$bad" ]; then
        line H10 RED "outcomes=$n; before declaration:${bad:- none}" "declaration merged $decl"
    else
        line H10 PASS "outcomes=$n, none before the declaration" "declaration merged $decl"
    fi
}

rule_h11() {
    local cli pub new_crates n_new wf bad_wf="" p
    cli=$(changed_names | awk '$0 == "contracts/apr-cli-commands-v1.yaml"')
    pub=$(changed_names | awk '/(^|\/)Cargo\.toml$/' | while IFS= read -r p; do
        g diff -U0 "$BASE...$HEAD_REF" -- "$p" | awk -v p="$p" '/^[+-][[:space:]]*publish[[:space:]]*=/ {print p; exit}'
    done | paste -sd ';' -)
    new_crates=$(added_names | awk '/^crates\/[^\/]+\/Cargo\.toml$/' | paste -sd ';' -)
    n_new=$(printf '%s' "$new_crates" | awk -F';' 'NF {print NF; exit} END {if (!NR) print 0}')
    wf=$(changed_names | awk '/^\.github\/workflows\/.*\.ya?ml$/')
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        g cat-file -e "$HEAD_REF:$p" 2>/dev/null || continue
        if g show "$HEAD_REF:$p" | awk '
            /^on:[[:space:]]*[^[:space:]]/ { if ($0 !~ /^on:[[:space:]]*\[?[[:space:]]*workflow_dispatch[[:space:]]*\]?[[:space:]]*$/) bad=1; next }
            /^on:/ {inon=1; next}
            inon && /^[^[:space:]#]/ {inon=0}
            inon && /^  [A-Za-z_]+:?/ { k=$1; sub(/:$/, "", k); if (k != "workflow_dispatch") bad=1 }
            END {exit (bad ? 0 : 1)}'; then
            bad_wf="$bad_wf $p"
        fi
    done <<<"$wf"
    if [ -n "$cli" ] || [ -n "$pub" ] || [ "$n_new" -gt 1 ] || [ -n "$bad_wf" ]; then
        line H11 RED "cli-contract:${cli:-no} publish:${pub:-no} new crates:$n_new workflows:${bad_wf:- ok}" \
            "new crates: ${new_crates:-none}"
    else
        line H11 PASS "no public surface added" "new crates: ${new_crates:-none}"
    fi
}

rule_h12() {
    local f s max=0 maxf="none" over=""
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        s=$(g cat-file -s "$HEAD_REF:$f" 2>/dev/null || echo 0)
        if [ "$s" -gt "$max" ]; then max=$s; maxf=$f; fi
        if [ "$s" -gt "$SIZE_CAP" ]; then over="$over $f($s)"; fi
    done < <(added_names)
    if [ -n "$over" ]; then
        line H12 RED "added file above $SIZE_CAP bytes:$over" "largest added: $maxf $max"
    else
        line H12 PASS "no added file above $SIZE_CAP bytes" "largest added: $maxf $max"
    fi
}

rule_h15() {
    local hits
    hits=$(added_runner_lines | awk -v a="$DIRTY_FLAG" -v s="$SSH_RE" -v re="$PUBLISH_RE" \
        '{ body = $0; sub(/^[^:]*:/, "", body) } index(body, a) || body ~ s || body ~ re' | head -n 5 | paste -sd ';' -)
    if [ -n "$hits" ]; then
        line H15 RED "forbidden line added: $hits" "runner lines scanned"
    else
        line H15 PASS "no dirty-publish, publish or ssh line added" \
            "$(added_runner_lines | wc -l) added runner lines scanned"
    fi
}

run_rules() {
    g rev-parse -q --verify "$BASE^{commit}" >/dev/null || { echo "base $BASE not found" >&2; exit 2; }
    g rev-parse -q --verify "$HEAD_REF^{commit}" >/dev/null || { echo "head $HEAD_REF not found" >&2; exit 2; }
    printf 'laya-train rules | base %s | head %s\n' \
        "$(g rev-parse --short "$BASE")" "$(g rev-parse --short "$HEAD_REF")"
    rule_h1; rule_h2; rule_h4; rule_h5; rule_h10; rule_h11; rule_h12; rule_h15
    if [ "$RED" -eq 0 ]; then echo "RESULT PASS"; else echo "RESULT RED"; fi
    return "$RED"
}

# ---------------------------------------------------------------- self-test

st_commit() { git -C "$ST" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm "$1"; }

st_case() {
    # st_case <name> <expect: PASS|RED> <rule> <setup...>: setup runs in $ST on a fresh branch.
    local name=$1 expect=$2 rule=$3 out rc=0
    shift 3
    git -C "$ST" checkout -q -B "$name" base
    (cd "$ST" && "$@")
    git -C "$ST" add -A
    st_commit "$name"
    out=$("$SELF" --repo "$ST" --base base --head "$name") || rc=$?
    local got
    got=$(printf '%s\n' "$out" | awk -v r="$rule" '$1 == r {print $2; exit}')
    if [ "$got" = "$expect" ] && { [ "$expect" = RED ] && [ "$rc" -eq 1 ] || [ "$expect" != RED ]; }; then
        printf 'ok   %-22s %s %s\n' "$name" "$rule" "$got"
    else
        printf 'FAIL %-22s %s want %s got %s rc=%s\n%s\n' "$name" "$rule" "$expect" "${got:-none}" "$rc" "$out"
        ST_FAIL=1
    fi
}

self_test() {
    SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
    ST=$(mktemp -d)
    case "$ST" in /tmp/*|"${TMPDIR:-/tmp}"/*) ;; *) echo "mktemp gave $ST" >&2; exit 2 ;; esac
    trap 'rm -rf -- "${ST:?}"' EXIT
    export GIT_CEILING_DIRECTORIES
    GIT_CEILING_DIRECTORIES=$(dirname "$ST")
    ST_FAIL=0
    git -C "$ST" init -q
    mkdir -p "$ST/crates/aprender-decide/src" "$ST/scripts" "$ST/.github/workflows"
    echo 'fn verify() {}' >"$ST/crates/aprender-decide/src/verify.rs"
    echo 'echo hi' >"$ST/scripts/a.sh"
    git -C "$ST" add -A
    st_commit base
    git -C "$ST" branch -q -M base

    st_case clean PASS H1 sh -c 'echo "# docs" > notes.md'
    st_case plant-py-file RED H1 sh -c 'echo "x = 1" > x.py'
    st_case plant-py-line RED H1 sh -c "echo '${PY_WORD}3 run.py' >> scripts/a.sh"
    st_case plant-judge-edit RED H2 sh -c 'echo "// x" >> crates/aprender-decide/src/verify.rs'
    st_case plant-crates-early RED H4 sh -c 'mkdir -p crates/x && echo "[package]" > crates/x/Cargo.toml'
    st_case lt1-own-receipt PASS H4 sh -c "mkdir -p crates/x evidence/laya-train/lt-1 && echo '[package]' > crates/x/Cargo.toml && echo '{\"engine\": \"core\"}' > $LT1_RECEIPT"
    st_case lt1-probe-only PASS H4 sh -c "mkdir -p crates/aprender-train/examples && echo 'fn main() {}' > $LT1_PROBE"
    st_case plant-probe-plus-crate RED H4 sh -c "mkdir -p crates/aprender-train/examples crates/x && echo 'fn main() {}' > $LT1_PROBE && echo '[package]' > crates/x/Cargo.toml"
    st_case plant-probe-lookalike RED H4 sh -c "mkdir -p crates/aprender-train/examples && echo 'fn main() {}' > crates/aprender-train/examples/laya_engine_probe2.rs"
    st_case plant-cli-contract RED H11 sh -c 'mkdir -p contracts && echo "x: 1" > contracts/apr-cli-commands-v1.yaml'
    st_case plant-two-crates RED H11 sh -c 'mkdir -p crates/a crates/b && echo "[package]" > crates/a/Cargo.toml && echo "[package]" > crates/b/Cargo.toml'
    st_case plant-push-trigger RED H11 sh -c 'mkdir -p .github/workflows && printf "on:\n  push:\n  workflow_dispatch:\njobs: {}\n" > .github/workflows/w.yml'
    st_case dispatch-only PASS H11 sh -c 'mkdir -p .github/workflows && printf "on:\n  workflow_dispatch:\njobs: {}\n" > .github/workflows/w.yml'
    st_case plant-big-file RED H12 sh -c 'head -c 1048577 /dev/zero > big.bin'
    st_case plant-dirty RED H15 sh -c "echo 'cargo package ${DIRTY_FLAG}' >> scripts/a.sh"
    st_case plant-ssh RED H15 sh -c "echo '${SSH_WORD}host true' >> scripts/a.sh"
    st_case plant-ssh-chained RED H15 sh -c "echo 'true; sudo ${SSH_WORD}host true' >> scripts/a.sh"
    st_case ssh-in-prose PASS H15 sh -c "echo '# no ${SSH_WORD}line is added here' >> scripts/a.sh"

    # H5 and H10 need LT-1's receipt and LT-9's declaration on the base.
    git -C "$ST" checkout -q base
    mkdir -p "$ST/evidence/laya-train/lt-1" "$ST/evidence/laya-train/lt-9"
    echo '{"engine": "core"}' >"$ST/$LT1_RECEIPT"
    echo '{"seeds": [13, 17, 23]}' >"$ST/$LT9_DECL"
    git -C "$ST" add -A
    st_commit "receipts"
    local nc="$NEW_CRATE"
    st_case crate-core-ok PASS H5 sh -c "mkdir -p $nc/src && echo '[package]' > $nc/Cargo.toml && echo 'use aprender::autograd::Tensor;' > $nc/src/lib.rs"
    st_case plant-other-engine RED H5 sh -c "mkdir -p $nc/src && echo '[package]' > $nc/Cargo.toml && echo 'use aprender_train::autograd::Tensor;' > $nc/src/lib.rs"
    st_case plant-second-tensor RED H5 sh -c "mkdir -p $nc/src && echo '[package]' > $nc/Cargo.toml && printf 'use aprender::autograd::Tensor as T;\npub struct Tensor;\nimpl Tensor { fn backward(&self) {} }\n' > $nc/src/lib.rs"
    st_case outcome-after PASS H10 sh -c "mkdir -p $LT10_DIR && echo '{\"run_started_utc\": \"2999-01-01T00:00:00Z\"}' > $LT10_DIR/outcome.json"
    st_case plant-outcome-before RED H10 sh -c "mkdir -p $LT10_DIR && echo '{\"run_started_utc\": \"2000-01-01T00:00:00Z\"}' > $LT10_DIR/outcome.json"
    st_case plant-two-outcomes RED H10 sh -c "mkdir -p $LT10_DIR && echo '{\"run_started_utc\": \"2999-01-01T00:00:00Z\"}' > $LT10_DIR/outcome.json && cp $LT10_DIR/outcome.json $LT10_DIR/outcome-2.json"

    if [ "$ST_FAIL" -eq 0 ]; then echo "SELF-TEST PASS"; else echo "SELF-TEST FAIL"; exit 1; fi
}

# ---------------------------------------------------------------- main

while [ $# -gt 0 ]; do
    case "$1" in
        --self-test) self_test; exit 0 ;;
        --repo) [ $# -ge 2 ] || usage; REPO=$2; shift 2 ;;
        --base) [ $# -ge 2 ] || usage; BASE=$2; shift 2 ;;
        --head) [ $# -ge 2 ] || usage; HEAD_REF=$2; shift 2 ;;
        *) usage ;;
    esac
done
if [ -z "$REPO" ]; then REPO=$(cd "$(dirname "$0")/.." && pwd); fi
run_rules
