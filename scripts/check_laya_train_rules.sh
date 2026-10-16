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
BUILD_WORD="car""go"   # split so guard_tree files this checker with the no-cargo guards: it runs no build tool
SSH_WORD="ss""h "
PUBLISH_RE='(cargo|make)[[:space:]]+'"publi""sh"
# H15 skips exactly this file by path: it must name
# the forbidden needles to plant them. The needle itself stays a plain substring.
H15_SELF="scripts/check_laya_train_rules.sh"

LT1_RECEIPT="evidence/laya-train/lt-1/receipt.json"
# The ONE path H4 exempts: the LT-1
# proxy-layer probe is measurement, not model code, and must merge before its receipt.
LT1_PROBE="crates/aprender-train/examples/laya_engine_probe.rs"
LT9_DECL="evidence/laya-train/lt-9/declaration.json"
LT10_DIR="evidence/laya-train/lt-10"
NEW_CRATE="crates/aprender-decide-train"
SIZE_CAP=1048576

REPO=""
BASE="origin/main"
HEAD_REF="HEAD"
RANGE=""   # set by run_rules: base...head, or base..head for a shallow merge head
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

changed_names() { g diff --name-only "$RANGE"; }
added_names() { g diff --name-only --diff-filter=A "$RANGE"; }

# Paths whose added lines count as "a Makefile, a justfile, a script or a workflow".
runner_paths() {
    changed_names | awk '/(^|\/)Makefile[^\/]*$|\.mk$|(^|\/)justfile$|\.sh$|\.bash$|^scripts\/|^\.github\/workflows\//'
}

# Added lines (without the leading '+') of runner paths, as "path:text".
added_runner_lines() {
    local p
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        g diff -U0 "$RANGE" -- "$p" \
            | awk -v p="$p" '/^\+\+\+/ {next} /^\+/ {print p ":" substr($0, 2)}'
    done < <(runner_paths)
}

rule_h1() {
    local files lines hits
    files=$(added_names | awk '/\.py$|(^|\/)pyproject\.toml$|(^|\/)uv\.lock$/')
    lines=$(added_runner_lines | awk -v a="$PY_WORD" -v b="$UV_WORD" -v c="$PIP_WORD" \
        'index($0, a) || index($0, b) || index($0, c)')
    hits=$(printf '%s\n%s\n' "$files" "$lines" | awk 'NF' | awk 'NR <= 5' | paste -sd ';' -)
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
        | awk 'NR <= 5' | paste -sd ';' -)
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
    # Model code of this spec: the new crate, or a crates/ path naming laya. A crates/
    # path of other work carried in the same diff (a dependency bump) is not this
    # spec's model code; it is counted in the arm and not judged.
    local exempt all
    all=$(changed_names | awk '/^crates\//' | wc -l)
    touches=$(changed_names | awk -v p="$LT1_PROBE" -v nc="$NEW_CRATE/" \
        '/^crates\// && $0 != p && (index($0, nc) == 1 || /laya/)' | wc -l)
    exempt=$(changed_names | awk -v p="$LT1_PROBE" '$0 == p' | wc -l)
    local arm="$LT1_RECEIPT base=$on_base head=$on_head; laya crates/ paths=$touches of $all crates/ paths; exempt=$exempt ($LT1_PROBE)"
    if [ "$touches" -gt 0 ] && [ "$on_base" = absent ] && [ "$on_head" = absent ]; then
        line H4 RED "touches laya model code under crates/ before LT-1's receipt" "$arm"
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
    # An absent receipt is an empty engine, not a pipefail exit.
    engine=$({ g show "$BASE:$LT1_RECEIPT" 2>/dev/null || true; } \
        | awk -F'"' '!d && /"engine"[[:space:]]*:/ {print $4; d = 1}')
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
        if grep -Eq -- "$other" <<<"$body"; then hits="$hits $f"; fi
        if [ "$engine" = core ]; then
            grep -Eq 'aprender::autograd' <<<"$body" && n_ok=$((n_ok + 1))
        else
            grep -Eq 'aprender_train::autograd|entrenar::autograd' <<<"$body" && n_ok=$((n_ok + 1))
        fi
    done <<<"$files"
    second_tensor=$(g diff -U0 "$RANGE" -- '*.rs' | awk '
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
            && [ -n "$(g ls-tree --name-only "$HEAD_REF" "$LT10_DIR/" | awk '/outcome/')" ]; then
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
        started=$(g show "$HEAD_REF:$f" | awk -F'"' '!d && /"run_started_utc"[[:space:]]*:/ {print $4; d = 1}')
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
        g diff -U0 "$RANGE" -- "$p" | awk -v p="$p" '!d && /^[+-][[:space:]]*publish[[:space:]]*=/ {print p; d = 1}'
    done | paste -sd ';' -)
    new_crates=$(added_names | awk '/^crates\/[^\/]+\/Cargo\.toml$/' | paste -sd ';' -)
    n_new=$(printf '%s' "$new_crates" | awk -F';' '!d && NF {print NF; d = 1} END {if (!NR) print 0}')
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
    hits=$(added_runner_lines | awk -v a="$DIRTY_FLAG" -v b="$SSH_WORD" -v re="$PUBLISH_RE" -v self="$H15_SELF:" \
        'index($0, self) == 1 {next} { body = $0; sub(/^[^:]*:/, "", body) } index(body, a) || index(body, b) || body ~ re' | awk 'NR <= 5' | paste -sd ';' -)
    if [ -n "$hits" ]; then
        line H15 RED "forbidden line added: $hits" "runner lines scanned; path skipped: $H15_SELF"
    else
        line H15 PASS "no dirty-publish, publish or ssh line added" \
            "$(added_runner_lines | wc -l) added runner lines scanned; path skipped: $H15_SELF"
    fi
}

# The rules are this spec's, so they judge only a laya-train diff: a head on a
# laya-train/ branch (the ref named by --head, or GITHUB_HEAD_REF in a pull request
# run), or a diff touching one of the spec's own paths. guard_tree runs this checker
# bare on every pull request; any other diff gets one SCOPE n/a line and passes.
SCOPE_RE='^crates/aprender-decide-train/|^evidence/laya-train/|^scripts/laya_train/|^scripts/check_laya_train_rules\.sh$|^crates/aprender-train/examples/laya_engine_probe|^docs/specifications/APR-LAYA-TRAIN-001|^crates/.*laya'

in_scope() {
    local b n hit
    b=$(g rev-parse -q --symbolic-full-name "$HEAD_REF" 2>/dev/null || true)
    b=${b#refs/heads/}
    b=${b#refs/remotes/origin/}
    if [[ "$b" == laya-train/* ]]; then
        line SCOPE in "branch $b" "head ref read"
        return 0
    fi
    if [[ "${GITHUB_HEAD_REF:-}" == laya-train/* ]]; then
        line SCOPE in "branch $GITHUB_HEAD_REF" "GITHUB_HEAD_REF read"
        return 0
    fi
    n=$(changed_names | awk 'END {print NR}')
    # The judge crate's own laya-named files (pack_laya.rs, src/laya/) are not laya-train
    # model code: a diff there is judged by its branch name (laya-train/), never by `laya`
    # in a path. H2 still reds any judge edit on a laya-train/ branch or beside a spec path.
    hit=$(changed_names | awk '$0 !~ /^crates\/aprender-decide\/.*laya/' \
        | SCOPE_RE="$SCOPE_RE" awk 'BEGIN {re = ENVIRON["SCOPE_RE"]} !d && $0 ~ re {print; d = 1}')
    if [ -n "$hit" ]; then
        line SCOPE in "path $hit" "$n changed paths scanned"
        return 0
    fi
    line SCOPE n/a "not a laya-train diff" "branch ${b:-detached}, GITHUB_HEAD_REF ${GITHUB_HEAD_REF:-unset}, $n changed paths scanned, none under the spec's paths"
    return 1
}

run_rules() {
    g rev-parse -q --verify "$BASE^{commit}" >/dev/null || { echo "base $BASE not found" >&2; exit 2; }
    g rev-parse -q --verify "$HEAD_REF^{commit}" >/dev/null || { echo "head $HEAD_REF not found" >&2; exit 2; }
    # A pull request's CI clone is shallow: HEAD is the merge commit (or, in the merge
    # queue, the squashed commit) and the base is its first parent, fetched alone, so
    # `base...head` has no merge base. Then, and only when the base IS head's first
    # parent (read from the raw commit, which a shallow graft does not hide), the tree
    # diff base..head is the pull request's diff.
    # Anything else has no defined diff and stops (exit 2).
    local how=merge-base p1
    RANGE="$BASE...$HEAD_REF"
    if ! g merge-base "$BASE" "$HEAD_REF" >/dev/null 2>&1; then
        p1=$(g cat-file -p "$(g rev-parse "$HEAD_REF")" | awk '!h && /^parent /{n++; if (n == 1) p = $2} /^$/{h = 1} END{if (n >= 1) print p}')
        if [ -n "$p1" ] && [ "$p1" = "$(g rev-parse "$BASE^{commit}")" ]; then
            RANGE="$BASE..$HEAD_REF"
            how=first-parent
        else
            echo "no merge base for $BASE...$HEAD_REF and $BASE is not the first parent of the head" >&2
            exit 2
        fi
    fi
    printf 'laya-train rules | base %s | head %s | diff %s\n' \
        "$(g rev-parse --short "$BASE")" "$(g rev-parse --short "$HEAD_REF")" "$how"
    if ! in_scope; then
        echo "RESULT PASS"
        return 0
    fi
    rule_h1; rule_h2; rule_h4; rule_h5; rule_h10; rule_h11; rule_h12; rule_h15
    if [ "$RED" -eq 0 ]; then echo "RESULT PASS"; else echo "RESULT RED"; fi
    return "$RED"
}

# ---------------------------------------------------------------- self-test

st_commit() { git -C "$ST" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm "$1"; }

st_case() {
    # st_case <name> <expect: PASS|RED> <rule> <setup...>: setup runs in $ST on a fresh branch.
    # The branch is laya-train/<name>, in scope by its name, unless ST_PREFIX says otherwise.
    local name=$1 expect=$2 rule=$3 out rc=0 br
    shift 3
    ST_ROW=$name
    br="${ST_PREFIX:-laya-train/}$name"
    git -C "$ST" checkout -q -B "$br" base
    (cd "$ST" && "$@")
    git -C "$ST" add -A
    st_commit "$name"
    out=$("$SELF" --repo "$ST" --base base --head "$br") || rc=$?
    local got
    got=$(printf '%s\n' "$out" | awk -v r="$rule" '!d && $1 == r {print $2; d = 1}')
    if [ "${got:-none}" = "$expect" ] && { [ "$expect" = RED ] && [ "$rc" -eq 1 ] || [ "$expect" != RED ]; }; then
        printf 'ok   %-22s %s %s\n' "$name" "$rule" "$got"
    else
        printf 'FAIL %-22s %s want %s got %s rc=%s\n%s\n' "$name" "$rule" "$expect" "${got:-none}" "$rc" "$out"
        ST_FAIL=1
    fi
}

st_shallow() {
    # st_shallow <name> <expect> <merge|squash> <pr-setup>: main moves on (adding a .py the
    # PR does not own), the PR runs <pr-setup> and lands as a merge commit (a pull request
    # run) or a single squashed commit (a merge queue run), and the check runs on a shallow
    # clone of that head.
    local name=$1 expect=$2 how=$3 setup=$4 out rc=0 got p1 sc
    ST_ROW=$name
    git -C "$ST" checkout -q -B "laya-train/$name-pr" base
    (cd "$ST" && sh -c "$setup")
    git -C "$ST" add -A; st_commit "$name pr"
    git -C "$ST" checkout -q -B "laya-train/$name-main" base
    echo "y = 1" >"$ST/main_only.py"
    git -C "$ST" add -A; st_commit "$name main"
    if [ "$how" = squash ]; then
        git -C "$ST" -c user.name=t -c user.email=t@t merge -q --squash "laya-train/$name-pr" >/dev/null
        st_commit "$name squash"
    else
        git -C "$ST" -c user.name=t -c user.email=t@t merge -q --no-ff --no-edit "laya-train/$name-pr" >/dev/null
    fi
    sc="$ST/shallow-$name"
    # A runner whose git speaks protocol v0 refuses to fetch an unadvertised sha (the
    # first parent is not a branch tip); allow it on the throwaway source repo.
    git -C "$ST" config uploadpack.allowReachableSHA1InWant true
    git clone -q --no-local --depth 1 --branch "laya-train/$name-main" "$ST" "$sc"
    p1=$(git -C "$sc" cat-file -p HEAD | awk '!d && /^parent /{print $2; d = 1}')
    git -C "$sc" fetch -q --depth 1 origin "$p1"
    out=$("$SELF" --repo "$sc" --base "$p1" --head HEAD) || rc=$?
    got=$(printf '%s\n' "$out" | awk '!d && $1 == "H1" {print $2; d = 1}')
    if [ "$got" = "$expect" ] && printf '%s\n' "$out" | awk '/diff first-parent/ {f=1} END {exit !f}'; then
        printf 'ok   %-22s %s %s\n' "$name" H1 "$got"
    else
        printf 'FAIL %-22s %s want %s got %s rc=%s\n%s\n' "$name" H1 "$expect" "${got:-none}" "$rc" "$out"
        ST_FAIL=1
    fi
    git -C "$ST" checkout -q base
}

self_test() {
    SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
    ST=$(mktemp -d)
    case "$ST" in /tmp/*|"${TMPDIR:-/tmp}"/*) ;; *) echo "mktemp gave $ST" >&2; exit 2 ;; esac
    # A self-test that dies mid-row names the row and how it ended; a runner kill or
    # a set -e exit otherwise leaves only the rows before it.
    ST_ROW=setup
    trap 'st_rc=$?; [ "$st_rc" -eq 0 ] || [ "$ST_ROW" = summary ] || printf "SELF-TEST DIED in row %s rc=%s\n" "$ST_ROW" "$st_rc"; rm -rf -- "${ST:?}"' EXIT
    trap 'printf "SELF-TEST KILLED by SIGTERM in row %s\n" "$ST_ROW"; exit 143' TERM
    trap 'printf "SELF-TEST KILLED by SIGHUP in row %s\n" "$ST_ROW"; exit 129' HUP
    trap 'printf "SELF-TEST KILLED by SIGINT in row %s\n" "$ST_ROW"; exit 130' INT
    # A pull request run's own branch name must not decide a fixture's scope.
    unset GITHUB_HEAD_REF
    export GIT_CEILING_DIRECTORIES
    GIT_CEILING_DIRECTORIES=$(dirname "$ST")
    ST_FAIL=0
    git -C "$ST" init -q
    mkdir -p "$ST/.git/info" && echo "shallow-*/" >>"$ST/.git/info/exclude"
    mkdir -p "$ST/crates/aprender-decide/src" "$ST/scripts" "$ST/.github/workflows"
    echo 'fn verify() {}' >"$ST/crates/aprender-decide/src/verify.rs"
    echo 'echo hi' >"$ST/scripts/a.sh"
    git -C "$ST" add -A
    st_commit base
    git -C "$ST" branch -q -M base

    st_case clean PASS H1 sh -c 'echo "# docs" > notes.md'

    # Scope: another branch's diff is not judged (the H11 mutant passes there), but it
    # is judged once it touches a path of the spec (the H4 mutant reds there).
    ST_PREFIX=other/
    st_case scope-out-push-trigger n/a SCOPE sh -c 'mkdir -p .github/workflows && printf "on:\n  push:\njobs: {}\n" > .github/workflows/w.yml'
    st_case scope-out-no-rules none H11 sh -c 'mkdir -p .github/workflows && printf "on:\n  push:\njobs: {}\n" > .github/workflows/w.yml'
    st_case scope-in-by-path RED H4 sh -c "mkdir -p $NEW_CRATE && echo '[package]' > $NEW_CRATE/Cargo.toml"
    st_case scope-in-by-path-line in SCOPE sh -c "mkdir -p $NEW_CRATE && echo '[package]' > $NEW_CRATE/Cargo.toml"
    # laya model code outside the new crate is in scope on any branch, and judged by H4.
    st_case scope-in-by-laya-crate-path RED H4 sh -c "mkdir -p crates/aprender-train/src/laya && echo 'pub fn f() {}' > crates/aprender-train/src/laya/mod.rs"
    st_case scope-in-by-laya-crate-path-line in SCOPE sh -c "mkdir -p crates/aprender-train/src/laya && echo 'pub fn f() {}' > crates/aprender-train/src/laya/mod.rs"
    # The judge crate's laya-named files are out of scope on a non-laya-train branch...
    st_case scope-out-judge-crate-laya-name n/a SCOPE sh -c "mkdir -p crates/aprender-decide/src/laya && echo 'pub fn f() {}' > crates/aprender-decide/src/laya/x.rs"
    # ...but a spec path beside them still brings the whole diff in, and H2 reds the judge edit.
    st_case scope-in-judge-crate-plus-spec-path RED H2 sh -c "mkdir -p $NEW_CRATE crates/aprender-decide/src/laya && echo '[package]' > $NEW_CRATE/Cargo.toml && echo 'pub fn f() {}' > crates/aprender-decide/src/laya/x.rs"
    # SCOPE_RE reaches awk through ENVIRON, so its "\." stays a literal dot.
    st_case scope-out-dot-lookalike n/a SCOPE sh -c "mkdir -p scripts && echo x > scripts/check_laya_train_rulesXsh"
    ST_PREFIX=
    st_case plant-py-file RED H1 sh -c 'echo "x = 1" > x.py'
    st_case plant-py-line RED H1 sh -c "echo '${PY_WORD}3 run.py' >> scripts/a.sh"
    st_case plant-judge-edit RED H2 sh -c 'echo "// x" >> crates/aprender-decide/src/verify.rs'
    st_case plant-crates-early RED H4 sh -c "mkdir -p $NEW_CRATE && echo '[package]' > $NEW_CRATE/Cargo.toml"
    st_case plant-laya-path-early RED H4 sh -c "mkdir -p crates/aprender-train/src/laya && echo 'pub fn f() {}' > crates/aprender-train/src/laya/mod.rs"
    st_case other-crate-not-judged PASS H4 sh -c 'mkdir -p crates/aprender-db && echo "[package]" > crates/aprender-db/Cargo.toml'
    ST_PREFIX=other/
    st_case carried-bump-not-judged PASS H4 sh -c "mkdir -p crates/aprender-db && echo '[package]' > crates/aprender-db/Cargo.toml && echo '# carried' >> $H15_SELF"
    ST_PREFIX=
    st_case lt1-own-receipt PASS H4 sh -c "mkdir -p $NEW_CRATE evidence/laya-train/lt-1 && echo '[package]' > $NEW_CRATE/Cargo.toml && echo '{\"engine\": \"core\"}' > $LT1_RECEIPT"
    st_case lt1-probe-only PASS H4 sh -c "mkdir -p crates/aprender-train/examples && echo 'fn main() {}' > $LT1_PROBE"
    st_case plant-probe-plus-crate RED H4 sh -c "mkdir -p crates/aprender-train/examples $NEW_CRATE && echo 'fn main() {}' > $LT1_PROBE && echo '[package]' > $NEW_CRATE/Cargo.toml"
    st_case plant-probe-lookalike RED H4 sh -c "mkdir -p crates/aprender-train/examples && echo 'fn main() {}' > crates/aprender-train/examples/laya_engine_probe2.rs"
    st_case plant-cli-contract RED H11 sh -c 'mkdir -p contracts && echo "x: 1" > contracts/apr-cli-commands-v1.yaml'
    # SIGPIPE falsifier: the publish line comes first and the diff runs ~240 KB past it, far
    # beyond a pipe buffer. A reader that exits on the first match kills git diff with
    # SIGPIPE every time, so the checker dies rc=141 before its H11 line; one that reads to
    # EOF prints H11 RED.
    st_case plant-publish-big-diff RED H11 sh -c "mkdir -p crates/aprender-db && { echo 'publish = false'; awk 'BEGIN {for (i = 0; i < 20000; i++) print \"# pad \" i}'; } > crates/aprender-db/Cargo.toml"
    st_case plant-two-crates RED H11 sh -c 'mkdir -p crates/a crates/b && echo "[package]" > crates/a/Cargo.toml && echo "[package]" > crates/b/Cargo.toml'
    st_case plant-push-trigger RED H11 sh -c 'mkdir -p .github/workflows && printf "on:\n  push:\n  workflow_dispatch:\njobs: {}\n" > .github/workflows/w.yml'
    st_case dispatch-only PASS H11 sh -c 'mkdir -p .github/workflows && printf "on:\n  workflow_dispatch:\njobs: {}\n" > .github/workflows/w.yml'
    st_case plant-big-file RED H12 sh -c 'head -c 1048577 /dev/zero > big.bin'
    st_case plant-dirty RED H15 sh -c "echo '${BUILD_WORD} package ${DIRTY_FLAG}' >> scripts/a.sh"
    st_case plant-ssh RED H15 sh -c "echo '${SSH_WORD}host true' >> scripts/a.sh"
    st_case plant-ssh-chained RED H15 sh -c "echo 'true; sudo ${SSH_WORD}host true' >> scripts/a.sh"
    st_case plant-ssh-yaml-run RED H15 sh -c "mkdir -p .github/workflows && echo '      run: ${SSH_WORD}host true' > .github/workflows/w.yml"
    st_case plant-ssh-then RED H15 sh -c "echo 'if true; then ${SSH_WORD}host true; fi' >> scripts/a.sh"
    st_case plant-ssh-do RED H15 sh -c "echo 'for h in a; do ${SSH_WORD}\$h true; done' >> scripts/a.sh"
    st_case plant-ssh-else RED H15 sh -c "echo 'if false; then :; else ${SSH_WORD}host true; fi' >> scripts/a.sh"
    st_case plant-ssh-brace RED H15 sh -c "echo '{ ${SSH_WORD}host true; }' >> scripts/a.sh"
    st_case plant-ssh-env RED H15 sh -c "echo 'env A=1 ${SSH_WORD}host true' >> scripts/a.sh"
    st_case plant-ssh-xargs RED H15 sh -c "echo 'echo h | xargs -I{} ${SSH_WORD}{} true' >> scripts/a.sh"
    st_case plant-ssh-nohup RED H15 sh -c "echo 'nohup ${SSH_WORD}host true &' >> scripts/a.sh"
    st_case plant-ssh-timeout RED H15 sh -c "echo 'timeout 5 ${SSH_WORD}host true' >> scripts/a.sh"
    st_case plant-ssh-abspath RED H15 sh -c "echo '/usr/bin/${SSH_WORD}host true' >> scripts/a.sh"
    st_case h15-self-path PASS H15 sh -c "echo '# ${SSH_WORD}is a needle here' >> $H15_SELF"
    st_case plant-ssh-self-lookalike RED H15 sh -c "echo '${SSH_WORD}host true' >> scripts/check_laya_train_rules2.sh"

    # A pull request's CI clone: a depth-1 merge head plus its first parent fetched
    # alone, so there is no merge base. Only the pull request's side is judged.
    st_shallow plant-py-shallow RED merge 'echo "x = 1" > x.py'
    st_shallow clean-shallow PASS merge 'echo "# docs" > notes.md'
    st_shallow plant-py-squash RED squash 'echo "x = 1" > x.py'
    st_shallow clean-squash PASS squash 'echo "# docs" > notes.md'

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

    ST_ROW=summary
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
