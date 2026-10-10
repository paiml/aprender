#!/usr/bin/env bash
# workspace_test_nightly.sh -- main's workspace tests, at night, on a clean-room host (#5002).
#
# WHY. A push to main reuses the merge-queue result for workspace-test, and the merge queue
# tests the crates a PR touches. #4912 broke aprender-core tests without touching
# aprender-core, so main's tree was red for ~46h and no run said so (#5001). This script runs,
# on main's head, test commands of ci/sections.yml's workspace-test-shard section: the FULL
# tier's workspace lib line, GPU crates, compute crate and explicit commands, and the QUICK
# tier's `--lib --tests` line (lane `tests`).
#
# The full tier alone is not enough. Its lib line is `--lib` only, and no explicit command
# names the integration targets #5001 broke (mutation_testing_tests,
# falsification_measurement_tests). Measured: those rows were green at #4912's commit, and the
# quick-tier line, scoped to aprender-core, failed the same five tests #5001 names.
#
# SCOPE of the tests lane (quorum q-5002d, Q1 KT). CI binds that line's `$pkgs` to the crates
# a PR touches. Here it is bound to TESTS_PKGS (aprender-core, the crate #5002 names), printed
# on every line of the lane. Widening it to the workspace is ticket #5027, after one measured
# workspace run.
#
# LAB, never a gate (operator C324). Only .github/workflows/workspace-test-nightly.yml runs it.
# That workflow is not "CI" and has no check named "workspace-test" or "ci / gate", and
# scripts/release/nightly_train.sh does not list it, so no release or merge reader sees it. A
# night that is not green opens or updates one ticket (workspace_test_nightly_ticket.sh).
#
# NO PYTHON (operator C301). The commands run on the runner host, not in the sovereign-ci
# container (the deep-nightly pattern). A difference that only the host has can make a red
# night that CI would not have; that opens the ticket like any other red. One such difference
# is closed (q-5002d, Q2 A): CI's container mounts no Hugging Face cache, and
# rosetta_dangerous_integration.rs reads $HF_HOME, else ~/.cache/huggingface, so the tests lane
# runs with HF_HOME set to an empty directory of its own (ADDED below). The other lanes run no
# integration target and get nothing added.
#
# DRIFT (quorum q-5002c, Q2). Before anything runs, ROWS and the env tables below are compared
# with the section. Any difference is NOT MEASURED (exit 2) and nothing runs:
#   (a) Set equality. Every non-comment line of the section that runs `cargo <subcommand>` or
#       ci_run_explicit_test_commands.sh is a row: RUN (run here) or SKIP (with its reason),
#       as many times as it appears there, and every row appears there.
#   (b) Env. For each RUN row, the -e set of the docker run that runs it ("host" outside one)
#       equals the row's ENVSET, and every token has one disposition: APPLY (exported here),
#       PASS (the job's own value) or OMIT (container-only, with its reason).
#   (c) The compute row runs as its CI text, pass rule included (`bash -c "$text"`).
#   (d) Comment lines (first non-blank character #) never count, as a command or as an env token.
#   (e) Mounts (q-5002d). For each RUN row, the container side of every -v of its docker run
#       equals the row's MOUNTS, each with one disposition. A new mount (a model cache, say)
#       is a difference the env check cannot see, so it stops the run too.
# A token that CI does not pass and this script adds (ADDED) must not be in the row's -e set.
# The workflow_dispatch refinement base pin must also appear verbatim. It runs first, as it does
# in CI, because lint_passes_on_real_contracts reads origin/main when CI is set (aprender-4502).
#
# EXIT  0 every command ran and passed · 1 a command ran and failed · 2 not measured (drift, a
#       missing tool, CI not "true", usage) · --self-test: 0 every case held, 1 otherwise
#
#   bash scripts/ci/workspace_test_nightly.sh --self-test
#   bash scripts/ci/workspace_test_nightly.sh --drift [--sections FILE]
#   bash scripts/ci/workspace_test_nightly.sh --run lib|gpu|compute|explicit|tests|all [--sections FILE]
#   bash scripts/ci/workspace_test_nightly.sh --run LANE --only-package PKG     (a proof, never a night)
#
# --only-package PKG narrows a run to one package for a pre-merge red/green proof: lib runs
# `-p PKG --lib`, gpu and compute run only for their own packages, explicit runs only the
# fragments that name `-p PKG`, tests binds `$pkgs` to ` -p PKG`. Every line it prints says
# PROOF-SCOPE, and the self-test fails
# if the nightly workflow ever passes it. Commands run in the current directory; --sections
# defaults to ./ci/sections.yml there.
set -euo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
SECTION=workspace-test-shard
BASE_PIN='git fetch --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main'
LANES='lib gpu compute explicit tests'
# The tests row's `$pkgs` (q-5002d Q1 KT). CI builds it from the crates a PR touches; here it is
# the crate #5002 names. Widening it to the workspace is #5027, after one measured run.
TESTS_PKGS=' -p aprender-core'

# ROWS: a header line, RUN <id> or SKIP <reason>, then the section's line, indented 4 spaces.
rows() {
    cat << 'ROWS'
RUN lib
    cargo nextest run --profile ci --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute --partition "hash:${SHARD}/${SHARDS}"
RUN gpu
    cargo nextest run --profile ci -p aprender-gpu -p aprender-cuda-edge --lib
RUN compute
    bash -c 'cargo test -p aprender-compute --lib 2>&1 | tee /tmp/compute-test.log; grep -q "test result: ok\." /tmp/compute-test.log && ! grep -q "test result: FAILED" /tmp/compute-test.log'
RUN explicit-list
    bash scripts/ci_run_explicit_test_commands.sh --list ci/explicit-test-commands.d
RUN explicit
    bash scripts/ci_run_explicit_test_commands.sh --run ci/explicit-test-commands.d --shard "${SHARD}/${SHARDS}"
SKIP the Σ-executed listing: it lists the tests a step owes and runs none
    cargo nextest list --profile ci --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute --message-format json > /sigma/lib.list.json
SKIP the Σ-executed listing: it lists the tests a step owes and runs none
    cargo nextest list --profile ci -p aprender-gpu -p aprender-cuda-edge --lib --message-format json > /sigma/gpu.list.json
SKIP the Σ-executed listing: it lists the tests a step owes and runs none
    cargo test -p aprender-compute --lib -- --list > /sigma/compute.list.txt
SKIP the Σ-executed listing: it lists the tests a step owes and runs none
    cargo test -p aprender-compute --lib -- --list --ignored > /sigma/compute.ignored.txt'
SKIP the Σ-executed listing: it lists the tests a step owes and runs none
    bash scripts/ci_run_explicit_test_commands.sh --list ci/explicit-test-commands.d > "$sig/explicit.list.txt"
SKIP shard 3's example build: it compiles the examples and runs no test
    cargo build --examples --workspace --keep-going
RUN tests
    bash -c "cargo nextest run --profile ci --lib --tests$pkgs"
SKIP the tree-reader filterset: a narrowing of targets the tests row runs (its scope: #5027; the registry's gap: #5028)
    cargo nextest run --profile ci $pkgs --lib --tests -E "$EXPR"
SKIP the quick tier's Σ-executed listing: it runs no test
    if [ -n "$SEL" ]; then cargo nextest list --profile ci --lib --tests $SEL --message-format json > /sigma/quick-sel.list.json; fi
SKIP the quick tier's Σ-executed listing: it runs no test
    cargo nextest list --profile ci $TREE --lib --tests -E "$EXPR" --message-format json > /sigma/quick-tree.list.json'
SKIP the quick tier's over-the-cap type check: it runs no test
    cargo check --workspace --all-targets --locked
ROWS
}

# What each RUN row drops from its CI text: the shard argument (the run here is whole).
declare -A STRIP=(
    [lib]=' --partition "hash:${SHARD}/${SHARDS}"'
    [explicit]=' --shard "${SHARD}/${SHARDS}"'
)

# Each RUN row's -e set, exactly as its docker run in the section has it.
GH='CI GITHUB_ACTIONS GITHUB_REF GITHUB_SHA GITHUB_REPOSITORY GITHUB_RUN_ID GITHUB_EVENT_NAME GITHUB_WORKFLOW'
BUILD='CARGO_TARGET_DIR=/workspace/target RUSTC_WRAPPER=/usr/local/cargo/bin/sccache SCCACHE_DIR=/sccache SCCACHE_CACHE_SIZE=50G CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8'
DEBUG='CARGO_PROFILE_TEST_DEBUG=line-tables-only CARGO_PROFILE_DEV_DEBUG=line-tables-only'
SAFE='GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/workspace'
declare -A ENVSET=(
    [lib]="$SAFE $GH $BUILD SHARD SHARDS $DEBUG"
    [gpu]="$GH $BUILD $DEBUG"
    [compute]="$GH $BUILD"
    [explicit-list]="host"
    [explicit]="$GH $BUILD SHARD SHARDS"
    [tests]="$SAFE $GH $BUILD $DEBUG CARGO_TERM_COLOR=never"
)

# Each RUN row's mounts: the container side of every -v of its docker run ("host" outside one).
VOLS='/workspace /usr/local/cargo/registry /workspace/target /sccache'
declare -A MOUNTS=(
    [lib]=$VOLS
    [gpu]=$VOLS
    [compute]=$VOLS
    [explicit-list]=host
    [explicit]=$VOLS
    [tests]=$VOLS
)

# What this script sets for a row that CI does not pass (names only; the value is made per run).
declare -A ADDED=(
    [tests]=HF_HOME
)

# One disposition per token: APPLY (exported for the row), PASS (the job's own value), OMIT
# (container-only), HOST (the row runs outside any container); for a mount, MAP (what stands
# for it here) or OMIT; ADDED (set here, never passed by CI).
dispositions() {
    cat << 'DISP'
CARGO_INCREMENTAL=0 APPLY as in CI
CARGO_TERM_COLOR=never APPLY as in CI
CARGO_BUILD_JOBS=8 APPLY as in CI
CARGO_PROFILE_TEST_DEBUG=line-tables-only APPLY as in CI
CARGO_PROFILE_DEV_DEBUG=line-tables-only APPLY as in CI
CI PASS the runner sets it in this job; --run refuses unless it is "true"
GITHUB_ACTIONS PASS the runner sets it in this job; -e only copies it into the container
GITHUB_REF PASS the runner sets it in this job; -e only copies it into the container
GITHUB_SHA PASS the runner sets it in this job; -e only copies it into the container
GITHUB_REPOSITORY PASS the runner sets it in this job; -e only copies it into the container
GITHUB_RUN_ID PASS the runner sets it in this job; -e only copies it into the container
GITHUB_EVENT_NAME PASS the runner sets it in this job; -e only copies it into the container
GITHUB_WORKFLOW PASS the runner sets it in this job; -e only copies it into the container
CARGO_TARGET_DIR=/workspace/target OMIT a path inside the container; this job builds in its own checkout's target/, which the checkout cleans
RUSTC_WRAPPER=/usr/local/cargo/bin/sccache OMIT the image's compiler cache; it changes build time, not what a test sees
SCCACHE_DIR=/sccache OMIT the compiler cache's mount inside the container
SCCACHE_CACHE_SIZE=50G OMIT the compiler cache's size
GIT_CONFIG_COUNT=1 OMIT safe.directory for a root container over a host-owned /workspace; this job's user owns its checkout
GIT_CONFIG_KEY_0=safe.directory OMIT as GIT_CONFIG_COUNT
GIT_CONFIG_VALUE_0=/workspace OMIT as GIT_CONFIG_COUNT
SHARD OMIT CI splits the row over 3 shards; here it runs whole, and STRIP drops the argument that reads it
SHARDS OMIT as SHARD
host HOST the row runs on the runner in CI too, outside any container
/workspace MAP the job's checkout, the current directory here
/usr/local/cargo/registry MAP the runner user's own cargo registry
/workspace/target MAP the checkout's own target/ (CARGO_TARGET_DIR is OMIT)
/sccache OMIT the compiler cache's directory; no compiler cache runs here (RUSTC_WRAPPER is OMIT)
HF_HOME ADDED an empty directory of the run's own: CI's container mounts no Hugging Face cache (MOUNTS), and the host user's cache made rosetta_dangerous red where CI skips it (q-5002d Q2 A)
DISP
}

# The section's command lines, as "CMD<TAB>env<TAB>text<TAB>mounts", then FOUND and PIN counts.
# shellcheck disable=SC2016 # awk program, not shell
EXTRACT='
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
BEGIN { sq = sprintf("%c", 39) }
/^jobs:[ \t]*$/ { injobs = 1; insec = 0; next }
/^[^ \t#]/ { injobs = 0; insec = 0 }
injobs && /^  [A-Za-z0-9_-]+:[ \t]*$/ {
    insec = (trim($0) == ENVIRON["WTN_SECTION"] ":"); if (insec) found++
    blk = 0; cmd = 0; env = ""; vols = ""; next
}
!insec { next }
/^[ \t]*#/ { next }
{
    t = trim($0)
    if (t ~ /^- (name|uses):/) { blk = 0; cmd = 0; env = ""; vols = "" }
    if (!cmd && t ~ /(^|[ \t])docker run( |$)/) { blk = 1; env = ""; vols = "" }
    if (blk && !cmd) {
        n = split(t, f, /[ \t]+/)
        for (i = 1; i <= n; i++) {
            if (f[i] == "-e" && i < n) env = env " " f[i + 1]
            # A mount is its container side: the host side holds runner paths, and what stands
            # for it here is a disposition in the table.
            else if (f[i] == "-v" && i < n) { v = f[i + 1]; gsub(/"/, "", v); sub(/^[^:]*:/, "", v); vols = vols " " v }
            # A joined or long form neither case reads is a token no table holds, so it drifts.
            else if (f[i] ~ /^-e=/ || f[i] ~ /^--env(-file)?(=|$)/) env = env " unparsed:" f[i]
            else if (f[i] ~ /^-v=/ || f[i] ~ /^--(volume|mount|tmpfs)(=|$)/) vols = vols " unparsed:" f[i]
        }
        if (index(t, "\"$IMAGE\"")) { cmd = 1; q = 0 }
    }
    if (t == ENVIRON["WTN_BASE_PIN"]) pin++
    # A step name is a label, not a command.
    if (t !~ /^(- )?name:/ && (t ~ /(^|[^A-Za-z0-9_.\/-])cargo +[a-z]/ || index(t, "ci_run_explicit_test_commands.sh")))
        print "CMD\t" (cmd ? (env == "" ? "none" : substr(env, 2)) : "host") "\t" t "\t" (cmd ? (vols == "" ? "none" : substr(vols, 2)) : "host")
    if (cmd) {
        q += split(t, parts, sq) - 1
        if (t !~ /\\$/ && q % 2 == 0) { blk = 0; cmd = 0; env = ""; vols = "" }
    }
}
END { print "FOUND\t" (found + 0); print "PIN\t" (pin + 0) }
'

sorted() { tr ' ' '\n' | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//'; }

declare -a KIND KEY TEXT
declare -A RUNTEXT DISP
load_tables() {
    local h t i=0
    KIND=() KEY=() TEXT=() RUNTEXT=() DISP=()
    while IFS= read -r h; do
        IFS= read -r t || { echo "TABLE: row '$h' has no text line"; return 2; }
        case "$t" in "    "?*) t=${t:4} ;; *) echo "TABLE: row '$h' text is not indented 4 spaces"; return 2 ;; esac
        case "$h" in
            "RUN "?*) KIND[i]=RUN; KEY[i]=${h#RUN } ;;
            "SKIP "?*) KIND[i]=SKIP; KEY[i]=${h#SKIP } ;;
            *) echo "TABLE: bad row header '$h'"; return 2 ;;
        esac
        TEXT[i]=$t
        if [ "${KIND[i]}" = RUN ]; then RUNTEXT[${KEY[i]}]=$t; fi
        i=$((i + 1))
    done < <(rows)
    while read -r tok d rest; do
        [ -z "${DISP[$tok]:-}" ] || { echo "TABLE: token $tok has two dispositions"; return 2; }
        DISP[$tok]="$d $rest"
    done < <(dispositions)
}

# table_check: the tables agree with themselves (one disposition per used token, none unused,
# APPLY carries a value, every STRIP is a suffix of its row, every lane has its rows, an ADDED
# name is not in its row's CI env set, the tests row holds `$pkgs` once).
table_check() {
    local id tok t bad=0
    declare -A used=()
    for id in "${!ENVSET[@]}"; do
        [ -n "${RUNTEXT[$id]:-}" ] || { echo "TABLE: ENVSET names $id, which is not a RUN row"; bad=1; }
        for tok in ${ENVSET[$id]}; do
            used[$tok]=1
            case "${DISP[$tok]:-}" in
                "APPLY "*) [[ $tok == *=* ]] || { echo "TABLE: APPLY token $tok has no value"; bad=1; } ;;
                "PASS "* | "OMIT "?* | "HOST "*) ;;
                *) echo "TABLE: env token $tok ($id) has no env disposition (APPLY, PASS, OMIT or HOST)"; bad=1 ;;
            esac
        done
    done
    for id in "${!MOUNTS[@]}"; do
        [ -n "${RUNTEXT[$id]:-}" ] || { echo "TABLE: MOUNTS names $id, which is not a RUN row"; bad=1; }
        for tok in ${MOUNTS[$id]}; do
            used[$tok]=1
            case "$tok:${DISP[$tok]:-}" in
                /*:"MAP "?* | /*:"OMIT "?* | host:"HOST "*) ;;
                *) echo "TABLE: mount $tok ($id) has no mount disposition (MAP or OMIT)"; bad=1 ;;
            esac
        done
    done
    for id in "${!ADDED[@]}"; do
        [ -n "${RUNTEXT[$id]:-}" ] || { echo "TABLE: ADDED names $id, which is not a RUN row"; bad=1; }
        for tok in ${ADDED[$id]}; do
            used[$tok]=1
            case "${DISP[$tok]:-}" in "ADDED "?*) ;; *) echo "TABLE: added $tok ($id) has no ADDED disposition"; bad=1 ;; esac
            # CI passing it would make it CI's value, not one this script adds.
            for t in ${ENVSET[$id]:-}; do [ "${t%%=*}" != "$tok" ] || { echo "TABLE: $tok is ADDED for $id and also in its CI env set"; bad=1; }; done
        done
    done
    for tok in "${!DISP[@]}"; do [ -n "${used[$tok]:-}" ] || { echo "TABLE: disposition for $tok, which no row uses"; bad=1; }; done
    for id in "${!RUNTEXT[@]}"; do
        [ -n "${ENVSET[$id]:-}" ] || { echo "TABLE: RUN row $id has no ENVSET"; bad=1; }
        [ -n "${MOUNTS[$id]:-}" ] || { echo "TABLE: RUN row $id has no MOUNTS"; bad=1; }
    done
    for id in "${!STRIP[@]}"; do
        [[ ${RUNTEXT[$id]:-} == *"${STRIP[$id]}" ]] || { echo "TABLE: STRIP for $id is not a suffix of its row"; bad=1; }
    done
    for id in $LANES explicit-list; do [ -n "${RUNTEXT[$id]:-}" ] || { echo "TABLE: no RUN row $id"; bad=1; }; done
    # The tests row is CI's text with `$pkgs` left as written; the script binds it once.
    t=${RUNTEXT[tests]:-}
    [[ $t == *'$pkgs'* && ${t#*'$pkgs'} != *'$pkgs'* ]] || { echo "TABLE: the tests row does not hold \$pkgs exactly once"; bad=1; }
    [[ $TESTS_PKGS =~ ^(\ -p\ [a-z0-9_-]+)+$ ]] || { echo "TABLE: TESTS_PKGS '$TESTS_PKGS' is not ' -p CRATE'..."; bad=1; }
    return $((bad * 2))
}

# drift FILE: 0 when the tables equal the section (a)-(e), else 2 with one DRIFT line per difference.
drift() {
    local file=$1 out found pin id t n want got bad=0
    load_tables || return 2
    table_check || return 2
    [ -r "$file" ] || { echo "DRIFT: cannot read $file"; return 2; }
    out=$(WTN_SECTION=$SECTION WTN_BASE_PIN=$BASE_PIN awk "$EXTRACT" "$file") || { echo "DRIFT: awk failed on $file"; return 2; }
    found=$(awk -F'\t' '$1 == "FOUND" {print $2}' <<< "$out")
    pin=$(awk -F'\t' '$1 == "PIN" {print $2}' <<< "$out")
    [ "$found" = 1 ] || { echo "DRIFT: $file has $found '$SECTION' jobs (want 1)"; return 2; }
    [ "$pin" = 1 ] || { echo "DRIFT: the base pin line appears $pin times in $SECTION (want 1): $BASE_PIN"; bad=1; }
    while IFS= read -r t; do echo "DRIFT: unclassified (no row): $t"; bad=1; done < <(
        LC_ALL=C comm -13 <(printf '%s\n' "${TEXT[@]}" | LC_ALL=C sort) <(awk -F'\t' '$1 == "CMD" {print $3}' <<< "$out" | LC_ALL=C sort))
    while IFS= read -r t; do echo "DRIFT: absent (a row the section no longer has): $t"; bad=1; done < <(
        LC_ALL=C comm -23 <(printf '%s\n' "${TEXT[@]}" | LC_ALL=C sort) <(awk -F'\t' '$1 == "CMD" {print $3}' <<< "$out" | LC_ALL=C sort))
    for id in "${!RUNTEXT[@]}"; do
        n=$(TXT=${RUNTEXT[$id]} awk -F'\t' '$1 == "CMD" && $3 == ENVIRON["TXT"]' <<< "$out" | wc -l)
        [ "$n" -eq 1 ] || { echo "DRIFT: RUN row $id appears $n times (want 1)"; bad=1; continue; }
        got=$(TXT=${RUNTEXT[$id]} awk -F'\t' '$1 == "CMD" && $3 == ENVIRON["TXT"] {print $2}' <<< "$out" | sorted)
        want=$(sorted <<< "${ENVSET[$id]}")
        [ "$got" = "$want" ] || { echo "DRIFT: env $id: the section has [$got], the table has [$want]"; bad=1; }
        got=$(TXT=${RUNTEXT[$id]} awk -F'\t' '$1 == "CMD" && $3 == ENVIRON["TXT"] {print $4}' <<< "$out" | sorted)
        want=$(sorted <<< "${MOUNTS[$id]}")
        [ "$got" = "$want" ] || { echo "DRIFT: mounts $id: the section has [$got], the table has [$want]"; bad=1; }
    done
    [ "$bad" = 0 ] || return 2
    echo "DRIFT ok: ${#TEXT[@]} rows (${#RUNTEXT[@]} run) equal $SECTION in $file; env and mounts checked for each run row; base pin present"
}

# lane_env ID: the APPLY tokens of a row, one per line.
lane_env() { local tok; for tok in ${ENVSET[$1]}; do case "${DISP[$tok]}" in "APPLY "*) echo "$tok" ;; esac; done; }

# ltag ID: the scope every line of a lane carries (none for a whole lane).
ltag() {
    if [ -n "$ONLY" ]; then echo " PROOF-SCOPE -p $ONLY"
    elif [ "$1" = tests ]; then echo " SCOPE$TESTS_PKGS"
    fi
}

# added_env ID DIR: a row's ADDED values, one per line; DIR is an empty directory of the run's own.
added_env() {
    local tok
    for tok in ${ADDED[$1]:-}; do
        case "$tok" in
            HF_HOME) echo "HF_HOME=$2" ;;
            *) echo "NOT-MEASURED: this script has no value for ADDED $tok"; return 2 ;;
        esac
    done
}

# run_row ID CMD [VAR=VALUE...]: run one command with its row's APPLY env and the given ADDED
# values; prints RESULT, returns 0, 1 or 2.
run_row() {
    local id=$1 cmd=$2 rc=0 tag
    shift 2
    tag=$(ltag "$id")
    local -a apply
    mapfile -t apply < <(lane_env "$id")
    echo "::group::$id$tag: $cmd"
    echo "ENV $id$tag: ${apply[*]:-none} (PASS and OMIT tokens: see --drift and this script's table)"
    [ $# = 0 ] || echo "ADDED $id$tag: ${*%%=*} (values made for this run; see this script's table)"
    env "${apply[@]}" "$@" bash -c "$cmd" < /dev/null || rc=$?
    echo "::endgroup::"
    echo "RESULT $id$tag rc=$rc"
    if [ "$rc" = 0 ]; then return 0; fi
    # The explicit runner's own vocabulary: 2 is "commands were not measured".
    if [ "$id" = explicit ] && [ "$rc" = 2 ]; then return 2; fi
    return 1
}

need_nextest() {
    cargo nextest --version > /dev/null 2>&1 || {
        echo "NOT-MEASURED: cargo-nextest is not installed on this host, so the $1 lane cannot run as CI runs it"; return 2; }
}

# explicit_scoped DIR: copy into DIR the fragments whose command names -p $ONLY; prints the count.
explicit_scoped() {
    local dir=$1 f n=0 line
    for f in ci/explicit-test-commands.d/*.cmd; do
        line=$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$f" || true)
        if [[ " $line " =~ [[:space:]]-p[[:space:]]+${ONLY}[[:space:]] ]]; then cp "$f" "$dir/"; n=$((n + 1)); fi
    done
    echo "$n"
}

lane() {
    local l=$1 rc=0 n tmp pk
    local -a added
    case "$l" in
        lib)
            if [ -z "$ONLY" ]; then need_nextest lib || return 2; run_row lib "${RUNTEXT[lib]%"${STRIP[lib]}"}" || rc=$?
            else
                case "$ONLY" in aprender-gpu | aprender-cuda-edge | aprender-compute) echo "RESULT lib PROOF-SCOPE -p $ONLY: not on the lib line, not run"; return 0 ;; esac
                need_nextest lib || return 2; run_row lib "cargo nextest run --profile ci -p $ONLY --lib" || rc=$?
            fi ;;
        gpu)
            if [ -n "$ONLY" ] && [ "$ONLY" != aprender-gpu ] && [ "$ONLY" != aprender-cuda-edge ]; then echo "RESULT gpu PROOF-SCOPE -p $ONLY: not a GPU crate, not run"; return 0; fi
            need_nextest gpu || return 2
            if [ -z "$ONLY" ]; then run_row gpu "${RUNTEXT[gpu]}" || rc=$?; else run_row gpu "cargo nextest run --profile ci -p $ONLY --lib" || rc=$?; fi ;;
        compute)
            if [ -n "$ONLY" ] && [ "$ONLY" != aprender-compute ]; then echo "RESULT compute PROOF-SCOPE -p $ONLY: not the compute crate, not run"; return 0; fi
            # CI's container starts with an empty /tmp. On a host, a log left by an earlier run
            # would be read by the pass rule if tee could not replace it, so it must be gone first.
            rm -f /tmp/compute-test.log
            [ ! -e /tmp/compute-test.log ] || { echo "NOT-MEASURED: cannot remove a stale /tmp/compute-test.log"; return 2; }
            run_row compute "${RUNTEXT[compute]}" || rc=$? ;;
        explicit)
            if [ -z "$ONLY" ]; then
                run_row explicit-list "${RUNTEXT[explicit-list]}" > /dev/null || { echo "NOT-MEASURED: the explicit command list does not parse"; return 2; }
                run_row explicit "${RUNTEXT[explicit]%"${STRIP[explicit]}"}" || rc=$?
            else
                tmp=$(mktemp -d)
                n=$(explicit_scoped "$tmp")
                if [ "$n" = 0 ]; then echo "RESULT explicit PROOF-SCOPE -p $ONLY: no fragment names it, not run"; rm -rf "${tmp:?}"; return 0; fi
                run_row explicit "bash scripts/ci_run_explicit_test_commands.sh --run $tmp" || rc=$?
                rm -rf "${tmp:?}"
            fi ;;
        tests)
            # `$pkgs` stays an unexported shell variable, as in CI's step; both values are
            # checked (TESTS_PKGS by table_check, ONLY by the argument parser), so quoting is safe.
            pk=${ONLY:+ -p $ONLY}
            pk=${pk:-$TESTS_PKGS}
            need_nextest tests || return 2
            echo "PKGS tests$(ltag tests): \$pkgs='$pk' (q-5002d Q1 KT: CI binds it to the crates a PR touches; the workspace is #5027)"
            tmp=$(mktemp -d) || { echo "NOT-MEASURED: mktemp failed"; return 2; }
            mapfile -t added < <(added_env tests "$tmp")
            if [[ ${added[*]:-} == *NOT-MEASURED* ]]; then printf '%s\n' "${added[@]}"; rm -rf "${tmp:?}"; return 2; fi
            run_row tests "pkgs='$pk'; ${RUNTEXT[tests]}" "${added[@]}" || rc=$?
            rm -rf "${tmp:?}" ;;
        *) echo "usage: unknown lane '$l' (one of: $LANES all)"; return 2 ;;
    esac
    return "$rc"
}

run() {
    local which=$1 sections=$2 l rc vtag worst=0
    drift "$sections" || { echo "VERDICT $which: NOT-MEASURED (drift: the table and $SECTION differ; nothing ran)"; return 2; }
    [ "${CI:-}" = true ] || { echo "VERDICT $which: NOT-MEASURED (CI is '${CI:-}', not 'true'; CI's container gets the runner's CI=true and tests read it)"; return 2; }
    [ -z "$ONLY" ] || echo "PROOF-SCOPE -p $ONLY: a narrowed run for a red/green proof, not a nightly verdict"
    # The pin is a shallow fetch. In a linked worktree it would write .git/shallow into the
    # common dir that every other worktree of that clone shares, so it runs only in a clone.
    [ "$(git rev-parse --absolute-git-dir)" = "$(git rev-parse --path-format=absolute --git-common-dir)" ] || {
        echo "VERDICT $which: NOT-MEASURED (a linked worktree: the shallow base pin would change every worktree of this clone; use a separate clone)"; return 2; }
    bash -c "$BASE_PIN" || { echo "VERDICT $which: NOT-MEASURED (the refinement base pin failed)"; return 2; }
    echo "refinement base: origin/main tip $(git rev-parse origin/main) (CI's workflow_dispatch arm)"
    for l in $([ "$which" = all ] && echo "$LANES" || echo "$which"); do
        rc=0; lane "$l" || rc=$?
        echo "LANE $l$(ltag "$l"): $(word "$rc")"
        if [ "$rc" = 1 ] || { [ "$rc" = 2 ] && [ "$worst" = 0 ]; }; then worst=$rc; fi
    done
    vtag=$(ltag "$which")
    [ "$which" != all ] || [ -n "$ONLY" ] || vtag=" (tests SCOPE$TESTS_PKGS)"
    echo "VERDICT $which$vtag: $(word "$worst")"
    return "$worst"
}

word() { case "$1" in 0) echo GREEN ;; 1) echo RED ;; *) echo NOT-MEASURED ;; esac; }

# ---------------------------------------------------------------------------------------------
# --self-test: a planted case for each of (a)-(e), the env and table checks, and the run paths
# (stub cargo on PATH, a fixture repo with a file:// origin). Run from the repo root.
self_test() {
    # T is global: the EXIT trap runs after this function returns, when a local T is unset.
    local real pass=0 fail=0
    real=$(pwd)/ci/sections.yml
    [ -r "$real" ] || { echo "self-test: run it from the repo root (no ci/sections.yml here)"; return 1; }
    T=$(mktemp -d)
    # The compute row's CI text writes /tmp/compute-test.log; the nightly runs this self-test in
    # its own job, before the lanes, so the stub's log never meets the real compute lane.
    trap 'rm -rf "${T:?}"; rm -f /tmp/compute-test.log' EXIT

    ok() { pass=$((pass + 1)); echo "  ok   $1"; }
    bad() { fail=$((fail + 1)); echo "  FAIL $1"; }
    # expect NAME RC PATTERN -- CMD...: CMD exits RC and its output matches PATTERN.
    expect() {
        local name=$1 want=$2 pat=$3 got=0 o; shift 3
        o=$("$@" 2>&1) || got=$?
        if [ "$got" = "$want" ] && grep -qE -- "$pat" <<< "$o"; then ok "$name"
        else bad "$name (rc $got, want $want; /$pat/)"; printf '%s\n' "$o" | tail -n 8 | sed 's/^/       | /'; fi
    }
    # plant OUT ACT TARGET [ANCHOR] [NEW] [OLD]: a copy of the real section with the first
    # non-comment line containing TARGET (after ANCHOR) deleted, doubled, commented out, followed
    # by NEW, or with OLD replaced by NEW. A plant that does not land fails its case.
    plant() {
        ACT=$2 TGT=$3 ANC=${4:-} NEW=${5:-} OLD=${6:-} awk '
            BEGIN { armed = (ENVIRON["ANC"] == "") }
            !armed && index($0, ENVIRON["ANC"]) { armed = 1 }
            armed && !done && index($0, ENVIRON["TGT"]) && $0 !~ /^[ \t]*#/ {
                done = 1; a = ENVIRON["ACT"]
                if (a == "del") next
                if (a == "dup") { print; print; next }
                if (a == "comment") { match($0, /^[ \t]*/); print substr($0, 1, RLENGTH) "# " substr($0, RLENGTH + 1); next }
                if (a == "after") { print; print ENVIRON["NEW"]; next }
                if (a == "sub") { i = index($0, ENVIRON["OLD"]); if (!i) exit 4; print substr($0, 1, i - 1) ENVIRON["NEW"] substr($0, i + length(ENVIRON["OLD"])); next }
                exit 5
            }
            { print }
            END { if (!done) exit 3 }' "$real" > "$T/$1"
    }
    drift_of() { bash "$SELF" --drift --sections "$T/$1"; }

    echo "self-test: drift"
    expect "control: the real section has no drift" 0 '^DRIFT ok' bash "$SELF" --drift --sections "$real"
    local lib='--partition "hash:${SHARD}/${SHARDS}"' gpu='cargo nextest run --profile ci -p aprender-gpu -p aprender-cuda-edge --lib'
    plant a-new.yml after "$lib" '' '            cargo test -p aprender-core --lib --features extra' \
        && expect "(a) a new test command with no row" 2 'unclassified.*--features extra' drift_of a-new.yml || bad "(a) plant a-new"
    plant a-gone.yml del "$gpu" && expect "(a) a row the section dropped" 2 'absent.*aprender-cuda-edge --lib$' drift_of a-gone.yml || bad "(a) plant a-gone"
    plant a-dup.yml dup 'cargo check --workspace --all-targets --locked' \
        && expect "(a) a SKIP line twice is not set-equal" 2 'unclassified.*cargo check' drift_of a-dup.yml || bad "(a) plant a-dup"
    plant a-edit.yml sub "$lib" '' '' ' --exclude aprender-compute' \
        && expect "(a) an edited RUN line" 2 'absent.*--exclude aprender-compute --partition' drift_of a-edit.yml || bad "(a) plant a-edit"
    plant b-add.yml after '-e CARGO_INCREMENTAL=0' '- name: Workspace lib tests' '            -e RUST_MIN_STACK=8388608 \' \
        && expect "(b) an env var added to the lib run" 2 'env lib:.*RUST_MIN_STACK' drift_of b-add.yml || bad "(b) plant b-add"
    plant b-del.yml del '-e CARGO_INCREMENTAL=0' '- name: Compute tests' && expect "(b) an env var dropped from the compute run" 2 'env compute:' drift_of b-del.yml || bad "(b) plant b-del"
    plant b-val.yml sub '-e CARGO_BUILD_JOBS=8' '- name: Integration tests' '-e CARGO_BUILD_JOBS=16' '-e CARGO_BUILD_JOBS=8' \
        && expect "(b) an env value changed in the explicit run" 2 'env explicit:.*CARGO_BUILD_JOBS=16' drift_of b-val.yml || bad "(b) plant b-val"
    plant c-rule.yml sub 'tee /tmp/compute-test.log' '' '' ' && ! grep -q "test result: FAILED" /tmp/compute-test.log' \
        && expect "(c) the compute pass rule weakened" 2 'absent.*FAILED' drift_of c-rule.yml || bad "(c) plant c-rule"
    plant d-cmt.yml comment "$gpu" && expect "(d) a RUN line commented out does not count" 2 'absent.*aprender-cuda-edge --lib$' drift_of d-cmt.yml || bad "(d) plant d-cmt"
    plant d-ghost.yml after '- name: Workspace lib tests' '' '        # cargo test -p ghost --lib' \
        && expect "(d) a command in a comment is not unclassified" 0 '^DRIFT ok' drift_of d-ghost.yml || bad "(d) plant d-ghost"
    plant d-env.yml after '-e CARGO_INCREMENTAL=0' '- name: Workspace lib tests' '            # -e RUST_MIN_STACK=8388608 \' \
        && expect "(d) an env var in a comment does not count" 0 '^DRIFT ok' drift_of d-env.yml || bad "(d) plant d-env"
    plant pin.yml del "$BASE_PIN" && expect "the base pin line removed" 2 'base pin line appears 0' drift_of pin.yml || bad "plant pin"
    # The quick tier's first step: the tests row (q-5002d).
    local quick='bash -c "cargo nextest run --profile ci --lib --tests$pkgs"' qname='- name: "Quick tier: lib + integration tests'
    local sc='-v "${SCCACHE_HOST_DIR}:/sccache"'
    plant a-tests.yml sub "$quick" '' '--lib$pkgs' '--lib --tests$pkgs' \
        && expect "(a) the tests row edited (--tests dropped)" 2 'absent.*--lib --tests.pkgs"$' drift_of a-tests.yml || bad "(a) plant a-tests"
    plant b-tests.yml after '-e CARGO_TERM_COLOR=never' "$qname" '            -e HF_HOME=/hf \' \
        && expect "(b) an env var added to the quick-tier run" 2 'env tests:.*HF_HOME=/hf' drift_of b-tests.yml || bad "(b) plant b-tests"
    plant e-add.yml after "$sc" "$qname" '            -v "${HF_CACHE}:/root/.cache/huggingface" \' \
        && expect "(e) a mount added to the quick-tier run" 2 'mounts tests:.*/root/.cache/huggingface' drift_of e-add.yml || bad "(e) plant e-add"
    plant e-del.yml del "$sc" '- name: Workspace lib tests' && expect "(e) a mount dropped from the lib run" 2 'mounts lib:' drift_of e-del.yml || bad "(e) plant e-del"
    plant e-long.yml after "$sc" "$qname" '            --mount type=bind,src=/x,dst=/y \' \
        && expect "(e) a mount in --mount form is not missed" 2 'mounts tests:.*unparsed:--mount' drift_of e-long.yml || bad "(e) plant e-long"
    plant e-cmt.yml after "$sc" "$qname" '            # -v "${HF_CACHE}:/root/.cache/huggingface" \' \
        && expect "(d) a mount in a comment does not count" 0 '^DRIFT ok' drift_of e-cmt.yml || bad "(d) plant e-cmt"
    sed 's/^\(    \[tests\]="\$SAFE \$GH \$BUILD \$DEBUG CARGO_TERM_COLOR=never\)"$/\1 HF_HOME"/' "$SELF" > "$T/added.sh"
    if cmp -s "$SELF" "$T/added.sh"; then bad "table: plant added.sh did not land"
    else expect "table: an ADDED name in its row's CI env set" 2 'TABLE: HF_HOME is ADDED for tests and also in its CI env set' bash "$T/added.sh" --drift --sections "$real"; fi

    echo "self-test: runs (stub cargo, fixture repo)"
    local R=$T/repo B=$T/bin
    mkdir -p "$R/ci/explicit-test-commands.d" "$R/scripts" "$B"
    cp "$real" "$R/ci/sections.yml"
    cp scripts/ci_run_explicit_test_commands.sh "$R/scripts/"
    printf 'true\n' > "$R/ci/explicit-test-commands.d/010-a.cmd"
    printf '%s\n' "cargo test -p aprender-core --test probe" > "$R/ci/explicit-test-commands.d/020-core.cmd"
    git init -q -b main "$T/origin" && git -C "$T/origin" -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -q --allow-empty -m o
    git init -q -b main "$R" && git -C "$R" -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -q --allow-empty -m r
    git -C "$R" remote add origin "file://$T/origin"
    cat > "$B/cargo" << 'STUB'
#!/usr/bin/env bash
hf=unset
if [ -n "${HF_HOME+x}" ]; then
    if [ -d "$HF_HOME" ] && [ -z "$(ls -A "$HF_HOME")" ]; then hf=empty; else hf="not-empty:$HF_HOME"; fi
    printf 'HFDIR %s\n' "$HF_HOME" >> "$STUB_LOG"
fi
printf 'ARGV %s | JOBS=%s INCR=%s HF=%s PKGS=%s\n' "$*" "${CARGO_BUILD_JOBS-unset}" "${CARGO_INCREMENTAL-unset}" "$hf" "${pkgs-unset}" >> "$STUB_LOG"
if [ "$1 $2" = "nextest --version" ]; then exit "${STUB_NEXTEST_RC:-0}"; fi
case "${STUB_MODE:-ok}" in
    ok) echo "test result: ok. 3 passed"; exit 0 ;;
    red) echo "test result: FAILED. 1 failed"; exit 100 ;;
    segv) echo "test result: ok. 3 passed"; exit 139 ;;
    okthenfail) echo "test result: ok. 3 passed"; echo "test result: FAILED. 1 failed"; exit 0 ;;
    silent) exit 0 ;;
esac
STUB
    chmod +x "$B/cargo"
    # in_repo ENV... -- CMD...: run CMD in the fixture repo with the stub first on PATH.
    # The caller's HF_HOME and pkgs are cleared, so a value the stub sees came from this script.
    in_repo() { (cd "$R" && env -u HF_HOME -u pkgs PATH="$B:$PATH" STUB_LOG="$T/stub.log" "$@"); }
    : > "$T/stub.log"
    expect "lib runs the CI line without its shard argument" 0 'VERDICT lib: GREEN' in_repo CI=true bash "$SELF" --run lib
    if grep -qxF 'ARGV nextest run --profile ci --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute | JOBS=8 INCR=0 HF=unset PKGS=unset' "$T/stub.log"; then ok "lib argv and APPLY env reach cargo; nothing ADDED"; else bad "lib argv/env: $(tail -n 1 "$T/stub.log")"; fi
    expect "a failing lib run is RED" 1 'VERDICT lib: RED' in_repo CI=true STUB_MODE=red bash "$SELF" --run lib
    expect "(c) compute: ok then FAILED is RED" 1 'LANE compute: RED' in_repo CI=true STUB_MODE=okthenfail bash "$SELF" --run compute
    expect "(c) compute: SIGSEGV at exit after ok is GREEN" 0 'LANE compute: GREEN' in_repo CI=true STUB_MODE=segv bash "$SELF" --run compute
    expect "(c) compute: no result line is RED" 1 'LANE compute: RED' in_repo CI=true STUB_MODE=silent bash "$SELF" --run compute
    expect "explicit runs every fragment" 0 'LANE explicit: GREEN' in_repo CI=true bash "$SELF" --run explicit
    printf 'false\n' > "$R/ci/explicit-test-commands.d/030-b.cmd"
    expect "a failing explicit fragment is RED" 1 'LANE explicit: RED' in_repo CI=true bash "$SELF" --run explicit
    expect "all: a red first lane does not stop the later lanes" 1 'LANE explicit: RED' in_repo CI=true STUB_MODE=red bash "$SELF" --run all
    git -C "$R" worktree add -q "$T/linked" 2> /dev/null
    : > "$T/stub.log"
    expect "a linked worktree: NOT-MEASURED before the shallow pin" 2 'linked worktree' \
        env -C "$T/linked" PATH="$B:$PATH" STUB_LOG="$T/stub.log" CI=true bash "$SELF" --run gpu --sections "$real"
    if [ -s "$T/stub.log" ]; then bad "a linked worktree: cargo was called"; else ok "a linked worktree: cargo was never called"; fi
    : > "$T/stub.log"
    expect "no cargo-nextest: NOT-MEASURED" 2 'cargo-nextest is not installed' in_repo CI=true STUB_NEXTEST_RC=101 bash "$SELF" --run gpu
    if grep -q 'nextest run' "$T/stub.log"; then bad "no cargo-nextest: a test ran anyway"; else ok "no cargo-nextest: nothing ran"; fi
    expect "CI not true: NOT-MEASURED" 2 'CI is .*not .true' in_repo CI= bash "$SELF" --run gpu
    cp "$T/a-new.yml" "$R/ci/sections.yml"; : > "$T/stub.log"
    expect "drift stops the run" 2 'NOT-MEASURED \(drift' in_repo CI=true bash "$SELF" --run gpu
    if [ -s "$T/stub.log" ]; then bad "drift: cargo was called"; else ok "drift: cargo was never called"; fi
    cp "$real" "$R/ci/sections.yml"; : > "$T/stub.log"
    expect "proof scope prints PROOF-SCOPE" 0 'VERDICT lib PROOF-SCOPE -p aprender-core: GREEN' in_repo CI=true bash "$SELF" --run lib --only-package aprender-core
    if grep -q '^ARGV nextest run --profile ci -p aprender-core --lib ' "$T/stub.log"; then ok "proof scope narrows lib to -p"; else bad "proof scope argv: $(tail -n 1 "$T/stub.log")"; fi
    expect "proof scope runs only the fragments naming -p" 0 'ran=1 skipped=0 total=1 failed=0' in_repo CI=true bash "$SELF" --run explicit --only-package aprender-core

    echo "self-test: the tests lane (q-5002d)"
    local o r=0 n hfdir tl='^(PKGS|ENV|ADDED|RESULT|LANE|VERDICT) tests'
    : > "$T/stub.log"
    o=$(in_repo CI=true bash "$SELF" --run tests 2>&1) || r=$?
    if [ "$r" = 0 ] && grep -qx 'VERDICT tests SCOPE -p aprender-core: GREEN' <<< "$o"; then ok "tests runs the quick-tier line with \$pkgs bound to TESTS_PKGS"
    else bad "tests verdict (rc $r)"; printf '%s\n' "$o" | tail -n 8 | sed 's/^/       | /'; fi
    if grep -qxF 'ARGV nextest run --profile ci --lib --tests -p aprender-core | JOBS=8 INCR=0 HF=empty PKGS=unset' "$T/stub.log"; then ok "tests argv, APPLY env and an empty HF_HOME reach cargo; \$pkgs is not exported"
    else bad "tests argv/env: $(tail -n 1 "$T/stub.log")"; fi
    n=$(grep -cE "$tl" <<< "$o" || true)
    if [ "$n" = 6 ] && ! grep -E "$tl" <<< "$o" | grep -qvE ' tests SCOPE -p aprender-core( |:)'; then ok "every line of the tests lane names its scope"
    else bad "tests lines: $n of 6, or one without its scope"; fi
    hfdir=$(sed -n 's/^HFDIR //p' "$T/stub.log" | tail -n 1)
    if [ -n "$hfdir" ] && [ ! -e "$hfdir" ]; then ok "the tests lane removes its HF_HOME directory"; else bad "HF_HOME directory '$hfdir' never made or left behind"; fi
    expect "a failing tests run is RED" 1 'LANE tests SCOPE -p aprender-core: RED' in_repo CI=true STUB_MODE=red bash "$SELF" --run tests
    : > "$T/stub.log"
    expect "proof scope binds \$pkgs to -p PKG" 0 'VERDICT tests PROOF-SCOPE -p aprender-serve: GREEN' in_repo CI=true bash "$SELF" --run tests --only-package aprender-serve
    if grep -q '^ARGV nextest run --profile ci --lib --tests -p aprender-serve |' "$T/stub.log"; then ok "proof scope narrows the tests lane"; else bad "tests proof argv: $(tail -n 1 "$T/stub.log")"; fi

    local wf=.github/workflows/workspace-test-nightly.yml l miss=""
    if [ ! -r "$wf" ]; then bad "the nightly workflow $wf is missing"
    elif grep -q -- '--only-package' "$wf"; then bad "the nightly workflow passes --only-package"
    else ok "the nightly workflow never passes --only-package"; fi
    for l in $LANES; do grep -qE "^ *- \{ lane: $l, timeout: [0-9]+ \}$" "$wf" || miss="$miss $l"; done
    if [ -z "$miss" ]; then ok "the nightly workflow has a matrix lane for each of: $LANES"; else bad "the nightly workflow has no matrix lane for:$miss"; fi

    echo "self-test: $pass ok, $fail failed"
    [ "$fail" = 0 ]
}

# ---------------------------------------------------------------------------------------------
ONLY=""
mode=${1:-}; shift || true
sections=ci/sections.yml lane_arg=""
case "$mode" in
    --run) lane_arg=${1:-}; shift || true ;;
esac
while [ $# -gt 0 ]; do
    case "$1" in
        --sections) sections=${2:?--sections needs a file}; shift 2 ;;
        --only-package) ONLY=${2:?--only-package needs a package}; shift 2 ;;
        *) echo "usage: unknown argument '$1'"; exit 2 ;;
    esac
done
case "$mode" in
    --self-test) self_test ;;
    --drift) drift "$sections" ;;
    --run)
        case " $LANES all " in *" $lane_arg "*) ;; *) echo "usage: --run $LANES|all"; exit 2 ;; esac
        [[ -z $ONLY || $ONLY =~ ^[a-z0-9_-]+$ ]] || { echo "usage: --only-package takes a package name"; exit 2; }
        load_tables || exit 2
        run "$lane_arg" "$sections" ;;
    *) sed -n '2,/^set -euo/p' "$SELF" | sed '$d'; exit 2 ;;
esac
