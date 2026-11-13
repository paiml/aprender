#!/usr/bin/env bash
# check_guard_shard_target_dirs.sh - no two guard-cargo shards share a host target dir (#7026).
#
# The shards run at once on one fat runner. When their host steps shared the runner's
# target dir, pv_bin.sh found the pv another shard had just built and refused it as
# WRONG-TREE (main run 43430038433, guard-cargo#m30 rc=3). This check does not read the
# wiring, it RUNS it: it takes the guard_shard_target_dir.sh line from the
# guard-cargo-shard job in ci/sections.yml, runs it once per pinned shard (plus 1 and 2,
# so the proof is never vacuous) against ONE shared inherited base, and is red unless
# every shard lands in its own dir, none of them the base, before the ci_guards.sh step.
#
#   bash scripts/check_guard_shard_target_dirs.sh              # check this tree
#   bash scripts/check_guard_shard_target_dirs.sh --self-test  # case table + mutations must turn it red
#
# EXIT 0 every shard has its own dir; 1 a shared dir or missing wiring; 2 usage.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB=guard-cargo-shard
HELPER=scripts/ci/guard_shard_target_dir.sh
TMPS=()
trap 'rm -rf -- "${TMPS[@]}"' EXIT

fail() { printf 'FAIL  %s\n' "$*"; return 1; }

# pinned_shards ROOT - the `shard:` values of matrix-pins.guard-cargo-shard, one per line.
pinned_shards() {
    awk -v j="  $JOB:" '
        /^[^ \t#]/ { inpins = ($0 ~ /^matrix-pins:/) ; injob = 0; next }
        inpins && /^  [^ \t#]/ { injob = ($0 == j) ; next }
        inpins && injob { while (match($0, /shard: *[0-9]+/)) { s = substr($0, RSTART, RLENGTH); sub(/shard: */, "", s); print s; $0 = substr($0, RSTART + RLENGTH) } }
    ' "$1/ci/sections.yml"
}

# check ROOT - rc 0 when ROOT's wiring gives every shard its own host target dir.
check() {
    local root=$1 td n hidx gidx line k out dir base seen=''
    td=$(mktemp -d) || return 2
    TMPS+=("$td")
    # The job under top-level `jobs:` (matrix-pins has a key of the same name), split into
    # steps at `      - `; comment lines are not code. No jq or yq: the CI image has neither.
    awk -v j="  $JOB:" -v h="$HELPER" '
        /^[^ \t#]/ { injobs = ($0 ~ /^jobs:[ \t]*$/); injob = 0; next }
        injobs && /^  [^ \t#]/ { injob = ($0 == j); next }
        !injob || /^[ \t]*#/ { next }
        /^      - / { step++ }
        index($0, h) { n++; hidx = step; sub(/^[ \t]+/, ""); line = $0 }
        /ci_guards\.sh .*guard-cargo/ && gidx == "" { gidx = step }
        END { printf "%d\t%s\t%s\t%s\n", n, hidx, gidx, line }
    ' "$root/ci/sections.yml" > "$td/scan"
    IFS=$'\t' read -r n hidx gidx line < "$td/scan"
    [ "$n" = 1 ] || { fail "$JOB: $n lines run $HELPER (need exactly 1)"; return; }
    [ -n "$gidx" ] || { fail "$JOB: no ci_guards.sh guard-cargo step"; return; }
    [ "$hidx" -lt "$gidx" ] || { fail "$JOB: $HELPER runs at step $hidx, not before the guard step $gidx"; return; }
    grep -qF 'GITHUB_ENV' <<< "$line" || { fail "$JOB: the $HELPER line does not write GITHUB_ENV: $line"; return; }
    base="$td/runner-target"
    mkdir -p "$base"
    for k in $( { pinned_shards "$root"; printf '1\n2\n'; } | sort -un); do
        out="$td/env.$k"
        : > "$out"
        local run=${line//'${{ matrix.shard }}'/$k}
        case "$run" in *'${{'*) fail "unexpected expression left in: $run"; return ;; esac
        (cd "$root" && env CARGO_TARGET_DIR="$base" GITHUB_ENV="$out" GITHUB_WORKSPACE="$root" bash -c "$run") \
            > "$td/log.$k" 2>&1 || { fail "shard $k: the line exited non-zero: $(tr '\n' ' ' < "$td/log.$k")"; return; }
        dir=$(sed -n 's/^CARGO_TARGET_DIR=//p' "$out" | tail -n 1)
        [ -n "$dir" ] || { fail "shard $k: no CARGO_TARGET_DIR written to GITHUB_ENV"; return; }
        [ "${dir%/}" != "$base" ] || { fail "shard $k keeps the runner's shared target dir $dir"; return; }
        case " $seen " in *" $dir "*) fail "shard $k shares target dir $dir with another shard"; return ;; esac
        seen="$seen $dir"
        printf 'ok    shard %s -> %s\n' "$k" "${dir#"$td"/}"
    done
    printf 'PASS  every %s shard has its own host target dir\n' "$JOB"
}

# copy_tree DST - the files check() reads, from this tree.
copy_tree() {
    mkdir -p "$1/ci" "$1/scripts/ci"
    cp "$ROOT/ci/sections.yml" "$1/ci/" && cp "$ROOT/$HELPER" "$1/$HELPER"
}

self_test() {
    local bad=0 td name want got rc
    row() { # row NAME WANT_RC CMD...  (WANT 0 = green, 1 = red)
        name=$1 want=$2; shift 2
        "$@" > /dev/null 2>&1; rc=$?
        got=$([ "$rc" = 0 ] && echo 0 || echo 1)
        if [ "$got" = "$want" ]; then printf 'ok    %-58s rc=%s\n' "$name" "$rc"; else printf 'FAIL  %-58s rc=%s want %s\n' "$name" "$rc" "$want"; bad=1; fi
    }
    helper() { CARGO_TARGET_DIR=$1 bash "$ROOT/$HELPER" "$2"; }
    td=$(mktemp -d) || return 2
    TMPS+=("$td")
    row "helper: shard 3 -> rc 0" 0 helper "$td/b" 3
    row "helper: empty shard -> refused" 1 helper "$td/b" ''
    row "helper: shard 0 -> refused" 1 helper "$td/b" 0
    row "helper: shard x -> refused" 1 helper "$td/b" x
    row "helper: base already a shard dir -> not nested" 0 test "$(helper "$td/b/guard-shard-1" 2)" = "CARGO_TARGET_DIR=$td/b/guard-shard-2"
    copy_tree "$td/ok" && row "this tree's wiring -> green" 0 check "$td/ok"
    copy_tree "$td/m1" && sed -i 's|guard_shard_target_dir.sh "${{ matrix.shard }}"|guard_shard_target_dir.sh "1"|' "$td/m1/ci/sections.yml"
    row "MUTANT every shard passes shard 1 (one shared dir) -> red" 1 check "$td/m1"
    copy_tree "$td/m2" && sed -i 's|guard-shard-\$shard|guard-shard|' "$td/m2/$HELPER"
    row "MUTANT helper drops the shard suffix -> red" 1 check "$td/m2"
    copy_tree "$td/m3" && sed -i '\|guard_shard_target_dir.sh "|d' "$td/m3/ci/sections.yml"
    row "MUTANT the step no longer calls the helper -> red" 1 check "$td/m3"
    copy_tree "$td/m4" && sed -i 's|printf .CARGO_TARGET_DIR=%s\\n. "\$dir"|printf "CARGO_TARGET_DIR=%s\\n" "$base"|' "$td/m4/$HELPER"
    row "MUTANT helper prints the inherited base -> red" 1 check "$td/m4"
    [ "$bad" = 0 ] && echo "PASS: self-test" || { echo "FAIL: self-test"; return 1; }
}

case "${1:-}" in
    --help | -h) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --self-test) self_test; exit $? ;;
    '') check "$ROOT"; exit $? ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
