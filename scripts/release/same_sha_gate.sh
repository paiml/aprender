#!/usr/bin/env bash
# same_sha_gate.sh — promotion and crates.io need the heavy suites green on the
# rc tag's EXACT commit (Y3, operator ruling 2026-09-25, #4415 follow-on to #4285).
#
#   bash scripts/release/same_sha_gate.sh vX.Y.Z[-rc.N] [--sha SHA] [--head]
#   bash scripts/release/same_sha_gate.sh --self-test
#
# WHY. rc_cut.sh cuts vX.Y.Z-rc.N as soon as `ci / gate` is green on the release
# branch tip: an rc is cheap, and the fleet should have it the moment the gate
# passes. It no longer waits for workspace-test. So an rc may exist whose commit
# never passed the test shards, and it must not become a final or reach crates.io.
# This gate is where that is enforced. It exits 0 only when ALL hold for the tag's
# commit C:
#
#   1. workspace-test: some CI run with head_sha == C has at least one job named
#      exactly `workspace-test`, and every job of that name in that run concluded
#      success. A run on another commit never counts, even one with the same tree.
#   2. clean-room: paiml/infra clean-room.yml tested exactly C and was green. This
#      is scripts/cascade-publish.sh's own clean_room_gate, extracted from that
#      file (the same way check_cascade_clean_room_gate.sh does), never a copy.
#   3. --sha SHA: the caller's commit (promote_rc.sh reads it from the API) is C.
#   4. --head: this checkout's HEAD is C. cargo publish packages the working tree,
#      so the tree that is published must be the tree that was tested.
#
# Exit: 0 all green · 1 refused (each failing condition prints a REFUSE line; all
# are evaluated, not just the first) · 2 ENV/usage (unreadable API, no tag).
set -uo pipefail
PROG=same_sha_gate
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${GITHUB_REPOSITORY:-paiml/aprender}"
WS_JOB="workspace-test"

# ws_decide — pure. W_JOBS lines "<run id>\t<job name>\t<conclusion>" for the CI
# runs on commit C. Prints "ok run <id>" or "refuse <reason>".
ws_decide() {
    printf '%s\n' "${W_JOBS:-}" | awk -F'\t' -v job="$WS_JOB" '
        NF >= 3 && $2 == job { seen[$1] = 1; if ($3 != "success") bad[$1] = bad[$1] (bad[$1] ? "," : "") $3 }
        END {
            n = 0
            for (r in seen) { n++; if (!(r in bad)) { print "ok run " r; exit 0 } }
            if (n == 0) { print "refuse no CI run on this commit has a `" job "` job"; exit 0 }
            msg = ""
            for (r in bad) msg = msg " run " r "=" bad[r]
            print "refuse `" job "` is not green on this commit:" msg
        }'
}

# load_clean_room — define clean_room_gate from cascade-publish.sh, or fail.
load_clean_room() {
    local src="${CASCADE_UNDER_TEST:-$ROOT/scripts/cascade-publish.sh}" fns fn
    fns=$(for fn in clean_room_parse_runs clean_room_parse_job clean_room_tested_abbrevs clean_room_tested_shas clean_room_gate; do
        sed -n "/^${fn}() {/,/^}/p" "$src"
    done)
    for fn in clean_room_parse_runs clean_room_parse_job clean_room_tested_abbrevs clean_room_tested_shas clean_room_gate; do
        grep -q "^${fn}() {" <<< "$fns" || { echo "$PROG: cannot extract $fn from $src" >&2; return 2; }
    done
    eval "$fns"
}

# fetch_ws_jobs SHA — W_JOBS for every CI run on SHA (all pages).
fetch_ws_jobs() {
    local sha=$1 ids id
    ids=$(gh api --paginate "repos/$REPO/actions/runs?head_sha=$sha&per_page=100" \
        --jq '.workflow_runs[] | select(.name == "CI") | .id') || return 2
    W_JOBS=''
    for id in $ids; do
        W_JOBS+=$(gh api --paginate "repos/$REPO/actions/runs/$id/jobs?filter=latest&per_page=100" \
            --jq ".jobs[] | [\"$id\", .name, (.conclusion // \"pending\")] | @tsv") || return 2
        W_JOBS+=$'\n'
    done
}

gate() {
    local tag=$1 want_sha=$2 want_head=$3 c head verdict rc=0
    command -v gh > /dev/null || { echo "$PROG: ENV gh missing" >&2; return 2; }
    c=$(git -C "$ROOT" rev-parse --verify --quiet "refs/tags/${tag}^{commit}") || {
        git -C "$ROOT" fetch -q origin "refs/tags/$tag:refs/tags/$tag" 2> /dev/null
        c=$(git -C "$ROOT" rev-parse --verify --quiet "refs/tags/${tag}^{commit}")
    } || { echo "$PROG: ENV tag $tag does not resolve to a commit" >&2; return 2; }
    echo "$PROG: $tag is commit $c"

    if [ -n "$want_sha" ] && [ "$want_sha" != "$c" ]; then
        echo "REFUSE sha: the caller's commit $want_sha is not $tag's commit $c"; rc=1
    fi
    if [ "$want_head" = 1 ]; then
        head=$(git -C "$ROOT" rev-parse HEAD)
        if [ "$head" != "$c" ]; then echo "REFUSE head: HEAD is $head, $tag is $c"; rc=1; fi
    fi

    fetch_ws_jobs "$c" || { echo "$PROG: ENV cannot read the CI runs on $c" >&2; return 2; }
    verdict=$(ws_decide)
    case "$verdict" in
        ok*) echo "ok     workspace-test: ${verdict#ok }" ;;
        *) echo "REFUSE workspace-test: ${verdict#refuse }"; rc=1 ;;
    esac

    load_clean_room || return 2
    if clean_room_gate "$ROOT" "$tag"; then echo "ok     clean-room on $c"; else rc=1; fi

    if [ "$rc" = 0 ]; then echo "$PROG: PASS $tag @ $c"; else echo "$PROG: REFUSED $tag @ $c"; fi
    return "$rc"
}

self_test() {
    local fail=0 n=0 got
    row() { # row <want prefix> <label>
        got=$(ws_decide); n=$((n + 1))
        if [ "${got#"$1"}" != "$got" ]; then printf 'ok   %s\n' "$2"
        else printf 'FAIL %s -- got: %s\n' "$2" "$got"; fail=1; fi
    }
    W_JOBS=$'1\tworkspace-test\tsuccess\n1\tci / gate\tfailure';            row 'ok run 1' 'green workspace-test passes (other jobs are not read)'
    W_JOBS=$'1\tci / gate\tsuccess';                                          row 'refuse no CI run' 'no workspace-test job at all refuses'
    W_JOBS='';                                                                row 'refuse no CI run' 'no CI run on the commit refuses'
    W_JOBS=$'1\tworkspace-test\tfailure';                                     row 'refuse `workspace-test` is not green' 'red workspace-test refuses'
    W_JOBS=$'1\tworkspace-test\tpending';                                     row 'refuse `workspace-test` is not green' 'unfinished workspace-test refuses'
    W_JOBS=$'1\tworkspace-test\tskipped';                                     row 'refuse `workspace-test` is not green' 'skipped is not success'
    W_JOBS=$'1\tworkspace-test\tsuccess\n1\tworkspace-test\tcancelled';      row 'refuse `workspace-test` is not green' 'every job of the name in the run must pass'
    W_JOBS=$'1\tworkspace-test\tfailure\n2\tworkspace-test\tsuccess';        row 'ok run 2' 'a re-run that went green on the same commit passes'
    W_JOBS=$'1\tworkspace-test-shard (1/3)\tsuccess';                         row 'refuse no CI run' 'a shard is not the aggregate job'
    # the real clean-room functions must extract from cascade-publish.sh
    n=$((n + 1))
    if (load_clean_room && declare -F clean_room_gate > /dev/null); then echo 'ok   clean_room_gate extracts from cascade-publish.sh'
    else echo 'FAIL clean_room_gate does not extract'; fail=1; fi
    n=$((n + 1))
    if (CASCADE_UNDER_TEST=/dev/null load_clean_room 2> /dev/null); then echo 'FAIL an empty source extracted'; fail=1
    else echo 'ok   an empty source fails closed'; fi
    # MUTANT: a decider that ignores conclusions must turn the red row
    n=$((n + 1))
    got=$(W_JOBS=$'1\tworkspace-test\tfailure' bash -c "$(declare -f ws_decide | sed 's/if (\$3 != \"success\")/if (0)/'); WS_JOB=$WS_JOB; ws_decide")
    if [ "${got#ok}" != "$got" ]; then echo 'ok   mutant (conclusion ignored) passes a red run: the red row can see it'
    else echo "FAIL mutant not killed-able: $got"; fail=1; fi
    if [ "$fail" = 0 ]; then echo "$PROG self-test: ${n}/${n} pass"; return 0; fi
    echo "$PROG self-test: FAIL"; return 1
}

main() {
    local tag='' sha='' head=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --self-test) self_test; return ;;
            --sha) sha=${2:-}; shift ;;
            --head) head=1 ;;
            -h|--help) sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; return 0 ;;
            -*) echo "$PROG: unknown flag $1" >&2; return 2 ;;
            *) tag=$1 ;;
        esac
        shift
    done
    [ -n "$tag" ] || { echo "$PROG: usage: same_sha_gate.sh TAG [--sha SHA] [--head] | --self-test" >&2; return 2; }
    gate "$tag" "$sha" "$head"
}

main "$@"
