#!/usr/bin/env bash
# ci_reclaim_target_dirs.sh — decide which per-RUN CI target dirs may be
# reclaimed under one PR's parent, and which must be spared because a sibling
# run is still LIVE.
#
# WHY THIS EXISTS
# ---------------
# The step this replaces removed the PARENT, not this run's directory:
#
#     docker run --rm \
#       -v "/mnt/nvme-raid0/targets/aprender-ci/${PR_OR_REF}:/workspace/target" \
#       "$IMAGE" \
#       bash -c 'rm -rf /workspace/target/* /workspace/target/.[!.]* ...'
#
# The tree is `aprender-ci/<PR>/run-<RUN_ID>` (#1693, 2026-05-15). `<PR>/` is
# therefore the shared parent of EVERY run for that PR, and `rm -rf <PR>/*`
# deletes every sibling `run-*` directory regardless of whether the run that
# owns it is still building. The code has no notion of sibling liveness at all;
# the blast radius is the whole PR tree while the intended radius is the state
# that can reach THIS run.
#
# The step's own five-whys comment names the premise it was written against —
# "the target dir is bind-mounted from a per-PR persistent path
# /mnt/nvme-raid0/targets/aprender-ci/<PR>/, so partial-compile state survives
# across runs". #1693 deepened that mount to `<PR>/run-<RUN_ID>` and the comment
# was never updated. Since #1693 no other run's state can reach this run, so the
# parent-wide removal buys nothing and can only take a sibling with it.
#
# MEASURED, not argued (2026-08-31, 36 runs sampled from the last 100):
#
#   * `workspace-test` and `guard-runner-labels` are two jobs of the SAME run on
#     two different runners of one host, and both mount `<PR>/run-<RUN_ID>`
#     (ci.yml, the `model-tests falsification suites` step). Their job windows
#     overlap by 21-29 minutes.
#   * The removal fires in practice: runs 33372476223 and 33375180919 both
#     printed "Previous run was cancelled; nuking target dir".
#   * The margin between the removal and the sibling job's cargo step on that
#     same shared directory is uncontrolled and CHANGES SIGN: +7.6 min at the
#     tightest (run 33369566809), and -10.8 min in run 33243677674, where the
#     removal came AFTER the sibling step instead of before it. Nothing orders
#     them. Today the ordering happens to miss; it is held open only by how long
#     the image pull takes, which ci.yml itself documents at up to 20 minutes.
#
# WHAT THIS DOES INSTEAD
# ----------------------
# Print, one per line on stdout, the basenames under <parent> that may be
# reclaimed. Never `run-<THIS_RUN_ID>` (a sibling JOB of this run shares it, per
# the overlap measured above). A sibling run's directory is reclaimed only when
# that run is provably not live.
#
# NOT A BLANKET SKIP. Sparing everything whenever liveness cannot be established
# would make the step inert exactly when it is needed — the failure mode ci.yml
# already warns about elsewhere. So when the run-status oracle is unavailable
# the decision falls back to age: the workspace-test job timeout is 100 minutes,
# so a directory untouched for APR_RECLAIM_STALE_MINUTES (default 180) cannot
# belong to a live run and is reclaimed anyway.
#
#   bash scripts/ci_reclaim_target_dirs.sh <parent-dir> <this-run-id>
#   bash scripts/ci_reclaim_target_dirs.sh --self-test
#
# Env:
#   APR_RUN_STATUS_CMD        command taking a run id, printing its status
#                             (default: gh api .../actions/runs/<id> --jq .status)
#   APR_RECLAIM_STALE_MINUTES age fallback when the oracle cannot answer (180)
#   GITHUB_REPOSITORY         owner/repo for the default oracle

set -uo pipefail

STALE_MINUTES="${APR_RECLAIM_STALE_MINUTES:-180}"

# Print the status of one workflow run, or the empty string when unknown.
run_status() {
    local id="$1" out=""
    if [ -n "${APR_RUN_STATUS_CMD:-}" ]; then
        out=$("${APR_RUN_STATUS_CMD}" "$id" 2>/dev/null)
        printf '%s' "$out"
        return 0
    fi
    if ! command -v gh > /dev/null 2>&1; then
        return 0
    fi
    out=$(gh api "repos/${GITHUB_REPOSITORY:-}/actions/runs/${id}" --jq '.status' 2>/dev/null)
    printf '%s' "$out"
}

# True when anything under $1 was modified in the last $STALE_MINUTES minutes.
recently_touched() {
    local d="$1" hit
    hit=$(find "$d" -maxdepth 3 -mmin "-${STALE_MINUTES}" -print -quit 2>/dev/null)
    [ -n "$hit" ]
}

# stdout: basenames to reclaim. stderr: one rationale line per directory.
reclaimable_in() {
    local parent="$1" this_id="$2"
    local d base id st
    [ -d "$parent" ] || {
        printf 'parent %s does not exist yet; nothing to reclaim\n' "$parent" >&2
        return 0
    }
    for d in "$parent"/run-*; do
        [ -d "$d" ] || continue
        base=$(basename "$d")
        id="${base#run-}"
        if [ "$id" = "$this_id" ]; then
            printf 'KEEP     %s  this run (shared with the concurrent guard-runner-labels job)\n' "$base" >&2
            continue
        fi
        st=$(run_status "$id")
        case "$st" in
            completed | not_found)
                printf 'RECLAIM  %s  run status: %s\n' "$base" "$st" >&2
                printf '%s\n' "$base"
                ;;
            queued | in_progress | waiting | requested | pending | action_required)
                printf 'KEEP     %s  run status: %s -- LIVE sibling\n' "$base" "$st" >&2
                ;;
            *)
                # Oracle silent. Age decides, so an API outage cannot turn this
                # step into a no-op.
                if recently_touched "$d"; then
                    printf 'KEEP     %s  status unknown, touched < %s min ago\n' "$base" "$STALE_MINUTES" >&2
                else
                    printf 'RECLAIM  %s  status unknown, untouched > %s min\n' "$base" "$STALE_MINUTES" >&2
                    printf '%s\n' "$base"
                fi
                ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Case table. Every row is a fixture on disk, and row 1 additionally executes
# the OLD one-liner so the RED half is demonstrated rather than described.
self_test() {
    local td fails=0 got
    td=$(mktemp -d) || return 1
    # shellcheck disable=SC2064
    trap "rm -rf '${td:?}'" EXIT

    cat > "$td/oracle.sh" <<'ORACLE'
#!/usr/bin/env bash
# Fixture oracle: reads $ORACLE_MAP, lines of "<run-id> <status>".
set -uo pipefail
while read -r id st; do
    [ "$id" = "$1" ] || continue
    printf '%s' "$st"
    exit 0
done < "${ORACLE_MAP}"
exit 0
ORACLE
    chmod +x "$td/oracle.sh"
    export APR_RUN_STATUS_CMD="$td/oracle.sh"
    export ORACLE_MAP="$td/map.txt"

    # -- row 1 ------------------------------------------------------------
    # A live sibling under the same parent. The OLD code must destroy it (RED);
    # the new plan must spare it, and must also spare this run's own dir.
    local p1="$td/pr-1"
    mkdir -p "$p1/run-100/debug" "$p1/run-200/debug"
    : > "$p1/run-100/debug/artifact"
    : > "$p1/run-200/debug/artifact"
    printf '100 in_progress\n200 in_progress\n' > "$ORACLE_MAP"
    # BACKDATE both. Without this the age fallback spares them for being fresh,
    # and a mutation that deletes the `in_progress` arm entirely still passes --
    # measured: dropping `in_progress` from the LIVE arm left this row green.
    # Backdated, only the run-status answer can save them.
    touch -d '1970-01-02 00:00:00' "$p1/run-100/debug/artifact" "$p1/run-100/debug" \
        "$p1/run-100" "$p1/run-200/debug/artifact" "$p1/run-200/debug" "$p1/run-200"

    # RED half: the removal this script replaces, verbatim in shape.
    local old="$td/pr-1-old"
    cp -a "$p1" "$old"
    # `${old:?}` is the guard SEC011 asks for; the SHAPE under test is the
    # unguarded parent-wide glob itself, which is preserved.
    rm -rf "${old:?}"/* "${old:?}"/.[!.]* 2> /dev/null
    if [ ! -d "$old/run-200" ]; then
        printf 'ok    row 1a OLD parent-wide removal destroys the live sibling run-200\n'
    else
        printf 'FAIL  row 1a OLD removal left run-200 behind; the fixture does not reproduce the defect\n'
        fails=1
    fi

    # GREEN half: the new plan names neither the live sibling nor this run.
    got=$(reclaimable_in "$p1" 100 2> /dev/null | tr '\n' ' ')
    if [ -z "$got" ]; then
        printf 'ok    row 1b new plan reclaims nothing while run-200 is live\n'
    else
        printf 'FAIL  row 1b new plan would reclaim [%s], expected nothing\n' "$got"
        fails=1
    fi
    if [ -d "$p1/run-200" ] && [ -d "$p1/run-100" ]; then
        printf 'ok    row 1c both the live sibling and this run survive on disk\n'
    else
        printf 'FAIL  row 1c a directory was removed by the planner itself\n'
        fails=1
    fi

    # -- row 2 ------------------------------------------------------------
    # NOT INERT. A genuinely orphaned sibling is still reclaimed. This row
    # matters as much as row 1: a blanket skip passes row 1 and is useless.
    local p2="$td/pr-2"
    mkdir -p "$p2/run-100/debug" "$p2/run-300/debug" "$p2/run-400/debug"
    printf '100 in_progress\n300 completed\n400 in_progress\n' > "$ORACLE_MAP"
    touch -d '1970-01-02 00:00:00' "$p2/run-100/debug" "$p2/run-100" \
        "$p2/run-300/debug" "$p2/run-300" "$p2/run-400/debug" "$p2/run-400"
    got=$(reclaimable_in "$p2" 100 2> /dev/null | tr '\n' ' ')
    if [ "$got" = "run-300 " ]; then
        printf 'ok    row 2a the completed orphan run-300 is reclaimed, run-400 is not\n'
    else
        printf 'FAIL  row 2a got [%s], expected [run-300 ]\n' "$got"
        fails=1
    fi

    # -- row 3 ------------------------------------------------------------
    # Oracle unavailable (API outage) + a directory older than the job timeout
    # => still reclaimed. Without this the outage path is a blanket skip.
    local p3="$td/pr-3"
    mkdir -p "$p3/run-500/debug"
    : > "$p3/run-500/debug/artifact"
    printf '' > "$ORACLE_MAP"
    touch -d '1970-01-02 00:00:00' "$p3/run-500/debug/artifact" "$p3/run-500/debug" "$p3/run-500"
    got=$(reclaimable_in "$p3" 100 2> /dev/null | tr '\n' ' ')
    if [ "$got" = "run-500 " ]; then
        printf 'ok    row 3  oracle silent + stale dir is reclaimed (outage does not disarm the step)\n'
    else
        printf 'FAIL  row 3  got [%s], expected [run-500 ]\n' "$got"
        fails=1
    fi

    # -- row 4 ------------------------------------------------------------
    # Oracle unavailable + a freshly written directory => spared. The age
    # fallback must not become a second parent-wide removal.
    local p4="$td/pr-4"
    mkdir -p "$p4/run-600/debug"
    : > "$p4/run-600/debug/artifact"
    printf '' > "$ORACLE_MAP"
    got=$(reclaimable_in "$p4" 100 2> /dev/null | tr '\n' ' ')
    if [ -z "$got" ]; then
        printf 'ok    row 4  oracle silent + fresh dir is spared\n'
    else
        printf 'FAIL  row 4  got [%s], expected nothing\n' "$got"
        fails=1
    fi

    # -- row 5 ------------------------------------------------------------
    # This run's own directory is never named, whatever the oracle says about
    # it. guard-runner-labels of this same run is writing to it concurrently.
    local p5="$td/pr-5"
    mkdir -p "$p5/run-100/debug"
    printf '100 completed\n' > "$ORACLE_MAP"
    touch -d '1970-01-02 00:00:00' "$p5/run-100/debug" "$p5/run-100"
    got=$(reclaimable_in "$p5" 100 2> /dev/null | tr '\n' ' ')
    if [ -z "$got" ]; then
        printf 'ok    row 5  this run own dir is never reclaimed, even if the API calls it completed\n'
    else
        printf 'FAIL  row 5  got [%s], expected nothing\n' "$got"
        fails=1
    fi

    # -- row 6 ------------------------------------------------------------
    # A missing parent is not an error and reclaims nothing.
    got=$(reclaimable_in "$td/pr-absent" 100 2> /dev/null | tr '\n' ' ')
    if [ -z "$got" ]; then
        printf 'ok    row 6  a parent that does not exist yet reclaims nothing\n'
    else
        printf 'FAIL  row 6  got [%s], expected nothing\n' "$got"
        fails=1
    fi

    if [ "$fails" -ne 0 ]; then
        printf '\nSELF-TEST FAILED\n'
        return 1
    fi
    printf '\nself-test passed (6 rows)\n'
    return 0
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    --self-test)
        self_test
        exit $?
        ;;
    "" | -h | --help)
        printf 'usage: %s <parent-dir> <this-run-id>\n' "$0" >&2
        printf '       %s --self-test\n' "$0" >&2
        exit 2
        ;;
esac

if [ "$#" -lt 2 ]; then
    printf 'usage: %s <parent-dir> <this-run-id>\n' "$0" >&2
    exit 2
fi

reclaimable_in "$1" "$2"
