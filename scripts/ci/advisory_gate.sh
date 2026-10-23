#!/usr/bin/env bash
# advisory_gate.sh -- the advisories step of guard-cargo (PMAT-5792, contract advisory-gate-diff-v1).
#
#   pull_request / merge_group  fail only on advisories the PR INTRODUCES against the pinned build base
#                               (refs/remotes/origin/main, pinned by pin_build_base.sh); an advisory
#                               already on the base is reported, never a block.
#   any other event (push)      the full `cargo deny check advisories`, as before: a red there is the
#                               main-resident andon, not a PR block.
# Fails closed: anything that cannot be checked exits 75 and fails the step. Both sides are scanned with the
# BASE deny.toml, from the BASE worktree's script, over ONE advisory-db fetched once by this job.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EV="${GITHUB_EVENT_NAME:-}"
case "$EV" in
    pull_request | merge_group) ;;
    *) exec cargo deny --all-features check advisories ;;
esac

base_sha() { # the build base the pin step named
    git rev-parse --verify -q refs/remotes/origin/main^{commit} || { echo "COULD-NOT-CHECK: origin/main is not pinned" >&2; return 75; }
}
BASE=$(base_sha) || exit 75
# merge_group: the pin must equal merge_group.base_sha; a drifted pin is refused, not guessed around.
if [ "$EV" = merge_group ] && [ -n "${MG_BASE_SHA:-}" ] && [ "$BASE" != "$MG_BASE_SHA" ]; then
    echo "COULD-NOT-CHECK: pinned base $BASE != merge_group base_sha $MG_BASE_SHA" >&2
    exit 75
fi
bash "$HERE/advisory_diff.sh" base-ref "$EV" "${MG_BASE_SHA:-}" "$BASE" >/dev/null || exit 75

W="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/advgate-base-$$"
trap 'rm -f "${W:?}.pr.ndjson" "${W:?}.base.ndjson"; git worktree remove --force "${W:?}" >/dev/null 2>&1 || true' EXIT
git worktree add --detach "$W" "$BASE" >/dev/null 2>&1 || { echo "COULD-NOT-CHECK: no worktree for base $BASE (shallow fetch?)" >&2; exit 75; }

# (before the subcommand) and replaced --disable-fetch with the global --offline, which `check` needs so both scans read the ONE advisory-db
# fetched above (without it 0.20 `check` may re-fetch the db and the two scans see different heads). --offline also makes cargo metadata need
# every crate in the registry cache (a cold runner exits rc=1 with log lines only, m70 #5793, measured 2026-10-21 on 0.20.2 with an empty
# CARGO_HOME), so `cargo fetch --locked` warms the cache for BOTH trees before either scan. Probe once, never assume the runner's version.
deny_help=$(cargo deny check --help 2>&1 || true)
if grep -qF -- '--config' <<<"$deny_help"; then DENY_OLD=1; else DENY_OLD=0; fi

# One DB for both scans: fetched with the BASE config (the db-path both scans read), HEAD logged.
CFG="$W/deny.toml"
# P9 option 4 (PMAT-6110): when the BASE carries ci/advisory-db.pin, both scans read that pinned snapshot, not the live
# DB. The pin and its script come from the BASE worktree, so a PR cannot move its own pin. A pin not verified for
# 7 days is RED (F5, rc 1); a missing/garbled pin or a failed fetch is 75. Only a cargo-deny whose db directory name
# was not derived (rc 3) keeps the live fetch below; the bootstrap merge (base has no pin) does too.
PINNED=0
if [ -f "$W/ci/advisory-db.pin" ] && [ -f "$W/scripts/advisory_pin.sh" ]; then
    pin_rc=0
    PD="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/advisory-db-pinned-$$"
    CFG=$(cd "$W" && bash scripts/advisory_pin.sh place "$W/deny.toml" "$PD") || pin_rc=$?
    case "$pin_rc" in
        0) PINNED=1; trap 'rm -rf -- "${PD:?}"; rm -f "${W:?}.pr.ndjson" "${W:?}.base.ndjson"; git worktree remove --force "${W:?}" >/dev/null 2>&1 || true' EXIT ;;
        3) CFG="$W/deny.toml"; echo "advisory-db: pinned snapshot unsupported on this cargo-deny; live fetch" ;;
        *) exit "$pin_rc" ;;
    esac
fi
if [ "$PINNED" = 1 ]; then
    echo "advisory-db pinned $(sed -n 's/^sha=//p' "$W/ci/advisory-db.pin") verified $(sed -n 's/^verified=//p' "$W/ci/advisory-db.pin")"
else
    if [ "$DENY_OLD" = 1 ]; then cargo deny fetch --config "$W/deny.toml" db; else cargo deny --config "$W/deny.toml" fetch db; fi \
        || { echo "COULD-NOT-CHECK: advisory-db fetch failed" >&2; exit 75; }
    for d in "$HOME"/.cargo/advisory-dbs/*; do
        [ -d "$d/.git" ] && echo "advisory-db $(basename "$d") HEAD $(git -C "$d" rev-parse --short HEAD 2>/dev/null) $(git -C "$d" log -1 --format=%cI 2>/dev/null)"
    done
fi

# The BASE script when it has one; the PR's copy only to bootstrap the merge that introduces the script.
GATE="$W/scripts/ci/advisory_diff.sh"
[ -f "$GATE" ] || GATE="$HERE/advisory_diff.sh"

bash "$GATE" ignores-added "$W/deny.toml" deny.toml || exit 1

scan() { # TREE OUT.ndjson -> prints the cargo-deny rc
    local rc=0
    if [ "$DENY_OLD" = 1 ]; then
        (cd "$1" && cargo deny --locked --all-features --format json check --config "$CFG" --disable-fetch advisories) \
            >/dev/null 2>"$2" || rc=$?
    else
        (cd "$1" && cargo deny --config "$CFG" --locked --offline --all-features --format json check advisories) \
            >/dev/null 2>"$2" || rc=$?
    fi
    echo "$rc"
}
# warm the registry cache for both trees (network allowed here, nothing else reads it); the scans then run offline
for t in "$PWD" "$W"; do (cd "$t" && cargo fetch --locked) >/dev/null 2>&1 || { echo "COULD-NOT-CHECK: cargo fetch --locked failed in $t" >&2; exit 75; }; done
PRRC=$(scan "$PWD" "$W.pr.ndjson")
BASERC=$(scan "$W" "$W.base.ndjson")
bash "$GATE" verdict "$W.pr.ndjson" "$PRRC" "$W.base.ndjson" "$BASERC"
