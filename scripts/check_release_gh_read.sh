#!/usr/bin/env bash
# check_release_gh_read.sh -- the release autopilot's GitHub reads survive two failed calls (#4939).
#
#   bash scripts/check_release_gh_read.sh            # the rows below; exit 0 only if every row holds
#
# THE PLANTED TEST (P1). A GitHub read that fails twice and then answers costs under 2 min and
# reruns no lane. A stub `gh` first on PATH fails its first two calls and answers the third. P1
# runs gh_read with the DEFAULT settings (no override shortens the waits), measures the wall
# time, and requires: the answer printed, exactly 3 calls, under 120 s, and inside
# gh_read_worst_s. "No lane rerun" is P5: autopilot.sh reads GitHub only through gh_read, so the
# retry happens inside the pass, and a blip no longer stops the pass to be restarted.
#
# THE ROWS
#   P1 fail-twice     default settings: answer, 3 calls, < 120 s, <= gh_read_worst_s.
#   P2 fail-thrice    3 failures: non-zero status, exactly 3 calls, no answer printed.
#   P3 hang           a call that never returns is killed at GH_READ_TIMEOUT_S and retried.
#   P4 writes         every write form is refused (status 2) and the stub is never called;
#                     every read form passes the write test.
#   P5 surface        autopilot.sh sources lib_gh_read.sh; every gh read it makes goes through
#                     gh_read, except the b2-gpu `--log` read, which has its own 6-try loop; no gh
#                     write goes through gh_read.
#   P6 bound          the defaults' worst case is under 2 min (gh_read_worst_s < 120).
#   M  mutants        a gh_read that never retries fails P1; one that retries without limit, or
#                     loses the failing status (rc read after an if), fails P2; one that retries
#                     writes fails P4. A guard that cannot turn red is not one.

set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
LIB="$ROOT/scripts/release/lib_gh_read.sh"
AP="$ROOT/scripts/release/autopilot.sh"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/gh-read.XXXXXX")
trap 'rm -rf -- "${TMP:?}"' EXIT
fails=0
row() { if [ "$2" = 0 ]; then printf 'ok    %s\n' "$1"; else printf 'FAIL  %s\n' "$1"; fails=$((fails + 1)); fi; }

# the stub: FAIL_FIRST failing calls (rc 1), then `answer`; HANG_FIRST calls that sleep for ever
mkdir -p "$TMP/bin"
cat > "$TMP/bin/gh" <<'STUB'
#!/usr/bin/env bash
n=$(( $(cat "$GH_STATE/calls" 2>/dev/null || echo 0) + 1 )); printf '%s\n' "$n" > "$GH_STATE/calls"
[ "$n" -le "${HANG_FIRST:-0}" ] && exec sleep 3600
[ "$n" -le "${FAIL_FIRST:-0}" ] && { printf 'stub: HTTP 502 on call %s\n' "$n" >&2; exit 1; }
printf 'answer\n'
STUB
chmod +x "$TMP/bin/gh"

# run LIB gh_read ARGS in a fresh state dir; sets OUT, RC, CALLS, SECS
try() {
    local lib=$1; shift
    local st; st=$(mktemp -d "$TMP/st.XXXXXX")
    local t0=$SECONDS
    set +e
    OUT=$(PATH="$TMP/bin:$PATH" GH_STATE="$st" bash -c '. "$1" || exit 9; shift; gh_read "$@"' _ "$lib" "$@" 2> "$st/err")
    RC=$?
    set -e
    SECS=$((SECONDS - t0))
    CALLS=$(cat "$st/calls" 2>/dev/null || echo 0)
}

p1() {  # LIB: 0 when the planted fail-twice read holds
    local worst
    worst=$(bash -c '. "$1"; gh_read_worst_s' _ "$1")
    FAIL_FIRST=2 try "$1" pr view 99 --json mergeCommit
    printf '      P1 measured: rc %s, %s calls, %s s (bound %s s)\n' "$RC" "$CALLS" "$SECS" "$worst"
    [ "$RC" = 0 ] && [ "$OUT" = answer ] && [ "$CALLS" = 3 ] && [ "$SECS" -lt 120 ] && [ "$SECS" -le "$worst" ]
}
p2() {
    FAIL_FIRST=9 GH_READ_WAIT_S=0 try "$1" run view 7 --json status
    [ "$RC" != 0 ] && [ -z "$OUT" ] && [ "$CALLS" = 3 ]
}
p3() {
    HANG_FIRST=1 GH_READ_TIMEOUT_S=1 GH_READ_WAIT_S=0 try "$1" release view v1 --json url
    [ "$RC" = 0 ] && [ "$OUT" = answer ] && [ "$CALLS" = 2 ] && [ "$SECS" -lt 30 ]
}
p4() {
    local w r
    while IFS= read -r w; do
        # shellcheck disable=SC2086 # the row is the argument list
        FAIL_FIRST=0 try "$1" $w
        { [ "$RC" = 2 ] && [ "$CALLS" = 0 ]; } || { printf '      P4 write not refused: gh %s (rc %s, %s calls)\n' "$w" "$RC" "$CALLS"; return 1; }
    done <<'W'
release create v1 --draft
release edit v1 --draft=false
workflow run b2-gpu.yml -f ref=v1
issue comment 1 --body x
issue close 1
pr create --base main
api -X PATCH repos/o/r/milestones/1 -f state=closed
api repos/o/r/issues -f title=x
api --method POST repos/o/r/dispatches
W
    while IFS= read -r r; do
        # shellcheck disable=SC2086
        bash -c '. "$1"; shift; gh_read_is_write "$@"' _ "$1" $r && { printf '      P4 read taken for a write: gh %s\n' "$r"; return 1; }
    done <<'R'
pr view 1 --json state
pr list --head b
run list --workflow x.yml
run view 1 --json jobs
release view v1 --json body
issue view 1 --json state
api repos/o/r/milestones/1 --jq .open_issues
api repos/o/r/issues?milestone=1 --paginate
R
    return 0
}

row "P1 fail-twice (default settings)" "$(p1 "$LIB" >&2; echo $?)"
row "P2 fail-thrice" "$(p2 "$LIB"; echo $?)"
row "P3 hang" "$(p3 "$LIB"; echo $?)"
row "P4 writes refused, reads read" "$(p4 "$LIB"; echo $?)"

# P5: the surface. Every executable `gh <read>` in autopilot.sh is a gh_read, bar the --log loop.
p5() {
    grep -qF '. "$REPO_ROOT/scripts/release/lib_gh_read.sh" || exit 2' "$AP" || { echo '      P5 lib not sourced'; return 1; }
    local bad
    bad=$(grep -nE '(^|[^_[:alnum:]])gh (pr (view|list)|run (view|list)|release view|issue view|api) ' "$AP" \
          | grep -vE '^[0-9]+:[[:space:]]*#' | grep -vF -- '--log >' | grep -vE 'gh api -X ' || true)
    [ -z "$bad" ] || { printf '      P5 bare gh read: %s\n' "$bad"; return 1; }
    bad=$(grep -nE 'gh_read (release (create|edit)|workflow run|issue (comment|close)|pr create|api -X)' "$AP" || true)
    [ -z "$bad" ] || { printf '      P5 write through gh_read: %s\n' "$bad"; return 1; }
    [ "$(grep -cE '\bgh_read (pr|run|release|issue|api) ' "$AP")" -ge 15 ] || { echo '      P5 fewer than 15 gh_read sites'; return 1; }
}
row "P5 autopilot reads through gh_read" "$(p5 >&2; echo $?)"
row "P6 default worst case < 120 s" "$( [ "$(bash -c '. "$1"; gh_read_worst_s' _ "$LIB")" -lt 120 ]; echo $?)"

# M: each mutant must turn its row red
mut() {  # NAME SED-EXPR ROWFN
    local m="$TMP/mut-$1.sh"
    sed -E "$2" "$LIB" > "$m"
    cmp -s "$m" "$LIB" && { printf 'FAIL  mutant %s did not apply\n' "$1"; fails=$((fails + 1)); return; }
    if "$3" "$m" > /dev/null 2>&1; then printf 'FAIL  mutant %s survived\n' "$1"; fails=$((fails + 1))
    else printf 'ok    mutant %s killed\n' "$1"; fi
}
mut no-retry   's/\[ "\$i" -ge "\$n" \] && return "\$rc"/return "$rc"/' p1
mut unbounded  's/\[ "\$i" -ge "\$n" \] && return "\$rc"/[ "$i" -ge 5 ] \&\& return "$rc"/' p2
mut retry-writes 's/^(    if gh_read_is_write "\$@"; then)$/    if false; then/' p4
mut rc-lost    's/^( +)rc=\$\?  #.*$/\1rc=0/' p2

[ "$fails" = 0 ] || { printf 'check_release_gh_read: %s row(s) red\n' "$fails"; exit 1; }
printf 'check_release_gh_read: all rows green\n'
