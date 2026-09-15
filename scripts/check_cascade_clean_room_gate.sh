#!/usr/bin/env bash
# The cascade's CLEAN-ROOM gate must refuse everything it cannot prove (PMAT-3318).
#
# `clean_room_gate` in scripts/cascade-publish.sh refuses to start a publish
# unless infra's clean-room.yml `clean-room (aprender)` job concluded success on
# EXACTLY the tag's commit. This is its falsifier: a hermetic case table, both
# polarities, run against the REAL functions (extracted from cascade-publish.sh,
# never re-implemented -- rename or delete them and extraction fails here).
#
# EVERY ROW RUNS AGAINST A STUB `gh` placed first on PATH, with GH_CONFIG_DIR
# pointed at an empty directory and GH_TOKEN/GITHUB_TOKEN unset, so a row that
# escaped the stub would be unauthenticated rather than a live API call. The
# stub logs every invocation and answers anything unexpected with rc 97, and
# the table fails if any such call happened. One row proves the stub engaged.
#
#   bash scripts/check_cascade_clean_room_gate.sh
set -euo pipefail

case "${1:-}" in
  -h|--help) echo "usage: $0   (runs the clean-room gate case table; no arguments)"; exit 0 ;;
  '') : ;;
  *) echo "usage: $0" >&2; exit 2 ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CASCADE="${CASCADE_UNDER_TEST:-$REPO_ROOT/scripts/cascade-publish.sh}"
echo "=== cascade clean-room gate (check_cascade_clean_room_gate.sh) ==="
[ -f "$CASCADE" ] || { echo "FAIL: $CASCADE not found"; exit 1; }

WORK=$(mktemp -d)
case "$WORK" in
  /tmp/*|/var/folders/*) : ;;
  *) echo "FAIL: mktemp -d returned an unexpected path '$WORK'; refusing to clean up"; exit 1 ;;
esac
trap 'rm -rf "${WORK:?}"' EXIT

FNS="$WORK/fns.sh"
: > "$FNS"
for fn in clean_room_parse_runs clean_room_parse_job clean_room_tested_abbrevs clean_room_gate; do
  sed -n "/^${fn}() {/,/^}/p" "$CASCADE" >> "$FNS"
  grep -q "^${fn}() {" "$FNS" || { echo "FAIL: could not extract '$fn' from $CASCADE"; exit 1; }
done
# shellcheck disable=SC1090
. "$FNS"

# ── fixture repository: base, the tagged commit A, a later commit B ──
FIX="$WORK/repo"
mkdir -p "$FIX"
fix_commit() {
  GIT_AUTHOR_DATE="$1" GIT_COMMITTER_DATE="$1" \
    git -C "$FIX" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t -c commit.gpgsign=false \
    commit -q --allow-empty -m "$2"
}
git -C "$FIX" init -q
fix_commit 2026-09-01T00:00:00Z base
fix_commit 2026-09-01T01:00:00Z tagged
git -C "$FIX" tag v1.2.3
fix_commit 2026-09-01T02:00:00Z later
SHA_A=$(git -C "$FIX" rev-parse 'v1.2.3^{commit}')
SHA_B=$(git -C "$FIX" rev-parse HEAD)
AB_A=${SHA_A:0:8}
AB_B=${SHA_B:0:8}

# ── the stub gh ──
BIN="$WORK/bin"
mkdir -p "$BIN" "$WORK/ghconfig"
cat > "$BIN/gh" <<'STUB'
#!/usr/bin/env bash
d=${STUB_DIR:?stub gh called without STUB_DIR}
printf '%s\n' "$*" >> "$d/calls"
case "$1" in
  auth)
    [ "${2:-}" = status ] || { echo "UNEXPECTED $*" >> "$d/calls"; exit 97; }
    exit "$(cat "$d/auth_rc" 2>/dev/null || echo 0)" ;;
  run)
    case "${2:-}" in
      list) if [ -f "$d/runlist_rc" ]; then exit "$(cat "$d/runlist_rc")"; fi
            cat "$d/runs.json" ;;
      view) cat "$d/jobs-${3:-x}.json" 2>/dev/null || exit 1 ;;
      *) echo "UNEXPECTED $*" >> "$d/calls"; exit 97 ;;
    esac ;;
  api)
    case "${2:-}" in
      repos/paiml/infra/actions/jobs/*/logs)
        jid=${2#repos/paiml/infra/actions/jobs/}; jid=${jid%/logs}
        cat "$d/log-$jid.txt" 2>/dev/null || exit 1 ;;
      *) echo "UNEXPECTED $*" >> "$d/calls"; exit 97 ;;
    esac ;;
  *) echo "UNEXPECTED $*" >> "$d/calls"; exit 97 ;;
esac
STUB
chmod +x "$BIN/gh"

# ── fixture writers ──
S=""   # the current row's stub directory
runs() { printf '[%s]\n' "$1" > "$S/runs.json"; }
run_obj() { printf '{"databaseId":%s,"status":"%s","conclusion":"%s","createdAt":"%s"}' "$1" "$2" "$3" "${4:-2026-09-02T00:00:00Z}"; }
jobs() { printf '{"jobs":[{"databaseId":1,"name":"matrix-setup","status":"completed","conclusion":"success"},{"databaseId":%s,"name":"clean-room (aprender)","status":"%s","conclusion":"%s"}]}\n' "$2" "$3" "$4" > "$S/jobs-$1.json"; }
log_commit() {
  {
    printf '2026-09-02T00:00:01.0000000Z Cloning into %s...\n' "'/tmp/clean-room-src/aprender'"
    printf '2026-09-02T00:03:00.0000000Z ==> Copying aprender source from /tmp/clean-room-src/aprender\n'
    printf '2026-09-02T00:03:00.1000000Z     version: 1.2.3\n'
    printf '2026-09-02T00:03:00.2000000Z     commit:  %s\r\n' "$2"
  } > "$S/log-$1.txt"
}

rc=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; rc=1; }
LAST_OUT=""
LAST_STUB=""

# row NAME EXPECT_RC NEEDLE SETUP [TAG]
row() {
  local name=$1 expect=$2 needle=$3 setup=$4 tag=${5:-v1.2.3} out got
  S="$WORK/stub-$name"; mkdir -p "$S"; : > "$S/calls"
  "$setup"
  got=0
  out=$(
    export PATH="$BIN:$PATH" STUB_DIR="$S" GH_CONFIG_DIR="$WORK/ghconfig"
    unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN
    clean_room_gate "$FIX" "$tag" 2>&1
  ) || got=$?
  LAST_OUT=$out; LAST_STUB=$S
  if grep -q '^UNEXPECTED' "$S/calls"; then
    fail "$name: the gate made a gh call the stub does not model: $(grep '^UNEXPECTED' "$S/calls" | head -1)"
  elif [ "$got" -ne "$expect" ]; then
    fail "$name: expected rc=$expect got rc=$got"; printf '      | %s\n' "$out"
  elif [[ "$out" != *"$needle"* ]]; then
    fail "$name: rc=$got but output never said '$needle'"; printf '      | %s\n' "$out"
  else
    pass "$name (rc=$got)"
  fi
}

s_green_tag()      { runs "$(run_obj 9001 completed success)"; jobs 9001 501 completed success; log_commit 501 "$AB_A"; }
s_green_other()    { runs "$(run_obj 9001 completed success)"; jobs 9001 501 completed success; log_commit 501 "$AB_B"; }
s_failed_tag()     { runs "$(run_obj 9001 completed failure)"; jobs 9001 501 completed failure; log_commit 501 "$AB_A"; }
s_cancelled_tag()  { runs "$(run_obj 9001 completed cancelled)"; jobs 9001 501 completed cancelled; log_commit 501 "$AB_A"; }
s_inprogress_tag() { runs "$(run_obj 9001 in_progress '')"; jobs 9001 501 in_progress ''; log_commit 501 "$AB_A"; }
s_queued()         { runs "$(run_obj 9001 queued '')"; printf '{"jobs":[]}\n' > "$S/jobs-9001.json"; }
s_no_runs()        { runs ""; }
s_unauth()         { echo 1 > "$S/auth_rc"; s_green_tag; }
s_runlist_err()    { echo 1 > "$S/runlist_rc"; s_green_tag; }
s_log_err()        { runs "$(run_obj 9001 completed success)"; jobs 9001 501 completed success; }
s_bad_runs()       { printf '<html>rate limited</html>\n' > "$S/runs.json"; }
s_bad_runs_shape() { printf '[{"databaseId":"9001","status":"completed","conclusion":"success","createdAt":"yesterday"}]\n' > "$S/runs.json"; }
s_bad_jobs()       { runs "$(run_obj 9001 completed success)"; printf '{"jobs":\n' > "$S/jobs-9001.json"; }
s_log_nocommit()   { s_green_tag; grep -v 'commit:' "$S/log-501.txt" > "$S/l" || true; mv "$S/l" "$S/log-501.txt"; }
s_log_two()        { s_green_tag; printf '2026-09-02T00:04:00.0000000Z     commit:  %s\n' "$AB_B" >> "$S/log-501.txt"; }
s_log_reworded()   { runs "$(run_obj 9001 completed success)"; jobs 9001 501 completed success
                     printf '2026-09-02T00:03:00.2000000Z commit: %s\n' "$AB_A" > "$S/log-501.txt"; }
s_predates()       { runs "$(run_obj 9001 completed success 2026-08-31T00:00:00Z)"; jobs 9001 501 completed success; log_commit 501 "$AB_A"; }
s_newer_red()      { runs "$(run_obj 9002 completed failure),$(run_obj 9001 completed success)"
                     jobs 9002 502 completed failure; log_commit 502 "$AB_B"
                     jobs 9001 501 completed success; log_commit 501 "$AB_A"; }
s_full_sha()       { runs "$(run_obj 9001 completed success)"; jobs 9001 501 completed success; log_commit 501 "$SHA_A"; }
s_no_aprender_job(){ runs "$(run_obj 9001 completed success)"
                     printf '{"jobs":[{"databaseId":7,"name":"clean-room (forjar)","status":"completed","conclusion":"success"}]}\n' > "$S/jobs-9001.json"; }

row green_on_tag_sha_proceeds            0 "CLEAN-ROOM PROCEED: run 9001 job 501"  s_green_tag
row green_on_different_sha_refuses       1 "tested $AB_B"                          s_green_other
row failed_on_tag_sha_refuses            1 "conclusion=failure"                    s_failed_tag
row cancelled_on_tag_sha_refuses         1 "conclusion=cancelled"                  s_cancelled_tag
row in_progress_on_tag_sha_refuses       1 "in_progress -- tested sha unknown"     s_inprogress_tag
row queued_run_refuses                   1 "no 'clean-room (aprender)' job"        s_queued
row no_runs_refuses                      1 "found none"                            s_no_runs
row gh_unauthenticated_refuses           1 "unauthenticated or erroring"           s_unauth
row gh_run_list_error_refuses            1 "gh run list --repo paiml/infra"        s_runlist_err
row gh_log_fetch_error_refuses           1 "could not read the log of run 9001"    s_log_err
row malformed_run_list_refuses           1 "run list could not be parsed"          s_bad_runs
row wrong_shape_run_list_refuses         1 "run list could not be parsed"          s_bad_runs_shape
row malformed_jobs_refuses               1 "jobs of run 9001 could not be parsed"  s_bad_jobs
row log_without_commit_line_refuses      1 "0 tested-commit record(s)"             s_log_nocommit
row log_with_two_commits_refuses         1 "2 tested-commit record(s)"             s_log_two
row reworded_commit_line_refuses         1 "0 tested-commit record(s)"             s_log_reworded
row run_created_before_commit_refuses    1 "(0 clean-room.yml run(s)"              s_predates
row only_other_repo_jobs_refuses         1 "no 'clean-room (aprender)' job"        s_no_aprender_job
row unknown_tag_refuses                  1 "does not resolve to a commit"          s_green_tag v9.9.9
row older_green_behind_newer_red_proceeds 0 "CLEAN-ROOM PROCEED: run 9001"         s_newer_red
row full_40_char_sha_line_proceeds       0 "tested $SHA_A = $SHA_A"                s_full_sha

# No bypass: the variables anyone would reach for change nothing.
s_bypass() { s_no_runs; }
SKIP_CLEAN_ROOM=1 CLEAN_ROOM_SKIP=1 SKIP_TESTS=1 CASCADE_SKIP_CLEAN_ROOM=1 \
  row bypass_env_vars_ignored            1 "found none"                            s_bypass

# The stub engaged: the green row's calls went to OUR gh, in the expected order.
row shim_engaged_probe                   0 "CLEAN-ROOM PROCEED"                    s_green_tag
resolved_gh=$(export PATH="$BIN:$PATH"; command -v gh)
calls=$(tr '\n' '|' < "$LAST_STUB/calls")
if [ "$resolved_gh" = "$BIN/gh" ] \
   && [[ "$calls" == "auth status|run list --repo paiml/infra --workflow clean-room.yml"*"|run view 9001 --repo paiml/infra --json jobs|api repos/paiml/infra/actions/jobs/501/logs|" ]]; then
  pass "gh_is_the_stub (resolved $resolved_gh; calls: $calls)"
else
  fail "gh_is_the_stub: gh resolved to '$resolved_gh', calls were '$calls'"
fi

# Wiring: the gate is called on the publishing path, BEFORE the preflight, and
# nothing in the script names a way to skip it.
gate_ln=$(grep -n 'if ! clean_room_gate "\$REPO_ROOT" "v\$TARGET_VERSION"; then' "$CASCADE" | head -1 | cut -d: -f1)
pre_ln=$(grep -n 'if ! bash "\$REPO_ROOT/scripts/check_publish_preflight.sh"; then' "$CASCADE" | head -1 | cut -d: -f1)
skip_hits=$(grep -ciE 'skip[-_]?clean[-_]?room|clean[-_]?room[-_]?skip|CLEAN_ROOM_(BYPASS|OVERRIDE|OFF)' "$CASCADE" || true)
if [ -n "$gate_ln" ] && [ -n "$pre_ln" ] && [ "$gate_ln" -lt "$pre_ln" ] && [ "${skip_hits:-0}" -eq 0 ]; then
  pass "gate_wired_before_preflight (gate line $gate_ln < preflight line $pre_ln; bypass tokens: 0)"
else
  fail "gate_wired_before_preflight: gate line '${gate_ln:-none}', preflight line '${pre_ln:-none}', bypass tokens ${skip_hits:-?}"
fi

if [ "$rc" -eq 0 ]; then
  echo "PASS  clean-room gate: every row held"
else
  echo "FAIL  clean-room gate: a row broke (see above)"
fi
exit "$rc"
