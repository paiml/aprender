#!/usr/bin/env bash
# One door to cargo publish (#4687): a clean-room run id that is missing, red, or
# tested another commit must refuse -- at BOTH doors, through ONE verifier.
#
# This is the case table. Every row runs through
#   door 1  scripts/release/publish_strict.sh <version>, the real script, against a
#           fixture tag checkout and state dir (RELEASE_AP), up to the point where
#           it enumerates crates: reaching the (stubbed) cascade_universe.py call means
#           every precondition, the clean-room one included, let it through;
#   door 2  clean_room_gate, extracted from scripts/cascade-publish.sh (never
#           re-implemented), with scripts/release/lib_clean_room.sh sourced.
# Rows marked `ps` have no meaning at door 2 (it searches runs; it reads no id file).
# R5c: the B2-gpu run id and the dry-run receipt commit get the same treatment at
# door 1 -- missing, red, another commit, or gh/git failing refuses; green on the
# tag's commit accepts. Each row names the gate that must refuse it.
#
# HERMETIC. gh, cargo, curl and python3 are stubs first on PATH; HOME, CARGO_HOME
# and GH_CONFIG_DIR point into a temp dir; GH tokens are unset. The gh stub serves
# canned JSON, logs every call, and applies --jq with jq as gh's built-in jq does.
# Any call the stub does not model, and any `cargo publish`, fails the table.
#
#   bash scripts/release/check_clean_room_verify.sh              # the table
#   bash scripts/release/check_clean_room_verify.sh --mutants    # each refusal dropped: table must go RED
#   CLEAN_ROOM_TREE=<checkout> bash ...                           # judge another tree (e.g. origin/main)
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
  '') MODE=table ;;
  --mutants) MODE=mutants ;;
  *) echo "usage: $0 [--mutants]" >&2; exit 2 ;;
esac

SELF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TREE=${CLEAN_ROOM_TREE:-$SELF_ROOT}
command -v jq > /dev/null || { echo "NOT_MEASURED: jq is not installed; the gh stub needs it to apply --jq"; exit 1; }

WORK=$(mktemp -d)
case "$WORK" in
  /tmp/*|/var/folders/*) : ;;
  *) echo "FAIL: mktemp -d returned an unexpected path '$WORK'; refusing to clean up"; exit 1 ;;
esac
trap 'rm -rf "${WORK:?}"' EXIT

# ── mutants: each drops ONE refusal from a copy of the tree; the table must go RED ──
# name@@file@@old@@new -- literal text. Lines sharing a name form one mutant.
# A mutant whose old text is absent is an ERROR, never a survivor.
# Not mutated, because equivalent: the B2-gpu record count and the receipt tag-row sha count.
# Two values joined by a newline never equal a 40-hex sha, so the next check refuses the same
# rows. Those checks exist to name the reason, and the b2_two_records and rcpt_two_shas rows pin it.
mutants() {
  cat <<'M'
ps_door_skips_verify@@scripts/release/publish_strict.sh@@cverdict=$(clean_room_verify_run "$crid" "$ctag" "$WT") || die "clean-room: run $crid is not proof for $T: $cverdict"@@cverdict=skipped
ps_door_back_to_nonempty_file@@scripts/release/publish_strict.sh@@crid=$(clean_room_read_run_id "$AP/cleanroom-run-id") ||@@[ -s "$AP/cleanroom-run-id" ] || die "clean-room: no run id"; crid=$(head -n 1 "$AP/cleanroom-run-id") ||
ps_door_back_to_nonempty_file@@scripts/release/publish_strict.sh@@cverdict=$(clean_room_verify_run "$crid" "$ctag" "$WT") || die "clean-room: run $crid is not proof for $T: $cverdict"@@cverdict=skipped
cascade_door_skips_verify@@scripts/cascade-publish.sh@@verdict=$(clean_room_verify_run "$rid" "$want" "$root") && vrc=0 || vrc=$?@@verdict="CLEAN-ROOM VERIFIED: forced"; vrc=0
cascade_not_measured_continues@@scripts/cascade-publish.sh@@*) echo "CLEAN-ROOM REFUSE: looked for $want (tag $tag); ${verdict#CLEAN-ROOM NOT-MEASURED: }"; return 1 ;;@@*) : ;;
id_missing_or_empty@@scripts/release/lib_clean_room.sh@@if [ ! -s "$f" ]; then@@if false; then
id_missing_or_empty@@scripts/release/lib_clean_room.sh@@if [ "$n" != 1 ]; then@@if false; then
id_missing_or_empty@@scripts/release/lib_clean_room.sh@@''|*[!0-9]*) echo "CLEAN-ROOM NOT-MEASURED: '$rid' in $f@@*[!0-9]*) echo "CLEAN-ROOM NOT-MEASURED: '$rid' in $f
id_missing_or_empty@@scripts/release/lib_clean_room.sh@@''|*[!0-9]*) echo "CLEAN-ROOM NOT-MEASURED: '$rid' is not a run id"@@*[!0-9]*) echo "CLEAN-ROOM NOT-MEASURED: '$rid' is not a run id"
id_more_than_one_line@@scripts/release/lib_clean_room.sh@@if [ "$n" != 1 ]; then@@if false; then
id_not_numeric@@scripts/release/lib_clean_room.sh@@''|*[!0-9]*) echo "CLEAN-ROOM NOT-MEASURED: '$rid' in $f@@__never__) echo "CLEAN-ROOM NOT-MEASURED: '$rid' in $f
id_not_numeric@@scripts/release/lib_clean_room.sh@@''|*[!0-9]*) echo "CLEAN-ROOM NOT-MEASURED: '$rid' is not a run id"@@__never__) echo "CLEAN-ROOM NOT-MEASURED: '$rid' is not a run id"
job_not_completed@@scripts/release/lib_clean_room.sh@@if [ "$jstatus" != "completed" ]; then@@if false; then
job_not_success@@scripts/release/lib_clean_room.sh@@if [ "$jconcl" != "success" ]; then@@if false; then
tested_other_commit@@scripts/release/lib_clean_room.sh@@if [ "$resolved" != "$want" ]; then@@if false; then
assert_step_not_green@@scripts/release/lib_clean_room.sh@@if [ "$jassert" != "completed/success" ]; then@@if false; then
more_than_one_job@@scripts/release/lib_clean_room.sh@@if ($h | length) > 1 then@@if false then
no_aprender_job@@scripts/release/lib_clean_room.sh@@elif ($h | length) == 0 then "NONE"@@elif ($h | length) == 0 then "OK\t1\tcompleted\tsuccess\tcompleted/success"
gh_view_fails_is_not_proof@@scripts/release/lib_clean_room.sh@@echo "CLEAN-ROOM NOT-MEASURED: gh run view $rid failed, or the jobs of run $rid could not be parsed"; return 2@@echo "run $rid: unread"; return 1
gh_log_fails_is_not_proof@@scripts/release/lib_clean_room.sh@@echo "CLEAN-ROOM NOT-MEASURED: could not read the log of run $rid job $jid"; return 2@@echo "run $rid: unread"; return 1
ps_b2_skips_verify@@scripts/release/publish_strict.sh@@gverdict=$(b2gpu_verify_run "$grid" "$ctag") || die "b2-gpu: run $grid is not proof for $T: $gverdict"@@gverdict=skipped
ps_b2_back_to_nonempty_file@@scripts/release/publish_strict.sh@@grid=$(b2gpu_read_run_id "$AP/b2gpu-run-id") ||@@[ -s "$AP/b2gpu-run-id" ] || die "b2-gpu: no run id"; grid=$(head -n 1 "$AP/b2gpu-run-id") ||
ps_b2_back_to_nonempty_file@@scripts/release/publish_strict.sh@@gverdict=$(b2gpu_verify_run "$grid" "$ctag") || die "b2-gpu: run $grid is not proof for $T: $gverdict"@@gverdict=skipped
ps_receipt_skips_verify@@scripts/release/publish_strict.sh@@dverdict=$(dryrun_receipt_verify "$AP/dryrun-receipt-commit" "$T" "$ctag" "$WT") || die "dry-run receipt: not proof for $T (T-4): $dverdict"@@dverdict=skipped
ps_receipt_back_to_nonempty_file@@scripts/release/publish_strict.sh@@dverdict=$(dryrun_receipt_verify "$AP/dryrun-receipt-commit" "$T" "$ctag" "$WT") || die "dry-run receipt: not proof for $T (T-4): $dverdict"@@[ -s "$AP/dryrun-receipt-commit" ] || die "dry-run receipt: none"; dverdict=skipped
b2_id_more_than_one_line@@scripts/release/lib_clean_room.sh@@if [ "$bn" != 1 ]; then@@if false; then
b2_more_than_one_job@@scripts/release/lib_clean_room.sh@@if ($h | length) > 1 then "BAD\t\($h | length) b2-gpu@@if false then "BAD\t\($h | length) b2-gpu
b2_no_job@@scripts/release/lib_clean_room.sh@@elif ($h | length) < 1 then "NONE"@@elif ($h | length) < 1 then "OK\t1\tcompleted\tsuccess\tcompleted/success"
b2_gh_view_fails_is_not_proof@@scripts/release/lib_clean_room.sh@@echo "B2-GPU NOT-MEASURED: gh run view $rid failed, or the jobs of run $rid could not be parsed"; return 2@@echo "B2-GPU VERIFIED: unread"; return 0
b2_gh_log_fails_is_not_proof@@scripts/release/lib_clean_room.sh@@echo "B2-GPU NOT-MEASURED: could not read the log of run $rid job $jid"; return 2@@echo "B2-GPU VERIFIED: unread"; return 0
b2_job_not_completed@@scripts/release/lib_clean_room.sh@@if [ "$gstatus" != "completed" ]; then@@if false; then
b2_assert_step_not_green@@scripts/release/lib_clean_room.sh@@if [ "$gassert" != "completed/success" ]; then@@if false; then
b2_tested_other_commit@@scripts/release/lib_clean_room.sh@@if [ "$gshas" != "$want" ]; then@@if false; then
b2_job_not_success@@scripts/release/lib_clean_room.sh@@if [ "$gconcl" != "success" ]; then@@if false; then
rcpt_more_than_one_line@@scripts/release/lib_clean_room.sh@@if [ "$rn" != 1 ]; then@@if false; then
rcpt_abbrev_sha@@scripts/release/lib_clean_room.sh@@if ! release_is_full_sha "$rsha"; then@@if false; then
rcpt_absent_commit_is_not_proof@@scripts/release/lib_clean_room.sh@@echo "DRY-RUN NOT-MEASURED: commit $rsha is not in $root (fetch it)"; return 2@@echo "DRY-RUN VERIFIED: unread"; return 0
rcpt_no_file@@scripts/release/lib_clean_room.sh@@echo "receipt commit $rsha: no $path in it"; return 1@@echo "DRY-RUN VERIFIED: unread"; return 0
rcpt_other_commit@@scripts/release/lib_clean_room.sh@@if [ "$tshas" != "$want" ]; then@@if false; then
rcpt_flags@@scripts/release/lib_clean_room.sh@@if ! grep -qE '(^|[`| ])cargo publish --workspace --dry-run --no-verify --locked([`| ]|$)' <<< "$row"; then@@if false; then
rcpt_flags_substring@@scripts/release/lib_clean_room.sh@@if ! grep -qE '(^|[`| ])cargo publish --workspace --dry-run --no-verify --locked([`| ]|$)' <<< "$row"; then@@if [[ "$row" != *--dry-run* || "$row" != *--no-verify* || "$row" != *--locked* ]]; then
rcpt_rc_unbounded@@scripts/release/lib_clean_room.sh@@rcs=$(grep -oE '(^|[ ,(|])rc=\**[0-9]+\**([ ,)|]|$)' <<< "$row" | grep -oE '[0-9]+'@@rcs=$(grep -oE 'rc=\**[0-9]+' <<< "$row" | grep -oE '[0-9]+'
rcpt_red@@scripts/release/lib_clean_room.sh@@if [ "$rcs" != 0 ]; then@@if false; then
rcpt_tree_not_clean@@scripts/release/lib_clean_room.sh@@if ! grep -qE '(^|, |\| *)tree clean after([ ,(|]|$)' <<< "$row"; then@@if false; then
rcpt_tree_substring@@scripts/release/lib_clean_room.sh@@if ! grep -qE '(^|, |\| *)tree clean after([ ,(|]|$)' <<< "$row"; then@@if [[ "$row" != *"tree clean"* ]]; then
rcpt_tree_qualified@@scripts/release/lib_clean_room.sh@@if grep -qiE '(^|[^a-z])(not|dirty|unclean)([^a-z]|$)' <<< "$row"; then@@if false; then
M
}

mutant_tree() { # mutant_tree DIR -- a copy of the four files the doors read
  local d=$1 f
  for f in scripts/release/publish_strict.sh scripts/release/lib_release_params.sh \
           scripts/release/lib_clean_room.sh scripts/cascade-publish.sh; do
    mkdir -p "$d/$(dirname "$f")"; cp "$TREE/$f" "$d/$f"
  done
}
mutate() { # mutate FILE OLD NEW -- exact literal replace of every occurrence; rc 1 if OLD is absent
  grep -qF -- "$2" "$1" || return 1
  OLD=$2 NEW=$3 perl -0pi -e 's/\Q$ENV{OLD}\E/$ENV{NEW}/g' "$1"
}
if [ "$MODE" = mutants ]; then
  killed=0; total=0; errors=0; names=$(mutants | sed 's/@@.*//' | awk '!seen[$0]++')
  while IFS= read -r name; do
    d="$WORK/mut-$name"; mutant_tree "$d"; applied=1
    while IFS= read -r l; do
      rest=${l#*@@}; file=${rest%%@@*}; rest=${rest#*@@}; old=${rest%%@@*}; new=${rest#*@@}
      mutate "$d/$file" "$old" "$new" || { applied=0; printf 'ERROR     %s: patch did not apply to %s: %s\n' "$name" "$file" "$old"; }
    done < <(mutants | grep -E "^${name}@@")
    if [ "$applied" -eq 0 ]; then errors=$((errors + 1)); continue; fi
    total=$((total + 1)); mrc=0
    out=$(CLEAN_ROOM_TREE=$d bash "$0" 2>&1) || mrc=$?
    # A kill is a bad input ACCEPTED (L25): a row that only BROKE says nothing about the refusal.
    if [ "$mrc" -ne 0 ] && grep -q '^FAIL  .*got ACCEPT$' <<< "$out"; then
      killed=$((killed + 1)); printf 'killed    %s -- %s\n' "$name" "$(grep '^FAIL  .*got ACCEPT$' <<< "$out" | head -1 | cut -c7-100)"
    else
      printf 'SURVIVED  %s\n' "$name"
    fi
  done <<< "$names"
  echo "MUTANTS killed $killed/$total, errors $errors"
  [ "$killed" -eq "$total" ] && [ "$errors" -eq 0 ] && [ "$total" -gt 0 ]
  exit $?
fi

echo "=== clean-room one door (check_clean_room_verify.sh) ==="
PS="$TREE/scripts/release/publish_strict.sh"
CASCADE="$TREE/scripts/cascade-publish.sh"
LIB="$TREE/scripts/release/lib_clean_room.sh"
[ -f "$PS" ] || { echo "FAIL  publish_strict.sh not found under the tree"; exit 1; }

# ── fixture: state dir AP, tag checkout AP/wt detached at v1.2.3, a later commit B ──
AP="$WORK/ap"; WT="$AP/wt"; FHOME="$WORK/home"; SBIN="$WORK/sbin"
mkdir -p "$WT" "$SBIN" "$FHOME/.cargo" "$WORK/ghconfig"
printf '[registry]\ntoken = "fixture-not-a-token"\n' > "$FHOME/.cargo/credentials.toml"
fix_commit() {
  GIT_AUTHOR_DATE="$1" GIT_COMMITTER_DATE="$1" \
    git -C "$WT" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t -c commit.gpgsign=false \
    commit -q --allow-empty -m "$2"
}
git -C "$WT" init -q
mkdir -p "$WT/scripts/release"; printf 'fixture-crate\n' > "$WT/scripts/release/publish-order.txt"
git -C "$WT" add scripts/release/publish-order.txt
fix_commit 2026-09-01T00:00:00Z base
fix_commit 2026-09-01T01:00:00Z tagged
git -C "$WT" -c core.hooksPath=/dev/null tag v1.2.3
fix_commit 2026-09-01T02:00:00Z later
SHA_B=$(git -C "$WT" rev-parse HEAD)
SHA_A=$(git -C "$WT" rev-parse 'v1.2.3^{commit}')
# One committed T-4 receipt variant per RC_* commit (the real receipts quote shas in backticks;
# the verifier reads the row either way)
RCPT=docs/audits/release/v1.2.3/dry-run-receipt.md
receipt_commit() { # receipt_commit VAR TAG_ROW_SHAS RC FLAGS AFTER -- sets VAR to the commit
  mkdir -p "$WT/$(dirname "$RCPT")"
  printf '# v1.2.3 packaging dry-run\n\n| step | record |\n|---|---|\n| tag / release commit | v1.2.3 = %s (detached checkout) |\n| packaging dry-run | cargo publish --workspace %s rc=**%s**, %s (summary.log) |\n' \
    "$2" "$4" "$3" "$5" > "$WT/$RCPT"
  git -C "$WT" add "$RCPT"; fix_commit 2026-09-01T03:00:00Z "receipt $1"
  printf -v "$1" '%s' "$(git -C "$WT" rev-parse HEAD)"
}
LOCKED="--dry-run --no-verify --locked"
receipt_commit RC_GOOD     "$SHA_A"          0   "$LOCKED"            "tree clean after"
receipt_commit RC_RED      "$SHA_A"          101 "$LOCKED"            "tree clean after"
receipt_commit RC_OTHER    "$SHA_B"          0   "$LOCKED"            "tree clean after"
receipt_commit RC_TWO_SHAS "$SHA_A / $SHA_B" 0   "$LOCKED"            "tree clean after"
receipt_commit RC_VERIFIED "$SHA_A"          0   "--dry-run --locked" "tree clean after"
receipt_commit RC_DIRTY    "$SHA_A"          0   "$LOCKED"            "tree dirty after"
# Quorum round 1 (R5c): substring tests took each of these as a green, clean, locked dry-run.
receipt_commit RC_NO_LOCKED   "$SHA_A"       0   "--dry-run --no-verify --no-locked"   "tree clean after"
receipt_commit RC_DRY_SKIPPED "$SHA_A"       0   "--dry-run-skipped --no-verify --locked" "tree clean after"
receipt_commit RC_HEX_RC      "$SHA_A"       0x1 "$LOCKED"            "tree clean after"
receipt_commit RC_SUBTREE     "$SHA_A"       0   "$LOCKED"            "subtree clean after"
receipt_commit RC_NOT_CLEAN   "$SHA_A"       0   "$LOCKED"            "tree is not tree clean after"
receipt_commit RC_CLEAN_NOT   "$SHA_A"       0   "$LOCKED"            "tree clean after (not checked before)"
git -C "$WT" checkout -q --detach v1.2.3
SHA_A=$(git -C "$WT" rev-parse 'v1.2.3^{commit}')
[ "$SHA_B" != "$SHA_A" ] || { echo "FAIL  fixture: no second commit"; exit 1; }

# ── stubs ──
cat > "$SBIN/gh" <<'STUB'
#!/usr/bin/env bash
d=${STUB_DIR:?stub gh called without STUB_DIR}
printf 'gh %s\n' "$*" >> "$d/calls"
jqx=""; all="$*"
case " $all" in *" --jq "*) jqx=${all##* --jq } ;; esac
serve() { if [ -n "$jqx" ]; then jq -r "$jqx" < "$1"; else cat "$1"; fi; }
case "$1 ${2:-}" in
  "auth status") exit 0 ;;
  "run list") serve "$d/runs.json" ;;
  "run view")
    if [[ " $all " == *" --repo paiml/aprender "* ]]; then
      if [ -f "$d/ajobs-${3:-}.json" ]; then serve "$d/ajobs-${3:-}.json"; else exit 1; fi
    elif [ -f "$d/jobs-${3:-}.json" ]; then serve "$d/jobs-${3:-}.json"
    elif [ -f "$d/jobs-any.json" ]; then serve "$d/jobs-any.json"
    else exit 1; fi ;;
  api\ repos/paiml/infra/actions/jobs/*/logs)
    jid=${2#repos/paiml/infra/actions/jobs/}; jid=${jid%/logs}
    cat "$d/log-$jid.txt" 2>/dev/null || exit 1 ;;
  api\ repos/paiml/aprender/actions/jobs/*/logs)
    jid=${2#repos/paiml/aprender/actions/jobs/}; jid=${jid%/logs}
    cat "$d/alog-$jid.txt" 2>/dev/null || exit 1 ;;
  *) echo "UNEXPECTED gh $*" >> "$d/calls"; exit 97 ;;
esac
STUB
for t in cargo curl python3; do
  printf '#!/usr/bin/env bash\nprintf "%%s %%s\\n" %s "$*" >> "${STUB_DIR:?}/calls"\nexit 97\n' "$t" > "$SBIN/$t"
done
chmod +x "$SBIN"/*

# ── fixture writers (S = the current row's stub dir) ──
S=""
runs() { printf '[%s]\n' "$1" > "$S/runs.json"; }
run_obj() { printf '{"databaseId":%s,"status":"completed","conclusion":"%s","createdAt":"2026-09-02T00:00:00Z"}' "$1" "$2"; }
# job_json JID STATUS CONCL ASSERT_STATUS ASSERT_CONCL  (ASSERT_STATUS "-" = a legacy job with no steps)
job_json() {
  if [ "$4" = "-" ]; then
    printf '{"databaseId":%s,"name":"clean-room (aprender)","status":"%s","conclusion":"%s"}' "$1" "$2" "$3"
  else
    printf '{"databaseId":%s,"name":"clean-room (aprender)","status":"%s","conclusion":"%s","steps":[{"name":"Clone aprender from GitHub","status":"completed","conclusion":"success"},{"name":"Assert the commit under test","status":"%s","conclusion":"%s"}]}' "$1" "$2" "$3" "$4" "$5"
  fi
}
jobs() { printf '{"jobs":[{"databaseId":1,"name":"matrix-setup","status":"completed","conclusion":"success"},%s]}\n' "$2" > "$S/jobs-$1.json"; }
# log JID STRUCTURED_SHA LEGACY_ABBREV  (either may be empty = absent)
log() {
  {
    printf '2026-09-02T00:00:02.0000000Z ==> Asserting the commit this clean-room job will test\n'
    [ -z "$2" ] || printf '2026-09-02T00:00:02.2000000Z     tested-sha: %s\r\n' "$2"
    printf '2026-09-02T00:03:00.1000000Z     version: 1.2.3\n'
    [ -z "$3" ] || printf '2026-09-02T00:03:00.2000000Z     commit:  %s\r\n' "$3"
  } > "$S/log-$1.txt"
}
id() { printf '%b' "$1" > "$S/id"; }
green() { jobs "$1" "$(job_json "$2" completed success completed success)"; log "$2" "$SHA_A" "${SHA_A:0:8}"; }

f_missing_id()        { runs "$(run_obj 9001 success)"; green 9001 501; cp "$S/jobs-9001.json" "$S/jobs-any.json"; }
f_empty_id()          { f_missing_id; id ''; }
f_garbage_id()        { f_missing_id; id 'pending\n'; }
f_two_ids()           { f_missing_id; id '9001\n9002\n'; }
f_failed_run()        { id '9001\n'; runs "$(run_obj 9001 failure)"; jobs 9001 "$(job_json 501 completed failure completed success)"; log 501 "$SHA_A" ""; }
f_cancelled_run()     { id '9001\n'; runs "$(run_obj 9001 cancelled)"; jobs 9001 "$(job_json 501 completed cancelled completed success)"; log 501 "$SHA_A" ""; }
f_in_progress_run()   { id '9001\n'; runs "$(run_obj 9001 '')"; jobs 9001 "$(job_json 501 in_progress '' completed success)"; log 501 "$SHA_A" ""; }
f_queued_says_success() { id '9001\n'; runs "$(run_obj 9001 '')"; jobs 9001 "$(job_json 501 queued success completed success)"; log 501 "$SHA_A" ""; }
f_other_commit_structured() { id '9001\n'; runs "$(run_obj 9001 success)"; jobs 9001 "$(job_json 501 completed success completed success)"; log 501 "$SHA_B" ""; }
f_other_commit_log_line() { id '9001\n'; runs "$(run_obj 9001 success)"; jobs 9001 "$(job_json 501 completed success -)"; log 501 "" "${SHA_B:0:8}"; }
f_assert_step_failed() { id '9001\n'; runs "$(run_obj 9001 success)"; jobs 9001 "$(job_json 501 completed success completed failure)"; log 501 "$SHA_A" ""; }
f_no_aprender_job()   { id '9001\n'; runs "$(run_obj 9001 success)"; log 1 "$SHA_A" ""
                        printf '{"jobs":[{"databaseId":7,"name":"clean-room (forjar)","status":"completed","conclusion":"success"}]}\n' > "$S/jobs-9001.json"; }
f_two_aprender_jobs() { id '9001\n'; runs "$(run_obj 9001 success)"; log 501 "$SHA_A" ""
                        jobs 9001 "$(job_json 501 completed success completed success),$(job_json 502 completed failure completed success)"; }
f_gh_view_fails()     { id '9002\n'; runs "$(run_obj 9002 success),$(run_obj 9001 success)"; green 9001 501; }
f_gh_log_fails()      { id '9002\n'; runs "$(run_obj 9002 success),$(run_obj 9001 success)"; green 9001 501
                        jobs 9002 "$(job_json 502 completed success completed success)"; }
f_green_on_tag()      { id '9001\n'; runs "$(run_obj 9001 success)"; green 9001 501; }
f_green_on_tag_log_line() { id '9001\n'; runs "$(run_obj 9001 success)"; jobs 9001 "$(job_json 501 completed success -)"; log 501 "" "${SHA_A:0:8}"; }

# B2-gpu (paiml/aprender b2-gpu.yml): run 7001, job 601 unless a row says otherwise.
b2_job() { # b2_job JID STATUS CONCL ASSERT_STATUS ASSERT_CONCL
  printf '{"databaseId":%s,"name":"b2-gpu (aprender-gpu, fixture runner)","status":"%s","conclusion":"%s","steps":[{"name":"Assert the commit under test","status":"%s","conclusion":"%s"}]}' "$1" "$2" "$3" "$4" "$5"
}
b2_jobs() { printf '{"jobs":[%s]}\n' "$2" > "$S/ajobs-$1.json"; }
b2_log() { # b2_log JID LINE...  -- each LINE is one `tested-sha:` value
  local j=$1; shift
  { printf '2026-09-02T00:00:01.0000000Z ==> b2-gpu\n'
    for t in "$@"; do printf '2026-09-02T00:00:02.0000000Z tested-sha: %s\r\n' "$t"; done; } > "$S/alog-$j.txt"
}
b2_id() { printf '%b' "$1" > "$S/b2id"; }
rcpt() { printf '%b' "$1" > "$S/rcpt"; }
defaults() {
  b2_id '7001\n'; b2_jobs 7001 "$(b2_job 601 completed success completed success)"; b2_log 601 "$SHA_A"
  rcpt "${RC_GOOD}\n"
}
f_b2_missing_id()      { f_green_on_tag; rm -f "${S:?}/b2id"; }
f_b2_garbage_id()      { f_green_on_tag; b2_id 'pending\n'; }
f_b2_two_ids()         { f_green_on_tag; b2_id '7001\n7002\n'; }
f_b2_failed_job()      { f_green_on_tag; b2_jobs 7001 "$(b2_job 601 completed failure completed success)"; }
f_b2_in_progress()     { f_green_on_tag; b2_jobs 7001 "$(b2_job 601 in_progress '' completed success)"; }
f_b2_queued_says_success() { f_green_on_tag; b2_jobs 7001 "$(b2_job 601 queued success completed success)"; }
f_b2_other_commit()    { f_green_on_tag; b2_log 601 "$SHA_B"; }
f_b2_no_record()       { f_green_on_tag; b2_log 601; }
f_b2_two_records()     { f_green_on_tag; b2_log 601 "$SHA_A" "$SHA_B"; }
f_b2_short_sha()       { f_green_on_tag; b2_log 601 "${SHA_A:0:12}"; }
f_b2_assert_failed()   { f_green_on_tag; b2_jobs 7001 "$(b2_job 601 completed success completed failure)"; }
f_b2_no_job()          { f_green_on_tag; b2_log 1 "$SHA_A"
                         printf '{"jobs":[{"databaseId":9,"name":"lint","status":"completed","conclusion":"success"}]}\n' > "$S/ajobs-7001.json"; }
f_b2_two_jobs()        { f_green_on_tag; b2_jobs 7001 "$(b2_job 601 completed success completed success),$(b2_job 602 completed failure completed success)"; }
f_b2_gh_view_fails()   { f_green_on_tag; rm -f "${S:?}/ajobs-7001.json"; }
f_b2_gh_log_fails()    { f_green_on_tag; rm -f "${S:?}/alog-601.txt"; }
f_rcpt_missing()       { f_green_on_tag; rm -f "${S:?}/rcpt"; }
f_rcpt_two_lines()     { f_green_on_tag; rcpt "${RC_GOOD}\n${RC_GOOD}\n"; }
f_rcpt_abbrev_sha()    { f_green_on_tag; rcpt "${RC_GOOD:0:12}\n"; }
f_rcpt_absent_commit() { f_green_on_tag; rcpt '0123456789abcdef0123456789abcdef01234567\n'; }
f_rcpt_no_file()       { f_green_on_tag; rcpt "$SHA_B\n"; }
f_rcpt_other_commit()  { f_green_on_tag; rcpt "${RC_OTHER}\n"; }
f_rcpt_two_shas()      { f_green_on_tag; rcpt "${RC_TWO_SHAS}\n"; }
f_rcpt_red()           { f_green_on_tag; rcpt "${RC_RED}\n"; }
f_rcpt_verified()      { f_green_on_tag; rcpt "${RC_VERIFIED}\n"; }
f_rcpt_dirty()         { f_green_on_tag; rcpt "${RC_DIRTY}\n"; }
f_rcpt_no_locked()     { f_green_on_tag; rcpt "${RC_NO_LOCKED}\n"; }
f_rcpt_dry_skipped()   { f_green_on_tag; rcpt "${RC_DRY_SKIPPED}\n"; }
f_rcpt_hex_rc()        { f_green_on_tag; rcpt "${RC_HEX_RC}\n"; }
f_rcpt_subtree()       { f_green_on_tag; rcpt "${RC_SUBTREE}\n"; }
f_rcpt_not_clean()     { f_green_on_tag; rcpt "${RC_NOT_CLEAN}\n"; }
f_rcpt_clean_not()     { f_green_on_tag; rcpt "${RC_CLEAN_NOT}\n"; }

# ── the two doors; each prints ACCEPT, REFUSE or BROKEN:<why> ──
door_ps() { # door_ps WHY -- REFUSE only when the STOP line names gate WHY
  local out why=$1
  rm -f "${AP:?}/cleanroom-run-id" "${AP:?}/b2gpu-run-id" "${AP:?}/dryrun-receipt-commit" "${AP:?}/STATUS"
  if [ -f "$S/id" ]; then cp "$S/id" "$AP/cleanroom-run-id"; fi
  if [ -f "$S/b2id" ]; then cp "$S/b2id" "$AP/b2gpu-run-id"; fi
  if [ -f "$S/rcpt" ]; then cp "$S/rcpt" "$AP/dryrun-receipt-commit"; fi
  out=$(env PATH="$SBIN:$PATH" HOME="$FHOME" CARGO_HOME="$FHOME/.cargo" RELEASE_AP="$AP" STUB_DIR="$S" \
        GH_CONFIG_DIR="$WORK/ghconfig" GH_TOKEN='' GITHUB_TOKEN='' bash "$PS" 1.2.3 2>&1) || true
  printf '%s\n' "$out" > "$S/ps.out"
  if grep -qE "STOP publish: $why: " <<< "$out"; then echo REFUSE
  elif grep -q '^python3 scripts/lib/cascade_universe.py' "$S/calls"; then echo ACCEPT
  else echo "BROKEN:$(tail -n 1 <<< "$out" | cut -c1-120)"; fi
}
FNS="$WORK/fns.sh"
{
  if [ -f "$LIB" ]; then cat "$LIB"; fi
  for fn in clean_room_runs_jq clean_room_gate; do sed -n "/^${fn}() {/,/^}/p" "$CASCADE"; done
} > "$FNS"
CASCADE_DOOR=1
for fn in clean_room_verify_run clean_room_runs_jq clean_room_gate; do
  grep -q "^${fn}() {" "$FNS" || CASCADE_DOOR=0
done
door_cascade() {
  local out grc=0
  [ "$CASCADE_DOOR" -eq 1 ] || { echo "BROKEN:cascade has no shared verifier"; return; }
  out=$(
    export PATH="$SBIN:$PATH" STUB_DIR="$S" GH_CONFIG_DIR="$WORK/ghconfig"
    unset GH_TOKEN GITHUB_TOKEN
    # shellcheck disable=SC1090
    . "$FNS"; clean_room_gate "$WT" v1.2.3 2>&1
  ) || grc=$?
  printf '%s\n' "$out" > "$S/cascade.out"
  if [ "$grc" -eq 0 ] && grep -q '^CLEAN-ROOM PROCEED' <<< "$out"; then echo ACCEPT
  elif [ "$grc" -eq 1 ] && grep -q '^CLEAN-ROOM REFUSE' <<< "$out"; then echo REFUSE
  else echo "BROKEN:rc=$grc $(head -1 <<< "$out" | cut -c1-80)"; fi
}

rc=0; bad_ps=0; bad_cascade=0; n_ps=0; n_cascade=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; rc=1; }
row() { # row NAME DOORS(ps|both) EXPECT(ACCEPT|REFUSE) [WHY: the gate that refuses; default clean-room]
  local name=$1 doors=$2 want=$3 why=${4:-clean-room} got
  S="$WORK/stub-$name"; mkdir -p "$S"; : > "$S/calls"
  defaults; "f_$name"
  got=$(door_ps "$why")
  if [ "$want" = REFUSE ]; then n_ps=$((n_ps + 1)); fi
  if [ "$got" = "$want" ]; then pass "$name @publish_strict ($got)"
  else
    fail "$name @publish_strict: want $want, got $got"
    if [ "$want" = REFUSE ] && [ "$got" = ACCEPT ]; then bad_ps=$((bad_ps + 1)); fi
  fi
  if [ "$doors" = both ]; then
    if [ "$want" = REFUSE ]; then n_cascade=$((n_cascade + 1)); fi
    got=$(door_cascade)
    if [ "$got" = "$want" ]; then pass "$name @cascade ($got)"
    else
      fail "$name @cascade: want $want, got $got"
      if [ "$want" = REFUSE ] && [ "$got" = ACCEPT ]; then bad_cascade=$((bad_cascade + 1)); fi
    fi
  fi
  if grep -qE '^(UNEXPECTED|cargo publish)' "$S/calls"; then
    fail "$name: forbidden call: $(grep -E '^(UNEXPECTED|cargo publish)' "$S/calls" | head -1)"
  fi
}

row missing_id               ps   REFUSE
row empty_id                 ps   REFUSE
row garbage_id               ps   REFUSE
row two_ids                  ps   REFUSE
row failed_run               both REFUSE
row cancelled_run            both REFUSE
row in_progress_run          both REFUSE
row queued_says_success      both REFUSE
row other_commit_structured  both REFUSE
row other_commit_log_line    both REFUSE
row assert_step_failed       both REFUSE
row no_aprender_job          both REFUSE
row two_aprender_jobs        both REFUSE
row gh_view_fails            both REFUSE
row gh_log_fails             both REFUSE
row b2_missing_id            ps   REFUSE b2-gpu
row b2_garbage_id            ps   REFUSE b2-gpu
row b2_two_ids               ps   REFUSE b2-gpu
row b2_failed_job            ps   REFUSE b2-gpu
row b2_in_progress           ps   REFUSE b2-gpu
row b2_queued_says_success   ps   REFUSE b2-gpu
row b2_other_commit          ps   REFUSE b2-gpu
row b2_no_record             ps   REFUSE b2-gpu
row b2_two_records           ps   REFUSE b2-gpu
row b2_short_sha             ps   REFUSE b2-gpu
row b2_assert_failed         ps   REFUSE b2-gpu
row b2_no_job                ps   REFUSE b2-gpu
row b2_two_jobs              ps   REFUSE b2-gpu
row b2_gh_view_fails         ps   REFUSE b2-gpu
row b2_gh_log_fails          ps   REFUSE b2-gpu
row rcpt_missing             ps   REFUSE "dry-run receipt"
row rcpt_two_lines           ps   REFUSE "dry-run receipt"
row rcpt_abbrev_sha          ps   REFUSE "dry-run receipt"
row rcpt_absent_commit       ps   REFUSE "dry-run receipt"
row rcpt_no_file             ps   REFUSE "dry-run receipt"
row rcpt_other_commit        ps   REFUSE "dry-run receipt"
row rcpt_two_shas            ps   REFUSE "dry-run receipt"
row rcpt_red                 ps   REFUSE "dry-run receipt"
row rcpt_verified            ps   REFUSE "dry-run receipt"
row rcpt_dirty               ps   REFUSE "dry-run receipt"
row rcpt_no_locked           ps   REFUSE "dry-run receipt"
row rcpt_dry_skipped         ps   REFUSE "dry-run receipt"
row rcpt_hex_rc              ps   REFUSE "dry-run receipt"
row rcpt_subtree             ps   REFUSE "dry-run receipt"
row rcpt_not_clean           ps   REFUSE "dry-run receipt"
row rcpt_clean_not           ps   REFUSE "dry-run receipt"
row green_on_tag             both ACCEPT
row green_on_tag_log_line    both ACCEPT

# The stub engaged: the accepted row's verdict came from OUR gh, through the shared verifier.
S="$WORK/stub-green_on_tag"
if grep -q '^gh run view 9001 --repo paiml/infra --json jobs --jq' "$S/calls" \
   && grep -q '^gh api repos/paiml/infra/actions/jobs/501/logs' "$S/calls" \
   && grep -q 'CLEAN-ROOM VERIFIED: run 9001 job 501' "$S/ps.out"; then
  pass "stub_engaged (publish_strict asked the stub gh and printed the verifier's line)"
else
  fail "stub_engaged: the green row's calls/output do not show the stub gh and the shared verifier"
fi
if grep -q '^gh run view 7001 --repo paiml/aprender --json jobs --jq' "$S/calls" \
   && grep -q '^gh api repos/paiml/aprender/actions/jobs/601/logs' "$S/calls" \
   && grep -q 'B2-GPU VERIFIED: run 7001 job 601' "$S/ps.out" \
   && grep -q "DRY-RUN VERIFIED: $RCPT at ${RC_GOOD}" "$S/ps.out"; then
  pass "stub_engaged_r5c (B2-gpu read from the stub gh, the receipt read from its commit)"
else
  fail "stub_engaged_r5c: the green row does not show the B2-gpu and dry-run verifiers"
fi
# One verifier: both doors call clean_room_verify_run, and neither keeps a private jobs parser.
if ! grep -q 'clean_room_parse_job' "$CASCADE" \
   && grep -q 'clean_room_verify_run' "$PS" && grep -q 'clean_room_verify_run' "$CASCADE"; then
  pass "one_verifier (both doors call clean_room_verify_run; no private jobs parser)"
else
  fail "one_verifier: a door does not call clean_room_verify_run, or keeps its own jobs parser"
fi

echo "BAD IDS ACCEPTED  publish_strict: $bad_ps/$n_ps   cascade: $bad_cascade/$n_cascade"
if [ "$rc" -eq 0 ]; then echo "PASS  clean-room one door: every row held at both doors"
else echo "FAIL  clean-room one door: a row broke (see above)"; fi
exit "$rc"
