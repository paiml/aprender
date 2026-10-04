# shellcheck shell=bash
# lib_clean_room.sh -- the ONE verifier of a clean-room run id (#4687, R5b).
#
# Sourced by scripts/release/publish_strict.sh and scripts/cascade-publish.sh,
# so both doors to cargo publish give one answer. Option-neutral: no `set`
# here (a sourced `set` changes the caller's shell); every function fails by
# return status. No python: gh's own --json/--jq, plus awk.
#
# WHAT A RUN TESTED (PMAT-3318, infra#621/#622). infra's clean-room.yml clones
# aprender; a run's headSha is an INFRA commit. The aprender commit is read from
# the JOB LOG: the structured `    tested-sha: <40 hex>` line written by the
# `Assert the commit under test` step, else the legacy `    commit:  <abbrev>`
# line printed by infra's Makefile `_copy-source`. The structured record wins,
# must be a full 40-char lowercase sha, must come from an assert step that
# itself succeeded, and must agree with the legacy line when both exist.
# The repository, job and step names are literals: an override would be a bypass.

# release_is_full_sha VALUE -- rc 0 iff VALUE is a full 40-char lowercase commit sha.
# A regex, not a length test against a literal (check_release_scripts_derive_identity.sh R3).
release_is_full_sha() { [[ "${1:-}" =~ ^[0-9a-f]{40}$ ]]; }

# stdin: a clean-room job log. Prints the DISTINCT legacy `commit:` values.
clean_room_tested_abbrevs() {
  tr -d '\r' \
    | { grep -E '^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z )?    commit:  [0-9a-f]{7,40}$' || true; } \
    | sed -E 's/^.*    commit:  //' \
    | sort -u
}

# stdin: a clean-room job log. Prints the DISTINCT structured `tested-sha:`
# values, captured LOOSELY on purpose: a truncated or uppercased field must
# reach the verifier as a malformed record it refuses by name.
clean_room_tested_shas() {
  tr -d '\r' \
    | { grep -E '^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z )?[[:space:]]*tested-sha: .+$' || true; } \
    | sed -E 's/^.*tested-sha: //' \
    | sed -E 's/[[:space:]]+$//' \
    | sort -u
}

# The --jq program for `gh run view --json jobs`. Prints exactly one line:
#   OK<TAB><job id><TAB><status><TAB><conclusion|none><TAB><assert state>
#   NONE                       no `clean-room (aprender)` job
#   BAD<TAB><why>              a shape the verifier will not read through
# The assert state is "<status>/<conclusion>", `absent` (no such step: every run
# before infra#622) or `ambiguous` (more than one).
clean_room_jobs_jq() {
  printf '%s' '
def nz: if . == null or . == "" then "none" else tostring end;
if (.jobs | type) != "array" then "BAD\tjobs is not a JSON array"
else [.jobs[] | select(type == "object" and .name == "clean-room (aprender)")] as $h
| if ($h | length) > 1 then "BAD\t\($h | length) jobs named clean-room (aprender)"
  elif ($h | length) == 0 then "NONE"
  else ($h | first) as $j
  | if ($j.databaseId | type) != "number" or ($j.databaseId | floor) != $j.databaseId then "BAD\tjob databaseId is not an integer"
    elif $j.steps != null and ($j.steps | type) != "array" then "BAD\tsteps is not a JSON array"
    else [($j.steps // [])[] | select(type == "object" and .name == "Assert the commit under test")] as $a
    | (if ($a | length) > 1 then "ambiguous" elif ($a | length) == 0 then "absent"
       else "\($a | first | .status | nz)/\($a | first | .conclusion | nz)" end) as $st
    | "OK\t\($j.databaseId)\t\($j.status | nz)\t\($j.conclusion | nz)\t\($st)"
    end
  end
end'
}

# clean_room_read_run_id FILE -- prints the one run id FILE records; rc 2 (with
# the reason on stdout) when it is missing, empty, more than one line, or not
# a decimal run id.
clean_room_read_run_id() {
  local f=${1:-} n rid=""
  if [ ! -s "$f" ]; then
    echo "CLEAN-ROOM NOT-MEASURED: no clean-room run id recorded at ${f:-<no file>}"
    return 2
  fi
  n=$(awk 'END { print NR }' "$f")
  if [ "$n" != 1 ]; then
    echo "CLEAN-ROOM NOT-MEASURED: $f has $n lines; it must record exactly one run id"
    return 2
  fi
  IFS= read -r rid < "$f" || [ -n "$rid" ]
  case "$rid" in
    ''|*[!0-9]*) echo "CLEAN-ROOM NOT-MEASURED: '$rid' in $f is not a run id"; return 2 ;;
  esac
  printf '%s\n' "$rid"
}

# clean_room_log_tested LOG JOB_ASSERT_STATE ROOT -- what one job log says it
# tested. rc 0 prints "<tested><TAB><display><TAB><source><TAB><resolved sha>";
# rc 1 prints why the log is not a single, well-formed record.
clean_room_log_tested() {
  local log=$1 jassert=$2 root=$3 shas ns abbrevs n resolved=""
  shas=$(clean_room_tested_shas <<< "$log")
  ns=$(grep -c . <<< "$shas" || true); ns=${ns:-0}
  abbrevs=$(clean_room_tested_abbrevs <<< "$log")
  n=$(grep -c . <<< "$abbrevs" || true); n=${n:-0}
  if [ "$ns" -gt 1 ]; then
    echo "$ns structured tested-sha record(s) in the log (need exactly 1)"; return 1
  fi
  if [ "$ns" -eq 0 ]; then
    if [ "$n" -ne 1 ]; then
      echo "$n tested-commit record(s) in the log (need exactly 1)"; return 1
    fi
    resolved=$(git -C "$root" rev-parse --verify --quiet "${abbrevs}^{commit}" 2>/dev/null) || resolved=""
    printf '%s\t%s\t%s\t%s\n' "$abbrevs" "$abbrevs${resolved:+ ($resolved)}" log-line "$resolved"
    return 0
  fi
  if ! release_is_full_sha "$shas"; then
    echo "structured tested-sha '$shas' is not a full 40-char lowercase sha"; return 1
  fi
  if [ "$jassert" != "completed/success" ]; then
    echo "structured tested-sha $shas but the 'Assert the commit under test' step is $jassert (need completed/success)"; return 1
  fi
  if [ "$n" -gt 1 ]; then
    echo "$n tested-commit record(s) in the log beside structured tested-sha $shas"; return 1
  fi
  # The abbreviation is `git rev-parse --short HEAD` of the SAME clone: agreement
  # is "the log line is a prefix of the structured sha".
  if [ "$n" -eq 1 ] && [[ "$shas" != "$abbrevs"* ]]; then
    echo "the two records disagree -- structured tested-sha says $shas, the log line says $abbrevs"; return 1
  fi
  printf '%s\t%s\t%s\t%s\n' "$shas" "$shas" structured "$shas"
}

# clean_room_verify_run RID WANT [ROOT] -- is infra run RID proof that
# clean-room passed on commit WANT? Prints ONE line and returns
#   0  "CLEAN-ROOM VERIFIED: run RID job J ... tested T = WANT, conclusion=success [...]"
#   1  "run RID ...: <why>"                 measured, and not proof
#   2  "CLEAN-ROOM NOT-MEASURED: <why>"     bad input, gh failing, unparseable
# WANT is a full 40-char lowercase sha; ROOT (default .) resolves a legacy
# abbreviated `commit:` line. Callers treat 1 and 2 as a refusal; never a pass.
clean_room_verify_run() {
  local rid=${1:-} want=${2:-} root=${3:-.}
  local repo="paiml/infra" job_name="clean-room (aprender)"
  local row kind jid jstatus jconcl jassert log rec tested tdisp tsource resolved
  case "$rid" in
    ''|*[!0-9]*) echo "CLEAN-ROOM NOT-MEASURED: '$rid' is not a run id"; return 2 ;;
  esac
  if ! release_is_full_sha "$want"; then
    echo "CLEAN-ROOM NOT-MEASURED: '$want' is not a full 40-char lowercase commit sha"; return 2
  fi
  if ! row=$(gh run view "$rid" --repo "$repo" --json jobs --jq "$(clean_room_jobs_jq)" 2>/dev/null); then
    echo "CLEAN-ROOM NOT-MEASURED: gh run view $rid failed, or the jobs of run $rid could not be parsed"; return 2
  fi
  case "$row" in
    NONE) echo "run $rid: no '$job_name' job"; return 1 ;;
    OK$'\t'*$'\n'*|OK$'\t'*$'\t'*$'\t'*$'\t'*$'\t'*) echo "CLEAN-ROOM NOT-MEASURED: the jobs of run $rid could not be parsed (extra fields)"; return 2 ;;
    OK$'\t'*$'\t'*$'\t'*$'\t'*) : ;;
    *) echo "CLEAN-ROOM NOT-MEASURED: the jobs of run $rid could not be parsed (${row#BAD?})"; return 2 ;;
  esac
  IFS=$'\t' read -r kind jid jstatus jconcl jassert <<< "$row"
  if [ "$jstatus" != "completed" ]; then
    echo "run $rid job $jid: $jstatus -- tested sha unknown, conclusion=$jconcl"; return 1
  fi
  if ! log=$(gh api "repos/$repo/actions/jobs/$jid/logs" 2>/dev/null); then
    echo "CLEAN-ROOM NOT-MEASURED: could not read the log of run $rid job $jid"; return 2
  fi
  if ! rec=$(clean_room_log_tested "$log" "$jassert" "$root"); then
    echo "run $rid job $jid: $rec, conclusion=$jconcl"; return 1
  fi
  IFS=$'\t' read -r tested tdisp tsource resolved <<< "$rec"
  if [ "$resolved" != "$want" ]; then
    echo "run $rid job $jid: tested $tdisp, conclusion=$jconcl [$tsource]"; return 1
  fi
  if [ "$jconcl" != "success" ]; then
    echo "run $rid job $jid: tested $tested = the wanted commit, conclusion=$jconcl [$tsource]"; return 1
  fi
  echo "CLEAN-ROOM VERIFIED: run $rid job $jid '$job_name' tested $tested = $want, conclusion=success [tested-sha source: $tsource]"
  return 0
}

# ---------------------------------------------------------------------------
# B2-gpu and the dry-run receipt (R5c): the other two publish preconditions,
# given the clean-room treatment. A file that merely exists is not proof.
# Each refusal is one line, so the case table's mutants can turn exactly one
# of them into an acceptance.

# The --jq program for a paiml/aprender `b2-gpu.yml` run. Same output shape as
# clean_room_jobs_jq; the job is matched by the name PREFIX `b2-gpu (aprender-gpu`
# (the suffix names the runner) and must be the only such job.
b2gpu_jobs_jq() {
  printf '%s' '
def nz: if . == null or . == "" then "none" else tostring end;
if (.jobs | type) != "array" then "BAD\tjobs is not a JSON array"
else [.jobs[] | select(type == "object" and (.name | type) == "string" and (.name | startswith("b2-gpu (aprender-gpu")))] as $h
| if ($h | length) > 1 then "BAD\t\($h | length) b2-gpu (aprender-gpu ...) jobs"
  elif ($h | length) < 1 then "NONE"
  else ($h | first) as $j
  | if ($j.databaseId | type) != "number" or ($j.databaseId | floor) != $j.databaseId then "BAD\tjob databaseId is not an integer"
    elif $j.steps != null and ($j.steps | type) != "array" then "BAD\tsteps is not a JSON array"
    else [($j.steps // [])[] | select(type == "object" and .name == "Assert the commit under test")] as $a
    | (if ($a | length) > 1 then "ambiguous" elif ($a | length) == 0 then "absent"
       else "\($a | first | .status | nz)/\($a | first | .conclusion | nz)" end) as $st
    | "OK\t\($j.databaseId)\t\($j.status | nz)\t\($j.conclusion | nz)\t\($st)"
    end
  end
end'
}

# b2gpu_read_run_id FILE -- clean_room_read_run_id's rules, B2-GPU's words.
b2gpu_read_run_id() {
  local bf=${1:-} bn brid=""
  if [ ! -s "$bf" ]; then echo "B2-GPU NOT-MEASURED: no b2-gpu run id recorded at ${bf:-<no file>}"; return 2; fi
  bn=$(awk 'END { print NR }' "$bf")
  if [ "$bn" != 1 ]; then echo "B2-GPU NOT-MEASURED: $bf has $bn lines; it must record exactly one run id"; return 2; fi
  IFS= read -r brid < "$bf" || [ -n "$brid" ]
  case "$brid" in
    ''|*[!0-9]*) echo "B2-GPU NOT-MEASURED: '$brid' in $bf is not a run id"; return 2 ;;
  esac
  printf '%s\n' "$brid"
}

# b2gpu_verify_run RID WANT -- is paiml/aprender run RID proof that the b2-gpu
# job passed on commit WANT? Same return contract as clean_room_verify_run.
# b2-gpu.yml has always written the structured `tested-sha:` record, so there
# is no legacy log-line fallback: no structured record is a refusal.
b2gpu_verify_run() {
  local rid=${1:-} want=${2:-}
  local repo="paiml/aprender" row kind jid gstatus gconcl gassert log gshas gns
  case "$rid" in
    ''|*[!0-9]*) echo "B2-GPU NOT-MEASURED: '$rid' is not a run id"; return 2 ;;
  esac
  if ! release_is_full_sha "$want"; then
    echo "B2-GPU NOT-MEASURED: '$want' is not a full 40-char lowercase commit sha"; return 2
  fi
  if ! row=$(gh run view "$rid" --repo "$repo" --json jobs --jq "$(b2gpu_jobs_jq)" 2>/dev/null); then
    echo "B2-GPU NOT-MEASURED: gh run view $rid failed, or the jobs of run $rid could not be parsed"; return 2
  fi
  case "$row" in
    NONE) echo "run $rid: no 'b2-gpu (aprender-gpu ...)' job"; return 1 ;;
    OK$'\t'*$'\n'*|OK$'\t'*$'\t'*$'\t'*$'\t'*$'\t'*) echo "B2-GPU NOT-MEASURED: the jobs of run $rid could not be parsed (extra fields)"; return 2 ;;
    OK$'\t'*$'\t'*$'\t'*$'\t'*) : ;;
    *) echo "B2-GPU NOT-MEASURED: the jobs of run $rid could not be parsed (${row#BAD?})"; return 2 ;;
  esac
  IFS=$'\t' read -r kind jid gstatus gconcl gassert <<< "$row"
  if [ "$gstatus" != "completed" ]; then echo "run $rid job $jid: $gstatus -- tested sha unknown, conclusion=$gconcl"; return 1; fi
  if ! log=$(gh api "repos/$repo/actions/jobs/$jid/logs" 2>/dev/null); then echo "B2-GPU NOT-MEASURED: could not read the log of run $rid job $jid"; return 2; fi
  gshas=$(clean_room_tested_shas <<< "$log")
  gns=$(grep -c . <<< "$gshas" || true); gns=${gns:-0}
  if [ "$gns" -ne 1 ]; then echo "run $rid job $jid: $gns structured tested-sha record(s) in the log (need exactly 1), conclusion=$gconcl"; return 1; fi
  if ! release_is_full_sha "$gshas"; then echo "run $rid job $jid: structured tested-sha '$gshas' is not a full 40-char lowercase sha, conclusion=$gconcl"; return 1; fi
  if [ "$gassert" != "completed/success" ]; then echo "run $rid job $jid: the 'Assert the commit under test' step is $gassert (need completed/success), conclusion=$gconcl"; return 1; fi
  if [ "$gshas" != "$want" ]; then echo "run $rid job $jid: tested $gshas, conclusion=$gconcl"; return 1; fi
  if [ "$gconcl" != "success" ]; then echo "run $rid job $jid: tested $gshas = the wanted commit, conclusion=$gconcl"; return 1; fi
  echo "B2-GPU VERIFIED: run $rid job $jid tested $gshas = $want, conclusion=success"
  return 0
}

# dryrun_receipt_verify FILE TAG WANT ROOT -- is the commit FILE records a
# committed T-4 receipt of a green packaging dry-run of WANT (TAG's commit)?
# The receipt is `docs/audits/release/TAG/dry-run-receipt.md` IN that commit,
# read with `git show`, never from a working tree. Its two table rows decide:
#   | tag / release commit | ... exactly one 40-hex sha, which must be WANT
#   | packaging dry-run    | ... the exact `cargo publish --workspace --dry-run --no-verify --locked`,
#                          one bounded rc= that is 0, ", tree clean after" unqualified
# Same return contract: 0 VERIFIED, 1 measured and not proof, 2 NOT-MEASURED.
dryrun_receipt_verify() {
  local rf=${1:-} tag=${2:-} want=${3:-} root=${4:-.} rn rsha="" path body row nrow tshas tns rcs nr
  if [ ! -s "$rf" ]; then echo "DRY-RUN NOT-MEASURED: no receipt commit recorded at ${rf:-<no file>}"; return 2; fi
  rn=$(awk 'END { print NR }' "$rf")
  if [ "$rn" != 1 ]; then echo "DRY-RUN NOT-MEASURED: $rf has $rn lines; it must record exactly one commit sha"; return 2; fi
  IFS= read -r rsha < "$rf" || [ -n "$rsha" ]
  if ! release_is_full_sha "$rsha"; then echo "DRY-RUN NOT-MEASURED: '$rsha' in $rf is not a full 40-char lowercase commit sha"; return 2; fi
  if ! release_is_full_sha "$want"; then
    echo "DRY-RUN NOT-MEASURED: '$want' is not a full 40-char lowercase commit sha"; return 2
  fi
  if ! git -C "$root" cat-file -e "$rsha^{commit}" 2>/dev/null; then echo "DRY-RUN NOT-MEASURED: commit $rsha is not in $root (fetch it)"; return 2; fi
  path="docs/audits/release/$tag/dry-run-receipt.md"
  if ! body=$(git -C "$root" show "$rsha:$path" 2>/dev/null); then echo "receipt commit $rsha: no $path in it"; return 1; fi
  body=$(tr -d '\r' <<< "$body")
  row=$(grep -E '^\| *tag / release commit *\|' <<< "$body" || true)
  nrow=$(grep -c . <<< "$row" || true)
  if [ "$nrow" != 1 ]; then echo "receipt commit $rsha: $path needs exactly one '| tag / release commit |' row"; return 1; fi
  tshas=$(grep -oE '\b[0-9a-f]{40}\b' <<< "$row" | sort -u || true)
  tns=$(grep -c . <<< "$tshas" || true); tns=${tns:-0}
  if [ "$tns" -ne 1 ]; then echo "receipt commit $rsha: the tag row names $tns full commit sha(s) (need exactly 1)"; return 1; fi
  if [ "$tshas" != "$want" ]; then echo "receipt commit $rsha: the dry-run ran on $tshas, not $tag = $want"; return 1; fi
  row=$(grep -E '^\| *packaging dry-run *\|' <<< "$body" || true)
  nrow=$(grep -c . <<< "$row" || true)
  if [ "$nrow" != 1 ]; then echo "receipt commit $rsha: $path needs exactly one '| packaging dry-run |' row"; return 1; fi
  # The exact command inside its backticks: a substring test took `--no-locked`, a space-bounded one
  # `--locked -p x`; reordered flags, an extra flag anywhere, or a bare command all refuse.
  if ! grep -qF '`cargo publish --workspace --dry-run --no-verify --locked`' <<< "$row"; then echo "receipt commit $rsha: the dry-run row does not record cargo publish --workspace --dry-run --no-verify --locked"; return 1; fi
  rcs=$(grep -oE '(^|[ ,(|])rc=\**[0-9]+\**([ ,)|]|$)' <<< "$row" | grep -oE '[0-9]+' | sort -u || true)
  nr=$(grep -c . <<< "$rcs" || true); nr=${nr:-0}
  if [ "$nr" -ne 1 ]; then echo "receipt commit $rsha: the dry-run row records $nr exit status(es) (need exactly 1)"; return 1; fi
  if [ "$rcs" != 0 ]; then echo "receipt commit $rsha: the dry-run exited rc=$rcs"; return 1; fi
  # "tree clean after" must open a clause and end it (" (", "|" or EOL); no not/dirty/unclean anywhere.
  if ! grep -qE '(^|, |\| *)tree clean after( \(| *\||$)' <<< "$row"; then echo "receipt commit $rsha: the dry-run row does not record the tree clean after"; return 1; fi
  if grep -qiE '(^|[^a-z])(not|dirty|unclean)([^a-z]|$)' <<< "$row"; then echo "receipt commit $rsha: the dry-run row qualifies its tree state (not/dirty/unclean)"; return 1; fi
  echo "DRY-RUN VERIFIED: $path at $rsha -- dry-run of $want rc=0, tree clean"
  return 0
}
