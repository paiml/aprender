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
  if [ "${#shas}" -ne 40 ] || [[ "$shas" == *[!0-9a-f]* ]]; then
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
  if [ "${#want}" -ne 40 ] || [[ "$want" == *[!0-9a-f]* ]]; then
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
