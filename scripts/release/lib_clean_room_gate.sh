#!/usr/bin/env bash
# lib_clean_room_gate.sh -- the clean-room gate, shared by the publish cascade and
# the release step.
#
#   . scripts/release/lib_clean_room_gate.sh || exit 1
#   clean_room_gate ROOT TAG          # scripts/cascade-publish.sh, before any upload, and
#                                     # scripts/release/release_gate.sh, before the GitHub release
#   clean_room_gate_sha ROOT SHA LBL  # the same gate on a commit named by its full sha
#
# Both answer the same question with the same code: is there a green
# `clean-room (aprender)` job in paiml/infra clean-room.yml that tested exactly
# this commit? clean_room_gate resolves the tag and asks clean_room_gate_sha.
# Moved here verbatim from scripts/cascade-publish.sh (PMAT-4805) so the release
# step asks the question with the code the cascade runs.
#
# Option-neutral: sourced, so no `set` at file scope. Failure is the return status.

# ==========================================================================
# THE CLEAN-ROOM GATE (PMAT-3318). The cascade refuses to start unless
# `clean-room.yml` is green on EXACTLY the tag's commit. Fail-closed, no flag,
# no environment bypass.
#
# WHY: clean-room was the first gate named in the release doctrine and was
# enforced nowhere -- none of the publish/cascade scripts referenced it. It ran
# red 8/8 from 2026-09-08 to 2026-09-15 and v0.66.0 and v0.67.0 both shipped
# over it.
#
# WHAT A RUN ACTUALLY TESTED. The workflow lives in paiml/infra, and the job
# does `git clone --depth 1 git@github.com:paiml/aprender.git`: it tests
# aprender's main HEAD AT CLONE TIME. A run's `headSha` is an INFRA commit, so
# comparing it to an aprender tag would compare two repositories. MEASURED
# (infra run 34915682258, job 104212693804): the only record of the aprender
# commit is one line printed by infra's Makefile `_copy-source`,
#
#     2026-09-15T01:06:22.6420228Z     commit:  030d9b14
#
# (`git rev-parse --short HEAD` of the clone). The result CSV has no sha column
# and no per-repo artifact is uploaded. So the gate reads that line from the
# job log, and it reads it STRICTLY: exactly one such line, 7-40 hex chars,
# which must resolve in THIS repository to exactly the tag commit
# (`git rev-parse --verify <abbrev>^{commit}` refuses an ambiguous prefix).
# Zero lines, two different lines, or a changed format all REFUSE -- if infra
# rewords the line, the release stops; it never silently passes.
#
# THE STRUCTURED RECORD (infra#621/#622, PMAT-3318) is that durable fix, and it
# is PREFERRED over the log line wherever it exists. infra's clean-room job now
# runs `Assert the commit under test` immediately after the clone: it refuses to
# build anything that is not the dispatched ref, and it records the tested
# commit as a full 40-char sha in three places -- the results.csv `tested_sha`
# column, the step summary, and the line
#
#     2026-09-16T00:52:43.0000000Z     tested-sha: <40 hex>
#
# in the JOB LOG. The gate reads the log copy, and only that copy, because it is
# the one that is BOTH per-run and reachable: results.csv holds one row per
# repo (the latest run, not this run) and the step summary text is not exposed
# by any REST endpoint. Reading it costs no new API surface -- it is the same
# `actions/jobs/<id>/logs` response the abbreviation is parsed from.
#
# PRECEDENCE, and what each path still has to prove:
#   structured present -> it must be a full 40-char LOWERCASE sha (an
#     abbreviation in that field is a refusal: being unabbreviated is the whole
#     point of the column), the `Assert the commit under test` step must itself
#     have concluded success (a recorded sha whose assertion did not pass is
#     not evidence), and if the old log line is there too the two must agree --
#     a disagreement REFUSES and prints both.
#   structured absent -> the strict log-line parse above, unchanged.
#   neither -> REFUSE, exactly as before.
# No path is looser than the one it replaces: all of them still require exactly
# one `clean-room (aprender)` job, conclusion `success`, and a tested commit
# EQUAL to the tag commit.
#
# Everything the gate cannot prove is a refusal: no run, a run still queued or
# in progress, cancelled or failed, a different sha, gh unauthenticated or
# erroring, any output it cannot parse. The repository, workflow and job name
# are literals, not parameters: an override would be a bypass.
# ==========================================================================

# stdin: `gh run list --json databaseId,status,conclusion,createdAt`.
# $1: the tag commit's committer time (epoch). A run CREATED before the commit
# existed cannot have cloned it. Prints one TSV row per run:
#   <id> <status> <conclusion|none> <createdAt> <after|before>
# Exits 3 on anything that is not that shape.
clean_room_parse_runs() {
  python3 -c '
import json, sys
from datetime import datetime, timezone
try:
    since = int(sys.argv[1])
    runs = json.load(sys.stdin)
    if not isinstance(runs, list):
        raise ValueError("run list is not a JSON array")
    rows = []
    for r in runs:
        rid = r["databaseId"]
        if isinstance(rid, bool) or not isinstance(rid, int):
            raise ValueError("databaseId is not an integer")
        created = datetime.strptime(r["createdAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        side = "after" if created.timestamp() >= since else "before"
        rows.append("%d\t%s\t%s\t%s\t%s" % (rid, str(r["status"]), r.get("conclusion") or "none", r["createdAt"], side))
except Exception as e:
    print("clean-room: unparseable run list: %s" % e, file=sys.stderr)
    sys.exit(3)
for row in rows:
    print(row)
' "$1"
}

# stdin: `gh run view <id> --json jobs`. $1: the exact job name, $2: the exact
# name of the step that asserts the tested commit.
# Prints "<job id> <status> <conclusion|none> <assert state>" for that job, or
# NONE. The assert state is "<status>/<conclusion>", or `absent` when the job
# has no such step (every run before infra#622), or `ambiguous` when it has
# more than one -- both of which the gate refuses to read a structured sha
# through. Exits 3 on malformed input or on more than one job of that name.
clean_room_parse_job() {
  python3 -c '
import json, sys
try:
    jobs = json.load(sys.stdin)["jobs"]
    if not isinstance(jobs, list):
        raise ValueError("jobs is not a JSON array")
    hits = [j for j in jobs if j.get("name") == sys.argv[1]]
    if len(hits) > 1:
        raise ValueError("%d jobs named %r" % (len(hits), sys.argv[1]))
    row = None
    if hits:
        jid = hits[0]["databaseId"]
        if isinstance(jid, bool) or not isinstance(jid, int):
            raise ValueError("job databaseId is not an integer")
        steps = hits[0].get("steps")
        if steps is None:
            state = "absent"
        elif not isinstance(steps, list):
            raise ValueError("steps is not a JSON array")
        else:
            hit = [st for st in steps if st.get("name") == sys.argv[2]]
            if len(hit) > 1:
                state = "ambiguous"
            elif not hit:
                state = "absent"
            else:
                state = "%s/%s" % (str(hit[0].get("status")), hit[0].get("conclusion") or "none")
        row = "%d\t%s\t%s\t%s" % (jid, str(hits[0]["status"]), hits[0].get("conclusion") or "none", state)
except Exception as e:
    print("clean-room: unparseable jobs: %s" % e, file=sys.stderr)
    sys.exit(3)
print(row if row else "NONE")
' "$1" "$2"
}

# stdin: a clean-room job log. Prints the DISTINCT aprender commits the log
# says it copied into the container -- the `_copy-source` line, optionally
# preceded by the Actions timestamp. Nothing else in the log is trusted.
clean_room_tested_abbrevs() {
  tr -d '\r' \
    | { grep -E '^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z )?    commit:  [0-9a-f]{7,40}$' || true; } \
    | sed -E 's/^.*    commit:  //' \
    | sort -u
}

# stdin: a clean-room job log. Prints the DISTINCT values infra's
# `Assert the commit under test` step recorded as the tested commit -- the
# structured `tested-sha:` field (infra#621/#622), whatever it says. The value
# is captured LOOSELY and validated by the caller ON PURPOSE: a truncated or
# uppercased field has to reach the gate as a malformed record it can refuse by
# name, not vanish and read as "this run predates the structured record".
clean_room_tested_shas() {
  tr -d '\r' \
    | { grep -E '^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+Z )?[[:space:]]*tested-sha: .+$' || true; } \
    | sed -E 's/^.*tested-sha: //' \
    | sed -E 's/[[:space:]]+$//' \
    | sort -u
}

# clean_room_gate ROOT TAG -- 0 only when a completed `clean-room (aprender)`
# job, whose log records exactly one tested commit resolving to TAG's commit,
# concluded `success`. Every other outcome prints a REFUSE line and returns 1.
clean_room_gate() {
  local root=$1 tag=$2 want
  want=$(git -C "$root" rev-parse --verify --quiet "refs/tags/${tag}^{commit}" 2>/dev/null) || want=""
  if [ -z "$want" ]; then
    echo "CLEAN-ROOM REFUSE: tag $tag does not resolve to a commit in $root -- cannot prove clean-room ran on it"
    return 1
  fi
  clean_room_gate_sha "$root" "$want" "tag $tag"
}

# clean_room_gate_sha ROOT SHA LABEL: the same gate on a commit named by its full
# sha (PMAT-4805). clean_room_gate resolves the tag and asks this; the checks are
# the cascade's, unchanged.
# LABEL only words the messages. An abbreviated or non-commit SHA refuses.
clean_room_gate_sha() {
  local root=$1 want=$2 what=$3
  local repo="paiml/infra" workflow="clean-room.yml" job_name="clean-room (aprender)"
  local assert_step="Assert the commit under test"
  local epoch runs_json runs examined=0 seen=""
  local rid rstatus rconcl rcreated side jobs_json job jid jstatus jconcl jassert
  local log abbrevs n shas ns malformed agrees tested tdisp tsource resolved

  if ! [[ "$want" =~ ^[0-9a-f]{40}$ ]] \
     || [ "$(git -C "$root" rev-parse --verify --quiet "${want}^{commit}" 2>/dev/null)" != "$want" ]; then
    echo "CLEAN-ROOM REFUSE: '$want' ($what) is not a full lowercase sha of a commit in $root -- cannot prove clean-room ran on it"
    return 1
  fi
  epoch=$(git -C "$root" log -1 --format=%ct "$want" 2>/dev/null) || epoch=""
  case "$epoch" in
    ''|*[!0-9]*) echo "CLEAN-ROOM REFUSE: looked for $want ($what); could not read its commit time"; return 1 ;;
  esac
  if ! gh auth status >/dev/null 2>&1; then
    echo "CLEAN-ROOM REFUSE: looked for $want ($what); gh is unauthenticated or erroring -- cannot prove clean-room ran on it"
    return 1
  fi
  if ! runs_json=$(gh run list --repo "$repo" --workflow "$workflow" --limit 30 \
        --json databaseId,status,conclusion,createdAt 2>/dev/null); then
    echo "CLEAN-ROOM REFUSE: looked for $want ($what); gh run list --repo $repo --workflow $workflow failed"
    return 1
  fi
  if ! runs=$(clean_room_parse_runs "$epoch" <<< "$runs_json" 2>/dev/null); then
    echo "CLEAN-ROOM REFUSE: looked for $want ($what); the $workflow run list could not be parsed"
    return 1
  fi

  while IFS=$'\t' read -r rid rstatus rconcl rcreated side; do
    [ -n "$rid" ] || continue
    [ "$side" = "after" ] || continue
    examined=$((examined + 1))
    if ! jobs_json=$(gh run view "$rid" --repo "$repo" --json jobs 2>/dev/null); then
      echo "CLEAN-ROOM REFUSE: looked for $want ($what); gh run view $rid failed"
      return 1
    fi
    if ! job=$(clean_room_parse_job "$job_name" "$assert_step" <<< "$jobs_json" 2>/dev/null); then
      echo "CLEAN-ROOM REFUSE: looked for $want ($what); the jobs of run $rid could not be parsed"
      return 1
    fi
    if [ "$job" = "NONE" ]; then
      seen="$seen"$'\n'"  - run $rid ($rstatus/$rconcl, $rcreated): no '$job_name' job"
      continue
    fi
    IFS=$'\t' read -r jid jstatus jconcl jassert <<< "$job"
    if [ "$jstatus" != "completed" ]; then
      seen="$seen"$'\n'"  - run $rid job $jid: $jstatus -- tested sha unknown, conclusion=$jconcl"
      continue
    fi
    if ! log=$(gh api "repos/$repo/actions/jobs/$jid/logs" 2>/dev/null); then
      echo "CLEAN-ROOM REFUSE: looked for $want ($what); could not read the log of run $rid job $jid"
      return 1
    fi
    shas=$(clean_room_tested_shas <<< "$log")
    ns=$(grep -c . <<< "$shas" || true); ns=${ns:-0}
    abbrevs=$(clean_room_tested_abbrevs <<< "$log")
    n=$(grep -c . <<< "$abbrevs" || true); n=${n:-0}
    tested=""; tdisp=""; tsource=""; resolved=""

    if [ "$ns" -gt 1 ]; then
      seen="$seen"$'\n'"  - run $rid job $jid: $ns structured tested-sha record(s) in the log (need exactly 1), conclusion=$jconcl"
      continue
    fi
    if [ "$ns" -eq 1 ]; then
      # The structured record wins -- after it proves it is what it claims.
      malformed=0
      case "${#shas}" in 40) : ;; *) malformed=1 ;; esac
      case "$shas" in *[!0-9a-f]*) malformed=1 ;; esac
      if [ "$malformed" -ne 0 ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: structured tested-sha '$shas' is not a full 40-char lowercase sha, conclusion=$jconcl"
        continue
      fi
      if [ "$jassert" != "completed/success" ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: structured tested-sha $shas but the '$assert_step' step is $jassert (need completed/success), conclusion=$jconcl"
        continue
      fi
      if [ "$n" -gt 1 ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: $n tested-commit record(s) in the log beside structured tested-sha $shas, conclusion=$jconcl"
        continue
      fi
      # The abbreviation is `git rev-parse --short HEAD` of the SAME clone, so
      # agreement is exactly "the log line is a prefix of the structured sha".
      # A quoted glob, not a substring expansion: the latter is SC2299.
      agrees=0
      case "$shas" in "$abbrevs"*) agrees=1 ;; esac
      if [ "$n" -eq 1 ] && [ "$agrees" -ne 1 ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: the two records disagree -- structured tested-sha says $shas, the log line says $abbrevs, conclusion=$jconcl"
        continue
      fi
      tested=$shas; tdisp=$shas; resolved=$shas; tsource="structured"
    else
      if [ "$n" -ne 1 ]; then
        seen="$seen"$'\n'"  - run $rid job $jid: $n tested-commit record(s) in the log (need exactly 1), conclusion=$jconcl"
        continue
      fi
      resolved=$(git -C "$root" rev-parse --verify --quiet "${abbrevs}^{commit}" 2>/dev/null) || resolved=""
      tested=$abbrevs; tdisp="$abbrevs${resolved:+ ($resolved)}"; tsource="log-line"
    fi

    if [ "$resolved" != "$want" ]; then
      seen="$seen"$'\n'"  - run $rid job $jid: tested $tdisp, conclusion=$jconcl [$tsource]"
      continue
    fi
    if [ "$jconcl" = "success" ]; then
      echo "CLEAN-ROOM PROCEED: run $rid job $jid '$job_name' tested $tested = $want ($what), conclusion=success [tested-sha source: $tsource]"
      return 0
    fi
    seen="$seen"$'\n'"  - run $rid job $jid: tested $tested = the tag commit, conclusion=$jconcl [$tsource]"
  done <<< "$runs"

  echo "CLEAN-ROOM REFUSE: looked for $want ($what); no green '$job_name' run tested it ($examined $workflow run(s) created after that commit examined)${seen:- -- found none}"
  return 1
}
