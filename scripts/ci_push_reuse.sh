#!/usr/bin/env bash
# ci_push_reuse.sh - the decision: may a push to main cite its PR's CI run instead of re-running
# x86-main, determinism and the workspace-test shards? (#7298; conditions C1..C6 and the never-reuse
# list are the design's; Refs #7272 #7273)
#
# PURE: it reads facts from one JSON (or YAML, via yq) file and prints exactly one line,
#   reuse=yes
#   reuse=no reason=<the first unproven condition>
# It always exits 0 from `decide`. It FAILS CLOSED: a missing, unreadable or malformed fact, a missing
# jq (or yq for a YAML file), a filter error, or a job result it does not know is `reuse=no`. The word
# `reuse=yes` is printed in ONE place, after the filter returned the single verdict "ok" with rc 0.
# Facts are gathered by scripts/ci_push_reuse_facts.sh (git and API reads); this file does no I/O but
# the facts file.
#
#   ci_push_reuse.sh decide --facts FILE|-   # FILE.json, FILE.yml, or - for stdin
#   ci_push_reuse.sh --self-test             # the case table (must-reuse and must-not-reuse rows)
#   ci_push_reuse.sh --mutants               # delete or invert each condition in a scratch copy; every mutant must turn the table RED
#
# Facts (all required): event ref sha tree parents[] before repo
#   pr{number merged merge_commit_sha base_ref head_ref is_fork head_repo draft head_sha}  (base_ref and head_ref are read from the PR object)
#   stamp{sha tree base head}  (MANDATORY: what the PR run recorded of its test merge commit M; null refuses)
#   m_derived{tree base head}|null (re-derived from M's own object; null when M is not fetchable, and null is refused)
#   expect_shards  runs_complete
#   run{id event path head_sha head_branch status conclusion newest_id created_epoch jobs_total sov_test("success"|...|null: read from the run's x86-main log) jobs[{name conclusion run_id head_sha}]}
#   paths[] (B..S changed paths)  now
#
# C1 push to refs/heads/main, one parent B, before == B.   C2 S bound to a merged same-repo non-draft PR.
# C3 the run tested this tree on main's tip. The stamp the run's workspace-test-ran job emitted is MANDATORY (no stamp, no reuse) and
#    binds the run to the push: stamp.tree == S^{tree}, stamp.base == B (S's parent: main's tip when the run tested), stamp.head == the PR's
#    own head.sha; M's own object must agree when fetchable. There is no fallback: a run's pull_requests[] is empty once the PR is merged.
# C4 the run is complete, real, from the PR's own head branch, the PR is into main (read from the PR object), newest, every named job
#    succeeded; advisories skipped; it did not itself reuse (no receipt-judge job that ran); AND it ran the heavy `sov.test` section:
#    ci/sections.yml sets heavy_on train, so on an ordinary PR x86-main is green with sov.test SKIPPED (printed NOT MEASURED) while a
#    push to main owes it. That result is read from the cited run's own x86-main log, never from a branch name or a label.
# C5 no gate path changed (every path under scripts/ is one).   C6 the run STARTED within 24 h.
# NEVER reused: advisories (the workflow always runs them), tags, release/publish, a cited reuse, a
# run that is not the newest on the head, forks, multi-commit/non-squash/direct pushes, gate-change
# PRs, a red/cancelled/skipped/partly read result.
set -uo pipefail

SELF="${BASH_SOURCE[0]}"

# ---- the decision filter. One line per condition, each ending in `# R-<NAME>` so --mutants can delete
# or invert it. chk() is false on an error, on no output and on more than one output: a condition that
# cannot be evaluated is UNPROVEN, never skipped.
read -r -d '' JQ_PROG <<'JQ'
def sha: type == "string" and test("\\A[0-9a-f]{40}\\z");
def isint: type == "number" and . == floor and . >= 0;
def cj(f): [try f catch false] | (length == 1 and .[0] == true);
def chk($n; f): [$n, ([try f catch false] | (length == 1 and .[0] == true))];
def named($n): [.run.jobs[] | select(.name == $n)];
def okjob($n): (named($n) | length) == 1 and (named($n)[0].conclusion == "success");
def gatepath:
  test("^(\\.github/|ci/|fleet/|machines/|arbiter/|\\.cargo/)")
  or test("^scripts/")
  or test("(^|/)(\\.github/|\\.githooks/)")
  or test("(^|/)[^/]*(-guard\\.sh|pre-job[^/]*|admission[^/]*|publish[^/]*\\.sh|release[^/]*\\.sh)$")
  or test("^(Makefile|rust-toolchain|deny\\.toml|\\.pmat-gates\\.toml)")
  or test("(ruleset|protection|allowlist|-ledger\\.txt)");
def c3a: # an absent or non-object stamp makes every field check below false, so no separate type check is needed
  cj((.stamp.sha | sha) and (.stamp.tree | sha) and (.stamp.base | sha) and (.stamp.head | sha)) and # R-C3A2
  cj(.stamp.tree == .tree) and # R-C3A3
  cj(.stamp.base == .parents[0]) and # R-C3A4
  cj(.stamp.head == .pr.head_sha) and # R-C3A5
  cj(.m_derived != null and (.m_derived.tree == .stamp.tree and .m_derived.base == .stamp.base and .m_derived.head == .stamp.head)); # R-C3A6
[
  chk("F-shape"; type == "object" and (keys | contains(["event","ref","sha","tree","parents","before","repo","pr","stamp","m_derived","expect_shards","runs_complete","run","paths","now"]))), # R-FSHAPE
  chk("C1-event"; .event == "push"), # R-C1EVENT
  chk("C1-ref"; .ref == "refs/heads/main"), # R-C1REF
  chk("C1-shas"; (.sha | sha) and (.tree | sha) and (.before | sha)), # R-C1SHAS
  chk("C1-one-parent"; (.parents | type == "array" and length == 1 and (.[0] | sha))), # R-C1PARENT
  chk("C1-before"; .before == .parents[0]), # R-C1BEFORE
  chk("C2-pr-number"; (.pr.number | type == "number" and . == floor and . > 0)), # R-C2NUM
  chk("C2-merged"; .pr.merged == true), # R-C2MERGED
  chk("C2-merge-sha"; (.pr.merge_commit_sha | sha) and .pr.merge_commit_sha == .sha), # R-C2MSHA
  chk("C2-fork"; .pr.is_fork == false and (.repo | type == "string" and length > 0) and .pr.head_repo == .repo), # R-C2FORK
  chk("C2-draft"; .pr.draft == false), # R-C2DRAFT
  chk("C2-head-sha"; (.pr.head_sha | sha)), # R-C2HEAD
  chk("C3-tree-identity"; c3a), # R-C3
  chk("C4-run-identity"; (.run.id | isint) and .run.event == "pull_request" and .run.head_sha == .pr.head_sha and .run.path == ".github/workflows/ci.yml"), # R-C4IDENT
  chk("C4-pr-base"; .pr.base_ref == "main"), # R-C4PRBASE
  chk("C4-run-branch"; (.run.head_branch | type == "string" and length > 0) and .run.head_branch == .pr.head_ref), # R-C4BRANCH
  chk("C4-run-newest"; .runs_complete == true and .run.id == .run.newest_id), # R-C4NEWEST
  chk("C4-run-done"; .run.status == "completed" and .run.conclusion == "success"), # R-C4DONE
  chk("C4-jobs-fully-read"; (.run.jobs | type == "array" and length > 0) and .run.jobs_total == (.run.jobs | length) and all(.run.jobs[]; (.name | type == "string") and .run_id == $__r and .head_sha == $__h)), # R-C4JOBS
  chk("C4-no-unknown-result"; all(.run.jobs[]; .conclusion == "success" or .conclusion == "skipped")), # R-C4KNOWN
  chk("C4-x86-main"; okjob("x86-main")), # R-C4X86
  chk("C4-determinism"; okjob("determinism")), # R-C4DET
  chk("C4-gate"; okjob("gate")), # R-C4GATE
  chk("C4-ci-gate"; okjob("ci / gate")), # R-C4CIGATE
  chk("C4-workspace-test"; okjob("workspace-test")), # R-C4WT
  chk("C4-workspace-test-ran"; okjob("workspace-test-ran")), # R-C4WTRAN
  chk("C4-shards"; (.expect_shards | type == "number" and . == floor and . >= 1) and (. as $r | $r.expect_shards as $n | (([$r.run.jobs[] | select(.name | startswith("workspace-test-shard"))] | length) == $n) and all(range(1; $n + 1); . as $i | $r | okjob("workspace-test-shard (\($i), \($n))")))), # R-C4SHARDS
  chk("C4-advisories-skipped"; (named("x86-main-advisories") | length) == 1 and named("x86-main-advisories")[0].conclusion == "skipped"), # R-C4ADV
  chk("C4-not-a-reuse"; all(.run.jobs[]; .name != "receipt-judge" or .conclusion == "skipped")), # R-C4TIER
  chk("C4-sov-test"; .run.sov_test == "success"), # R-C4SOV
  chk("C5-paths-judgeable"; (.paths | type == "array" and length > 0) and all(.paths[]; type == "string" and length > 0 and (test("^/|^\\./|\\.\\.|\"|\\\\|\\n") | not))), # R-C5JUDGE
  chk("C5-no-gate-path"; all(.paths[]; gatepath | not)), # R-C5GATE
  chk("C6-fresh"; (.now | isint) and (.run.created_epoch | isint) and (.now - .run.created_epoch) >= 0 and (.now - .run.created_epoch) <= 86400) # R-C6
]
| map(select(.[1] | not))
| if length == 0 then "ok" else .[0][0] end
JQ

no() { printf 'reuse=no reason=%s\n' "$1"; }

decide() {
  local facts="" json verdict rc
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --facts) facts="${2-}"; shift 2 || { no "usage: --facts needs a value"; return 0; } ;;
      *) no "usage: unknown argument"; return 0 ;;
    esac
  done
  [ -n "$facts" ] || { no "no facts file given"; return 0; }
  command -v jq > /dev/null 2>&1 || { no "jq is missing"; return 0; }
  case "$facts" in
    -) json=$(cat) || { no "facts on stdin unreadable"; return 0; } ;;
    *.yml | *.yaml)
      command -v yq > /dev/null 2>&1 || { no "yq is missing (YAML facts)"; return 0; }
      [ -r "$facts" ] || { no "facts file unreadable"; return 0; }
      json=$(yq -o=json '.' "$facts" 2> /dev/null) || { no "facts YAML unreadable"; return 0; } ;;
    *)
      [ -r "$facts" ] || { no "facts file unreadable"; return 0; }
      json=$(cat "$facts") || { no "facts file unreadable"; return 0; } ;;
  esac
  [ -n "$json" ] || { no "facts are empty"; return 0; }
  # $__r and $__h: the run id and head sha every job must carry (read from the same facts, bound before the filter runs)
  verdict=$(jq -r --argjson __r "$(jq -c '.run.id // null' <<< "$json" 2> /dev/null || echo null)" \
    --arg __h "$(jq -r '.pr.head_sha // ""' <<< "$json" 2> /dev/null || true)" "$JQ_PROG" <<< "$json" 2> /dev/null)
  rc=$?
  [ "$rc" -eq 0 ] || { no "decision filter failed (malformed facts)"; return 0; }
  case "$verdict" in
    '') no "decision filter returned nothing"; return 0 ;;
    *$'\n'*) no "decision filter returned several lines"; return 0 ;;
    ok) printf 'reuse=yes\n'; return 0 ;; # the only `yes`
    *) [[ $verdict =~ ^[A-Za-z0-9-]+$ ]] && no "$verdict" || no "decision filter returned an unknown verdict"; return 0 ;;
  esac
}
# ---- the case table (S2b: base facts, must-reuse, C1/C2/C3 rows, malformed input, and the two lane-2 regressions of #7426).
# Each row edits the all-green base facts with one jq expression and states the one-line verdict it must print.
self_test() {
  command -v jq > /dev/null 2>&1 || { echo "ENV: jq missing" >&2; return 2; }
  local base n=0 fails=0 s1 s2 s3 s4 s5 s6
  s1=$(printf '%040d' 1); s2=$(printf '%040d' 2); s3=$(printf '%040d' 3); s4=$(printf '%040d' 4); s5=$(printf '%040d' 5); s6=$(printf '%040d' 6)
  base=$(jq -nc --arg S "$s1" --arg T "$s2" --arg B "$s3" --arg H "$s4" --arg M "$s5" '
    {event:"push", ref:"refs/heads/main", sha:$S, tree:$T, parents:[$B], before:$B, repo:"paiml/aprender",
     pr:{number:7000, merged:true, merge_commit_sha:$S, base_ref:"main", head_ref:"feat/x", is_fork:false, head_repo:"paiml/aprender", draft:false, head_sha:$H},
     stamp:{sha:$M, tree:$T, base:$B, head:$H}, m_derived:{tree:$T, base:$B, head:$H},
     expect_shards:2, runs_complete:true,
     run:{id:100, event:"pull_request", path:".github/workflows/ci.yml", head_sha:$H, head_branch:"feat/x", status:"completed", conclusion:"success",
          newest_id:100, created_epoch:1000000, jobs_total:9, sov_test:"success",
          jobs:(["x86-main","determinism","gate","ci / gate","workspace-test","workspace-test-ran","workspace-test-shard (1, 2)","workspace-test-shard (2, 2)"]
                | map({name:., conclusion:"success", run_id:100, head_sha:$H}))
               + [{name:"x86-main-advisories", conclusion:"skipped", run_id:100, head_sha:$H}]},
     paths:["crates/aprender-core/src/lib.rs"], now:1000100}') || return 2
  row() { # name, jq edit of the base, the verdict line it must print
    local out; out=$(jq -c "$2" <<< "$base" | bash "$SELF" decide --facts -)
    n=$((n + 1)); [ "$out" = "$3" ] || { echo "FAIL $1: got '$out', want '$3'"; fails=$((fails + 1)); }
  }
  raw() { # name, stdin text, the verdict line it must print
    local out; out=$(printf '%s' "$2" | bash "$SELF" decide --facts -)
    n=$((n + 1)); [ "$out" = "$3" ] || { echo "FAIL $1: got '$out', want '$3'"; fails=$((fails + 1)); }
  }
  row must-reuse '.' 'reuse=yes'
  row c1-event '.event="pull_request"' 'reuse=no reason=C1-event'
  row c1-ref '.ref="refs/heads/dev"' 'reuse=no reason=C1-ref'
  row c1-two-parents ".parents+=[\"$s6\"]" 'reuse=no reason=C1-one-parent'
  row c1-before ".before=\"$s6\"" 'reuse=no reason=C1-before'
  row c2-number '.pr.number=0' 'reuse=no reason=C2-pr-number'
  row c2-merged '.pr.merged=false' 'reuse=no reason=C2-merged'
  row c2-merge-sha ".pr.merge_commit_sha=\"$s6\"" 'reuse=no reason=C2-merge-sha'
  row c2-fork '.pr.is_fork=true' 'reuse=no reason=C2-fork'
  row c2-draft '.pr.draft=true' 'reuse=no reason=C2-draft'
  row c3-stamp-null '.stamp=null' 'reuse=no reason=C3-tree-identity'
  row c3-stamp-tree ".stamp.tree=\"$s6\"" 'reuse=no reason=C3-tree-identity'
  row c3-stamp-base ".stamp.base=\"$s6\"" 'reuse=no reason=C3-tree-identity'
  row c3-stamp-head ".stamp.head=\"$s6\"" 'reuse=no reason=C3-tree-identity'
  row c3-m-derived-null '.m_derived=null' 'reuse=no reason=C3-tree-identity'
  row c3-m-derived-tree ".m_derived.tree=\"$s6\"" 'reuse=no reason=C3-tree-identity'
  # lane-2 regressions at 0ac867c: a trailing newline on a sha, and a ./-prefixed path past the ^-anchored gate rules
  row sha-trailing-newline '.pr.head_sha+="\n" | .run.head_sha+="\n" | .stamp.head+="\n" | .m_derived.head+="\n"' 'reuse=no reason=C2-head-sha'
  row path-dot-slash '.paths=["./scripts/ci_push_reuse.sh"]' 'reuse=no reason=C5-paths-judgeable'
  row path-gate '.paths=["scripts/ci_push_reuse.sh"]' 'reuse=no reason=C5-no-gate-path'
  raw empty-stdin '' 'reuse=no reason=facts are empty'
  raw empty-object '{}' 'reuse=no reason=F-shape'
  raw not-json 'not json' 'reuse=no reason=decision filter failed (malformed facts)'
  echo "ci_push_reuse self-test: $n rows, $fails failed"
  [ "$fails" -eq 0 ]
}
case "${1-}" in
  decide) shift; decide "$@" ;;
  --self-test) self_test ;;
  -h | --help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "usage: ci_push_reuse.sh decide --facts FILE|- | --self-test | --mutants" >&2; exit 2 ;;
esac
