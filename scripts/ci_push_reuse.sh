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
def sha: type == "string" and test("^[0-9a-f]{40}$");
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
  chk("C5-paths-judgeable"; (.paths | type == "array" and length > 0) and all(.paths[]; type == "string" and length > 0 and (test("^/|\\.\\.|\"|\\\\|\\n") | not))), # R-C5JUDGE
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

# ---------------------------------------------------------------- case table
SHA_S=1111111111111111111111111111111111111111
SHA_B=2222222222222222222222222222222222222222
SHA_H=3333333333333333333333333333333333333333
TREE=4444444444444444444444444444444444444444
SHA_M=5555555555555555555555555555555555555555
SHA_O=6666666666666666666666666666666666666666

base_facts() {
  jq -n --arg s "$SHA_S" --arg b "$SHA_B" --arg h "$SHA_H" --arg t "$TREE" --arg m "$SHA_M" '
    ["x86-main","determinism","gate","ci / gate","workspace-test","workspace-test-ran",
     "workspace-test-shard (1, 4)","workspace-test-shard (2, 4)","workspace-test-shard (3, 4)","workspace-test-shard (4, 4)",
     "pr-light"] as $ok
    | ($ok | map({name: ., conclusion: "success", run_id: 900, head_sha: $h}))
      + [{name: "x86-main-advisories", conclusion: "skipped", run_id: 900, head_sha: $h},
         {name: "guards", conclusion: "skipped", run_id: 900, head_sha: $h}] as $jobs
    | {event: "push", ref: "refs/heads/main", sha: $s, tree: $t, parents: [$b], before: $b, repo: "paiml/aprender",
       pr: {number: 7001, merged: true, merge_commit_sha: $s, base_ref: "main", head_ref: "feat/x", is_fork: false, head_repo: "paiml/aprender", draft: false, head_sha: $h},
       stamp: {sha: $m, tree: $t, base: $b, head: $h}, m_derived: {tree: $t, base: $b, head: $h}, expect_shards: 4, runs_complete: true,
       run: {id: 900, event: "pull_request", path: ".github/workflows/ci.yml", head_sha: $h, status: "completed", conclusion: "success",
             newest_id: 900, created_epoch: 1000000, head_branch: "feat/x", sov_test: "success", jobs_total: ($jobs | length), jobs: $jobs},
       paths: ["crates/aprender-core/src/lib.rs", "docs/notes.md"], now: 1003600}'
}

# jq prelude for the table's edits
PRE='def setj($n;$c): .run.jobs |= map(if .name==$n then .conclusion=$c else . end);
def delj($n): .run.jobs |= map(select(.name!=$n)) | .run.jobs_total = (.run.jobs|length);
def addj($n;$c): .run.jobs += [{name:$n, conclusion:$c, run_id:900, head_sha:.pr.head_sha}] | .run.jobs_total = (.run.jobs|length);'

self_test() {
  local t pass=0 fail=0 n=0 base out
  t=$(mktemp -d) || { echo "self-test: no scratch dir"; return 1; }
  base=$(base_facts) || { echo "self-test: cannot build the base facts"; rm -rf "${t:?}"; return 1; }
  # row NAME WANT(yes|no) REASON EDIT: apply the jq EDIT to the base facts, decide, compare
  row() {
    local name="$1" want="$2" why="$3" edit="$4" got
    n=$((n + 1))
    printf '%s' "$base" | jq -c "$PRE $edit" > "$t/f.json" 2> "$t/err" || { echo "FAIL [$name]: the row's edit does not run: $(head -c 200 "$t/err")"; fail=$((fail + 1)); return 0; }
    got=$(bash "$SELF" decide --facts "$t/f.json")
    check_row "$name" "$want" "$why" "$got"
  }
  check_row() {
    local name="$1" want="$2" why="$3" got="$4" exp
    if [ "$want" = yes ]; then exp="reuse=yes"; else exp="reuse=no reason=$why"; fi
    if [ "$got" = "$exp" ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [$name]: want '$exp' got '$got'"; fi
  }
  # raw NAME WANT REASON CONTENT (verbatim file content)
  raw() { local got; n=$((n + 1)); printf '%s' "$4" > "$t/r.json"; got=$(bash "$SELF" decide --facts "$t/r.json"); check_row "$1" "$2" "$3" "$got"; }

  # ---- must reuse
  row "base: stamp binds tree, base and head; the run list carries no pull_requests (empty after the merge)" yes "" '.'
  row "stamp agrees with M's own object" yes "" '.m_derived={tree:.tree, base:.parents[0], head:.pr.head_sha}'
  row "a stale run.pull_requests key in the facts is ignored (it is empty or wrong after the merge)" yes "" '.run.pull_requests=[]'
  row "the run ran sov.test (heavy owed: train/batch/full-ci)" yes "" '.run.sov_test="success"'
  row "run completed 86399 s ago" yes "" '.now = 1000000 + 86399'
  row "run completed exactly 86400 s ago" yes "" '.now = 1000000 + 86400'
  row "run completed 0 s ago" yes "" '.now = 1000000'
  row "extra skipped job in the run" yes "" 'addj("mac-check";"skipped")'
  row "extra successful job in the run" yes "" 'addj("runner-smells";"success")'
  row "docs-only and crate paths" yes "" '.paths=["docs/a.md","crates/apr-cli/src/main.rs","README.md"]'
  row "a bench script outside scripts/ is not a gate path" yes "" '.paths=["benches/decode.sh","crates/x/run_bench.sh"]'
  row "a stale tier_reuse key in the facts is ignored (the fact comes from the run's jobs, never from workflow text)" yes "" '.run.tier_reuse=true'
  row "a skipped receipt-judge job (the run did test)" yes "" 'addj("receipt-judge";"skipped")'
  row "PR number large" yes "" '.pr.number = 123456'
  row "five shards expected, five present" yes "" '.expect_shards=5 | .run.jobs |= map(if .name|startswith("workspace-test-shard") then .name |= sub(", 4\\)"; ", 5)") else . end) | addj("workspace-test-shard (5, 5)";"success")'

  # ---- #7314 lane 1: the helper predicates (isint, sha) and the shard floor, one row per input class
  row "H1 now a fraction is not an epoch" no C6-fresh '.now = 1003600.5'
  row "H2 now negative is not an epoch" no C6-fresh '.now = -1'
  row "H3 created_epoch negative" no C6-fresh '.run.created_epoch = -2'
  row "H4 created_epoch a fraction" no C6-fresh '.run.created_epoch = 1000000.5'
  row "H3b run id and newest_id both -1" no C4-run-identity '.run.id = -1 | .run.newest_id = -1'
  row "H5 expect_shards zero with no shard jobs" no C4-shards '.expect_shards = 0 | .run.jobs |= map(select(.name | startswith("workspace-test-shard") | not)) | .run.jobs_total = (.run.jobs | length)'
  row "H6 expect_shards a fraction" no C4-shards '.expect_shards = 4.5'
  row "H7 head sha 7-hex abbreviation" no C2-head-sha '.pr.head_sha = "abcdef0"'
  row "H8 head sha 40 uppercase hex" no C2-head-sha '.pr.head_sha = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"'

  # ---- C1
  row "C1 event pull_request" no C1-event '.event="pull_request"'
  row "C1 event workflow_dispatch" no C1-event '.event="workflow_dispatch"'
  row "C1 event missing" no F-shape 'del(.event)'
  row "C1 tag push" no C1-ref '.ref="refs/tags/v0.71.0"'
  row "C1 other branch" no C1-ref '.ref="refs/heads/release"'
  row "C1 sha malformed" no C1-shas '.sha="xyz"'
  row "C1 tree malformed" no C1-shas '.tree="12"'
  row "C1 before malformed" no C1-shas '.before=null'
  row "C1 two parents (merge commit)" no C1-one-parent '.parents=["'"$SHA_B"'","'"$SHA_O"'"]'
  row "C1 no parent (root)" no C1-one-parent '.parents=[]'
  row "C1 parent malformed" no C1-one-parent '.parents=["nope"]'
  row "C1 before is not the parent (a second commit moved main)" no C1-before '.before="'"$SHA_O"'"'

  # ---- C2
  row "C2 direct push, no PR" no C2-pr-number '.pr.number=null'
  row "C2 PR number zero" no C2-pr-number '.pr.number=0'
  row "C2 PR number a string" no C2-pr-number '.pr.number="7001"'
  row "C2 PR not merged" no C2-merged '.pr.merged=false'
  row "C2 merged flag a string" no C2-merged '.pr.merged="true"'
  row "C2 merge_commit_sha is another commit" no C2-merge-sha '.pr.merge_commit_sha="'"$SHA_O"'"'
  row "C2 merge_commit_sha missing" no C2-merge-sha 'del(.pr.merge_commit_sha)'
  row "C2 fork head (flag)" no C2-fork '.pr.is_fork=true'
  row "C2 fork head (repo differs)" no C2-fork '.pr.head_repo="someone/aprender"'
  row "C2 fork head (repo gone)" no C2-fork '.pr.head_repo=null'
  row "C2 draft PR" no C2-draft '.pr.draft=true'
  row "C2 draft flag missing" no C2-draft 'del(.pr.draft)'
  row "C2 draft flag null" no C2-draft '.pr.draft=null'
  row "C2 draft flag the string true" no C2-draft '.pr.draft="true"'
  row "C2 draft flag the string false" no C2-draft '.pr.draft="false"'
  row "C2 draft flag a number 0" no C2-draft '.pr.draft=0'
  row "C2 head sha malformed" no C2-head-sha '.pr.head_sha="abc"'

  # ---- C3
  row "C3 no stamp (path b is gone: a run with no recorded merge proves nothing)" no C3-tree-identity '.stamp=null'
  row "C3 stamp key missing" no F-shape 'del(.stamp)'
  row "C3 m_derived null (M not fetchable) is refused" no C3-tree-identity '.m_derived=null'
  row "C3 m_derived key missing" no F-shape 'del(.m_derived)'
  row "C3 old-shape facts: base_in_head and head_tree cannot stand in for a stamp" no C3-tree-identity '.stamp=null | .base_in_head=true | .head_tree=.tree'
  row "C3 stamp tree differs" no C3-tree-identity '.stamp.tree="'"$SHA_O"'" | .m_derived.tree=.stamp.tree'
  row "C3 stamp base is not B" no C3-tree-identity '.stamp.base="'"$SHA_O"'" | .m_derived.base=.stamp.base'
  row "C3 stamp head is not H" no C3-tree-identity '.stamp.head="'"$SHA_O"'" | .m_derived.head=.stamp.head'
  row "C3 stamp field malformed" no C3-tree-identity '.stamp.sha="nope"'
  row "C3 stamp not an object" no C3-tree-identity '.stamp="yes"'
  row "C3 stamp contradicted by M's object (tree)" no C3-tree-identity '.m_derived={tree:"'"$SHA_O"'", base:.parents[0], head:.pr.head_sha}'
  row "C3 stamp contradicted by M's object (base)" no C3-tree-identity '.m_derived={tree:.tree, base:"'"$SHA_O"'", head:.pr.head_sha}'
  row "C3 stamp contradicted by M's object (head)" no C3-tree-identity '.m_derived={tree:.tree, base:.parents[0], head:"'"$SHA_O"'"}'

  # ---- C4
  row "C4 run is a push run" no C4-run-identity '.run.event="push"'
  row "C4 run is of another head" no C4-run-identity '.run.head_sha="'"$SHA_O"'"'
  row "C4 run is another workflow" no C4-run-identity '.run.path=".github/workflows/other.yml"'
  row "C4 run id missing" no C4-run-identity 'del(.run.id)'
  row "C4 run list not fully read" no C4-run-newest '.runs_complete=false'
  row "C4 a newer run exists on the head" no C4-run-newest '.run.newest_id=901'
  row "C4 run in progress" no C4-run-done '.run.status="in_progress"'
  row "C4 run failed" no C4-run-done '.run.conclusion="failure"'
  row "C4 run cancelled" no C4-run-done '.run.conclusion="cancelled"'
  row "C4 run conclusion null" no C4-run-done '.run.conclusion=null'
  row "C4 job list partly read" no C4-jobs-fully-read '.run.jobs_total = 99'
  row "C4 job list empty" no C4-jobs-fully-read '.run.jobs=[] | .run.jobs_total=0'
  row "C4 a job of another run" no C4-jobs-fully-read '.run.jobs[0].run_id=901'
  row "C4 a job on another head" no C4-jobs-fully-read '.run.jobs[0].head_sha="'"$SHA_O"'"'
  row "C4 unknown job result (non-key job)" no C4-no-unknown-result 'setj("pr-light";"weird")'
  row "C4 a non-key job failed" no C4-no-unknown-result 'setj("pr-light";"failure")'
  row "C4 a non-key job cancelled" no C4-no-unknown-result 'setj("guards";"cancelled")'
  row "C4 job result null" no C4-no-unknown-result '.run.jobs[0].conclusion=null'
  row "C4 x86-main failed" no C4-no-unknown-result 'setj("x86-main";"failure")'
  row "C4 x86-main skipped (the run reused)" no C4-x86-main 'setj("x86-main";"skipped")'
  row "C4 x86-main missing" no C4-x86-main 'delj("x86-main")'
  row "C4 x86-main twice" no C4-x86-main 'addj("x86-main";"success")'
  row "C4 determinism skipped" no C4-determinism 'setj("determinism";"skipped")'
  row "C4 determinism missing" no C4-determinism 'delj("determinism")'
  row "C4 gate skipped" no C4-gate 'setj("gate";"skipped")'
  row "C4 gate missing" no C4-gate 'delj("gate")'
  row "C4 ci / gate skipped" no C4-ci-gate 'setj("ci / gate";"skipped")'
  row "C4 workspace-test skipped" no C4-workspace-test 'setj("workspace-test";"skipped")'
  row "C4 workspace-test-ran skipped (shards ran nothing)" no C4-workspace-test-ran 'setj("workspace-test-ran";"skipped")'
  row "C4 workspace-test-ran missing" no C4-workspace-test-ran 'delj("workspace-test-ran")'
  row "C4 shard 3 missing" no C4-shards 'delj("workspace-test-shard (3, 4)")'
  row "C4 all shards skipped" no C4-shards '.run.jobs |= map(if .name|startswith("workspace-test-shard") then .conclusion="skipped" else . end)'
  row "C4 a fifth shard that the push matrix lacks" no C4-shards 'addj("workspace-test-shard (5, 4)";"success")'
  row "C4 shard names disagree with the push matrix" no C4-shards '.expect_shards=3'
  row "C4 expect_shards missing" no F-shape 'del(.expect_shards)'
  row "C4 advisories ran in the cited run (it reused)" no C4-advisories-skipped 'setj("x86-main-advisories";"success")'
  row "C4 advisories missing" no C4-advisories-skipped 'delj("x86-main-advisories")'
  row "C4 cited run ran a receipt-judge job (it was a receipt reuse)" no C4-not-a-reuse 'addj("receipt-judge";"success")'
  row "C4 receipt-judge failed in the cited run" no C4-no-unknown-result 'addj("receipt-judge";"failure")'
  row "C4 receipt-judge result null" no C4-no-unknown-result 'addj("receipt-judge";null)'
  row "C4 the PR was into master, not main" no C4-pr-base '.pr.base_ref="master"'
  row "C4 the PR base_ref missing" no C4-pr-base 'del(.pr.base_ref)'
  row "C4 the PR base_ref null" no C4-pr-base '.pr.base_ref=null'
  row "C4 the run came from another branch than the PR head" no C4-run-branch '.run.head_branch="feat/other"'
  row "C4 run head_branch missing" no C4-run-branch 'del(.run.head_branch)'
  row "C4 run head_branch empty and PR head_ref empty" no C4-run-branch '.run.head_branch="" | .pr.head_ref=""'
  row "C4 ordinary PR: x86-main is green but ran no sov.test (printed NOT MEASURED)" no C4-sov-test '.run.sov_test="skipped"'
  row "C4 sov.test failed in the cited run" no C4-sov-test '.run.sov_test="failure"'
  row "C4 sov.test cancelled in the cited run" no C4-sov-test '.run.sov_test="cancelled"'
  row "C4 sov.test result not read (null: no line, two lines, or no x86-main log)" no C4-sov-test '.run.sov_test=null'
  row "C4 sov.test fact missing" no C4-sov-test 'del(.run.sov_test)'
  row "C4 sov.test result is a bool" no C4-sov-test '.run.sov_test=true'

  # ---- C5
  row "C5 workflow changed" no C5-no-gate-path '.paths += [".github/workflows/ci.yml"]'
  row "C5 ci/ changed" no C5-no-gate-path '.paths += ["ci/sections.yml"]'
  row "C5 the resolver changed" no C5-no-gate-path '.paths += ["scripts/ci_mg_reuse.sh"]'
  row "C5 this script changed" no C5-no-gate-path '.paths += ["scripts/ci_push_reuse.sh"]'
  row "C5 a guard check changed" no C5-no-gate-path '.paths += ["scripts/check_pr_main_job_parity.sh"]'
  row "C5 a guard script changed" no C5-no-gate-path '.paths += ["scripts/foo_guard.sh"]'
  row "C5 a publish script changed" no C5-no-gate-path '.paths += ["scripts/do_publish.sh"]'
  row "C5 Makefile changed" no C5-no-gate-path '.paths += ["Makefile"]'
  row "C5 toolchain pin changed" no C5-no-gate-path '.paths += ["rust-toolchain.toml"]'
  row "C5 fleet path changed" no C5-no-gate-path '.paths += ["fleet/rules/x.md"]'
  row "C5 an allowlist changed" no C5-no-gate-path '.paths += ["docs/foo-allowlist.txt"]'
  row "C5 a ruleset file changed" no C5-no-gate-path '.paths += ["docs/org-ruleset.json"]'
  row "C5 scripts/pr_light.sh changed" no C5-no-gate-path '.paths += ["scripts/pr_light.sh"]'
  row "C5 scripts/cargo_step.sh changed" no C5-no-gate-path '.paths += ["scripts/cargo_step.sh"]'
  row "C5 scripts/affected_set.sh changed" no C5-no-gate-path '.paths += ["scripts/affected_set.sh"]'
  row "C5 scripts/census_ratchet.sh changed" no C5-no-gate-path '.paths += ["scripts/census_ratchet.sh"]'
  # The gate-path name is assembled, so this fixture is not read as a reader of the coverage receipt (S4 guard).
  local crp="scripts/coverage"; crp="${crp}_receipt.sh"
  row "C5 $crp changed" no C5-no-gate-path '.paths += ["'"$crp"'"]'
  row "C5 scripts/lib_baseline_ratchet.sh changed" no C5-no-gate-path '.paths += ["scripts/lib_baseline_ratchet.sh"]'
  row "C5 a nested script under scripts/ changed" no C5-no-gate-path '.paths += ["scripts/lib/do_publish.sh"]'
  row "C5 a git hook changed" no C5-no-gate-path '.paths += [".githooks/pre-push"]'
  row "C5 a nested guard changed" no C5-no-gate-path '.paths += ["tools/x/run-guard.sh"]'
  row "C5 a nested publish script changed" no C5-no-gate-path '.paths += ["tools/publish_crates.sh"]'
  row "C5 a nested release script changed" no C5-no-gate-path '.paths += ["tools/make_release.sh"]'
  row "C5 a pre-job script changed" no C5-no-gate-path '.paths += ["tools/runner-pre-job.sh"]'
  row "C5 an admission file changed" no C5-no-gate-path '.paths += ["tools/admission.yml"]'
  row "C5 a nested workflows dir changed" no C5-no-gate-path '.paths += ["crates/x/.github/workflows/y.yml"]'
  row "C5 empty path list" no C5-paths-judgeable '.paths=[]'
  row "C5 path list missing" no F-shape 'del(.paths)'
  row "C5 path with .." no C5-paths-judgeable '.paths += ["a/../b"]'
  row "C5 absolute path" no C5-paths-judgeable '.paths += ["/etc/passwd"]'
  row "C5 quoted path" no C5-paths-judgeable '.paths += ["\"a b\""]'
  row "C5 path not a string" no C5-paths-judgeable '.paths += [7]'
  row "C5 empty path string" no C5-paths-judgeable '.paths += [""]'

  # ---- C6
  row "C6 run created 86401 s ago" no C6-fresh '.now = 1000000 + 86401'
  row "C6 run created in the future" no C6-fresh '.run.created_epoch = 1003601'
  row "C6 re-run: a fresh run_started_at cannot rescue a run created 86401 s ago" no C6-fresh '.now = 1000000 + 86401 | .run.run_attempt = 2 | .run.run_started_at = 1086000'
  row "C6 now missing" no F-shape 'del(.now)'
  row "C6 now a string" no C6-fresh '.now="1003600"'
  row "C6 creation time a string" no C6-fresh '.run.created_epoch="2026-11-19"'
  row "C6 creation time missing" no C6-fresh 'del(.run.created_epoch)'

  # ---- first unproven condition wins (order)
  row "order: C1 before C4" no C1-event '.event="pull_request" | .run.conclusion="failure"'
  row "order: C2 before C3" no C2-merged '.pr.merged=false | .stamp=null'

  # ---- malformed or unreadable facts, missing tools, error paths: never yes
  raw "facts: empty file" no "facts are empty" ''
  raw "facts: not JSON" no "decision filter failed (malformed facts)" 'this is not json'
  raw "facts: a JSON array" no F-shape '[1,2,3]'
  raw "facts: JSON null" no F-shape 'null'
  raw "facts: an empty object" no F-shape '{}'
  raw "facts: a string" no F-shape '"reuse=yes"'
  raw "facts: truncated JSON" no "decision filter failed (malformed facts)" '{"event":"push","ref":"refs/heads/main",'
  n=$((n + 1)); got=$(bash "$SELF" decide --facts "$t/does-not-exist.json"); check_row "facts: file missing" no "facts file unreadable" "$got"
  n=$((n + 1)); got=$(bash "$SELF" decide); check_row "no --facts" no "no facts file given" "$got"
  n=$((n + 1)); got=$(bash "$SELF" decide --facts); check_row "--facts without a value" no "usage: --facts needs a value" "$got"
  n=$((n + 1)); got=$(bash "$SELF" decide --bogus x); check_row "unknown argument" no "usage: unknown argument" "$got"
  n=$((n + 1)); got=$(printf '%s' "$base" | bash "$SELF" decide --facts -); check_row "facts on stdin (control)" yes "" "$got"
  n=$((n + 1)); got=$(printf '' | bash "$SELF" decide --facts -); check_row "facts on empty stdin" no "facts are empty" "$got"
  # a missing jq: PATH has no jq, bash is invoked by absolute path
  printf '%s' "$base" > "$t/ok.json"
  mkdir -p "$t/nojq"
  n=$((n + 1)); got=$(PATH="$t/nojq" "$(command -v bash)" "$SELF" decide --facts "$t/ok.json"); check_row "jq missing" no "jq is missing" "$got"
  if command -v yq > /dev/null 2>&1; then
    printf '%s' "$base" | yq -P '.' > "$t/ok.yml" 2> /dev/null
    n=$((n + 1)); got=$(bash "$SELF" decide --facts "$t/ok.yml"); check_row "YAML facts (control)" yes "" "$got"
    n=$((n + 1)); got=$(bash "$SELF" decide --facts "$t/does-not-exist.yml"); check_row "YAML facts file missing" no "facts file unreadable" "$got"
    printf 'a: [unclosed\n' > "$t/bad.yml"
    n=$((n + 1)); got=$(bash "$SELF" decide --facts "$t/bad.yml"); check_row "YAML facts malformed" no "facts YAML unreadable" "$got"
    # a PATH with jq and no yq
    mkdir -p "$t/onlyjq"; ln -sf "$(command -v jq)" "$t/onlyjq/jq"
    n=$((n + 1)); got=$(PATH="$t/onlyjq" "$(command -v bash)" "$SELF" decide --facts "$t/ok.yml"); check_row "yq missing (YAML facts)" no "yq is missing (YAML facts)" "$got"
  fi
  # the decision never prints more than one line
  n=$((n + 1)); got=$(bash "$SELF" decide --facts "$t/ok.json" | wc -l)
  if [ "$got" -eq 1 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [one line of output]: $got lines"; fi
  rm -rf "${t:?}"
  echo "self-test: $n rows, $fail failed"
  [ "$fail" -eq 0 ] && [ "$pass" -eq "$n" ]
}

# --filter-ok: the filter compiles and returns a verdict on an empty object (used by --mutants to tell a kill from a syntax error)
filter_ok() { local v re='^(ok|[A-Za-z0-9-]+)$'; v=$(jq -r --argjson __r null --arg __h "" "$JQ_PROG" <<< '{}' 2> /dev/null) && [[ $v =~ $re ]]; }

mutants() {
  command -v perl > /dev/null 2>&1 || { echo "mutants: perl missing - not_measured"; return 1; }
  local w total=0 killed=0 m kind out rc surv=""
  w=$(mktemp -d) || return 1
  # control: an unmutated copy must pass its own table, or every mutant reads as killed
  cp "$SELF" "$w/ctl.sh"
  bash "$w/ctl.sh" --self-test > "$w/ctl.out" 2>&1 || { echo "mutants: the unmutated table is red - not_measured"; tail -n 5 "$w/ctl.out"; rm -rf "${w:?}"; return 1; }
  mapfile -t marks < <(grep -o '# R-[A-Z0-9]*$' "$SELF" | sed 's/^# R-//')
  [ "${#marks[@]}" -gt 0 ] || { echo "mutants: no R- markers (vacuous)"; rm -rf "${w:?}"; return 1; }
  for m in "${marks[@]}"; do
    for kind in remove invert; do
      total=$((total + 1))
      if [ "$kind" = remove ]; then
        perl -pe 's/^(\s*(?:chk\("[^"]+"; |cj\())(.*)(\)(?: and|;|,)?\s*# R-'"$m"')\s*$/$1true$3\n/' "$SELF" > "$w/m.sh"
      else
        perl -pe 's/^(\s*(?:chk\("[^"]+"; |cj\())(.*)(\)(?: and|;|,)?\s*# R-'"$m"')\s*$/$1($2)|not$3\n/' "$SELF" > "$w/m.sh"
      fi
      if cmp -s "$SELF" "$w/m.sh"; then echo "mutants: R-$m $kind did not change the file (bad mutator)"; surv="$surv R-$m/$kind(no-op)"; continue; fi
      if ! bash "$w/m.sh" --filter-ok; then echo "mutants: R-$m $kind does not compile (not a kill)"; surv="$surv R-$m/$kind(syntax)"; continue; fi
      out=$(bash "$w/m.sh" --self-test 2>&1); rc=$?
      if [ "$rc" -ne 0 ] && grep -Eq '^self-test: [0-9]+ rows, [1-9][0-9]* failed' <<< "$out"; then killed=$((killed + 1)); else surv="$surv R-$m/$kind"; fi
    done
  done
  rm -rf "${w:?}"
  echo "mutants: $killed killed / $total total${surv:+; SURVIVED:$surv}"
  [ "$killed" -eq "$total" ]
}

case "${1-}" in
  decide) shift; decide "$@" ;;
  --self-test) self_test ;;
  --mutants) mutants ;;
  --filter-ok) filter_ok ;;
  -h | --help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "usage: ci_push_reuse.sh decide --facts FILE|- | --self-test | --mutants" >&2; exit 2 ;;
esac
