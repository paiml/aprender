#!/usr/bin/env bash
# check_pr_main_job_parity.sh - RED MAIN guard: no job may run on a push to main that no
# pull_request and no merge_group run covers (ticket in the PR body; cause class "a": the job
# does not run before the merge, so main is the test bench).
#
# For every workflow with a push trigger that reaches main, each job is evaluated on three
# events: push, pull_request, merge_group. A job is on an event when the workflow triggers
# on it and the job's `if:` does not exclude it. The `if:` is evaluated per event with `&&`, `||`, `!` and parentheses;
# a term about `github.event_name ==|!= 'X'` is true or false, any other term is "may run". A job whose `if:` names
# github.event_name in a form not understood, or that does not parse, is RED (unjudgeable), never read as running everywhere.
# FAIL CLOSED: a workflow yq cannot read, a directory with no *.yml or *.yaml, and an unjudgeable `if:` each exit 1 with the cause.
#
# VIOLATION 1: job is on push, and not on pull_request. A merge_group run does not count: the queue
# is a different event and may be off, and a job that first runs after the merge makes main the test bench.
# VIOLATION 2: a job's run script returns success on a pull_request before it has checked the heavy jobs
# (an `if` on the pull_request event whose body reaches `exit 0`, or `[ .. = pull_request ] && exit 0`).
# An aggregate job that passes a PR on a light check alone makes the heavy jobs optional there.
# VIOLATION 3 (draft parity): in a workflow where a heavy job (timeout-minutes >= 100 in any branch of its expression) that runs on
# push to main and on pull_request carries a `github.event.pull_request.draft` clause, every other such heavy job must carry it
# too. A heavy job without it runs on a draft PR while its siblings skip, so a draft burns the pool the others spare.
# --live is kept as an alias of the plain scan (merge_group-only coverage is now RED offline, so no ruleset read is needed).
# EXEMPT: ci/push-only-jobs.txt, one `<workflow-file>:<job> <reason>` per line. An entry
# whose job is not push-only, or does not exist, is stale and fails.
#
#   bash scripts/check_pr_main_job_parity.sh              # scan .github/workflows
#   bash scripts/check_pr_main_job_parity.sh --live       # alias of the plain scan
#   bash scripts/check_pr_main_job_parity.sh --self-test  # case table on throwaway workflows
#   env: PARITY_WORKFLOWS (dir), PARITY_EXEMPT (file)
#
# KNOWN LIMITS (forms this guard does not judge; each is refused or ignored, never read as parity):
#   - `if:` terms other than github.event_name, github.ref and `github.event.pull_request ==|!= null` are "may run"; a job that
#     only runs on a PR through another context (a label, an actor, an input) is read as on the PR.
#   - A `needs:` edge counts only inside one workflow file; a need that names no job there reads as not on the event (RED).
#   - A `pull_request.types` list is judged only for the three defaults (opened, synchronize, reopened) being present.
#   - A `paths`/`paths-ignore` filter must equal the push filter (a set, or as written when it holds a `!` entry); a looser PR filter is RED.
#   - The PR-shortcut scan reads run scripts line by line: a shortcut built across a function, an eval, a sourced file or a variable
#     holding the event name is not seen. It matches `exit 0` and a bare `exit` after a pull_request test; it does not read `exit $rc`.
#   - A reusable workflow (`uses:` job) is judged by its caller's `if:` only; the callee's own steps are not read.
#   - A step-level `if:` is not read: a step skipped on a pull_request by `if: github.event_name != 'pull_request'` reads as run.
#     (Reading it flagged three live jobs whose step `if:` legitimately names the event, so it stays a limit.)
set -uo pipefail
command -v yq >/dev/null || { echo "ENV: yq missing"; exit 2; }
command -v jq >/dev/null || { echo "ENV: jq missing"; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WF="${PARITY_WORKFLOWS:-$ROOT/.github/workflows}"
EX="${PARITY_EXEMPT:-$ROOT/ci/push-only-jobs.txt}"


# Every read below FAILS CLOSED: a workflow yq or jq cannot read, an empty workflow set, and an `if:` the judge cannot
# evaluate are each a RED (non-zero, with the cause named), never an OK. Reads go through a variable, never a
# process substitution, because `< <(cmd)` discards cmd's exit status.
WFILES=()
wfiles() { # sets WFILES to every *.yml and *.yaml directly in $WF; non-zero when the directory is unreadable or holds none
  local out
  [ -d "$WF" ] || { echo "RED: workflow directory $WF does not exist (nothing was judged)" >&2; return 1; }
  out=$(find "$WF" -maxdepth 1 \( -type f -o -type l \) \( -name '*.yml' -o -name '*.yaml' \) | LC_ALL=C sort) || { echo "RED: cannot list $WF (nothing was judged)" >&2; return 1; }
  [ -n "$out" ] || { echo "RED: no *.yml or *.yaml workflow in $WF (nothing was judged, a vacuous pass is refused)" >&2; return 1; }
  mapfile -t WFILES <<< "$out"
}

# The judge. Input: "<file>\t<job>\t<pushmain>\t<pr trigger>\t<mg trigger>\t<if expr>" lines. For each job on a push to main it
# evaluates the `if:` once per event with three values: T, F, U (a term that is not about github.event_name is U, "may run").
# `&&`, `||`, `!` and parentheses are understood; function calls are folded into one U term. A term that names github.event_name
# in any form but `github.event_name ==|!= 'lit'`, an unbalanced or empty expression, or a call that names it, is unjudgeable:
# the job prints as `ERR\t<file>\t<job>\t<reason>` and scan() refuses. A job is on an event when the workflow triggers on it and
# its `if:` is not F there. Output: "<file>\t<job>\t1\t<on pr>\t<on mg>" for jobs on push.
JUDGE='
function setor(v, w) { return (v == "T" || w == "T") ? "T" : ((v == "U" || w == "U") ? "U" : "F") }
function setand(v, w) { return (v == "F" || w == "F") ? "F" : ((v == "U" || w == "U") ? "U" : "T") }
function tokenize(s,   i, c, n, a, j) {
  nt = 0; i = 1; n = length(s)
  while (i <= n && err == "") {
    c = substr(s, i, 1)
    if (c == " ") { i++; continue }
    if (c == "(" || c == ")") { t[++nt] = c; i++; continue }
    if (substr(s, i, 2) == "&&" || substr(s, i, 2) == "||") { t[++nt] = substr(s, i, 2); i += 2; continue }
    if (c == "!" && substr(s, i + 1, 1) != "=") { t[++nt] = "!"; i++; continue }
    a = ""
    while (i <= n) {
      c = substr(s, i, 1)
      if (c == "\047") { j = index(substr(s, i + 1), "\047"); if (j == 0) { err = "unterminated quote"; return } a = a substr(s, i, j + 1); i += j + 1; continue }
      if (c == "(" || c == ")") break
      if (substr(s, i, 2) == "&&" || substr(s, i, 2) == "||") break
      a = a c; i++
    }
    sub(/ +$/, "", a); t[++nt] = "A:" a
  }
}
function atom(a,   lit, eq, la) {
  if (a == "") { err = "empty term"; return "U" }
  # GitHub compares strings and property names case-insensitively ('PUSH' == 'push' is true), so judge on the lower-cased term.
  la = tolower(a)
  if (la ~ /event_name/) {
    if (la !~ /^github\.event_name *(==|!=) *\047[a-z_]+\047$/) { err = "github.event_name used in a form the check does not understand: " a; return "U" }
    match(la, /\047[a-z_]+\047/); lit = substr(la, RSTART + 1, RLENGTH - 2); eq = (lit == ev)
    if (a ~ /!=/) return eq ? "F" : "T"
    return eq ? "T" : "F"
  }
  # github.ref differs per event (push to main: refs/heads/main; a PR: refs/pull/N/merge; the queue: refs/heads/gh-readonly-queue/...),
  # so `github.ref == refs/heads/main` is false on a PR. Only the exact `github.ref ==|!= '"'"'lit'"'"'` form is judged; any other
  # use of github.ref* is unjudgeable.
  if (la ~ /github\.ref/) {
    if (la !~ /^github\.ref *(==|!=) *\047[^\047]+\047$/) { err = "github.ref used in a form the check does not understand: " a; return "U" }
    match(la, /\047[^\047]+\047/); lit = substr(la, RSTART + 1, RLENGTH - 2); eq = (lit == tolower(rf))
    if (a ~ /!=/) return eq ? "F" : "T"
    return eq ? "T" : "F"
  }
  # github.event.pull_request is null on push and merge_group, an object on pull_request; only the exact `== null` / `!= null` form is judged.
  if (la ~ /^github\.event\.pull_request *(==|!=) *null$/) {
    eq = (ev != "pull_request")
    if (a ~ /!=/) return eq ? "F" : "T"
    return eq ? "T" : "F"
  }
  # Any other use of github.event.pull_request that is not a plain field access (`.draft`, `.head.repo...`) is unjudgeable, not "may run":
  # a bare `!github.event.pull_request`, `github.event[\047pull_request\047] == null`, `null == github.event.pull_request`, or a field
  # compared to null / the empty string is push-only (null on push, an object on a PR) in a spelling the exact form above does not match.
  if (la ~ /github\.event *\[/ || la ~ /github\.event\.pull_request([^.a-z_0-9]|$)/ || (la ~ /github\.event\.pull_request\./ && la ~ /(==|!=) *(null|\047\047)|(null|\047\047) *(==|!=)/)) {
    err = "github.event.pull_request used in a form the check does not understand (only == null, != null and plain field reads are judged): " a; return "U"
  }
  if (la == "true") return "T"   # GitHub reads TRUE/True/true alike
  if (la == "false") return "F"
  if (la ~ /^(-?0|-?0\.0+|-?0x0+|-?0e[+-]?0+|nan|\047\047|null)$/) return "F"   # the falsy literals: 0, -0, 0.0, empty string, null
  return "U"
}
function pprim(   v, a) {
  if (pos > nt) { err = "expression ends early"; return "U" }
  if (t[pos] == "(") { pos++; v = por(); if (err != "") return "U"; if (pos > nt || t[pos] != ")") { err = "unbalanced parenthesis"; return "U" } pos++; return v }
  if (substr(t[pos], 1, 2) == "A:") { a = substr(t[pos], 3); pos++; return atom(a) }
  err = "unexpected " t[pos]; return "U"
}
function pnot(   v) {
  if (pos > nt) { err = "expression ends early"; return "U" }
  if (t[pos] == "!") { pos++; v = pnot(); return (v == "T") ? "F" : ((v == "F") ? "T" : "U") }
  return pprim()
}
function pand(   v, w) { v = pnot(); while (err == "" && pos <= nt && t[pos] == "&&") { pos++; w = pnot(); v = setand(v, w) } return v }
function por(   v, w) { v = pand(); while (err == "" && pos <= nt && t[pos] == "||") { pos++; w = pand(); v = setor(v, w) } return v }
function fold(s,   m) { # function calls become one U term; one that names github.event_name is unjudgeable
  gsub(/[$][{][{]/, "", s); gsub(/[}][}]/, "", s)
  while (match(s, /[A-Za-z_][A-Za-z0-9_]*\([^()]*\)/)) {
    m = substr(s, RSTART, RLENGTH)
    if (tolower(m) ~ /event_name/) err = "a call names github.event_name: " m
    else if (tolower(m) ~ /github\.ref/) err = "a call names github.ref: " m
    s = substr(s, 1, RSTART - 1) "CALL" substr(s, RSTART + RLENGTH)
  }
  return s
}
function judge(e, evn,   v) {
  err = ""; ev = evn; pos = 1
  rf = (evn == "push") ? "refs/heads/main" : ((evn == "pull_request") ? "refs/pull/N/merge" : "refs/heads/gh-readonly-queue/main/pr-1-0")
  if (tolower(e) ~ /github[ ]*\[/) { err = "github[...] bracket dereference: " e; return "U" }
  e = fold(e)
  if (e ~ /^ *$/) return "T"
  if (err != "") return "U"
  tokenize(e)
  if (err != "") return "U"
  v = por()
  if (err == "" && pos <= nt) err = "trailing " t[pos]
  return v
}
BEGIN { FS = "\t" }
{
  nj++; ex = $6; bad = ""; onpr = 0; onmg = 0; pv = "F"
  if ($3 == "1") { pv = judge(ex, "push"); if (err != "") bad = err }
  if ($4 == "1" && bad == "") { v = judge(ex, "pull_request"); if (err != "") bad = err; else onpr = (v != "F") }
  if ($5 == "1" && bad == "") { v = judge(ex, "merge_group"); if (err != "") bad = err; else onmg = (v != "F") }
  FL[nj] = $1; JB[nj] = $2; PM[nj] = ($3 == "1"); ONPUSH[nj] = ($3 == "1" && pv != "F"); BAD[nj] = bad
  P[nj] = onpr; M[nj] = onmg; NEEDS[nj] = $7
  # A job whose if: names always(), cancelled() or failure() runs whatever its needs did; any other job is skipped when a needed job is skipped.
  # Quoted literals are dropped first (a string holding "always()" is no status call). Only always() and a negated cancelled()/failure()
  # stay true when a need was skipped; a bare cancelled()/failure() is false then, so such a job keeps needs: propagation (RED, never OK).
  lx = tolower(ex); gsub(/\047[^\047]*\047/, "", lx)
  # A success() call anywhere makes the job need its needs to have succeeded (a skipped need is not success), so it keeps propagation.
  ST[nj] = ((lx ~ /always[ ]*\(/ || lx ~ /![ ]*(cancelled|failure)[ ]*\(/) && lx !~ /success[ ]*\(/)
  KEY[$1 SUBSEP $2] = nj
}
END {
  # needs: a job without a status-check function is not on an event when a job it needs is not (a skipped need skips it); a need that
  # names no job of the file is unjudgeable and reads as not on the event. Iterated to a fixed point (the relation only ever clears).
  for (r = 0; r < nj; r++) for (i = 1; i <= nj; i++) {
    if (ST[i] || NEEDS[i] == "") continue
    n = split(NEEDS[i], nd, ",")
    for (k = 1; k <= n; k++) {
      if (!((FL[i] SUBSEP nd[k]) in KEY)) { P[i] = 0; M[i] = 0; continue }
      x = KEY[FL[i] SUBSEP nd[k]]
      if (!P[x]) P[i] = 0
      if (!M[x]) M[i] = 0
    }
  }
  for (i = 1; i <= nj; i++) {
    if (!PM[i]) continue
    if (BAD[i] != "") { print "ERR\t" FL[i] "\t" JB[i] "\t" BAD[i]; continue }
    if (ONPUSH[i]) print FL[i] "\t" JB[i] "\t1\t" P[i] "\t" M[i]
  }
}'

# rows: "<file>\t<job>\t1\t<pr>\t<mg>" with 1/0 for each job on push, or "ERR\t<file>\t<job>\t<reason>"
rows() {
  local f raw acc=""
  for f in "${WFILES[@]}"; do
    raw=$(yq -o=json '.' "$f") || { echo "RED: $(basename "$f") is not readable by yq (malformed YAML), so it was not judged" >&2; return 1; }
    raw=$(printf '%s' "$raw" | jq -r --arg f "$(basename "$f")" '
      def arr: if type=="array" then . elif type=="null" then [] else [.] end;
      def gl: gsub("(?<c>[|$^(){}])";"\\\(.c)") | gsub("\\.";"\\.") | gsub("\\*\\*";"@@DS@@") | gsub("\\*";"[^/]*") | gsub("@@DS@@";".*") | gsub("\\?";"[^/]");
      # A paths filter on pull_request must be non-empty and equal the push filter. A filter holding a negation is order-sensitive,
      # so it is compared as written; one without is compared as a set.
      def pseq($a; $b): ($a|arr) as $x | ($b|arr) as $y | ($x|length) > 0 and (if ($x+$y|map(tostring|startswith("!"))|any) then $x == $y else ($x|sort) == ($y|sort) end);
      def pats($ps; $b): reduce ($ps|arr)[] as $p (false; if ($p|startswith("!")) then (if ($b|test("^"+($p[1:]|gl)+"$")) then false else . end) else (if ($b|test("^"+($p|gl)+"$")) then true else . end) end);
      (.on // {}) as $on
      | (if ($on|type)=="string" then [$on] elif ($on|type)=="array" then $on else ($on|keys) end) as $ev
      | (if ($on|type)=="object" and ($on.push|type)=="object"
           then (($on.push.branches // null) as $b | ($on.push.tags // null) as $t | ($on.push["branches-ignore"] // null) as $bi
                 | if $b==null and $t!=null and $bi==null then false
                   else (($b==null or pats($b; "main") or ($b|arr|map(select(.=="main" or .=="master" or test("[*?]")))|length>0))
                         and ($bi==null or (pats($bi; "main")|not))) end)
           else ($ev|index("push")!=null) end) as $pushmain
      | (if ($ev|index("pull_request")) == null then false
         elif ($on|type)=="object" and ($on.pull_request|type)=="object" then
           ($on.pull_request) as $p
           | (($on.push // {}) | if type=="object" then . else {} end) as $q
           | (if $p.branches != null then pats($p.branches; "main") else true end)
             and (if $p["branches-ignore"] != null then (pats($p["branches-ignore"]; "main") | not) else true end)
             and (if $p.types != null then (($p.types|arr) as $ty | (["opened","synchronize","reopened"] | all(. as $d | $ty | index($d) != null))) else true end)
             and (if $p.paths != null then pseq($p.paths; $q.paths) else true end)
             and (if $p["paths-ignore"] != null then pseq($p["paths-ignore"]; $q["paths-ignore"]) else true end)
         else true end) as $prmain
      | (.jobs // {}) | to_entries[]
      | [$f, .key, (if $pushmain then "1" else "0" end), (if $prmain then "1" else "0" end),
         (if $ev|index("merge_group") then "1" else "0" end),
         (((.value.if | if . == null then "" else . end)|tostring)|gsub("[\t\n\r]+";" ")),
         ((.value.needs // []) | arr | map(tostring) | join(","))] | @tsv') || { echo "RED: $(basename "$f") has a shape jq cannot read, so it was not judged" >&2; return 1; }
    [ -z "$raw" ] || acc="$acc$raw
"
  done
  printf '%s' "$acc" | awk "$JUDGE"
}

# VIOLATION 2 rows: "<file>\t<job>" for each job whose run script exits 0 on a pull_request branch.
shortcuts() {
  local f raw j b body
  for f in "${WFILES[@]}"; do
    raw=$(yq -o=json '.' "$f" | jq -r '(.jobs // {}) | to_entries[] | [.key, ([.value.steps[]?.run // empty] | join("\n") | @base64)] | @tsv') \
      || { echo "RED: $(basename "$f") could not be read for the pull_request shortcut check" >&2; return 1; }
    while IFS=$'\t' read -r j b; do
      [ -n "$j" ] || continue
      body=$(printf '%s' "$b" | base64 -d) || { echo "RED: $(basename "$f"):$j run script is not decodable" >&2; return 1; }
      if printf '%s' "$body" | awk '
          # A PR "shortcut" is any path that returns success on a pull_request before the heavy checks. Judged conservatively (over-matching
          # is a RED, never an OK): `[ .. pull_request .. ] && exit 0`, `[ .. pull_request .. ] || .. exit 0` (either polarity), an if/elif
          # on pull_request whose block (nested ifs counted by depth, an else included) reaches exit 0, and a `case` arm naming pull_request
          # that reaches exit 0.
          function words(l, w,   n, a, i, c) { gsub(/"[^"]*"/, "", l); gsub(/\047[^\047]*\047/, "", l); n = split(l, a, /[^A-Za-z0-9_]+/); c = 0; for (i = 1; i <= n; i++) if (a[i] == w) c++; return c }
          /^[[:space:]]*#/ { next }
          {
            l = $0; nif = words(l, "if"); nfi = words(l, "fi")
            # `exit 0` and a bare `exit` (status of the last command, success after an echo) both return success.
            ex = (l ~ /(^|[^A-Za-z0-9_-])exit[[:space:]]+["\047]?0["\047]?([^0-9]|$)/ || l ~ /(^|[^A-Za-z0-9_-])exit[[:space:]]*([;&|})#]|$)/)
            if (l ~ /(\[|test[[:space:]]).*pull_request.*(&&|\|\|).*(exit)/ && ex) hit = 1
            # `[ .. pull_request .. ] && {` / `|| {` (or a line that continues): the brace block, or the next line, is the guarded body.
            if (blk > 0 && ex) hit = 1
            if (blk == 1 && l !~ /(&&|\|\||\\)[[:space:]]*$/ && l !~ /\{[[:space:]]*$/) blk = 0
            if (blk == 2 && l ~ /^[[:space:]]*\}/) blk = 0
            if (blk == 1 && l ~ /^[[:space:]]*\{[[:space:]]*$/) blk = 2
            if (l ~ /pull_request/ && l ~ /(&&|\|\|)[[:space:]]*\{[^}]*$/) blk = 2
            if (l ~ /^[[:space:]]*(\[|test[[:space:]]).*pull_request/ && l ~ /(&&|\|\||\\)[[:space:]]*\{?[[:space:]]*$/ ) blk = (l ~ /\{[[:space:]]*$/) ? 2 : 1
            if (l ~ /^[[:space:]]*(if|elif)[[:space:]].*pull_request/) { if (ex) hit = 1 }
            else if (prd > 0 && ex) hit = 1
            d += nif - nfi
            if (l ~ /^[[:space:]]*(if|elif)[[:space:]].*pull_request/ && prd == 0 && d > 0 && (nif > nfi || l ~ /^[[:space:]]*elif/)) prd = d
            if (prd > 0 && d < prd) prd = 0
            if (d < 0) d = 0
            if (l !~ /^[[:space:]]*(if|elif)[[:space:]]/ && l ~ /(^|[[:space:]|(])["\047]?pull_request[*"\047]*[[:space:]]*[|)]/) { carm = 1; if (ex) hit = 1 }
            else if (carm && ex) hit = 1
            if (carm && l ~ /;;/) carm = 0
          }
          END { exit !hit }'; then
        printf '%s\t%s\n' "$(basename "$f")" "$j"
      fi
    done <<< "$raw"
  done
}

# VIOLATION 3 rows: "<file>\t<job>" for each heavy push+PR job lacking the draft clause a heavy sibling carries.
# $1 = the rows() output (read once by scan).
drafts() {
  local f pr_jobs raw j
  for f in "${WFILES[@]}"; do
    pr_jobs="$(printf '%s\n' "$1" | awk -F'\t' -v f="$(basename "$f")" '$1==f && $4=="1" {print $2}')"
    [ -n "$pr_jobs" ] || continue
    raw=$(yq -o=json '.' "$f" | jq -r --arg pj "$pr_jobs" '
      ($pj | split("\n")) as $pr
      | [ (.jobs // {}) | to_entries[]
          | select(.key as $k | $pr | index($k))
          | select(((.value["timeout-minutes"] // 0) | tostring | [scan("[0-9]+") | tonumber] | max // 0) >= 100)
          | {k: .key, d: (((.value.if | if . == null then "" else . end) | tostring) | contains("github.event.pull_request.draft"))} ] as $h
      | if ($h | map(select(.d)) | length) > 0 then $h[] | select(.d | not) | .k else empty end') \
      || { echo "RED: $(basename "$f") could not be read for the draft-parity check" >&2; return 1; }
    while read -r j; do
      [ -n "$j" ] || continue
      printf '%s\t%s\n' "$(basename "$f")" "$j"
    done <<< "$raw"
  done
}

scan() {
  local fail=0 f j p pr mg key reason rws sc dr
  local exempt="" used=""
  wfiles || return 1
  rws=$(rows) || { echo "RED: the workflow rows could not be read; the scan is refused, not passed"; return 1; }
  sc=$(shortcuts) || { echo "RED: the pull_request shortcut scan could not be read; refused, not passed"; return 1; }
  dr=$(drafts "$rws") || { echo "RED: the draft-parity scan could not be read; refused, not passed"; return 1; }
  if [ -f "$EX" ]; then
    while read -r key reason || [ -n "$key" ]; do
      case "$key" in ''|\#*) continue ;; esac
      [ -n "$reason" ] || { echo "STALE: $key has no reason"; fail=1; }
      exempt="$exempt
$key"
    done < "$EX"
  fi
  while IFS=$'\t' read -r f j p pr mg; do
    [ -n "$f" ] || continue
    if [ "$f" = ERR ]; then
      echo "RED: $j:$p has an if: the check cannot judge ($pr${mg:+ $mg}); an unjudgeable if: is refused, never read as running everywhere"
      fail=1; continue
    fi
    key="$f:$j"
    if [ "$pr" = 0 ]; then
      if grep -qxF -- "$key" <<< "$exempt"; then used="$used
$key"; continue; fi
      if [ "$mg" = 1 ]; then echo "RED: $key runs on push to main but is skipped on pull_request by an if: event guard (merge_group does not cover it)"
      else echo "RED: $key runs on push to main and not on any pull_request run"; fi
      fail=1
    fi
  done <<< "$rws"
  while IFS=$'\t' read -r f j; do
    [ -n "$f" ] || continue
    echo "RED: $f:$j passes a pull_request (exit 0) before it checks the heavy jobs"
    fail=1
  done <<< "$sc"
  while IFS=$'\t' read -r f j; do
    [ -n "$f" ] || continue
    echo "RED: $f:$j is a heavy job on push and pull_request without the draft clause its heavy siblings carry (a draft PR runs it)"
    fail=1
  done <<< "$dr"
  for key in $(printf "%s\n" "$exempt" | tr " " "_" | grep -v "^$"); do
    grep -qxF -- "$key" <<< "$used" || { echo "STALE: exemption $key is not a push-only job"; fail=1; }
  done
  [ "$fail" = 0 ] && echo "OK: every job that runs on push to main also runs on a pull_request, and no job passes a PR before its heavy checks"
  return "$fail"
}

self_test() {
  local d fails=0 w rc
  d="$(mktemp -d)"; w="$d/wf"; mkdir -p "$w"
  t() { # name, expected rc, exempt content, workflow body
    printf '%s' "$4" > "$w/ci.yml"; printf '%s' "$3" > "$d/ex"
    PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; local rc=$?
    [ "$rc" = "$2" ] && echo "ok   $1" || { echo "FAIL $1 (rc $rc, want $2)"; fails=1; }
  }
  local H='on: {push: {branches: [main]}, pull_request: {}, merge_group: {}}
jobs:'
  t "push-only job is RED" 1 "" "$H
  a: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
"
  t "job on pull_request is green" 0 "" "$H
  a: {runs-on: x, steps: []}
"
  t "!= pull_request but on merge_group is RED (the queue does not cover it)" 1 "" "$H
  a: {runs-on: x, if: \"github.event_name != 'pull_request'\", steps: []}
"
  t "!= pull_request with a push-only exemption is green" 0 "ci.yml:a ticket-maker, cannot block" "$H
  a: {runs-on: x, if: \"github.event_name != 'pull_request'\", steps: []}
"
  t "draft guard only is green" 0 "" "$H
  a: {runs-on: x, if: \"github.event.pull_request.draft != true\", steps: []}
"
  t "aggregate exits 0 on a PR before the heavy checks is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            echo light passed; exit 0
          fi
          check_heavy
"
  t "aggregate one-liner [ pull_request ] && exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" = pull_request ] && exit 0
          check_heavy
"
  t "aggregate that checks light then falls through is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            [ \"\${PRL:-}\" = success ] || exit 1
            echo light passed
          fi
          check_heavy
"
  t "draft exit 0 and an echo naming pull_request are green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${DRAFT:-}\" != true ] || { echo draft; exit 0; }
          echo \"pull_request: nothing owed\"; exit 0
"
  t "heavy job missing the draft clause its sibling carries is RED" 1 "" "$H
  a: {runs-on: x, timeout-minutes: 240, if: \"github.event.pull_request.draft != true\", steps: []}
  b: {runs-on: x, timeout-minutes: 120, steps: []}
"
  t "heavy jobs both carrying the draft clause are green" 0 "" "$H
  a: {runs-on: x, timeout-minutes: 240, if: \"github.event.pull_request.draft != true\", steps: []}
  b: {runs-on: x, timeout-minutes: 120, if: \"!cancelled() && github.event.pull_request.draft != true\", steps: []}
"
  t "light job without the draft clause beside a heavy one is green" 0 "" "$H
  a: {runs-on: x, timeout-minutes: 240, if: \"github.event.pull_request.draft != true\", steps: []}
  b: {runs-on: x, timeout-minutes: 10, steps: []}
"
  t "== push with || is read as everywhere" 0 "" "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || x\", steps: []}
"
  t "exempt push-only job is green" 0 "ci.yml:a ticket-maker, cannot block" "$H
  a: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
"
  t "stale exemption is RED" 1 "ci.yml:a was push only" "$H
  a: {runs-on: x, steps: []}
"
  t "exemption without reason is RED" 1 "ci.yml:a" "$H
  a: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
"
  t "tag-only push is not main" 0 "" "on: {push: {tags: ['v*']}}
jobs:
  a: {runs-on: x, steps: []}
"
  t "push-only workflow, no if, is RED" 1 "" "on: {push: {branches: [main]}}
jobs:
  a: {runs-on: x, steps: []}
"
  # FAIL CLOSED rows (#7314 lane 2): each was a silent exit 0 before. tf = like t, but names the file and clears the dir first.
  tf() { # name, expected rc, file name, workflow body
    rm -f "${w:?}"/*; printf '%s' "$4" > "$w/$3"; : > "$d/ex"
    PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; local rc=$?
    [ "$rc" = "$2" ] && echo "ok   $1" || { echo "FAIL $1 (rc $rc, want $2)"; fails=1; }
  }
  tf "malformed YAML is RED, not OK" 1 ci.yml 'on: [push
jobs:
  bad: [
'
  printf '%s' "$H
  a: {runs-on: x, steps: []}
" > "$w/good.yml"; printf 'on: [push\njobs:\n  bad: [\n' > "$w/bad.yml"; : > "$d/ex"
  PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; rc=$?
  [ "$rc" = 1 ] && echo "ok   one malformed workflow beside a clean one is RED" || { echo "FAIL one malformed workflow beside a clean one (rc $rc, want 1)"; fails=1; }
  rm -f "${w:?}"/*; : > "$d/ex"
  PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; rc=$?
  [ "$rc" = 1 ] && echo "ok   an empty workflow directory is RED (no vacuous pass)" || { echo "FAIL an empty workflow directory (rc $rc, want 1)"; fails=1; }
  PARITY_WORKFLOWS="$d/no-such-dir" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; rc=$?
  [ "$rc" = 1 ] && echo "ok   a missing workflow directory is RED" || { echo "FAIL a missing workflow directory (rc $rc, want 1)"; fails=1; }
  tf "a .yaml push-only workflow is scanned and RED" 1 ci.yaml "on: {push: {branches: [main]}}
jobs:
  a: {runs-on: x, steps: []}
"
  tf "a clean .yaml workflow is scanned and green" 0 ci.yaml "$H
  a: {runs-on: x, steps: []}
"
  tf "push || merge_group (never on a PR) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || github.event_name == 'merge_group'\", steps: []}
"
  tf "(push || merge_group) && other is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"(github.event_name == 'push' || github.event_name == 'merge_group') && vars.X == 'y'\", steps: []}
"
  tf "event_name == 'push' || a pull_request term is green" 0 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || github.event_name == 'pull_request'\", steps: []}
"
  tf "!= pull_request || draft guard is green (it runs on a PR when not a draft)" 0 ci.yml "$H
  a: {runs-on: x, if: \"\${{ github.event_name != 'pull_request' || !github.event.pull_request.draft }}\", steps: []}
"
  tf "NO-FORK guard form is green" 0 ci.yml "$H
  a: {runs-on: x, if: \"github.event.pull_request.draft != true && (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)\", steps: []}
"
  tf "!cancelled() with a draft clause is green" 0 ci.yml "$H
  a: {runs-on: x, if: \"!cancelled() && github.event.pull_request.draft != true\", steps: []}
"
  tf "upper-case event_name literal 'PUSH' is judged as push, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'PUSH'\", steps: []}
"
  tf "mixed-case literal != 'Pull_Request' is judged as pull_request, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name != 'Pull_Request'\", steps: []}
"
  tf "upper-case property github.EVENT_NAME == 'push' is judged, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.EVENT_NAME == 'push'\", steps: []}
"
  tf "mixed-case literal == 'Pull_Request' is green" 0 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'PUSH' || github.event_name == 'Pull_Request'\", steps: []}
"
  tf "a call naming github.EVENT_NAME is unjudgeable, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"contains(fromJSON('[\\\"push\\\"]'), github.EVENT_NAME)\", steps: []}
"
  tf "a call naming github.event_name is unjudgeable, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"contains(fromJSON('[\\\"push\\\"]'), github.event_name)\", steps: []}
"
  tf "reversed literal == github.event_name is unjudgeable, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"'push' == github.event_name\", steps: []}
"
  tf "an unbalanced parenthesis is unjudgeable, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"(github.event_name == 'push'\", steps: []}
"
  tf "a dangling || is unjudgeable, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' ||\", steps: []}
"
  tf "a negated group that removes PR is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"!(github.event_name == 'pull_request' || github.event_name == 'merge_group')\", steps: []}
"
  # #7314 round 4 (agy lanes 1+2 on ce76a90, lane 1+2 on b8fd832): each row below was a silent exit 0 before.
  tf "[ != pull_request ] || exit 0 is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${EVT:-}\" != pull_request ] || exit 0
          check_heavy
"
  tf "[ != pull_request ] || { ..; exit 0; } is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${EVT:-}\" != pull_request ] || { echo light passed; exit 0; }
          check_heavy
"
  tf "case pull_request) exit 0 on one line is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\${EVT:-}\" in pull_request) exit 0 ;; esac
          check_heavy
"
  tf "case arm on its own line reaching exit 0 is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\${EVT:-}\" in
            push|pull_request)
              echo light
              exit 0 ;;
          esac
          check_heavy
"
  tf "case arm on pull_request that does not exit 0 is green" 0 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\${EVT:-}\" in
            pull_request) echo light ;;
            *) exit 0 ;;
          esac
          check_heavy
"
  tf "nested if before exit 0 inside the pull_request block is RED (depth, not a flag)" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            if [ \"\${PRL:-}\" != success ]; then
              exit 1
            fi
            echo light passed; exit 0
          fi
          check_heavy
"
  tf "single-line if pull_request; fi then a draft exit 0 is green (no leaked state)" 0 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then echo ok; fi
          [ \"\$DRAFT\" != true ] || { echo draft; exit 0; }
"
  tf "pull_request.branches that drops main is RED" 1 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {branches: [release-only]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.branches-ignore main is RED" 1 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {branches-ignore: [main]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.branches [**, !main] drops main, RED" 1 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {branches: ["**", "!main"]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.branches [main] is green" 0 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {branches: [main]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.branches [**] is green" 0 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {branches: ["**"]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.paths the push trigger lacks is RED (a PR can skip the job)" 1 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {paths: ["src/**"]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.paths-ignore the push trigger lacks is RED" 1 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {paths-ignore: ["docs/**"]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "the same paths on push and pull_request is green" 0 ci.yml 'on:
  push: {branches: [main], paths: ["src/**", "b"]}
  pull_request: {paths: ["b", "src/**"]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "push || FALSE (upper-case literal) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || FALSE\", steps: []}
"
  tf "!= pull_request || FALSE is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name != 'pull_request' || FALSE\", steps: []}
"
  tf "push || ! TRUE is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || ! TRUE\", steps: []}
"
  tf "github.ref == refs/heads/main never holds on a PR, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.ref == 'refs/heads/main'\", steps: []}
"
  tf "push || github.ref == refs/heads/main is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || github.ref == 'refs/heads/main'\", steps: []}
"
  tf "github.REF == REFS/HEADS/MAIN (case) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.REF == 'REFS/HEADS/MAIN'\", steps: []}
"
  tf "github.ref != refs/heads/main is green (true on a PR)" 0 ci.yml "$H
  a: {runs-on: x, if: \"github.ref != 'refs/heads/main' || github.event_name == 'push'\", steps: []}
"
  tf "a call naming github.ref is unjudgeable, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"startsWith(github.ref, 'refs/heads/')\", steps: []}
"
  tf "github.ref_name is unjudgeable, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.ref_name == 'main'\", steps: []}
"
  # needs: a skipped need skips its dependent unless the dependent's if: names always(), cancelled() or failure()
  t "needs a push-only (exempt) job, no status function: skipped on a PR, RED" 1 "ci.yml:p push-only ticket maker" "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  x: {runs-on: x, needs: [p], steps: []}
"
  t "needs a push-only job as a bare string, RED" 1 "ci.yml:p push-only ticket maker" "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  x: {runs-on: x, needs: p, steps: []}
"
  t "needs a push-only job, transitively through a middle job, RED" 1 "ci.yml:p push-only ticket maker" "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  m: {runs-on: x, needs: [p], if: \"success()\", steps: []}
  x: {runs-on: x, needs: [m], steps: []}
"
  t "needs a push-only job with !cancelled() runs on a PR, green" 0 "ci.yml:p push-only ticket maker" "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  x: {runs-on: x, needs: [p], if: \"\${{ !cancelled() }}\", steps: []}
"
  t "needs a job the file does not define is unjudgeable, RED" 1 "" "$H
  x: {runs-on: x, needs: [ghost], steps: []}
"
  t "needs a job that runs on a PR is green" 0 "" "$H
  p: {runs-on: x, steps: []}
  x: {runs-on: x, needs: [p], steps: []}
"
  # pull_request.types and an empty or reordered-negated paths filter
  tf "pull_request.types [closed] never runs on an open PR, RED" 1 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {branches: [main], types: [closed]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.types [labeled] is RED" 1 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {types: [labeled]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.types with the three defaults plus more is green" 0 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {types: [opened, synchronize, reopened, ready_for_review]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.paths [] against no push paths is RED" 1 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {paths: []}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "pull_request.paths-ignore [] against no push filter is RED" 1 ci.yml 'on:
  push: {branches: [main]}
  pull_request: {paths-ignore: []}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "the same negated paths in a different order is RED (order decides)" 1 ci.yml 'on:
  push: {branches: [main], paths: ["!src/foo", "src/**"]}
  pull_request: {paths: ["src/**", "!src/foo"]}
jobs:
  a: {runs-on: x, steps: []}
'
  tf "the same negated paths in the same order is green" 0 ci.yml 'on:
  push: {branches: [main], paths: ["src/**", "!src/foo"]}
  pull_request: {paths: ["src/**", "!src/foo"]}
jobs:
  a: {runs-on: x, steps: []}
'
  # more shortcut forms
  tf "[ != pull_request ] || { ...; exit 0; } across lines is RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${EVT:-}\" != pull_request ] || {
            echo light passed
            exit 0
          }
          check_heavy
"
  tf "[ = pull_request ] && { ...; exit 0; } across lines is RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${EVT:-}\" = pull_request ] && {
            echo light passed
            exit 0
          }
          check_heavy
"
  tf "a single-quoted case arm 'pull_request') exit 0 is RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\${EVT:-}\" in
            'pull_request') exit 0 ;;
          esac
          check_heavy
"
  tf "test = pull_request && exit 0 is RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          test \"\${EVT:-}\" = pull_request && exit 0
          check_heavy
"
  tf "a bare exit inside the pull_request if block is RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            echo light passed
            exit
          fi
          check_heavy
"
  tf "exit 1 inside the pull_request if block is green (a failure is not a pass)" 0 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            exit 1
          fi
          check_heavy
"
  # falsy literals and the null form
  tf "push || 0 is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || 0\", steps: []}
"
  tf "push || '' is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || ''\", steps: []}
"
  tf "push || null is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || null\", steps: []}
"
  tf "github.event.pull_request == null is RED (null on push, an object on a PR)" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event.pull_request == null\", steps: []}
"
  tf "github.event.pull_request != null is green (an object on a PR)" 0 ci.yml "$H
  a: {runs-on: x, if: \"github.event.pull_request != null || github.event_name == 'push'\", steps: []}
"
  tf "github.event['pull_request'] == null (bracket form of the push-only test) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event['pull_request'] == null\", steps: []}
"
  tf "!github.event.pull_request (negated bare object) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"!github.event.pull_request\", steps: []}
"
  tf "null == github.event.pull_request (reversed operands) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"null == github.event.pull_request\", steps: []}
"
  tf "github.event.pull_request == '' (empty string compare) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event.pull_request == ''\", steps: []}
"
  tf "github.event.pull_request.number == null (a field null on push) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event.pull_request.number == null\", steps: []}
"
  tf "github.event.pull_request.number == '' (a field empty on push) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event.pull_request.number == ''\", steps: []}
"
  tf "github.event[dq pull_request dq] (double-quoted bracket) is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event[\\\"pull_request\\\"]\", steps: []}
"
  tf "a push-only term inside a wrapped expression is RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"\${{ !github.event.pull_request && vars.X == 'y' }}\", steps: []}
"
  tf "github.event.pull_request.draft read as a plain field stays green" 0 ci.yml "$H
  a: {runs-on: x, if: \"!github.event.pull_request.draft\", steps: []}
"
  # symlinked workflow files are scanned (find -type f skipped them)
  rm -f "${w:?}"/*; : > "$d/ex"
  printf 'on: {push: {branches: [main]}}\njobs:\n  hidden: {runs-on: x, steps: []}\n' > "$d/hidden.yml"; ln -s "$d/hidden.yml" "$w/link.yml"
  PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; rc=$?
  [ "$rc" = 1 ] && echo "ok   a symlinked push-only workflow is scanned and RED" || { echo "FAIL a symlinked push-only workflow (rc $rc, want 1)"; fails=1; }
  rm -f "${w:?}"/*; printf '%s\n  a: {runs-on: x, steps: []}\n' "$H" > "$d/clean.yml"; ln -s "$d/clean.yml" "$w/link.yaml"
  PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; rc=$?
  [ "$rc" = 0 ] && echo "ok   a symlinked clean .yaml workflow is scanned and green" || { echo "FAIL a symlinked clean workflow (rc $rc, want 0)"; fails=1; }
  rm -f "${w:?}"/*; ln -s "$d/no-such-target.yml" "$w/dangling.yml"
  PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; rc=$?
  [ "$rc" = 1 ] && echo "ok   a dangling symlinked workflow is RED" || { echo "FAIL a dangling symlinked workflow (rc $rc, want 1)"; fails=1; }
  rm -f "${w:?}"/*
  # #7314 lane 2 on a192475: each row below was a silent exit 0 before it was fixed.
  tx() { # name, expected rc, workflow body; job p is the exempt push-only one
    rm -f "${w:?}"/*; printf '%s' "$3" > "$w/ci.yml"; printf '%s' "ci.yml:p push only" > "$d/ex"
    PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" bash "$0" >/dev/null 2>&1; local rc=$?
    [ "$rc" = "$2" ] && echo "ok   $1" || { echo "FAIL $1 (rc $rc, want $2)"; fails=1; }
  }
  tf "github['ref'] bracket dereference is unjudgeable, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || github['ref'] == 'refs/heads/main'\", steps: []}
"
  tf "startsWith(github['ref'], ..) bracket dereference is unjudgeable, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || startsWith(github['ref'], 'refs/heads/')\", steps: []}
"
  tf "uppercase GITHUB['ref'] bracket dereference is unjudgeable, RED (#7314 lane 2 on cb90e8c)" 1 ci.yml "$H
  a: {runs-on: x, if: \"GITHUB['ref'] == 'refs/heads/main'\", steps: []}
"
  tf "if: false (a boolean) is a disabled job, not an unconditional one, green (#7314 lane 1 on cb90e8c)" 0 ci.yml "$H
  a: {runs-on: x, timeout-minutes: 240, if: \"github.event.pull_request.draft != true\", steps: []}
  b: {runs-on: x, timeout-minutes: 240, if: false, steps: []}
"
  tf "push || NaN (a falsy literal) never runs on a PR, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || NaN\", steps: []}
"
  tf "push || -0.0 (a falsy literal) never runs on a PR, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || -0.0\", steps: []}
"
  tf "push || 0x0 (a falsy literal) never runs on a PR, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || 0x0\", steps: []}
"
  tf "push || 0e0 (a falsy literal) never runs on a PR, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || 0e0\", steps: []}
"
  tf "push || github.ref == a PR merge ref never runs on a PR, RED" 1 ci.yml "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || github.ref == 'refs/pull/1/merge'\", steps: []}
"
  tx "needs a PR-skipped job, if success() || failure(), RED" 1 "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  a: {runs-on: x, needs: [p], if: \"\${{ success() || failure() }}\", steps: []}
"
  tx "needs a PR-skipped job, if success() || cancelled(), RED" 1 "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  a: {runs-on: x, needs: [p], if: \"\${{ success() || cancelled() }}\", steps: []}
"
  tx "needs a PR-skipped job, if holds a quoted always() literal, RED" 1 "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  a: {runs-on: x, needs: [p], if: \"vars.MODE != 'always()'\", steps: []}
"
  tx "needs a PR-skipped job, if always() && success(), RED (#7314 lane 2 on cb90e8c)" 1 "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  a: {runs-on: x, needs: [p], if: \"\${{ always() && success() }}\", steps: []}
"
  tx "needs a PR-skipped job, if !cancelled() && success(), RED (#7314 lane 2 on cb90e8c)" 1 "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  a: {runs-on: x, needs: [p], if: \"\${{ !cancelled() && success() }}\", steps: []}
"
  tx "needs a PR-skipped job, if always() is green (always runs on a skipped need)" 0 "$H
  p: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
  a: {runs-on: x, needs: [p], if: \"always()\", steps: []}
"
  tf "pull_request.branches with a literal | never matches main, RED" 1 ci.yml "on: {push: {branches: [main]}, pull_request: {branches: [\"release|main\"]}}
jobs:
  a: {runs-on: x, steps: []}
"
  tf "pull_request.branches with a literal $ never matches main, RED" 1 ci.yml "on: {push: {branches: [main]}, pull_request: {branches: [\"main\$\"]}}
jobs:
  a: {runs-on: x, steps: []}
"
  tf "push.branches with a ? glob reaches main, push-only job RED" 1 ci.yml "on: {push: {branches: [\"mai?\"]}}
jobs:
  a: {runs-on: x, steps: []}
"
  tf "push with branches-ignore and tags reaches main, push-only job RED" 1 ci.yml "on: {push: {branches-ignore: [dev], tags: [\"v*\"]}}
jobs:
  a: {runs-on: x, steps: []}
"
  tf "push with branches-ignore: [main] does not reach main, green" 0 ci.yml "on: {push: {branches-ignore: [main]}}
jobs:
  a: {runs-on: x, steps: []}
"
  tf "brace block with code on the opening line is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${EVT:-}\" != pull_request ] || { echo light passed
            exit 0
          }
          check_heavy
"
  tf "backslash-continued { block is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${EVT:-}\" != pull_request ] || \\
          {
            echo light passed
            exit 0
          }
          check_heavy
"
  tf "case arm with spaces round | is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\${EVT:-}\" in
            push | pull_request ) exit 0 ;;
          esac
          check_heavy
"
  tf "case arm with a glob is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\${EVT:-}\" in
            pull_request*) exit 0 ;;
          esac
          check_heavy
"
  tf "bare exit followed by a comment is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            echo light passed
            exit # ok
          fi
          check_heavy
"
  tf "exit \"0\" quoted is a PR shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            exit \"0\"
          fi
          check_heavy
"
  tf "a chained command before [ pull_request ] && exit 0 is a shortcut, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          echo checking && [ \"\${EVT:-}\" = pull_request ] && exit 0
          check_heavy
"
  tf "fi inside quoted text does not close the if, RED" 1 ci.yml "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            echo \"wi-fi light passed\"
            exit 0
          fi
          check_heavy
"
  # Fail closed on a missing tool: a clean workflow and an empty exemption file would be green,
  # so rc 2 here can only come from the ENV check (without it, no rows and no exemptions print OK).
  local sh yqp jqp
  sh="$(command -v bash)"; yqp="$(command -v yq)"; jqp="$(command -v jq)"
  mkdir -p "$d/noyq" "$d/nojq"; ln -sf "$jqp" "$d/noyq/jq"; ln -sf "$yqp" "$d/nojq/yq"
  te() { # name, expected rc, PATH dir
    printf '%s' "$H
  a: {runs-on: x, steps: []}
" > "$w/ci.yml"; : > "$d/ex"
    PATH="$3" PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" "$sh" "$0" >/dev/null 2>&1; local rc=$?
    [ "$rc" = "$2" ] && echo "ok   $1" || { echo "FAIL $1 (rc $rc, want $2)"; fails=1; }
  }
  te "yq missing fails closed (rc 2), clean workflow and empty exemptions" 2 "$d/noyq"
  te "jq missing fails closed (rc 2), clean workflow and empty exemptions" 2 "$d/nojq"
  rm -rf "${d:?}"
  return "$fails"
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --live) scan; exit $? ;;
  -h|--help) sed -n "2,24p" "$0" | sed 's/^# \{0,1\}//' ;;
  *) scan; exit $? ;;
esac
