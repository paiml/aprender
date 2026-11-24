#!/usr/bin/env bash
# check_pr_main_job_parity.sh - RED MAIN guard: no job may run on a push to main that no
# pull_request and no merge_group run covers (ticket in the PR body; cause class "a": the job
# does not run before the merge, so main is the test bench).
#
# For every workflow with a push trigger that reaches main, each job is evaluated on three
# events: push, pull_request, merge_group. A job is on an event when the workflow triggers
# on it and the job's `if:` does not exclude it. The `if:` is read for `github.event_name`
# literals only: `!= 'X'` removes X; `== 'X'` (no `||` in the expression) keeps only the
# listed events; a `||` expression is the UNION of its disjuncts (a disjunct naming no event runs everywhere); `'X' == github.event_name`
# is read like `github.event_name == 'X'`; any other use of github.event_name (inside a function call such as contains(.., github.event_name), or compared to anything but a quoted literal) is unreadable and fails closed (rc 2); quoted strings are masked first so a ( ) ! | inside one is not read as grammar.
# A missing or empty workflow dir, a workflow that does not parse, or a workflows set with no job also fail closed (rc 2, never OK).
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
set -uo pipefail
command -v yq >/dev/null || { echo "ENV: yq missing"; exit 2; }
command -v jq >/dev/null || { echo "ENV: jq missing"; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WF="${PARITY_WORKFLOWS:-$ROOT/.github/workflows}"
EX="${PARITY_EXEMPT:-$ROOT/ci/push-only-jobs.txt}"

# eval_if EXPR -> prints 1 when the if: expression CAN be true, 0 when it cannot. EXPR holds only 0 1 u && || ! ( ) and spaces
# (rows() has already replaced every atom; u is an unknown atom). Every u is tried as both 0 and 1, so a negation of an unknown
# (!(u), !u && 1) is read exactly, not as false. Anything else, a parse failure or more than 12 unknowns is rc 1 (fail closed).
eval_if() {
  [[ "$1" =~ ^[01u\&\|\!\(\)[:space:]]+$ ]] || return 1
  local e="$1" u n mask i x v limit=1
  u="${e//[^u]/}"; n="${#u}"
  [ "$n" -le 12 ] || return 1
  for ((i = 0; i < n; i++)); do limit=$((limit * 2)); done
  for ((mask = 0; mask < limit; mask++)); do
    x="$e"; i=0
    while [[ "$x" == *u* ]]; do x="${x/u/$(( (mask >> i) & 1 ))}"; i=$((i + 1)); done
    v="$(bash -c 'echo $(( $1 ))' _ "$x" 2>/dev/null)"
    case "$v" in 1) echo 1; return 0 ;; 0) ;; *) return 1 ;; esac
  done
  echo 0
}

# rows: "<file>\t<job>\t<push>\t<pr>\t<mg>" with 1/0. Each job's if: is evaluated once per event with a real &&, ||, ! and ()
# grammar: github.event_name comparisons become 1 or 0 for that event, every other atom (a ref, a needs result, a status function)
# is unknown (u) and eval_if tries it as both 0 and 1, so the guard skips an event only when no value of the unknowns makes it true.
rows() {
  local f line fl j pm pr_on mg_on ep epr emg bp bpr bmg
  for f in "$WF"/*.yml "$WF"/*.yaml; do
    [ -e "$f" ] || continue
    while IFS=$'\t' read -r fl j pm pr_on mg_on ep epr emg; do
      [ -n "$j" ] || continue
      bp=0; bpr=0; bmg=0
      if [ "$pm" = 1 ]; then bp="$(eval_if "$ep")" || return 1; fi
      if [ "$pr_on" = 1 ]; then bpr="$(eval_if "$epr")" || return 1; fi
      if [ "$mg_on" = 1 ]; then bmg="$(eval_if "$emg")" || return 1; fi
      [ "$bp" = 1 ] || continue
      printf '%s\t%s\t1\t%s\t%s\n' "$fl" "$j" "$bpr" "$bmg"
    done < <(yq -o=json '.' "$f" | jq -r --arg f "$(basename "$f")" '
      def ex($e; $x):
        ($e | gsub("\\$\\{\\{|\\}\\}"; " ")
            | gsub("github\\s*\\[\\s*\u0027event_name\u0027\\s*\\]|github\\s*\\.\\s*event_name"; "github.event_name"; "i")
            | gsub("github\\.event_name\\s*==\\s*\u0027(?<n>[a-z_]+)\u0027"; if .n == $x then "@1" else "@0" end)
            | gsub("\u0027(?<n>[a-z_]+)\u0027\\s*==\\s*github\\.event_name"; if .n == $x then "@1" else "@0" end)
            | gsub("github\\.event_name\\s*!=\\s*\u0027(?<n>[a-z_]+)\u0027"; if .n == $x then "@0" else "@1" end)
            | gsub("\u0027(?<n>[a-z_]+)\u0027\\s*!=\\s*github\\.event_name"; if .n == $x then "@0" else "@1" end)
            | gsub("\u0027[^\u0027]*\u0027"; " S ")
            | if test("github\\.event_name") or test("\\bgithub\\b(?!\\s*\\.)";"i") then "?" else . end
            | gsub("[A-Za-z_]+\\([^()]*\\)"; " u ")
            | gsub("[A-Za-z_]+\\([^()]*\\)"; " u ")
            | gsub("[A-Za-z_]+\\([^()]*\\)"; " u ")
            | gsub("[A-Za-z_]+\\([^()]*\\)"; " u ")) as $c
        | ("\\(|\\)|&&|\\|\\||!(?!=)|(?:[^()&|!]|!(?==))+") as $re
        | [
            $c | scan($re)
            | select(test("\\S"))
            | if . == "(" or . == ")" or . == "&&" or . == "||" or . == "!" then .
              elif test("^\\s*@[01]\\s*$") then gsub("[\\s@]"; "")
              elif test("^\\s*true\\s*$") then "1"
              elif test("^\\s*false\\s*$") then "0"
              elif test("[?@]") then "?"
              else "u" end ]
        | if length == 0 then "1" else join(" ") end;
      (.on // {}) as $on
      | (if ($on|type)=="string" then [$on] elif ($on|type)=="array" then $on else ($on|keys) end) as $ev
      | (def mainish: .=="main" or .=="master" or test("[*?+\\[]");
         if ($on|type)=="object" and ($on.push|type)=="object"
           then (($on.push.branches // null) as $b | ($on.push["branches-ignore"] // null) as $bi | ($on.push.tags // null) as $t
                 | if $b!=null then ($b|map(select(mainish))|length>0)
                   elif $bi!=null then (($bi|map(select(.=="main" or .=="master"))|length)==0)
                   elif $t!=null then false
                   else true end)
           else ($ev|index("push")!=null) end) as $pushmain
      | (if ($on|type)=="object" and ($on.pull_request|type)=="object"
           then (($on.pull_request.branches // null) as $b | ($on.pull_request["branches-ignore"] // null) as $bi
                 | if $b!=null then ($b|map(select(.=="main" or .=="master" or test("[*?+\\[]")))|length>0)
                   elif $bi!=null then (($bi|map(select(.=="main" or .=="master"))|length)==0)
                   else true end)
           else true end) as $prbr
      | (.jobs // {}) | to_entries[]
      | .key as $j
      | ((.value|if type=="object" and has("if") then (.if|tostring) else "" end)) as $je
      | ((.value|if type=="object" and (.steps|type)=="array" and (.steps|length)>0 and all(.steps[]; type=="object" and has("if"))
            then (.steps | map("(" + (.if|tostring) + ")") | join(" || ")) else "" end)) as $se
      | (if $je == "" then $se elif $se == "" then $je else "(" + $je + ") && (" + $se + ")" end) as $e
      | [$f, $j, (if $pushmain then "1" else "0" end), (if ($ev|index("pull_request")) and $prbr then "1" else "0" end),
         (if $ev|index("merge_group") then "1" else "0" end), ex($e;"push"), ex($e;"pull_request"), ex($e;"merge_group")] | @tsv') || return 1
  done
}

# SHORTCUT_AWK: a small statement-level reader of one job's run script (see shortcuts()). It splits the script into statements
# outside quotes and $( ), keeps a stack of if / case / { } / loop frames, and gives every branch a PR state: 1 = runs on a
# pull_request, -1 = cannot, 0 = neutral. An exit 0 in a state-1 branch is a hit. A condition on pull_request that it cannot
# classify (compound, other operators) is "U": its then AND else branches are both read as state 1 (fail closed). A script that
# names pull_request and uses a construct this reader does not follow (heredoc, eval, unbalanced blocks) is a hit too.
SHORTCUT_AWK='
function trim(s) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", s); return s }
function kind(c) {
  gsub(/\$\{\{[^}]*\}\}/, "X", c); gsub(/["\047]/, "", c); c = trim(c)
  if (c !~ /pull_request/) return "N"
  if (c !~ /^(\[\[?|test)[[:space:]]/ || c ~ /&&|\|\||;|[[:space:]]-[ao][[:space:]]/) return "U"
  if (c ~ /!=/) return (c ~ /(!=[[:space:]]*pull_request([[:space:]]|\]|$))|(pull_request[[:space:]]*!=)/) ? "F" : "U"
  if (c ~ /(==?[[:space:]]*pull_request([[:space:]]|\]|$))|(pull_request[[:space:]]*==?)/) return "T"
  return "U"
}
function maybe(p) { return (p == -1) ? -1 : 1 }
function branch(p, k) {   # sets tp (state of the then branch) and rm (state of the remaining branches)
  if (k == "T") { tp = maybe(p); rm = -1 }
  else if (k == "F") { tp = -1; rm = maybe(p) }
  else if (k == "U") { tp = maybe(p); rm = maybe(p) }
  else { tp = p; rm = p }
}
function cur() { return sp > 0 ? fpr[sp] : 0 }
function push(t, p) { sp++; ft[sp] = t; fpr[sp] = p; frm[sp] = p; fbase[sp] = p }
function pop(t) { if (sp > 0 && ft[sp] == t) sp--; else bad = 1 }
function exits(s,   r, m) {
  gsub(/"[^"]*"/, "", s); gsub(/\047[^\047]*\047/, "", s)
  r = s
  while (match(r, /(^|[[:space:]&|{(])exit/)) {
    m = substr(r, RSTART + RLENGTH); r = m
    if (m ~ /^[A-Za-z0-9_]/) continue
    gsub(/[0-9]*[<>]+&?[^[:space:];&|)}]*/, "", m)
    if (m ~ /^([[:space:]]+0+)?[[:space:]]*($|[}&|)])/) return 1
  }
  return 0
}
function docond(kw, c,   p, k) {
  k = kind(c)
  if (kw == "if") { p = cur(); push("I", p); branch(p, k); fpr[sp] = tp; frm[sp] = rm }
  else if (ft[sp] == "I") { branch(frm[sp], k); fpr[sp] = tp; frm[sp] = rm }
  else bad = 1
}
function stmt(s,   h, op, rest, k, eff, p) {
  s = trim(s); if (s == "") return
  if (pend != "") { p = pend; pend = ""; docond(p, s); return }
  if (s ~ /^(then|do)([[:space:]]|$)/) {
    if (s ~ /^do/) { push("L", (lst == "") ? cur() : lst); lst = "" }
    sub(/^(then|do)[[:space:]]*/, "", s); stmt(s); return
  }
  if (s ~ /^(if|elif)([[:space:]]|$)/) {
    op = (s ~ /^if/) ? "if" : "elif"; sub(/^(if|elif)[[:space:]]*/, "", s)
    if (s == "") pend = op; else docond(op, s)
    return
  }
  if (s ~ /^else([[:space:]]|$)/) {
    if (ft[sp] == "I") fpr[sp] = frm[sp]; else bad = 1
    sub(/^else[[:space:]]*/, "", s); stmt(s); return
  }
  if (s ~ /^fi([[:space:]]|$)/) { pop("I"); return }
  if (s ~ /^done([[:space:]]|$)/) { pop("L"); return }
  if (s ~ /^esac([[:space:]]|$)/) { pop("C"); return }
  if (s ~ /^(while|until)[[:space:]]/) {
    # the body runs while the condition holds (while) or while it fails (until)
    op = (s ~ /^while/); rest = s; sub(/^(while|until)[[:space:]]+/, "", rest); k = kind(rest); p = cur()
    if (k == "T") lst = op ? maybe(p) : -1
    else if (k == "F") lst = op ? -1 : maybe(p)
    else if (k == "U") lst = maybe(p)
    else lst = p
    if (lst == 0) lst = "0"
    return
  }
  if (s ~ /^for[[:space:]]/) return
  if (s ~ /^case[[:space:]]/) {
    push("C", cur())
    if (match(s, /[[:space:]]in([[:space:]]|$)/)) stmt(substr(s, RSTART + RLENGTH))
    return
  }
  if (s == ";;") return
  if (ft[sp] == "C" && match(s, /^\(?[^()]*\)/)) {
    h = substr(s, 1, RLENGTH - 1); rest = substr(s, RLENGTH + 1)
    gsub(/["\047]/, "", h)
    fpr[sp] = (h ~ /(^|[^A-Za-z_])pull_request([^A-Za-z_]|$)/) ? maybe(fbase[sp]) : fbase[sp]
    stmt(rest); return
  }
  if (s ~ /^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(\)[[:space:]]*\{/) { sub(/^[^{]*\{[[:space:]]*/, "", s); push("G", 0); stmt(s); return }
  if (s ~ /^\}/) { pop("G"); sub(/^\}[[:space:]]*/, "", s); stmt(s); return }
  if (s ~ /^\{/) { push("G", cur()); sub(/^\{[[:space:]]*/, "", s); stmt(s); return }
  if (match(s, /(&&|\|\|)[[:space:]]*\{/)) {
    h = substr(s, 1, RSTART - 1); op = substr(s, RSTART, 2); rest = substr(s, RSTART + RLENGTH)
    push("G", chainstate(h, op)); stmt(rest); return
  }
  if (split_chain(s) > 1) { chainstate(s, ""); return }
  if (cur() == 1 && exits(s)) hit = 1
}
function un(a, b) { return (a == 1 || b == 1) ? 1 : ((a == 0 || b == 0) ? 0 : -1) }
# split_chain(s): splits s at && and || outside quotes, $( ), ( ) and [[ ]]; operands in copnd[1..n], the operator before
# operand i in cop[i]. Returns n.
function split_chain(s,   i, n, c, q, d, b, cn, two) {
  nop = 1; copnd[1] = ""; cop[1] = ""; q = ""; d = 0; b = 0; n = length(s)
  for (i = 1; i <= n; i++) {
    c = substr(s, i, 1); two = substr(s, i, 2)
    if (q != "") { copnd[nop] = copnd[nop] c; if (c == q) q = ""; continue }
    if (c == "\"" || c == "\047") { q = c; copnd[nop] = copnd[nop] c; continue }
    if (two == "[[") b++
    if (two == "]]" && b > 0) b--
    if (c == "(") d++
    if (c == ")" && d > 0) d--
    if (d == 0 && b == 0 && (two == "&&" || two == "||")) { nop++; cop[nop] = two; copnd[nop] = ""; i++; continue }
    copnd[nop] = copnd[nop] c
  }
  return nop
}
# chainstate(h, op): evaluates the && / || chain h under the current frame; ok / ko are the PR states under which the chain so far
# succeeded / failed. An operand that exits 0 in a PR state is a hit. Returns the PR state under which the operand after op runs.
function chainstate(h, op,   i, r, k, sc, fl, ok, ko, o) {
  split_chain(h); ok = cur(); ko = -1
  for (i = 1; i <= nop; i++) {
    o = trim(copnd[i])
    r = (i == 1) ? cur() : ((cop[i] == "&&") ? ok : ko)
    k = kind(o)
    if (k == "T") { sc = maybe(r); fl = -1 }
    else if (k == "F") { sc = -1; fl = maybe(r) }
    else if (k == "U") { sc = maybe(r); fl = maybe(r) }
    else { sc = r; fl = r }
    if (r == 1 && exits(o)) hit = 1
    if (i == 1) { ok = sc; ko = fl }
    else if (cop[i] == "&&") { ok = sc; ko = un(ko, fl) }
    else { ok = un(ok, sc); ko = fl }
  }
  return (op == "&&") ? ok : ko
}
# feed(line): one script line. Quote state (q) and the open statement (cur_s) persist across lines, so a quote that spans lines and
# a statement continued by a trailing && || | or backslash are read as one. A \ escapes the next character (so \" does not close a
# string); a # that starts a word ends the line, and a comment after a trailing && still continues the statement.
function feed(line,   i, c, n, bsl) {
  n = length(line); bsl = 0
  for (i = 1; i <= n; i++) {
    c = substr(line, i, 1)
    if (q != "") {
      if (q == "\"" && c == "\\" && i < n) { cur_s = cur_s c substr(line, i + 1, 1); i++; continue }
      cur_s = cur_s c; if (c == q) q = ""; continue
    }
    if (c == "\\") { if (i == n) { bsl = 1; break } cur_s = cur_s c substr(line, i + 1, 1); i++; continue }
    if (c == "\"" || c == "\047") { q = c; cur_s = cur_s c; continue }
    if (c == "#" && (i == 1 || substr(line, i - 1, 1) ~ /[[:space:]]/) && dep == 0) break
    if (c == "$" && substr(line, i + 1, 1) == "(") { dep++; cur_s = cur_s c; continue }
    if (c == ")" && dep > 0) { dep--; cur_s = cur_s c; continue }
    if (c == ";" && dep == 0) {
      if (substr(line, i + 1, 1) == ";") { stmt(cur_s); cur_s = ""; stmt(";;"); i++ }
      else { stmt(cur_s); cur_s = "" }
      continue
    }
    cur_s = cur_s c
  }
  if (q != "") { cur_s = cur_s "\n"; return }
  if (bsl) { cur_s = cur_s " "; return }
  if (cur_s ~ /(&&|\|\||\|)[[:space:]]*$/) { cur_s = cur_s " "; return }
  stmt(cur_s); cur_s = ""
}
{
  raw = $0
  if (q == "" && raw ~ /^[[:space:]]*#/) next
  t = raw; gsub(/\$\{\{[^}]*\}\}/, "X", t)
  if (t ~ /pull_request/) mention = 1
  if (raw ~ /\$\{\{[^}]*["\047]pull_request["\047][^}]*\}\}/) { mention = 1; opaque = 1 }
  if (t ~ /<<|(^|[[:space:]&|;{(])eval[[:space:]]|^[[:space:]]*function[[:space:]]/) opaque = 1
  feed(raw)
}
END { if (cur_s != "") stmt(cur_s); if (mention && (opaque || bad || sp != 0 || pend != "")) hit = 1; exit !hit }
'

# VIOLATION 2 rows: "<file>\t<job>" for each job whose run script exits 0 on a pull_request branch.
shortcuts() {
  local f j b js
  for f in "$WF"/*.yml "$WF"/*.yaml; do
    [ -e "$f" ] || continue
    js="$(yq -o=json '.' "$f" | jq -r '(.jobs // {}) | to_entries[] | [.key, ([.value.steps[]?.run // empty] | join("\n") | @base64)] | @tsv')" || return 1
    while IFS=$'\t' read -r j b; do
      [ -n "$j" ] || continue
      if printf '%s' "$b" | base64 -d | awk "$SHORTCUT_AWK"; then
        printf '%s\t%s\n' "$(basename "$f")" "$j"
      fi
    done <<<"$js"
  done
}

# VIOLATION 3 rows: "<file>\t<job>" for each heavy push+PR job lacking the draft clause a heavy sibling carries.
drafts() {
  local f pr_jobs
  for f in "$WF"/*.yml "$WF"/*.yaml; do
    [ -e "$f" ] || continue
    pr_jobs="$(printf '%s\n' "$ROWS" | awk -F'\t' -v f="$(basename "$f")" '$1==f && $4=="1" {print $2}')"
    [ -n "$pr_jobs" ] || continue
    yq -o=json '.' "$f" | jq -r --arg pj "$pr_jobs" '
      ($pj | split("\n")) as $pr
      | ((.jobs // {}) | to_entries
          | map(select(.key as $k | $pr | index($k))
                | select(((.value["timeout-minutes"] // 0) | tostring | [scan("[0-9]+") | tonumber] | max // 0) >= 100)
                | {k: .key, d: (((.value.if // "") | tostring) | contains("github.event.pull_request.draft"))})) as $h
      | if ($h | map(select(.d)) | length) > 0 then ($h | .[]) | select(.d | not) | .k else empty end' \
      | while read -r j; do printf '%s\t%s\n' "$(basename "$f")" "$j"; done || return 1
  done
}

# Fail closed (rc 2, never OK): a missing or empty workflow dir, a workflow that does not parse, a jobs map that yields no row.
preflight() {
  local f n=0 jn=0 c
  [ -d "$WF" ] || { echo "ENV: workflow dir $WF is missing"; return 2; }
  for f in "$WF"/*.yml "$WF"/*.yaml; do
    [ -e "$f" ] || continue
    n=$((n + 1))
    c="$(yq -o=json '.' "$f" 2>/dev/null | jq -e -r 'if type == "object" then ((.jobs // {}) | length) else empty end' 2>/dev/null)" || { echo "ENV: $f does not parse as a YAML mapping"; return 2; }
    jn=$((jn + c))
  done
  [ "$n" -gt 0 ] || { echo "ENV: no workflow file under $WF"; return 2; }
  [ "$jn" -gt 0 ] || { echo "ENV: the workflows under $WF define no job"; return 2; }
}

scan() {
  local fail=0 f j p pr mg key reason SHORTS DRAFTS
  local exempt="" used=""
  preflight || return 2
  ROWS="$(rows)" || { echo "ENV: could not read the jobs and if: conditions of $WF"; return 2; }
  SHORTS="$(shortcuts)" || { echo "ENV: could not read the run scripts of $WF"; return 2; }
  DRAFTS="$(drafts)" || { echo "ENV: could not read the heavy-job draft clauses of $WF"; return 2; }
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
    key="$f:$j"
    if [ "$pr" = 0 ]; then
      if grep -qxF -- "$key" <<<"$exempt"; then used="$used
$key"; continue; fi
      if [ "$mg" = 1 ]; then echo "RED: $key runs on push to main but is skipped on pull_request by an if: event guard (merge_group does not cover it)"
      else echo "RED: $key runs on push to main and not on any pull_request run"; fi
      fail=1
    fi
  done <<<"$ROWS"
  while IFS=$'\t' read -r f j; do
    [ -n "$f" ] || continue
    echo "RED: $f:$j passes a pull_request (exit 0) before it checks the heavy jobs"
    fail=1
  done <<<"$SHORTS"
  while IFS=$'\t' read -r f j; do
    [ -n "$f" ] || continue
    echo "RED: $f:$j is a heavy job on push and pull_request without the draft clause its heavy siblings carry (a draft PR runs it)"
    fail=1
  done <<<"$DRAFTS"
  for key in $(printf "%s\n" "$exempt" | tr " " "_" | grep -v "^$"); do
    grep -qxF -- "$key" <<<"$used" || { echo "STALE: exemption $key is not a push-only job"; fail=1; }
  done
  [ "$fail" = 0 ] && echo "OK: every job that runs on push to main also runs on a pull_request, and no job passes a PR before its heavy checks"
  return "$fail"
}

self_test() {
  local d fails=0 w
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
  t "|| guard that skips pull_request is RED" 1 "" "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || github.event_name == 'merge_group'\", steps: []}
"
  t "|| guard that keeps pull_request is green" 0 "" "$H
  a: {runs-on: x, if: \"github.event_name == 'push' || github.event_name == 'pull_request'\", steps: []}
"
  t "reversed operand 'pull_request' != github.event_name is RED" 1 "" "$H
  a: {runs-on: x, if: \"'pull_request' != github.event_name\", steps: []}
"
  t "negated group !(event_name == pull_request) skips a PR: RED" 1 "" "$H
  a: {runs-on: x, if: \"!(github.event_name == 'pull_request')\", steps: []}
"
  t "unparseable workflow fails closed (rc 2)" 2 "" "on: [push, pull_request]
jobs:
  a: {runs-on: [unclosed
"
  t "empty jobs map fails closed (rc 2)" 2 "" "$H {}
"
  t "[ != pull_request ] || { exit 0; } is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\${EVT:-}\" != pull_request ] || { echo ok; exit 0; }
          check_heavy
"
  t "case pull_request) exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            pull_request) exit 0 ;;
          esac
          check_heavy
"
  t "bare exit on a pull_request if is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then exit; fi
          check_heavy
"
  t "a commented-out pull_request shortcut is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          # [ \"\$EVT\" = pull_request ] && exit 0
          check_heavy
"
  t "an exit 0 after the pull_request if block has closed is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            [ \"\${PRL:-}\" = success ] || exit 1
          fi
          [ \"\${DRAFT:-}\" != true ] || exit 0
          check_heavy
"
  t "a comment naming exit 0 inside the pull_request if block is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then
            # never exit 0 here
            [ \"\${PRL:-}\" = success ] || exit 1
          fi
          check_heavy
"
  t "an exit 0 in a later case branch, after pull_request) closed, is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            pull_request) [ \"\${PRL:-}\" = success ] || exit 1 ;;
            *) exit 0 ;;
          esac
          check_heavy
"
  t "a multi-line pull_request) branch closed by ;; before a later exit 0 branch is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            pull_request)
              [ \"\${PRL:-}\" = success ] || exit 1
              ;;
            *)
              exit 0
              ;;
          esac
          check_heavy
"
  t "&& with a parenthesised || that skips pull_request is RED (lane 1 finding 1)" 1 "" "$H
  a: {runs-on: x, if: \"github.event_name != 'pull_request' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master')\", steps: []}
"
  t "(!cancelled() || always()) && event_name == 'push' is RED" 1 "" "$H
  a: {runs-on: x, if: \"(!cancelled() || always()) && github.event_name == 'push'\", steps: []}
"
  t "&& with a parenthesised || that keeps pull_request is green" 0 "" "$H
  a: {runs-on: x, if: \"github.event_name != 'merge_group' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master')\", steps: []}
"
  t "single-line if pull_request; then echo; fi does not leak into a later exit 0 (lane 1 finding 2a)" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then echo ok; fi
          [ \"\${DRAFT:-}\" != true ] || exit 0
          check_heavy
"
  t "[[ != pull_request ]] || exit 0 is RED (lane 1 finding 2b)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [[ \"\$EVT\" != pull_request ]] || exit 0
          check_heavy
"
  t "[ = pull_request ] && { newline exit 0 } is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" = pull_request ] && {
            exit 0
          }
          check_heavy
"
  t "case push|pull_request) exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in push|pull_request) exit 0 ;; esac
          check_heavy
"
  t "a negated unknown atom (!github.event.pull_request.draft) is unknown, not false: green" 0 "" "$H
  a: {runs-on: x, if: \"github.event_name != 'pull_request' || !github.event.pull_request.draft\", steps: []}
"
  t "!(<unknown>) && event_name == 'push' is RED (lane 1 finding 1)" 1 "" "$H
  a: {runs-on: x, if: \"!(github.event.pull_request.draft == true) && github.event_name == 'push'\", steps: []}
"
  t "!(cancelled()) && event_name == 'push' is RED" 1 "" "$H
  a: {runs-on: x, if: \"!(cancelled()) && github.event_name == 'push'\", steps: []}
"
  t "!contains(fromJSON(..), github.event_name) && event_name == 'push' (nested call) fails closed (rc 2)" 2 "" "$H
  a: {runs-on: x, if: \"!contains(fromJSON('[\\\"merge_group\\\"]'), github.event_name) && github.event_name == 'push'\", steps: []}
"
  t "!(<unknown>) && event_name != 'pull_request' is RED (lane 2 finding 1)" 1 "" "$H
  a: {runs-on: x, if: \"!(github.ref == 'refs/heads/skip') && github.event_name != 'pull_request'\", steps: []}
"
  t "!(event_name == 'pull_request' && draft) keeps pull_request: green (no false RED)" 0 "" "$H
  a: {runs-on: x, if: \"!(github.event_name == 'pull_request' && github.event.pull_request.draft)\", steps: []}
"
  t "nested if/fi inside a pull_request branch then exit 0 is RED (lane 2 finding 2a)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then
            if [ \"\$X\" = 1 ]; then
              echo ok
            fi
            exit 0
          fi
          check_heavy
"
  t "quoted multi-line case label \"pull_request\") exit 0 is RED (lane 2 finding 2b)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            \"pull_request\")
              exit 0
              ;;
          esac
          check_heavy
"
  t "echo ok; fi does not leak into a later exit 0 (lane 2 finding 2c)" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\${EVT:-}\" = pull_request ]; then
            echo ok; fi
          [ \"\${DRAFT:-}\" != true ] || exit 0
          check_heavy
"
  t "test = pull_request && exit 0 is RED (lane 1 finding 2a)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          test \"\$EVT\" = pull_request && exit 0
          check_heavy
"
  t "contains(fromJSON(['push']), github.event_name) is unreadable: fails closed (rc 2, lane 1 finding 1)" 2 "" "$H
  a: {runs-on: x, if: \"contains(fromJSON('[\\\"push\\\"]'), github.event_name)\", steps: []}
"
  t "startsWith(github.event_name, 'push') is unreadable: fails closed (rc 2)" 2 "" "$H
  a: {runs-on: x, if: \"startsWith(github.event_name, 'push')\", steps: []}
"
  t "a string holding ( ) in a push-only if: is masked, job is RED (lane 1 finding 1)" 1 "" "$H
  a: {runs-on: x, if: \"github.ref == '(foo)' && github.event_name == 'push'\", steps: []}
"
  t "a string holding ! and | in a push-only if: is masked, job is RED" 1 "" "$H
  a: {runs-on: x, if: \"github.head_ref == '!a|b' && github.event_name == 'push'\", steps: []}
"
  t "reversed 'push' == github.event_name is RED (kills the reversed-== deletion mutant)" 1 "" "$H
  a: {runs-on: x, if: \"'push' == github.event_name\", steps: []}
"
  t "unparseable if: on a push+pull_request workflow fails closed (rc 2, kills the bp error-swallow mutant)" 2 "" "$H
  a: {runs-on: x, if: \"github.event_name == 'push' && (\", steps: []}
"
  t "unparseable if: on a pull_request-only workflow fails closed (rc 2, kills the bpr error-swallow mutant)" 2 "" "on: {pull_request: {}}
jobs:
  a: {runs-on: x, if: \"github.event_name == 'pull_request' && (\", steps: []}
"
  t "unparseable if: on a merge_group-only workflow fails closed (rc 2, kills the bmg error-swallow mutant)" 2 "" "on: {merge_group: {}}
jobs:
  a: {runs-on: x, if: \"github.event_name == 'merge_group' && (\", steps: []}
"
  t "single-quoted [ != 'pull_request' ] || exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" != 'pull_request' ] || exit 0
          check_heavy
"
  t "reversed [ pull_request != \"\$EVT\" ] || exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ pull_request != \"\$EVT\" ] || exit 0
          check_heavy
"
  t "line-continued [ = pull_request ] &&<newline> exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" = pull_request ] &&
            exit 0
          check_heavy
"
  t "multi-line if / cond / then / exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if
            [ \"\$EVT\" = pull_request ]
          then
            exit 0
          fi
          check_heavy
"
  t "case pull_request) echo ok; exit 0 ;; esac is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in pull_request) echo ok; exit 0 ;; esac
          check_heavy
"
  t "case 'push | pull_request)' with spaces and exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            push | pull_request) exit 0 ;;
          esac
          check_heavy
"
  t "case pull_request|workflow_dispatch) exit 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            pull_request|workflow_dispatch)
              exit 0
              ;;
          esac
          check_heavy
"
  t "else branch of a != pull_request test that exits 0 is RED" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" != pull_request ]; then
            check_other
          else
            exit 0
          fi
          check_heavy
"
  t "an if whose pull_request test is compound (unclassifiable) with exit 0 is RED (fail closed)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ] && [ -n \"\$X\" ]; then
            exit 0
          fi
          check_heavy
"
  t "a script naming pull_request and using a heredoc is RED (fail closed, not followed)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          cat <<EOF
          x
          EOF
          [ \"\$EVT\" = pull_request ] || true
          check_heavy
"
  t "a pull_request block left open at the end of the script is RED (fail closed)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then
            check_light
"
  t "if pull_request; then check || exit 1; else exit 0; fi is green (exit 0 is the non-PR branch)" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then
            check_pr || exit 1
          else
            exit 0
          fi
          check_heavy
"
  t "elif [ = push ]; then exit 0 after a pull_request branch is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then
            check_pr
          elif [ \"\$EVT\" = push ]; then
            exit 0
          fi
          check_heavy
"
  t "if [ != pull_request ]; then exit 0; fi (non-PR exit) is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" != pull_request ]; then exit 0; fi
          check_heavy
"
  t "[ = pull_request ] || exit 0 (non-PR exit) is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" = pull_request ] || exit 0
          check_heavy
"
  t "[ = pull_request ] && { check || exit 1; } then a draft exit 0 is green (no brace leak)" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" = pull_request ] && {
            check_pr || exit 1; }
          [ \"\${DRAFT:-}\" != true ] || exit 0
          check_heavy
"
  t "a final pull_request) case branch without ;; then a draft exit 0 is green (no case leak)" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          case \"\$EVT\" in
            push) check_push ;;
            pull_request) check_pr
          esac
          [ \"\${DRAFT:-}\" != true ] || exit 0
          check_heavy
"
  t "a shell function body is not the pull_request branch: green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          res() { printf '%s' \"\$1\"; }
          [ \"\$EVT\" = pull_request ] || exit 0
          check_heavy
"
  t "[ = pull_request ] || { exit 0; } (non-PR exit in a brace group) is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" = pull_request ] || { echo x; exit 0; }
          check_heavy
"
  t "[ != pull_request ] && { exit 0; } (non-PR exit in a brace group) is green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" != pull_request ] && { echo x; exit 0; }
          check_heavy
"
  t "an else of a pull_request test nested inside a pull_request block is the non-PR branch: green" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then
            if [ \"\$EVT\" = pull_request ]; then
              check_pr
            else
              exit 0
            fi
          fi
          check_heavy
"
  t "unparseable if: on a push-only workflow fails closed (rc 2, kills the bp error-swallow mutant)" 2 "" "on: {push: {branches: [main]}}
jobs:
  a: {runs-on: x, if: \"github.event_name == 'push' && (\", steps: []}
"
  t "github['event_name'] == 'push' (index syntax) is read like the dot form: RED (rc 1; lane 1 finding 1a)" 1 "" "$H
  a: {runs-on: x, if: \"github['event_name'] == 'push'\", steps: []}
"
  t "GitHub.event_name == 'push' (case-insensitive context name) is read like github.event_name: RED (rc 1; lane 1 finding 1a)" 1 "" "$H
  a: {runs-on: x, if: \"GitHub.event_name == 'push'\", steps: []}
"
  t "github [ 'EVENT_NAME' ] == 'push' (spaces, upper-case key) is read like github.event_name: RED (rc 1)" 1 "" "$H
  a: {runs-on: x, if: \"github [ 'EVENT_NAME' ] == 'push'\", steps: []}
"
  t "github . event_name == 'push' (spaces round the dot) is read like github.event_name: RED (rc 1)" 1 "" "$H
  a: {runs-on: x, if: \"github . event_name == 'push'\", steps: []}
"
  t "contains(toJSON(github), 'push') reads the whole context: fails closed (rc 2)" 2 "" "$H
  a: {runs-on: x, if: \"contains(toJSON(github), 'push') && github.ref == 'x'\", steps: []}
"
  t "GITHUB['Event_Name'] == 'push' (every part mixed case) is read like github.event_name: RED (rc 1)" 1 "" "$H
  a: {runs-on: x, if: \"GITHUB['Event_Name'] == 'push'\", steps: []}
"
  t "github[\\\"event_name\\\"] == 'push' (double-quoted index, not valid GHA) fails closed (rc 2)" 2 "" "$H
  a: {runs-on: x, if: \"github[\\\"event_name\\\"] == 'push'\", steps: []}
"
  t "github[env.KEY] == 'push' (computed index) fails closed (rc 2)" 2 "" "$H
  a: {runs-on: x, if: \"github[env.KEY] == 'push'\", steps: []}
"
  t "github.ref != 'x' && github.event_name == 'push' keeps != as one atom: RED (rc 1, kills the != split mutant)" 1 "" "$H
  a: {runs-on: x, if: \"github.ref != 'refs/heads/dev' && github.event_name == 'push'\", steps: []}
"
  t "\${{ github.event_name == 'push' }} wrapper is stripped: RED (rc 1, kills the wrapper-strip mutant)" 1 "" "$H
  a: {runs-on: x, if: \"\${{ github.event_name == 'push' }}\", steps: []}
"
  t "13 unknown atoms fail closed (rc 2, kills the 12-unknown cap mutant)" 2 "" "$H
  a: {runs-on: x, if: \"github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a' || github.ref == 'a'\", steps: []}
"
  t "four nested calls are all reduced to an unknown: RED (rc 1, kills a deleted nested-call pass)" 1 "" "$H
  a: {runs-on: x, if: \"f(f(f(f(github.ref)))) && github.event_name == 'push'\", steps: []}
"
  t "push branches [master] is main: push-only job is RED (rc 1, kills the master mutant)" 1 "" "on: {push: {branches: [master]}, pull_request: {}, merge_group: {}}
jobs:
  a: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
"
  t "push branches ['**'] is a glob that can reach main: push-only job is RED (rc 1, kills the glob mutant)" 1 "" "on: {push: {branches: ['**']}, pull_request: {}, merge_group: {}}
jobs:
  a: {runs-on: x, if: \"github.event_name == 'push'\", steps: []}
"
  t "a test joined by -o with heavy and exit 0 in its two branches is read as unclassified (both branches PR): RED (rc 1, kills the -o/-a mutant)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request -o \"\$X\" = y ]; then
            check_heavy
          else
            exit 0
          fi
"
  t "echo \"a exit 0 }\" inside quotes is not an exit: green (rc 0, kills the quote-strip-in-exits mutant)" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          if [ \"\$EVT\" = pull_request ]; then
            echo \"a exit 0 }\"
          fi
          check_heavy
"
  t "a \$( if .. fi ) command substitution is one statement: green (rc 0, kills the \$( depth mutant)" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          v=\$(if [ \"\$EVT\" = pull_request ]; then echo 1; else echo 2; fi)
          exit 0
"
  t "a # inside \$( ) is not a comment: green (rc 0, kills the comment-depth mutant)" 0 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          v=\$(echo a # b)
          if [ \"\$EVT\" != pull_request ]; then exit 0; fi
          check_heavy
"
  local ev_kw="ev""al"
  t "eval in a script that names pull_request is opaque: RED (rc 1, kills the eval mutant)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          ${ev_kw} \"\$CMD\"
          if [ \"\$EVT\" != pull_request ]; then echo x; fi
"
  local fn_kw="func""tion"
  t "function keyword in a script that names pull_request is opaque: RED (rc 1)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          ${fn_kw} g { :; }
          if [ \"\$EVT\" != pull_request ]; then echo x; fi
"
  t "[ != pull_request ] || newline exit 0 joins the continuation: RED (rc 1, kills the || continuation mutant)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          [ \"\$EVT\" != pull_request ] ||
            exit 0
          check_heavy
"
  t "a stray fi in a script that names pull_request is unbalanced: RED (rc 1, kills the bad-flag END mutant)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          echo pull_request
          fi
"
  t "a dangling if in a script that names pull_request is unbalanced: RED (rc 1, kills the pend END mutant)" 1 "" "$H
  a:
    runs-on: x
    steps:
      - run: |
          echo pull_request
          if
"
  # Quorum round 3 (lane 1 findings 1b-1e, lane 1 findings 2a-2e): helpers take literal text, no shell escaping.
  tj() { local b; b="$(cat)"; t "$1" "$2" "" "$H"$'\n'"$b"$'\n'; }    # job yaml on stdin, default on: header
  tw() { local b; b="$(cat)"; t "$1" "$2" "" "$b"$'\n'; }            # whole workflow on stdin
  ts() { # name, expected rc; run script on stdin
    local b; b="$(sed 's/^/          /')"
    t "$1" "$2" "" "$H"$'\n  a:\n    runs-on: x\n    steps:\n      - run: |\n'"$b"$'\n'
  }
  tj "1b chained comparison on event_name fails closed (rc 2)" 2 <<'EOF'
  a: {runs-on: x, if: "github.event_name != 'pull_request' == true", steps: []}
EOF
  tj "1b event_name inside a comparison with an unknown fails closed (rc 2)" 2 <<'EOF'
  a: {runs-on: x, if: "github.event_name == 'push' == needs.x.result", steps: []}
EOF
  tj "1b a number containing 1 or 0 next to an unknown stays unknown, not an event result: green" 0 <<'EOF'
  a: {runs-on: x, if: "github.run_number == 10", steps: []}
EOF
  tj "1e false literal on the PR disjunct is false: push-only job is RED" 1 <<'EOF'
  a: {runs-on: x, if: "github.event_name == 'push' || (github.event_name == 'pull_request' && false)", steps: []}
EOF
  tj "1e true literal does not exclude the PR: green" 0 <<'EOF'
  a: {runs-on: x, if: "github.event_name == 'push' || (github.event_name == 'pull_request' && true)", steps: []}
EOF
  tj "1e if: true is green" 0 <<'EOF'
  a: {runs-on: x, if: true, steps: []}
EOF
  tj "1e if: false (YAML boolean) never runs on push, so nothing to cover: green" 0 <<'EOF'
  a: {runs-on: x, if: false, steps: []}
  b: {runs-on: x, steps: []}
EOF
  tj "1e && false on a push job means it never runs on push, so nothing to cover: green" 0 <<'EOF'
  a: {runs-on: x, if: "github.event_name == 'push' && false", steps: []}
  b: {runs-on: x, steps: []}
EOF
  tw "1c push branches glob without a star (mai?) reaches main: push-only job is RED" 1 <<'EOF'
on: {push: {branches: ["mai?"]}, pull_request: {}}
jobs:
  a: {runs-on: x, if: "github.event_name == 'push'", steps: []}
EOF
  tw "1c push branches [dev] never reaches main: green" 0 <<'EOF'
on: {push: {branches: [dev]}, pull_request: {}}
jobs:
  a: {runs-on: x, if: "github.event_name == 'push'", steps: []}
  b: {runs-on: x, steps: []}
EOF
  tw "1c push tags plus branches-ignore (not main) still runs on a main push: RED" 1 <<'EOF'
on: {push: {tags: ["v*"], branches-ignore: [docs]}, pull_request: {}}
jobs:
  a: {runs-on: x, if: "github.event_name == 'push'", steps: []}
EOF
  tw "1c push branches-ignore [main] never runs on main: green" 0 <<'EOF'
on: {push: {branches-ignore: [main]}, pull_request: {}}
jobs:
  a: {runs-on: x, if: "github.event_name == 'push'", steps: []}
  b: {runs-on: x, steps: []}
EOF
  tw "1c pull_request filtered to another base branch does not cover a push to main: RED" 1 <<'EOF'
on: {push: {branches: [main]}, pull_request: {branches: [release]}}
jobs:
  a: {runs-on: x, steps: []}
EOF
  tw "1c pull_request branches-ignore [main] does not cover a push to main: RED" 1 <<'EOF'
on: {push: {branches: [main]}, pull_request: {branches-ignore: [main]}}
jobs:
  a: {runs-on: x, steps: []}
EOF
  tw "1c pull_request branches [main] covers it: green" 0 <<'EOF'
on: {push: {branches: [main]}, pull_request: {branches: [main]}}
jobs:
  a: {runs-on: x, steps: []}
EOF
  tj "1d every step skipped on a PR by a step if: the job is a no-op there: RED" 1 <<'EOF'
  a:
    runs-on: x
    steps:
      - {run: a, if: "github.event_name != 'pull_request'"}
      - {run: b, if: "github.event_name != 'pull_request'"}
EOF
  tj "1d one step without an if: runs on a PR: green" 0 <<'EOF'
  a:
    runs-on: x
    steps:
      - {run: a, if: "github.event_name != 'pull_request'"}
      - {run: b}
EOF
  tj "1d a step if: that can hold on a PR (always()) keeps the job on a PR: green" 0 <<'EOF'
  a:
    runs-on: x
    steps:
      - {run: a, if: "github.event_name != 'pull_request'"}
      - {run: b, if: "always()"}
EOF
  tj "1d job if on a PR and a step set that does not run on a PR: RED" 1 <<'EOF'
  a:
    runs-on: x
    if: "github.event_name != 'merge_group'"
    steps:
      - {run: a, if: "github.event_name == 'push'"}
EOF
  ts "2a [ other ] && [ pr ] && exit 0 is RED" 1 <<'EOF'
[ "${PRL:-}" = success ] && [ "${EVT:-}" = pull_request ] && exit 0
check_heavy
EOF
  ts "2a [ != pr ] && cmd || exit 0 passes a PR through the || arm: RED" 1 <<'EOF'
[ "$EVT" != pull_request ] && check_heavy || exit 0
check_heavy
EOF
  ts "2a [ other ] && [ != pr ] && exit 0 only exits off a PR: green" 0 <<'EOF'
[ "${PRL:-}" = success ] && [ "${EVT:-}" != pull_request ] && exit 0
check_heavy
EOF
  ts "2a cmd && [ pr ] || exit 0 exits when the test fails, which is off a PR: green" 0 <<'EOF'
check_heavy && [ "$EVT" = pull_request ] || exit 0
check_heavy
EOF
  ts "2a a && b && c chain with a [[ x && y ]] operand is not split inside the brackets: green" 0 <<'EOF'
[[ "$A" = 1 && "$B" = 2 ]] && echo ok
check_heavy
EOF
  ts "2b while [ pr ]; do exit 0; done is RED" 1 <<'EOF'
while [ "$EVT" = pull_request ]; do exit 0; done
check_heavy
EOF
  ts "2b while [ != pr ]; do exit 0 is off a PR: green" 0 <<'EOF'
while [ "$EVT" != pull_request ]; do exit 0; done
check_heavy
EOF
  ts "2b until [ != pr ]; do exit 0 runs on a PR: RED" 1 <<'EOF'
until [ "$EVT" != pull_request ]; do exit 0; done
check_heavy
EOF
  ts "2b until [ pr ]; do exit 0 is off a PR: green" 0 <<'EOF'
until [ "$EVT" = pull_request ]; do exit 0; done
check_heavy
EOF
  ts "2b while true; do ... done inside a PR branch keeps the PR state: RED" 1 <<'EOF'
if [ "$EVT" = pull_request ]; then
  while true; do exit 0; done
fi
EOF
  ts "2c \${{ event_name == 'pull_request' }} spliced into a test is RED (opaque)" 1 <<'EOF'
if [ "${{ github.event_name == 'pull_request' }}" = true ]; then exit 0; fi
check_heavy
EOF
  ts "2c \${{ github.event.pull_request.number }} alone is not a PR test: green" 0 <<'EOF'
bash free.sh "${{ github.event.pull_request.number || github.ref_name }}"
check_heavy
EOF
  ts "2d exit 00 on a PR is RED" 1 <<'EOF'
if [ "$EVT" = pull_request ]; then exit 00; fi
check_heavy
EOF
  ts "2d exit 0 >/dev/null on a PR is RED" 1 <<'EOF'
if [ "$EVT" = pull_request ]; then exit 0 >/dev/null; fi
check_heavy
EOF
  ts "2d exit 0 2>&1 on a PR is RED" 1 <<'EOF'
[ "$EVT" = pull_request ] && exit 0 2>&1
check_heavy
EOF
  ts "2d exit 10 and exit 1 on a PR are failures: green" 0 <<'EOF'
if [ "$EVT" = pull_request ]; then exit 10; fi
if [ "$EVT" = pull_request ]; then exit 1; fi
check_heavy
EOF
  ts "2d a word starting with exit (exit_code=0) is not an exit: green" 0 <<'EOF'
if [ "$EVT" = pull_request ]; then exit_code=0; fi
check_heavy
EOF
  ts "2e inline comment after a trailing && still continues to exit 0: RED" 1 <<'EOF'
[ "$EVT" = pull_request ] && # skip on PR
  exit 0
check_heavy
EOF
  ts "2e escaped quote inside a string does not hide the next test: RED" 1 <<'EOF'
echo "x\""; [ "$EVT" = pull_request ] && exit 0
check_heavy
EOF
  ts "2e a quote that spans lines keeps the following test readable: RED" 1 <<'EOF'
echo "a
b"; [ "$EVT" = pull_request ] && exit 0
check_heavy
EOF
  ts "2e pull_request && exit 0 text inside a multi-line quote is data, not code: green" 0 <<'EOF'
echo "on pull_request
[ x = pull_request ] && exit 0
done"
check_heavy
EOF
  ei() { # name, expected rc, eval_if input: calls eval_if directly (--eval-if)
    local o; o="$(bash "$0" --eval-if "$3" 2>/dev/null)"; local rc=$?
    [ "$rc" = "$2" ] && echo "ok   $1" || { echo "FAIL $1 (rc $rc, want $2, out $o)"; fails=1; }
  }
  ei "eval_if 1 && !0 is 1 (rc 0)" 0 "1 && !0"
  ei "eval_if rejects 11 (value other than 0 or 1, rc 1)" 1 "11"
  ei "eval_if rejects 1 1 (arithmetic error, rc 1)" 1 "1 1"
  ei "eval_if rejects a ternary (char whitelist, rc 1)" 1 "1 ? 1 : 1"
  # Fail closed on a missing tool: a clean workflow and an empty exemption file would be green,
  # so rc 2 here can only come from the ENV check (without it, no rows and no exemptions print OK).
  local sh yqp jqp rc out
  sh="$(command -v bash)"; yqp="$(command -v yq)"; jqp="$(command -v jq)"
  mkdir -p "$d/noyq" "$d/nojq"; ln -sf "$jqp" "$d/noyq/jq"; ln -sf "$yqp" "$d/nojq/yq"
  te() { # name, expected rc, PATH dir, missing tool named in the ENV line
    printf '%s' "$H
  a: {runs-on: x, steps: []}
" > "$w/ci.yml"; : > "$d/ex"
    local out; out="$(PATH="$3" PARITY_WORKFLOWS="$w" PARITY_EXEMPT="$d/ex" "$sh" "$0" 2>&1)"; local rc=$?
    case "$out" in *"ENV: $4 missing"*) ;; *) rc="no-ENV-msg-$rc" ;; esac
    [ "$rc" = "$2" ] && echo "ok   $1" || { echo "FAIL $1 (rc $rc, want $2)"; fails=1; }
  }
  te "yq missing fails closed (rc 2), clean workflow and empty exemptions" 2 "$d/noyq" yq
  te "jq missing fails closed (rc 2), clean workflow and empty exemptions" 2 "$d/nojq" jq
  : > "$d/ex"
  out="$(PARITY_WORKFLOWS="$d/missing" PARITY_EXEMPT="$d/ex" bash "$0" 2>&1)"; rc=$?
  case "$out" in *"ENV: workflow dir"*) ;; *) rc="no-ENV-msg-$rc" ;; esac
  [ "$rc" = 2 ] && echo "ok   missing workflow dir fails closed (rc 2)" || { echo "FAIL missing workflow dir (rc $rc, want 2)"; fails=1; }
  mkdir -p "$d/emptydir"
  out="$(PARITY_WORKFLOWS="$d/emptydir" PARITY_EXEMPT="$d/ex" bash "$0" 2>&1)"; rc=$?
  case "$out" in *"ENV: no workflow file"*) ;; *) rc="no-ENV-msg-$rc" ;; esac
  [ "$rc" = 2 ] && echo "ok   empty workflow dir fails closed (rc 2)" || { echo "FAIL empty workflow dir (rc $rc, want 2)"; fails=1; }
  bash "$0" --help >"$d/help" 2>&1; rc=$?
  if [ "$rc" = 0 ] \
    && grep -qF -- '--self-test' "$d/help" \
    && ! grep -qE '^(#|!|set |command |[A-Za-z_][A-Za-z0-9_]*=|ENV:|OK:|RED:|STALE:|[A-Za-z_][A-Za-z0-9_]*\(\)|[{}]|eval_if |SHORTCUT_AWK:|rows:)' "$d/help"; then
    echo "ok   --help prints usage including --self-test and no non-usage lines"
  else
    echo "FAIL --help prints usage including --self-test and no non-usage lines (rc $rc)"
    fails=1
  fi
  rm -rf "${d:?}"
  return "$fails"
}

case "${1:-}" in
  --self-test) self_test; exit $? ;;
  --live) scan; exit $? ;;
  --eval-if) eval_if "${2:-}"; exit $? ;;
  -h|--help) awk 'NR == 1 && /^#!/ { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0" ;;
  *) scan; exit $? ;;
esac
