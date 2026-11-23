#!/usr/bin/env bash
# ONE-GATE parity falsifier (APR-GREEN-001): diffs push-to-main jobs vs PR-required set.
# Usage: check_one_gate_parity.sh [--workflows DIR] [--allow FILE] [--live] [--repo OWNER/REPO] [--rulesets-json FILE] | --self-test | --help
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
wf_dir="$root/.github/workflows"; allow="$root/.github/one-gate-allow.txt"; mode=check
live=0; repo="${GITHUB_REPOSITORY:-paiml/aprender}"; rulesets_json=""
SCRATCH=(); trap 'rm -rf -- "${SCRATCH[@]}"' EXIT
usage() { echo "usage: $0 [--workflows DIR] [--allow FILE] [--live] [--repo OWNER/REPO] [--rulesets-json FILE] | --self-test"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --workflows) [ $# -ge 2 ] || { usage >&2; exit 2; }; wf_dir=$2; shift 2 ;;
    --allow) [ $# -ge 2 ] || { usage >&2; exit 2; }; allow=$2; shift 2 ;;
    --repo) [ $# -ge 2 ] || { usage >&2; exit 2; }; repo=$2; shift 2 ;;
    --rulesets-json) [ $# -ge 2 ] || { usage >&2; exit 2; }; rulesets_json=$2; live=1; shift 2 ;;
    --live) live=1; shift ;;
    --self-test) mode=self-test; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

live_has_merge_queue() {
  local json_file=$1 raw ids id item items=()
  command -v jq >/dev/null 2>&1 || { echo "one-gate: --live requires jq" >&2; return 2; }
  if [ -n "$json_file" ]; then
    [ -r "$json_file" ] || { echo "one-gate: unreadable rulesets json $json_file" >&2; return 2; }
    raw=$(cat "$json_file") || return 2
  else
    command -v gh >/dev/null 2>&1 || { echo "one-gate: --live requires gh" >&2; return 2; }
    raw=$(gh api "repos/${repo}/rulesets" 2>/dev/null) || { echo "one-gate: gh api repos/${repo}/rulesets failed" >&2; return 2; }
    ids=$(jq -r '.[] | select(.enforcement == "active" and .target == "branch") | .id' <<< "$raw" 2>/dev/null) || { echo "one-gate: invalid rulesets list json" >&2; return 2; }
    for id in $ids; do
      item=$(gh api "repos/${repo}/rulesets/${id}" 2>/dev/null) || { echo "one-gate: gh api repos/${repo}/rulesets/${id} failed" >&2; return 2; }
      items+=("$item")
    done
    if [ "${#items[@]}" -eq 0 ]; then raw="[]"; else raw=$(printf '%s\n' "${items[@]}" | jq -s '.' 2>/dev/null) || return 2; fi
  fi
  local rc=0
  jq -e '
    if type == "array" then . else [.] end | any(
      .enforcement == "active"
      and .target == "branch"
      and ((.conditions.ref_name.include // []) | any(. == "refs/heads/main" or . == "~DEFAULT_BRANCH" or . == "refs/heads/*" or . == "refs/heads/**" or . == "**"))
      and (((.conditions.ref_name.exclude // []) | any(. == "refs/heads/main" or . == "~DEFAULT_BRANCH")) | not)
      and ((.rules // []) | any(.type == "merge_queue"))
    )
  ' <<< "$raw" >/dev/null 2>&1 || rc=$?
  case "$rc" in
    0) echo 1 ;;
    1) echo 0 ;;
    *) echo "one-gate: invalid rulesets json" >&2; return 2 ;;
  esac
}

run_check() {
  local dir=$1 al=$2 has_mq=0 files=() f
  [ -d "$dir" ] || { echo "one-gate: no workflows dir $dir" >&2; return 2; }
  [ -r "$al" ] || { echo "one-gate: unreadable allow-list $al" >&2; return 2; }
  for f in "$dir"/*.yml "$dir"/*.yaml; do [ -f "$f" ] && files+=("$f"); done
  [ "${#files[@]}" -gt 0 ] || { echo "one-gate: no workflows in $dir (fail closed)" >&2; return 2; }
  if [ "$live" -eq 1 ]; then has_mq=$(live_has_merge_queue "$rulesets_json") || return 2; fi
  awk -v live="$live" -v has_mq="$has_mq" -v alfile="$al" -f <(cat <<'AWK'
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
function unq(s) { s = trim(s); if ((s ~ /^".*"$/) || (s ~ /^\047.*\047$/)) s = substr(s, 2, length(s)-2); return s }
function strip_c(s,    i, c, q, out) {
  q = ""; out = ""
  for (i = 1; i <= length(s); i++) {
    c = substr(s, i, 1)
    if (q == "") { if (c == "\047" || c == "\"") q = c; else if (c == "#") break }
    else if (c == q) {
      if (q == "\047" && substr(s, i+1, 1) == "\047") { out = out "\047\047"; i++; continue }
      if (q == "\"" && substr(s, i-1, 1) == "\\") { out = out c; continue }
      q = ""
    }
    out = out c
  }
  return trim(out)
}
function norm_list(s,    n, i, a, out) { s = trim(s); gsub(/^\[|\]$/, "", s); n = split(s, a, ","); out = ""; for (i = 1; i <= n; i++) out = out (i > 1 ? " " : "") unq(a[i]); return out }
function br_match_main(pat,    neg, rx, i, ch) {
  if (pat == "") return -1
  neg = 0; if (substr(pat, 1, 1) == "!") { neg = 1; pat = substr(pat, 2) }
  rx = "^"
  for (i = 1; i <= length(pat); i++) {
    ch = substr(pat, i, 1)
    if (ch == "*") {
      if (substr(pat, i+1, 1) == "*") { rx = rx ".*"; i++ } else rx = rx "[^/]*"
    } else if (ch == "?") rx = rx "[^/]"
    else if (index(".+^$()[]{}|\\", ch) > 0) rx = rx "\\" ch
    else rx = rx ch
  }
  rx = rx "$"
  return ("main" ~ rx) ? (neg ? 0 : 1) : -1
}
function truthy(v) { return (v != "" && v != "false" && v != "0") }
function lookup(n) {
  if (n == "true" || n == "false" || n ~ /^[0-9]+$/) return n
  if (n == "github.event_name") return EV
  if (n == "github.ref") return (EV == "push" ? "refs/heads/main" : (EV == "pull_request" ? "refs/pull/1/merge" : "refs/heads/gh-readonly-queue/main/pr-1"))
  if (n == "github.ref_name") return (EV == "push" ? "main" : (EV == "pull_request" ? "1/merge" : "gh-readonly-queue/main/pr-1"))
  if (n == "github.repository") return "paiml/aprender"
  if (n == "github.event.pull_request.draft") return (EV == "pull_request" ? "false" : "")
  if (n == "github.event.pull_request.head.repo.full_name") return (EV == "pull_request" ? "paiml/aprender" : "")
  if (n ~ /^needs\.[^.]+\.result$/) return "success"; unk = 1; return ""
}
function tok(s,    m) {
  nt = 0; s = trim(s)
  while (s != "") {
    if (match(s, /^\047([^\047]|\047\047)*\047/)) { m = substr(s, 2, RLENGTH-2); gsub(/\047\047/, "\047", m); tk[++nt] = "S" m; s = substr(s, RLENGTH+1) }
    else if (match(s, /^(==|!=|&&|\|\|)/)) { tk[++nt] = "O" substr(s, 1, 2); s = substr(s, 3) }
    else if (match(s, /^[!(),]/)) { tk[++nt] = "O" substr(s, 1, 1); s = substr(s, 2) }
    else if (match(s, /^[A-Za-z0-9_.*-]+/)) { tk[++nt] = "I" substr(s, 1, RLENGTH); s = substr(s, RLENGTH+1) }
    else if (match(s, /^[ \t]+/)) { s = substr(s, RLENGTH+1) } else { bad = 1; return }
  }
}
function isop(o) { return (tp <= nt && tk[tp] == "O" o) }
function pOr(   v, w) { v = pAnd(); while (isop("||")) { tp++; w = pAnd(); if (!truthy(v)) v = w }; return v }
function pAnd(   v, w) { v = pCmp(); while (isop("&&")) { tp++; w = pCmp(); if (truthy(v)) v = w }; return v }
function pCmp(   v, w, o) { v = pPrim(); while (isop("==") || isop("!=")) { o = substr(tk[tp], 2); tp++; w = pPrim(); v = ((o == "==") == (v == w)) ? "true" : "false" }; return v }
function pPrim(   t, n, v, a, b, u0, ua, ub) {
  if (tp > nt) { bad = 1; return "" }; t = tk[tp++]
  if (t == "O!") { v = pPrim(); return truthy(v) ? "false" : "true" }
  if (t == "O(") { v = pOr(); if (isop(")")) tp++; else bad = 1; return v }
  if (substr(t, 1, 1) == "S") return substr(t, 2)
  if (substr(t, 1, 1) == "I") {
    n = substr(t, 2)
    if (isop("(")) {
      tp++; u0 = unk; unk = 0; a = ""; b = ""; ua = 0; ub = 0
      if (!isop(")")) {
        a = pOr(); ua = unk
        if (isop(",")) { tp++; unk = 0; b = pOr(); ub = unk }
        while (isop(",")) { tp++; pOr() }
      }
      if (isop(")")) tp++; else bad = 1
      if (n == "cancelled" || n == "failure") { unk = u0; return "false" }
      if (n == "always" || n == "success") { unk = u0; return "true" }
      if (n == "contains") { unk = u0; return (ua || ub) ? "true" : ((index(tolower(a), tolower(b)) > 0) ? "true" : "false") }
      if (n == "startsWith") { unk = u0; return (ua || ub) ? "true" : ((b == "" || tolower(substr(a, 1, length(b))) == tolower(b)) ? "true" : "false") }
      if (n == "endsWith") { unk = u0; return (ua || ub) ? "true" : ((b == "" || (length(a) >= length(b) && tolower(substr(a, length(a) - length(b) + 1)) == tolower(b))) ? "true" : "false") }
      if (n == "fromJSON" || n == "toJSON" || n == "format") { unk = (u0 || ua || ub); return a }
      unk = u0; return ""
    }
    return lookup(n)
  }
  bad = 1; return ""
}
function ev_expr(e, ev,    v) { EV = ev; bad = 0; unk = 0; tok(e); tp = 1; if (bad) return "UNPARSEABLE"; v = pOr(); if (bad || tp <= nt) return "UNPARSEABLE"; gsub(/ /, "", v); return v }
function resolve(raw, ev,    e) { raw = unq(raw); if (match(raw, /^\$\{\{.*\}\}$/)) { e = substr(raw, 4, length(raw)-5); return ev_expr(e, ev) }; gsub(/ /, "", raw); return raw }
function ifval(raw, ev,    e) { raw = trim(raw); if (raw == "") return 1; if (match(raw, /^\$\{\{.*\}\}$/)) raw = substr(raw, 4, length(raw)-5); e = ev_expr(raw, ev); if (e == "UNPARSEABLE") { unparse[wf_ "|" job_] = raw; return 1 }; return truthy(e) }
function viol(kind, w, j, d, msg,    key) { gsub(/ /, "_", d); key = kind "|" w "|" j "|" d; if (kind != "live-mq-off" && (key in allowed)) { used[key] = 1; return }; if (kind == "live-mq-off") printf "RED %s: %s %s: %s  [enable merge_queue ruleset]\n", kind, w, (j == "-" ? "" : j), msg; else printf "RED %s: %s %s: %s  [allow %s %s %s %s]\n", kind, w, (j == "-" ? "" : j), msg, kind, w, j, d; bad_n++ }
FNR == 1 { top = ""; ontrig = ""; job = ""; jsec = ""; sidx = 0; in_mat = 0; in_br = 0; in_paths = 0; wf = FILENAME; sub(/.*\//, "", wf) }
FILENAME == alfile {
  if ($0 ~ /^[ \t]*(#|$)/) next; na = split($0, a, " ")
  if (a[1] == "required") { req[a[2] "|" a[3]] = 1; next }
  if (a[1] == "allow") { if (a[2] == "live-mq-off") { printf "RED allow-list cannot exempt live-mq-off: %s\n", $0; bad_n++; next }; key = a[2] "|" a[3] "|" a[4] "|" a[5]; if (index($0, " -- ") == 0 || a[6] != "--" || na < 7) { printf "RED allow-list missing reason: %s\n", $0; bad_n++ }; allowed[key] = 1; next }
  printf "RED allow-list unknown line: %s\n", $0; bad_n++; next
}
{ raw = $0; line = strip_c($0); if (line == "") next }
raw ~ /^on:[ \t]*(.*)/ { top = "on"; v = trim(substr(line, 4)); if (v ~ /push/) { trig[wf "|push"] = 1; push_main[wf] = 1 }; if (v ~ /pull_request/) trig[wf "|pull_request"] = 1; if (v ~ /merge_group/) trig[wf "|merge_group"] = 1; next }
raw ~ /^jobs:/ && line ~ /^jobs:[ \t]*$/ { top = "jobs"; ontrig = ""; next }
raw ~ /^[^ \t#]/ { top = ""; ontrig = ""; job = ""; next }
top == "on" && raw ~ /^  [A-Za-z0-9_-]+:/ { ontrig = line; sub(/:.*/, "", ontrig); in_br = 0; in_paths = 0; if (ontrig == "push" || ontrig == "pull_request" || ontrig == "merge_group") trig[wf "|" ontrig] = 1; if (ontrig == "push") push_main[wf] = 1; next }
top == "on" && ontrig == "push" && raw ~ /^    branches:/ { v = trim(substr(line, index(line, ":")+1)); in_paths = 0; push_main[wf] = 0; if (v != "") { n = split(norm_list(v), ba, " "); for (i=1; i<=n; i++) { bm = br_match_main(ba[i]); if (bm >= 0) push_main[wf] = bm } } else in_br = 1; next }
top == "on" && ontrig == "push" && in_br && raw ~ /^      -[ \t]*/ { bm = br_match_main(unq(substr(line, 3))); if (bm >= 0) push_main[wf] = bm; next }
top == "on" && (ontrig == "push" || ontrig == "pull_request") && raw ~ /^    paths:/ { v = trim(substr(line, index(line, ":")+1)); in_br = 0; in_paths = 0; if (v != "") { n = split(norm_list(v), pa, " "); for (i=1; i<=n; i++) if (pa[i]!="") ppaths[wf "|" ontrig "|" pa[i]] = 1 } else in_paths = 1; next }
top == "on" && in_paths && raw ~ /^      -[ \t]*/ { ppaths[wf "|" ontrig "|" unq(substr(line, 3))] = 1; next }
top == "on" && raw ~ /^    [A-Za-z0-9_-]+:/ { in_br = 0; in_paths = 0; next }
top == "jobs" && raw ~ /^  [A-Za-z0-9_-]+:/ && line ~ /^[A-Za-z0-9_-]+:[ \t]*$/ { job = line; sub(/:.*/, "", job); jobs[wf "|" job] = 1; jsec = ""; sidx = 0; in_mat = 0; next }
top == "jobs" && job != "" && raw ~ /^    if:[ \t]*/ { v = trim(substr(line, index(line, ":")+1)); jsec = (v == ">-" || v == "|-") ? "ifml" : ""; if (jsec == "") jif[wf "|" job] = v; next }
top == "jobs" && jsec == "ifml" && raw ~ /^      / { jif[wf "|" job] = trim(jif[wf "|" job] " " line); next }
top == "jobs" && job != "" && raw ~ /^    needs:[ \t]*/ { v = trim(substr(line, index(line, ":")+1)); if (v != "") { needs[wf "|" job] = norm_list(v); jsec = "" } else jsec = "needs"; next }
top == "jobs" && jsec == "needs" && raw ~ /^      -[ \t]*/ { needs[wf "|" job] = trim(needs[wf "|" job] " " unq(substr(line, 3))); next }
top == "jobs" && job != "" && raw ~ /^    runs-on:[ \t]*/ { v = trim(substr(line, index(line, ":")+1)); if (v != "") { rawattr[wf "|" job "|runs-on"] = v; jsec = "" } else { jsec = "runs-on"; rawattr[wf "|" job "|runs-on"] = "" }; next }
top == "jobs" && jsec == "runs-on" && raw ~ /^      -[ \t]*/ { v = unq(substr(line, 3)); cur = rawattr[wf "|" job "|runs-on"]; rawattr[wf "|" job "|runs-on"] = (cur == "" ? "[" v "]" : substr(cur, 1, length(cur)-1) "," v "]"); next }
top == "jobs" && job != "" && raw ~ /^    (uses|container|image):[ \t]*/ { pinv[wf "|" job, ++npin[wf "|" job]] = trim(substr(line, index(line, ":")+1)); jsec = ""; next }
top == "jobs" && job != "" && raw ~ /^    env:/ && line ~ /^env:[ \t]*$/ { jsec = "env"; next }
top == "jobs" && jsec == "env" && raw ~ /^      [A-Za-z0-9_-]+:[ \t]*/ { k = line; sub(/:.*/, "", k); rawattr[wf "|" job "|env." k] = trim(substr(line, index(line, ":")+1)); next }
top == "jobs" && job != "" && raw ~ /^    strategy:/ && line ~ /^strategy:[ \t]*$/ { jsec = "strategy"; in_mat = 0; next }
top == "jobs" && jsec == "strategy" && raw ~ /^      matrix:/ && line ~ /^matrix:[ \t]*$/ { in_mat = 1; next }
top == "jobs" && jsec == "strategy" && in_mat && raw ~ /^        [A-Za-z0-9_-]+:[ \t]*/ { k = line; sub(/:.*/, "", k); v = trim(substr(line, index(line, ":")+1)); if (k != "include" && k != "exclude" && v != "") rawattr[wf "|" job "|matrix." k] = v; next }
top == "jobs" && job != "" && raw ~ /^    steps:/ && line ~ /^steps:[ \t]*$/ { jsec = "steps"; sidx = 0; next }
top == "jobs" && jsec == "steps" && raw ~ /^      -[ \t]*/ { sidx++; nstep[wf "|" job] = sidx; v = trim(substr(line, 2)); if (v ~ /^name:[ \t]*/) sname[wf "|" job, sidx] = unq(substr(v, 6)); else if (v ~ /^uses:[ \t]*/) { s = trim(substr(v, 6)); suses[wf "|" job, sidx] = s; pinv[wf "|" job, ++npin[wf "|" job]] = s } else if (v ~ /^if:[ \t]*/) sif[wf "|" job, sidx] = trim(substr(v, 4)); next }
top == "jobs" && jsec == "steps" && sidx > 0 && raw ~ /^        (name|uses|if):[ \t]*/ { k = line; sub(/:.*/, "", k); v = trim(substr(line, index(line, ":")+1)); if (k == "name") sname[wf "|" job, sidx] = unq(v); else if (k == "uses") { suses[wf "|" job, sidx] = v; pinv[wf "|" job, ++npin[wf "|" job]] = v } else if (k == "if") sif[wf "|" job, sidx] = v; next }
top == "jobs" && jsec == "steps" && sidx > 0 && raw ~ /^          toolchain:[ \t]*/ { pinv[wf "|" job, ++npin[wf "|" job]] = trim(substr(line, index(line, ":")+1)); next }
top == "jobs" && job != "" && raw ~ /^    [A-Za-z0-9_-]+:/ { jsec = ""; in_mat = 0; next }
END {
  for (w in push_main) if (push_main[w]) { pwfs[w] = 1; n_pw++ }
  if (n_pw == 0) { print "one-gate: no push-to-main workflows found (fail closed)" > "/dev/stderr"; exit 2 }
  split("push pull_request merge_group", EVS, " ")
  for (key in jobs) { split(key, p, "|"); w = p[1]; j = p[2]; if (!(w in pwfs)) continue; wf_ = w; job_ = j; for (e = 1; e <= 3; e++) ran[key "|" EVS[e]] = (((w "|" EVS[e]) in trig) && ifval(jif[key], EVS[e])) ? 1 : 0 }
  for (rk in req) { stack[++sn] = rk; inclos[rk] = 1 }
  while (sn > 0) { cur = stack[sn--]; n = split(needs[cur], nn, " "); split(cur, cp, "|"); for (i = 1; i <= n; i++) { nk = cp[1] "|" nn[i]; if (!(nk in inclos)) { inclos[nk] = 1; stack[++sn] = nk } } }
  for (key in jobs) {
    split(key, p, "|"); w = p[1]; j = p[2]; if (!(w in pwfs)) continue
    is_push = ran[key "|push"]; is_mg_only = (ran[key "|merge_group"] && !is_push && !ran[key "|pull_request"])
    if (!is_push && !is_mg_only) continue; wf_ = w; job_ = j
    if (is_push) {
      if (!(ran[key "|pull_request"] || ran[key "|merge_group"])) viol("push-only", w, j, "-", "runs on push to main but on neither pull_request nor merge_group")
      else {
        if (!(key in inclos)) viol("not-required", w, j, "-", "runs on push to main but not in needs closure of any required job")
        if (live && !has_mq && !ran[key "|pull_request"] && ran[key "|merge_group"]) viol("live-mq-off", w, j, "-", "merge_group is the only cover for push-to-main job and no active ruleset has a merge_queue rule")
      }
      for (e = 2; e <= 3; e++) {
        ev = EVS[e]; if (!ran[key "|" ev]) continue
        for (ak in rawattr) { split(ak, ap, "|"); if (ap[1] == w && ap[2] == j) { a1 = resolve(rawattr[ak], "push"); a2 = resolve(rawattr[ak], ev); if (a1 != a2) viol("attr", w, j, ap[3], ap[3] " differs: push=" a1 " vs " ev "=" a2) } }
        for (si = 1; si <= nstep[key]; si++) { snm = (sname[key, si] != "" ? sname[key, si] : (suses[key, si] != "" ? suses[key, si] : "#" si)); if (ifval(sif[key, si], "push") && !ifval(sif[key, si], ev)) viol("step", w, j, snm, "step runs on push but not on " ev) }
      }
      for (ak in rawattr) { split(ak, ap, "|"); if (ap[1] == w && ap[2] == j && resolve(rawattr[ak], "push") == "UNPARSEABLE") viol("attr", w, j, ap[3], ap[3] " unparseable (fail closed)") }
      for (i = 1; i <= npin[key]; i++) if (pinv[key, i] ~ /github\.(event_name|ref|ref_name|event\.pull_request)/) viol("pin", w, j, "-", "event-dependent pin: " pinv[key, i])
    } else {
      viol("mg-only", w, j, "-", "runs on merge_group only (neither push nor pull_request)")
      if (!(key in inclos)) viol("not-required", w, j, "-", "runs on merge_group only but not in needs closure of any required job")
      if (live && !has_mq) viol("live-mq-off", w, j, "-", "runs on merge_group only and no active ruleset has a merge_queue rule")
    }
  }
  for (key in unparse) { split(key, p, "|"); viol("attr", p[1], p[2], "if", "if unparseable: " unparse[key]) }
  for (pk in ppaths) { split(pk, q, "|"); if (q[1] in pwfs) has_p[q[1] "|" q[2]] = 1 }
  for (w in pwfs) if (((w "|pull_request") in trig) && ((w "|pull_request") in has_p) && !((w "|push") in has_p)) viol("paths", w, "-", "push-unfiltered", "pull_request has paths filter but push has none")
  for (pk in ppaths) { split(pk, q, "|"); if ((q[1] in pwfs) && q[2] == "push" && ((q[1] "|pull_request") in trig) && ((q[1] "|pull_request") in has_p) && !((q[1] "|pull_request|" q[3]) in ppaths)) viol("paths", q[1], "-", q[3], "push path missing from pull_request paths: " q[3]) }
  for (key in allowed) if (!(key in used)) { printf "RED stale allow-list line: allow %s\n", key; bad_n++ }
  if (bad_n > 0) { printf "one-gate parity: %d difference(s)\n", bad_n; exit 1 }
  print "one-gate parity: OK"
}
AWK
) "$al" "${files[@]}"
}

if [ "$mode" = check ]; then run_check "$wf_dir" "$allow"; exit $?; fi
st=$(mktemp -d); SCRATCH+=("$st"); fail=0
base_ci() { printf 'name: CI\non:\n  push:\n    branches: [main]\n  pull_request:\n    branches: [main]\n  merge_group:\njobs:\n  build:\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    if: ${{ github.event.pull_request.draft != true }}\n    steps: [{name: compile, run: echo hi}]\n  shard:\n    needs: [build]\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    strategy:\n      matrix:\n        shard: ${{ fromJSON(\047[1,2,3]\047) }}\n    steps:\n      - name: test\n        run: echo hi\n  gate:\n    needs: [build, shard]\n    if: always()\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    steps: [{name: verdict, run: echo hi}]\n'; }
case_run() {
  local name=$1 want=$2 extra=$3 mut=$4 rc=0
  rm -rf -- "${st:?}/wf"; mkdir -p "$st/wf"; base_ci >"$st/wf/ci.yml"
  [ -z "$mut" ] || sed -i "$mut" "$st/wf/ci.yml"
  printf 'required ci.yml gate\n%s' "$extra" >"$st/allow.txt"
  "$0" --workflows "$st/wf" --allow "$st/allow.txt" >"$st/out" 2>&1 || rc=$?
  if [ "$rc" = "$want" ]; then printf 'ok   %s (rc=%s)\n' "$name" "$rc"
  else printf 'FAIL %s: want rc=%s got rc=%s\n' "$name" "$want" "$rc"; sed 's/^/     /' "$st/out"; fail=1; fi
}
Q="'"
case_run "baseline parity holds" 0 "" ""
case_run "M1 push-only job" 1 "" "s/^  shard:\$/  shard:\n    if: github.event_name == ${Q}push${Q}/"
case_run "M2 runs-on differs on PR" 1 "" "0,/^  shard:\$/! s/^    runs-on: \[self-hosted, Linux, X64, clean-room\]\$/    runs-on: \${{ github.event_name == ${Q}pull_request${Q} \&\& fromJSON(${Q}[\"self-hosted\",\"Linux\",\"clean-room\"]${Q}) || fromJSON(${Q}[\"self-hosted\",\"Linux\",\"X64\",\"clean-room\"]${Q}) }}/"
case_run "M3 matrix differs push vs merge_group" 1 "" "s/fromJSON(${Q}\[1,2,3\]${Q})/fromJSON(github.event_name == ${Q}merge_group${Q} \&\& ${Q}[1,2,3]${Q} || ${Q}[1,2,3,4]${Q})/"
case_run "M4 push job not in required needs" 1 "" 's/^    needs: \[build, shard\]$/    needs: [build]/'
case_run "M5 step push-only" 1 "" "s/^      - name: test\$/      - if: github.event_name == ${Q}push${Q}\n        name: test/"
case_run "M6 event-dependent pin" 1 "" "s/^  build:\$/  build:\n    container: \${{ github.event_name == ${Q}push${Q} \&\& ${Q}img:1${Q} || ${Q}img:2${Q} }}/"
case_run "M7 unparseable if fails closed" 1 "" 's/^    if: always()$/    if: ${{ @@@ }}/'
case_run "M8 env differs by event" 1 "" "s/^  build:\$/  build:\n    env:\n      X: \${{ github.event_name == ${Q}push${Q} \&\& ${Q}a${Q} || ${Q}b${Q} }}/"
case_run "M9 stale allow line is RED" 1 $'allow attr ci.yml build runs-on -- unused\n' ""
case_run "M10 allow without reason is RED" 1 $'allow push-only ci.yml shard -\n' "s/^  shard:\$/  shard:\n    if: github.event_name == ${Q}push${Q}/"
case_run "A1 allow-listed exception passes" 0 $'allow push-only ci.yml shard - -- reviewed exception\n' "s/^  shard:\$/  shard:\n    if: github.event_name == ${Q}push${Q}/"
case_run "M11 startsWith(github.ref) push-only job" 1 "" "s/^  shard:\$/  shard:\n    if: startsWith(github.ref, ${Q}refs\/heads\/main${Q})/"
case_run "M12 contains(github.event_name) push-only step" 1 "" "s/^      - name: test\$/      - if: contains(github.event_name, ${Q}push${Q})\n        name: test/"
case_run "M13 endsWith(github.ref) push-only job" 1 "" "s/^  shard:\$/  shard:\n    if: endsWith(github.ref, ${Q}\/main${Q})/"
case_run "M14 asymmetric paths (PR filtered, push unfiltered)" 1 "" 's/^  pull_request:$/  pull_request:\n    paths: ["book\/**"]/'
case_run "M15 asymmetric paths (push path missing from PR)" 1 "" 's/^  push:$/  push:\n    paths: ["src\/**"]/; s/^  pull_request:$/  pull_request:\n    paths: ["book\/**"]/'
case_run "M17 wildcard branches [*] catches push-only" 1 "" "s/branches: \[main\]/branches: [${Q}*${Q}]/; s/^  shard:\$/  shard:\n    if: github.event_name == ${Q}push${Q}/"
case_run "M18 wildcard branches [**] catches push-only" 1 "" "s/branches: \[main\]/branches: [${Q}**${Q}]/; s/^  shard:\$/  shard:\n    if: github.event_name == ${Q}push${Q}/"
case_run "M19 github.ref event-dependent pin" 1 "" "s/^  build:\$/  build:\n    container: \${{ github.ref == ${Q}refs\/heads\/main${Q} \&\& ${Q}img:1${Q} || ${Q}img:2${Q} }}/"
case_run "M20 job header with inline comment" 1 "" "s/^  shard:\$/  shard: # inline comment\n    if: github.event_name == ${Q}push${Q}/"
case_run "A2 unknown-context contains() stays truthy on push" 0 $'allow step ci.yml shard test -- post-merge step\n' "s/^      - name: test\$/      - if: always() \&\& github.event_name == ${Q}push${Q} \&\& contains(steps.wait.outputs.deferred, ${Q}codecov${Q})\n        name: test/"
case_run "M21 mg-only guards job without allow is RED" 1 "" "s/^  gate:\$/  guards:\n    if: github.event_name == ${Q}merge_group${Q}\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    steps: [{name: g, run: echo g}]\n  gate:/; s/^    needs: \[build, shard\]\$/    needs: [build, shard, guards]/"
case_run "A3 mg-only guards job with allow passes offline" 0 $'allow mg-only ci.yml guards - -- merge_group-only guards\n' "s/^  gate:\$/  guards:\n    if: github.event_name == ${Q}merge_group${Q}\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    steps: [{name: g, run: echo g}]\n  gate:/; s/^    needs: \[build, shard\]\$/    needs: [build, shard, guards]/"
case_live() {
  local name=$1 want=$2 extra=$3 mut=$4 rs=$5 rc=0
  rm -rf -- "${st:?}/wf"; mkdir -p "$st/wf"; base_ci >"$st/wf/ci.yml"
  [ -z "$mut" ] || sed -i "$mut" "$st/wf/ci.yml"
  printf 'required ci.yml gate\n%s' "$extra" >"$st/allow.txt"
  printf '%s\n' "$rs" >"$st/rs.json"
  "$0" --workflows "$st/wf" --allow "$st/allow.txt" --live --rulesets-json "$st/rs.json" >"$st/out" 2>&1 || rc=$?
  if [ "$rc" = "$want" ]; then printf 'ok   %s (rc=%s)\n' "$name" "$rc"
  else printf 'FAIL %s: want rc=%s got rc=%s\n' "$name" "$want" "$rc"; sed 's/^/     /' "$st/out"; fail=1; fi
}
RS_OFF='[{"id":17836320,"enforcement":"active","target":"branch","conditions":{"ref_name":{"include":["refs/heads/main"],"exclude":[]}},"rules":[]}]'
RS_ON='[{"id":17836320,"enforcement":"active","target":"branch","conditions":{"ref_name":{"include":["refs/heads/main"],"exclude":[]}},"rules":[{"type":"merge_queue"}]}]'
RS_DIS='[{"id":17836320,"enforcement":"disabled","target":"branch","conditions":{"ref_name":{"include":["refs/heads/main"],"exclude":[]}},"rules":[{"type":"merge_queue"}]}]'
case_live "L1 --live fails when merge_group is only cover and merge_queue is off" 1 "" "s/^  shard:\$/  shard:\n    if: github.event_name != ${Q}pull_request${Q}/" "$RS_OFF"
case_live "L2 --live fails on allow-listed mg-only guards when merge_queue is off" 1 $'allow mg-only ci.yml guards - -- merge_group-only guards\n' "s/^  gate:\$/  guards:\n    if: github.event_name == ${Q}merge_group${Q}\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    steps: [{name: g, run: echo g}]\n  gate:/; s/^    needs: \[build, shard\]\$/    needs: [build, shard, guards]/" "$RS_OFF"
case_live "L3 --live passes when active ruleset has merge_queue" 0 $'allow mg-only ci.yml guards - -- merge_group-only guards\n' "s/^  shard:\$/  shard:\n    if: github.event_name != ${Q}pull_request${Q}/; s/^  gate:\$/  guards:\n    if: github.event_name == ${Q}merge_group${Q}\n    runs-on: [self-hosted, Linux, X64, clean-room]\n    steps: [{name: g, run: echo g}]\n  gate:/; s/^    needs: \[build, shard\]\$/    needs: [build, shard, guards]/" "$RS_ON"
case_live "L4 --live fails when merge_queue ruleset is disabled" 1 "" "s/^  shard:\$/  shard:\n    if: github.event_name != ${Q}pull_request${Q}/" "$RS_DIS"
case_live "L5 allow live-mq-off in allow-list is refused" 1 $'allow live-mq-off ci.yml shard - -- cannot suppress\n' "s/^  shard:\$/  shard:\n    if: github.event_name != ${Q}pull_request${Q}/" "$RS_ON"
case_live "V3 --live fails closed on invalid rulesets json" 2 "" "" "not-json"
rm -rf -- "${st:?}/wf"; mkdir -p "$st/wf"; base_ci >"$st/wf/ci.yml"; : >"$st/empty.txt"
rc=0; "$0" --workflows "$st/wf" --allow "$st/empty.txt" >"$st/out" 2>&1 || rc=$?
[ "$rc" -eq 1 ] && grep -q '^RED not-required: ci.yml build:' "$st/out" && ! grep -q '^RED allow-list' "$st/out" && printf 'ok   M22 empty allow-list fails without misreading workflow (rc=%s)\n' "$rc" || { printf 'FAIL M22: want rc=1 got rc=%s\n' "$rc"; sed 's/^/     /' "$st/out"; fail=1; }
rm -rf -- "${st:?}/wf"; mkdir -p "$st/wf"; base_ci >"$st/wf/ci.yml"
printf 'on:\n  push:\n    branches: [main]\n  pull_request:\n    branches: [main]\njobs:\n  extra:\n    runs-on: [self-hosted]\n    steps: [{name: ok, run: echo ok}]\n' >"$st/wf/extra.yaml"
printf 'required ci.yml gate\n' >"$st/allow.txt"
rc=0; "$0" --workflows "$st/wf" --allow "$st/allow.txt" >"$st/out" 2>&1 || rc=$?
[ "$rc" -eq 1 ] && grep -q '^RED not-required: extra.yaml extra:' "$st/out" && printf 'ok   M16 not-required without required entry + .yaml (rc=%s)\n' "$rc" || { printf 'FAIL M16: want rc=1 got rc=%s\n' "$rc"; sed 's/^/     /' "$st/out"; fail=1; }
mkdir -p "$st/empty"; rc=0; "$0" --workflows "$st/empty" --allow "$st/allow.txt" >"$st/out" 2>&1 || rc=$?
[ "$rc" -eq 2 ] && printf 'ok   V1 empty workflows fails closed (rc=%s)\n' "$rc" || { printf 'FAIL V1: want rc=2 got rc=%s\n' "$rc"; fail=1; }
rc=0; "$0" --workflows >"$st/out" 2>&1 || rc=$?
[ "$rc" -eq 2 ] && printf 'ok   V2 missing --workflows arg fails closed (rc=%s)\n' "$rc" || { printf 'FAIL V2: want rc=2 got rc=%s\n' "$rc"; sed 's/^/     /' "$st/out"; fail=1; }
[ "$fail" -eq 0 ] || { echo "one-gate self-test: FAILED"; exit 1; }
echo "one-gate self-test: all cases pass"
