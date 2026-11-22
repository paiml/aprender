#!/usr/bin/env bash
# ONE-GATE parity falsifier (APR-GREEN-001): diffs push-to-main jobs vs PR-required set.
# Usage: check_one_gate_parity.sh [--workflows DIR] [--allow FILE] | --self-test | --help
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
wf_dir="$root/.github/workflows"; allow="$root/.github/one-gate-allow.txt"; mode=check
SCRATCH=(); trap 'rm -rf -- "${SCRATCH[@]}"' EXIT
while [ $# -gt 0 ]; do
  case "$1" in
    --workflows) wf_dir=$2; shift 2 ;; --allow) allow=$2; shift 2 ;;
    --self-test) mode=self-test; shift ;;
    -h|--help) echo "usage: $0 [--workflows DIR] [--allow FILE] | --self-test"; exit 0 ;;
    *) echo "usage: $0 [--workflows DIR] [--allow FILE] | --self-test" >&2; exit 2 ;;
  esac
done

run_check() {
  local dir=$1 al=$2 files=() f
  [ -d "$dir" ] || { echo "one-gate: no workflows dir $dir" >&2; return 2; }
  [ -r "$al" ] || { echo "one-gate: unreadable allow-list $al" >&2; return 2; }
  for f in "$dir"/*.yml; do [ -f "$f" ] && files+=("$f"); done
  [ "${#files[@]}" -gt 0 ] || { echo "one-gate: no workflows in $dir (fail closed)" >&2; return 2; }
  awk -f <(cat <<'AWK'
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
function truthy(v) { return (v != "" && v != "false" && v != "0") }
function lookup(n) {
  if (n == "true" || n == "false" || n ~ /^[0-9]+$/) return n
  if (n == "github.event_name") return EV
  if (n == "github.ref") return (EV == "push" ? "refs/heads/main" : (EV == "pull_request" ? "refs/pull/1/merge" : "refs/heads/gh-readonly-queue/main/pr-1"))
  if (n == "github.ref_name") return (EV == "push" ? "main" : ""); if (n == "github.repository") return "paiml/aprender"
  if (n == "github.event.pull_request.draft") return (EV == "pull_request" ? "false" : "")
  if (n == "github.event.pull_request.head.repo.full_name") return (EV == "pull_request" ? "paiml/aprender" : "")
  if (n ~ /^needs\.[^.]+\.result$/) return "success"; return ""
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
function pPrim(   t, n, v, a) {
  if (tp > nt) { bad = 1; return "" }; t = tk[tp++]
  if (t == "O!") { v = pPrim(); return truthy(v) ? "false" : "true" }
  if (t == "O(") { v = pOr(); if (isop(")")) tp++; else bad = 1; return v }
  if (substr(t, 1, 1) == "S") return substr(t, 2)
  if (substr(t, 1, 1) == "I") {
    n = substr(t, 2)
    if (isop("(")) {
      tp++; a = ""; if (!isop(")")) { a = pOr(); while (isop(",")) { tp++; pOr() } }; if (isop(")")) tp++; else bad = 1
      if (n == "cancelled" || n == "failure") return "false"
      if (n == "always" || n == "success" || n == "contains" || n == "startsWith" || n == "endsWith") return "true"
      if (n == "fromJSON" || n == "toJSON" || n == "format") return a; return ""
    }
    return lookup(n)
  }
  bad = 1; return ""
}
function ev_expr(e, ev,    v) { EV = ev; bad = 0; tok(e); tp = 1; if (bad) return "UNPARSEABLE"; v = pOr(); if (bad || tp <= nt) return "UNPARSEABLE"; gsub(/ /, "", v); return v }
function resolve(raw, ev,    e) { raw = unq(raw); if (match(raw, /^\$\{\{.*\}\}$/)) { e = substr(raw, 4, length(raw)-5); return ev_expr(e, ev) }; gsub(/ /, "", raw); return raw }
function ifval(raw, ev,    e) { raw = trim(raw); if (raw == "") return 1; if (match(raw, /^\$\{\{.*\}\}$/)) raw = substr(raw, 4, length(raw)-5); e = ev_expr(raw, ev); if (e == "UNPARSEABLE") { unparse[wf_ "|" job_] = raw; return 1 }; return truthy(e) }
function viol(kind, w, j, d, msg,    key) { gsub(/ /, "_", d); key = kind "|" w "|" j "|" d; if (key in allowed) { used[key] = 1; return }; printf "RED %s: %s %s: %s  [allow %s %s %s %s]\n", kind, w, (j == "-" ? "" : j), msg, kind, w, j, d; bad_n++ }
FNR == 1 { fno++; top = ""; ontrig = ""; job = ""; jsec = ""; sidx = 0; in_mat = 0; in_br = 0; in_paths = 0; wf = FILENAME; sub(/.*\//, "", wf) }
fno == 1 {
  if ($0 ~ /^[ \t]*(#|$)/) next; na = split($0, a, " ")
  if (a[1] == "required") { req[a[2] "|" a[3]] = 1; next }
  if (a[1] == "allow") { key = a[2] "|" a[3] "|" a[4] "|" a[5]; if (index($0, " -- ") == 0 || a[6] != "--" || na < 7) { printf "RED allow-list missing reason: %s\n", $0; bad_n++ }; allowed[key] = 1; next }
  printf "RED allow-list unknown line: %s\n", $0; bad_n++; next
}
{ raw = $0; line = strip_c($0); if (line == "") next }
raw ~ /^on:[ \t]*(.*)/ { top = "on"; v = trim(substr(line, 4)); if (v ~ /push/) { trig[wf "|push"] = 1; push_main[wf] = 1 }; if (v ~ /pull_request/) trig[wf "|pull_request"] = 1; if (v ~ /merge_group/) trig[wf "|merge_group"] = 1; next }
raw ~ /^jobs:[ \t]*$/ { top = "jobs"; ontrig = ""; next }
raw ~ /^[^ \t#]/ { top = ""; ontrig = ""; job = ""; next }
top == "on" && raw ~ /^  [A-Za-z0-9_-]+:/ { ontrig = line; sub(/:.*/, "", ontrig); in_br = 0; in_paths = 0; if (ontrig == "push" || ontrig == "pull_request" || ontrig == "merge_group") trig[wf "|" ontrig] = 1; if (ontrig == "push") push_main[wf] = 1; next }
top == "on" && ontrig == "push" && raw ~ /^    branches:/ { v = trim(substr(line, index(line, ":")+1)); in_paths = 0; push_main[wf] = 0; if (v != "") { if ((" " norm_list(v) " ") ~ / main /) push_main[wf] = 1 } else in_br = 1; next }
top == "on" && ontrig == "push" && in_br && raw ~ /^      -[ \t]*/ { if (unq(substr(line, 3)) == "main") push_main[wf] = 1; next }
top == "on" && (ontrig == "push" || ontrig == "pull_request") && raw ~ /^    paths:/ { v = trim(substr(line, index(line, ":")+1)); in_br = 0; in_paths = 0; if (v != "") { n = split(norm_list(v), pa, " "); for (i=1; i<=n; i++) if (pa[i]!="") ppaths[wf "|" ontrig "|" pa[i]] = 1 } else in_paths = 1; next }
top == "on" && in_paths && raw ~ /^      -[ \t]*/ { ppaths[wf "|" ontrig "|" unq(substr(line, 3))] = 1; next }
top == "on" && raw ~ /^    [A-Za-z0-9_-]+:/ { in_br = 0; in_paths = 0; next }
top == "jobs" && raw ~ /^  [A-Za-z0-9_-]+:[ \t]*$/ { job = line; sub(/:.*/, "", job); jobs[wf "|" job] = 1; jsec = ""; sidx = 0; in_mat = 0; next }
top == "jobs" && job != "" && raw ~ /^    if:[ \t]*/ { v = trim(substr(line, index(line, ":")+1)); jsec = (v == ">-" || v == "|-") ? "ifml" : ""; if (jsec == "") jif[wf "|" job] = v; next }
top == "jobs" && jsec == "ifml" && raw ~ /^      / { jif[wf "|" job] = trim(jif[wf "|" job] " " line); next }
top == "jobs" && job != "" && raw ~ /^    needs:[ \t]*/ { v = trim(substr(line, index(line, ":")+1)); if (v != "") { needs[wf "|" job] = norm_list(v); jsec = "" } else jsec = "needs"; next }
top == "jobs" && jsec == "needs" && raw ~ /^      -[ \t]*/ { needs[wf "|" job] = trim(needs[wf "|" job] " " unq(substr(line, 3))); next }
top == "jobs" && job != "" && raw ~ /^    runs-on:[ \t]*/ { v = trim(substr(line, index(line, ":")+1)); if (v != "") { rawattr[wf "|" job "|runs-on"] = v; jsec = "" } else { jsec = "runs-on"; rawattr[wf "|" job "|runs-on"] = "" }; next }
top == "jobs" && jsec == "runs-on" && raw ~ /^      -[ \t]*/ { v = unq(substr(line, 3)); cur = rawattr[wf "|" job "|runs-on"]; rawattr[wf "|" job "|runs-on"] = (cur == "" ? "[" v "]" : substr(cur, 1, length(cur)-1) "," v "]"); next }
top == "jobs" && job != "" && raw ~ /^    (uses|container|image):[ \t]*/ { pinv[wf "|" job, ++npin[wf "|" job]] = trim(substr(line, index(line, ":")+1)); jsec = ""; next }
top == "jobs" && job != "" && raw ~ /^    env:[ \t]*$/ { jsec = "env"; next }
top == "jobs" && jsec == "env" && raw ~ /^      [A-Za-z0-9_-]+:[ \t]*/ { k = line; sub(/:.*/, "", k); rawattr[wf "|" job "|env." k] = trim(substr(line, index(line, ":")+1)); next }
top == "jobs" && job != "" && raw ~ /^    strategy:[ \t]*$/ { jsec = "strategy"; in_mat = 0; next }
top == "jobs" && jsec == "strategy" && raw ~ /^      matrix:[ \t]*$/ { in_mat = 1; next }
top == "jobs" && jsec == "strategy" && in_mat && raw ~ /^        [A-Za-z0-9_-]+:[ \t]*/ { k = line; sub(/:.*/, "", k); v = trim(substr(line, index(line, ":")+1)); if (k != "include" && k != "exclude" && v != "") rawattr[wf "|" job "|matrix." k] = v; next }
top == "jobs" && job != "" && raw ~ /^    steps:[ \t]*$/ { jsec = "steps"; sidx = 0; next }
top == "jobs" && jsec == "steps" && raw ~ /^      -[ \t]*/ { sidx++; nstep[wf "|" job] = sidx; v = trim(substr(line, 2)); if (v ~ /^name:[ \t]*/) sname[wf "|" job, sidx] = unq(substr(v, 6)); else if (v ~ /^uses:[ \t]*/) { s = trim(substr(v, 6)); suses[wf "|" job, sidx] = s; pinv[wf "|" job, ++npin[wf "|" job]] = s } else if (v ~ /^if:[ \t]*/) sif[wf "|" job, sidx] = trim(substr(v, 4)); next }
top == "jobs" && jsec == "steps" && sidx > 0 && raw ~ /^        (name|uses|if):[ \t]*/ { k = line; sub(/:.*/, "", k); v = trim(substr(line, index(line, ":")+1)); if (k == "name") sname[wf "|" job, sidx] = unq(v); else if (k == "uses") { suses[wf "|" job, sidx] = v; pinv[wf "|" job, ++npin[wf "|" job]] = v } else if (k == "if") sif[wf "|" job, sidx] = v; next }
top == "jobs" && jsec == "steps" && sidx > 0 && raw ~ /^          toolchain:[ \t]*/ { pinv[wf "|" job, ++npin[wf "|" job]] = trim(substr(line, index(line, ":")+1)); next }
top == "jobs" && job != "" && raw ~ /^    [A-Za-z0-9_-]+:/ { jsec = ""; in_mat = 0; next }
END {
  for (w in push_main) if (push_main[w]) { pwfs[w] = 1; n_pw++ }
  if (n_pw == 0) { print "one-gate: no push-to-main workflows found (fail closed)" > "/dev/stderr"; exit 2 }
  split("push pull_request merge_group", EVS, " ")
  for (key in jobs) { split(key, p, "|"); w = p[1]; j = p[2]; if (!(w in pwfs)) continue; wf_ = w; job_ = j; for (e = 1; e <= 3; e++) ran[key "|" EVS[e]] = (((w "|" EVS[e]) in trig) && ifval(jif[key], EVS[e])) ? 1 : 0 }
  for (rk in req) { split(rk, q, "|"); hasreq[q[1]] = 1; stack[++sn] = rk; inclos[rk] = 1 }
  while (sn > 0) { cur = stack[sn--]; n = split(needs[cur], nn, " "); split(cur, cp, "|"); for (i = 1; i <= n; i++) { nk = cp[1] "|" nn[i]; if (!(nk in inclos)) { inclos[nk] = 1; stack[++sn] = nk } } }
  for (key in jobs) {
    split(key, p, "|"); w = p[1]; j = p[2]; if (!(w in pwfs) || !ran[key "|push"]) continue; wf_ = w; job_ = j
    if (!(ran[key "|pull_request"] || ran[key "|merge_group"])) viol("push-only", w, j, "-", "runs on push to main but on neither pull_request nor merge_group")
    if ((w in hasreq) && !(key in inclos)) viol("not-required", w, j, "-", "runs on push to main but not in needs closure of any required job")
    for (e = 2; e <= 3; e++) {
      ev = EVS[e]; if (!ran[key "|" ev]) continue
      for (ak in rawattr) { split(ak, ap, "|"); if (ap[1] == w && ap[2] == j) { a1 = resolve(rawattr[ak], "push"); a2 = resolve(rawattr[ak], ev); if (a1 != a2) viol("attr", w, j, ap[3], ap[3] " differs: push=" a1 " vs " ev "=" a2) } }
      for (si = 1; si <= nstep[key]; si++) { snm = (sname[key, si] != "" ? sname[key, si] : (suses[key, si] != "" ? suses[key, si] : "#" si)); if (ifval(sif[key, si], "push") && !ifval(sif[key, si], ev)) viol("step", w, j, snm, "step runs on push but not on " ev) }
    }
    for (ak in rawattr) { split(ak, ap, "|"); if (ap[1] == w && ap[2] == j && resolve(rawattr[ak], "push") == "UNPARSEABLE") viol("attr", w, j, ap[3], ap[3] " unparseable (fail closed)") }
    for (i = 1; i <= npin[key]; i++) if (pinv[key, i] ~ /github\.event_name/) viol("pin", w, j, "-", "event-dependent pin: " pinv[key, i])
  }
  for (key in unparse) { split(key, p, "|"); viol("attr", p[1], p[2], "if", "if unparseable: " unparse[key]) }
  for (pk in ppaths) { split(pk, q, "|"); if (q[2] == "push" && (q[1] "|pull_request") in trig) { has_pr = 0; for (qk in ppaths) { split(qk, qq, "|"); if (qq[1] == q[1] && qq[2] == "pull_request") { has_pr = 1; break } }; if (has_pr && !((q[1] "|pull_request|" q[3]) in ppaths)) viol("paths", q[1], "-", q[3], "push path missing from pull_request paths: " q[3]) } }
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
mkdir -p "$st/empty"; rc=0; "$0" --workflows "$st/empty" --allow "$st/allow.txt" >"$st/out" 2>&1 || rc=$?
[ "$rc" -eq 2 ] && printf 'ok   V1 empty workflows fails closed (rc=%s)\n' "$rc" || { printf 'FAIL V1: want rc=2 got rc=%s\n' "$rc"; fail=1; }
[ "$fail" -eq 0 ] || { echo "one-gate self-test: FAILED"; exit 1; }
echo "one-gate self-test: all cases pass"
