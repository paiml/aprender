#!/usr/bin/env bash
# ONE-GATE parity falsifier (APR-GREEN-001 item 2, 2026-11-21).
#
# Root cause it closes: #7356 passed its PR checks and then broke main, because
# main ran jobs/shards/runner labels the PR-side gate did not. This script reads
# every workflow that triggers on a push to main and diffs what runs on that push
# against what runs on the PR side (pull_request + merge_group, the merge queue
# being the required pre-merge run). It FAILS on ANY difference that is not listed,
# with a reason, in .github/one-gate-allow.txt. Nothing is matched by pattern: a
# difference it cannot positively find in the allow-list is RED (fail closed), and
# an allow-list line that no longer matches a difference is RED too (no stale waivers).
#
# Differences checked, per workflow that has `on: push` to main:
#   push-only    a job that runs on push but on neither pull_request nor merge_group
#   not-required a job that runs on push but is not in the `needs` closure of a
#                required verdict job (`required:` lines; only for workflows that list one)
#   attr         runs-on labels, matrix values or env values of a job differ between
#                push and the PR side (resolved per event, so a ternary on
#                github.event_name is seen)
#   step         a step runs on push but not on the PR side (resolved `if`)
#   paths        a push `paths` filter entry missing from the PR `paths` filter
#   pin          an event-dependent tool pin (uses/container/image containing
#                github.event_name)
#
# Usage: check_one_gate_parity.sh [--workflows DIR] [--allow FILE]
#        check_one_gate_parity.sh --self-test     # planted-defect mutants, must go RED
# Exit: 0 parity, 1 differences, 2 usage / unreadable input.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
wf_dir="$root/.github/workflows"
allow="$root/.github/one-gate-allow.txt"
mode=check

while [ $# -gt 0 ]; do
  case "$1" in
    --workflows) wf_dir=$2; shift 2 ;;
    --allow) allow=$2; shift 2 ;;
    --self-test) mode=self-test; shift ;;
    *) echo "usage: $0 [--workflows DIR] [--allow FILE] | --self-test" >&2; exit 2 ;;
  esac
done

command -v yq >/dev/null 2>&1 || { echo "one-gate: yq not found" >&2; exit 2; }

# emit_records DIR: tab-separated records, one fact per line:
#   wf <TAB> job <TAB> kind <TAB> value [<TAB> paths for kind=trigger]
emit_records() {
  local dir=$1 f wf br ev
  for f in "$dir"/*.yml; do
    [ -f "$f" ] || continue
    wf=$(basename "$f")
    # Only workflows that trigger on a push to main are judged.
    yq -e '.on.push' "$f" >/dev/null 2>&1 || continue
    br=$(yq -o=json -I=0 '.on.push.branches // ["main"]' "$f")
    case "$br" in *'"main"'*) ;; *) continue ;; esac
    printf '%s\t-\ttrigger\tpush\t%s\n' "$wf" "$(yq -o=json -I=0 '.on.push.paths // []' "$f")"
    for ev in pull_request merge_group; do
      if yq -e ".on | has(\"$ev\")" "$f" >/dev/null 2>&1; then
        printf '%s\t-\ttrigger\t%s\t%s\n' "$wf" "$ev" "$(yq -o=json -I=0 ".on.$ev.paths // []" "$f")"
      fi
    done
    yq -r '.jobs | to_entries | .[] | .key as $j | (
        [$j, "if", (.value.if // "")],
        [$j, "runs-on", (.value["runs-on"] | to_json(0))],
        [$j, "needs", ((.value.needs // []) | [.] | flatten | join(" "))],
        [$j, "pin", ((.value.uses // "") + " " + ((.value.container // "") | tostring) + " " + ((.value.image // "") | tostring))],
        ((.value.strategy.matrix // {}) | to_entries | .[] | select(.key != "include" and .key != "exclude") | [$j, "matrix." + .key, (.value | to_json(0))]),
        ((.value.env // {}) | to_entries | .[] | [$j, "env." + .key, (.value | tostring)]),
        ((.value.steps // []) | to_entries | .[] | [$j, "step", ((.value.if // "") | tostring) + "@@" + ((.value.name // .value.uses // ("#" + (.key | tostring))) | tostring)]),
        ((.value.steps // []) | .[] | select(.uses != null) | [$j, "pin", (.uses + " " + ((.with.toolchain // "") | tostring))]),
        ((.value.steps // []) | .[] | select(((.run // "") | tostring) | test("--sections")) | [$j, "sect", ((.if // "") | tostring) + "@@" + (.run | sub("\n"; "~NL~"))])
      ) | join("\t")' "$f" | awk -v wf="$wf" -F'\t' 'NF >= 3 && $1 != "" { printf "%s\t%s\t%s\t%s\n", wf, $1, $2, $3 }'
  done
}

# The resolver and comparer. Pure awk: a small GitHub-expression evaluator (JS-like
# && / ||, ==, !=, !, calls), evaluated once per event, then the diff.
# Unknown contexts: needs.*.result = success, needs.*.outputs.* / vars.* / steps.* = ''.
compare_awk() {
  cat <<'AWK'
function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
function truthy(v) { return (v != "" && v != "false" && v != "0") }
function lookup(n) {
  if (n == "true") return "true"
  if (n == "false") return "false"
  if (n == "github.event_name") return EV
  if (n == "github.ref") return (EV == "push" ? "refs/heads/main" : (EV == "pull_request" ? "refs/pull/1/merge" : "refs/heads/gh-readonly-queue/main/pr-1"))
  if (n == "github.ref_name") return (EV == "push" ? "main" : "")
  if (n == "github.repository") return "paiml/aprender"
  if (n == "github.event.pull_request.draft") return (EV == "pull_request" ? "false" : "")
  if (n == "github.event.pull_request.head.repo.full_name") return (EV == "pull_request" ? "paiml/aprender" : "")
  if (n ~ /^needs\.[^.]+\.result$/) return "success"
  if (n ~ /^[0-9]+$/) return n
  return ""
}
function tokenize(s,    m) {
  nt = 0; s = trim(s)
  while (s != "") {
    if (match(s, /^'([^']|'')*'/)) { m = substr(s, 2, RLENGTH - 2); gsub(/''/, "'", m); tk[++nt] = "S" m; s = substr(s, RLENGTH + 1) }
    else if (match(s, /^(==|!=|&&|\|\|)/)) { tk[++nt] = "O" substr(s, 1, 2); s = substr(s, 3) }
    else if (match(s, /^[!(),]/)) { tk[++nt] = "O" substr(s, 1, 1); s = substr(s, 2) }
    else if (match(s, /^[A-Za-z0-9_.*-]+/)) { tk[++nt] = "I" substr(s, 1, RLENGTH); s = substr(s, RLENGTH + 1) }
    else if (match(s, /^[ \t]+/)) { s = substr(s, RLENGTH + 1) }
    else { bad = 1; return }
  }
}
function isop(o) { return (tp <= nt && tk[tp] == "O" o) }
function pOr(   v, w) { v = pAnd(); while (isop("||")) { tp++; w = pAnd(); if (!truthy(v)) v = w } return v }
function pAnd(   v, w) { v = pCmp(); while (isop("&&")) { tp++; w = pCmp(); if (truthy(v)) v = w } return v }
function pCmp(   v, w, o) {
  v = pPrim()
  while (isop("==") || isop("!=")) { o = substr(tk[tp], 2); tp++; w = pPrim(); v = ((o == "==") == (v == w)) ? "true" : "false" }
  return v
}
function pPrim(   t, n, v, a) {
  if (tp > nt) { bad = 1; return "" }
  t = tk[tp++]
  if (t == "O!") { v = pPrim(); return truthy(v) ? "false" : "true" }
  if (t == "O(") { v = pOr(); if (isop(")")) tp++; else bad = 1; return v }
  if (substr(t, 1, 1) == "S") return substr(t, 2)
  if (substr(t, 1, 1) == "I") {
    n = substr(t, 2)
    if (isop("(")) {
      tp++; a = ""
      if (!isop(")")) { a = pOr(); while (isop(",")) { tp++; pOr() } }
      if (isop(")")) tp++; else bad = 1
      if (n == "cancelled" || n == "failure") return "false"
      if (n == "always" || n == "success" || n == "contains" || n == "startsWith" || n == "endsWith") return "true"
      if (n == "fromJSON" || n == "toJSON" || n == "format") return a
      return ""
    }
    return lookup(n)
  }
  bad = 1; return ""
}
function ev_expr(e, ev,    v) {
  EV = ev; bad = 0; tokenize(e); tp = 1
  if (bad) return "UNPARSEABLE"
  v = pOr()
  if (bad || tp <= nt) return "UNPARSEABLE"
  gsub(/ /, "", v)
  return v
}
# resolve(raw): a whole-value ${{ expr }} is evaluated; anything else is literal.
function resolve(raw, ev,    e) {
  raw = trim(raw)
  if (substr(raw, 1, 1) == "\"" && substr(raw, length(raw), 1) == "\"") { raw = substr(raw, 2, length(raw) - 2); gsub(/\\"/, "\"", raw) }
  if (match(raw, /^\$\{\{.*\}\}$/)) { e = substr(raw, 4, length(raw) - 5); return ev_expr(e, ev) }
  gsub(/ /, "", raw)
  return raw
}
function ifval(raw, ev,    e) {
  raw = trim(raw)
  if (raw == "") return 1
  if (match(raw, /^\$\{\{.*\}\}$/)) raw = substr(raw, 4, length(raw) - 5)
  e = ev_expr(raw, ev)
  if (e == "UNPARSEABLE") { unparse[wf_ "|" job_] = raw; return 1 }
  return truthy(e)
}
# sections_of(key, ev): the fat_driver --sections items the job runs on event ev.
# A `run` block may branch on "$EVT" (if [ "$EVT" = X ]; then ... else ... fi): the branch is honoured.
function add_items(list, ev, key,    i, c, depth, cur, n) {
  gsub(/\$\{\{[^}]*\}\}/, "{}", list)
  depth = 0; cur = ""
  for (i = 1; i <= length(list) + 1; i++) {
    c = substr(list, i, 1)
    if (c == "[") depth++
    if (c == "]") depth--
    if ((c == "," && depth == 0) || i > length(list)) {
      if (cur != "") { secset[ev "|" cur] = 1; secjob[ev "|" cur] = key }
      cur = ""
    } else cur = cur c
  }
}
function sections_of(key, ev,    i, sv, sp, sif, run, nl, lines, l, ctx, X, m, lst) {
  for (i = 1; i <= nsect[key]; i++) {
    sv = sectraw[key, i]; sp = index(sv, "@@"); sif = substr(sv, 1, sp - 1); run = substr(sv, sp + 2)
    if (!ifval(sif, ev)) continue
    nl = split(run, lines, "~NL~"); ctx = "all"; X = ""
    for (l = 1; l <= nl; l++) {
      if (match(lines[l], /\[ "\$EVT" = [a-z_]+ \]/)) { m = substr(lines[l], RSTART, RLENGTH); sub(/.*= /, "", m); sub(/ \].*/, "", m); X = m; ctx = "is"; continue }
      if (lines[l] ~ /^[ ]*else[ ]*$/) { ctx = "not"; continue }
      if (lines[l] ~ /^[ ]*fi[ ]*$/) { ctx = "all"; continue }
      if (match(lines[l], /--sections +'[^']*'/)) {
        lst = substr(lines[l], RSTART, RLENGTH); sub(/^--sections +'/, "", lst); sub(/'$/, "", lst)
        if (ctx == "all" || (ctx == "is" && X == ev) || (ctx == "not" && X != ev)) add_items(lst, ev, key)
      }
    }
  }
}
function viol(kind, w, j, d, msg,    key) {
  gsub(/ /, "_", d)
  key = kind "|" w "|" j "|" d
  if (key in allowed) { used[key] = 1; return }
  printf "RED %s: %s %s: %s  [allow key: allow %s %s %s %s]\n", kind, w, (j == "-" ? "" : j), msg, kind, w, j, d
  bad_n++
}
FNR == 1 { fileno++ }
# file 1: allow-list
fileno == 1 {
  if ($0 ~ /^[ \t]*(#|$)/) next
  na = split($0, a, " ")
  if (a[1] == "required") { req[a[2] "|" a[3]] = 1; next }
  if (a[1] == "allow") {
    key = a[2] "|" a[3] "|" a[4] "|" a[5]
    if (index($0, " -- ") == 0 || a[6] != "--" || na < 7) { printf "RED allow-list line without a reason (-- <reason>): %s\n", $0; bad_n++ }
    allowed[key] = 1; next
  }
  printf "RED allow-list: unknown line kind: %s\n", $0; bad_n++; next
}
# file 2: records
fileno == 2 {
  w = $1; j = $2; k = $3; v = $4
  for (i = 5; i <= NF && k != "trigger"; i++) v = v "\t" $i
  wfs[w] = 1
  if (k == "trigger") { trig[w "|" v] = 1; tpaths[w "|" v] = $5; next }
  jobs[w "|" j] = 1
  if (k == "if") jif[w "|" j] = v
  else if (k == "needs") needs[w "|" j] = v
  else if (k == "pin") { npin[w "|" j]++; pinv[w "|" j, npin[w "|" j]] = v }
  else if (k == "runs-on" || k ~ /^matrix\./ || k ~ /^env\./) rawattr[w "|" j "|" k] = v
  else if (k == "sect") { nsect[w "|" j]++; sectraw[w "|" j, nsect[w "|" j]] = v }
  else if (k == "step") { nstep[w "|" j]++; stepraw[w "|" j, nstep[w "|" j]] = v }
  next
}
END {
  split("push pull_request merge_group", EVS, " ")
  for (key in jobs) {
    split(key, p, "|"); w = p[1]; j = p[2]; wf_ = w; job_ = j
    for (e = 1; e <= 3; e++) { ev = EVS[e]; ran[key "|" ev] = (((w "|" ev) in trig) && ifval(jif[key], ev)) ? 1 : 0 }
  }
  for (rk in req) { split(rk, q, "|"); hasreq[q[1]] = 1; stack[++sn] = rk; inclos[rk] = 1 }
  while (sn > 0) {
    cur = stack[sn--]
    n = split(needs[cur], nn, " ")
    split(cur, cp, "|")
    for (i = 1; i <= n; i++) { nk = cp[1] "|" nn[i]; if (!(nk in inclos)) { inclos[nk] = 1; stack[++sn] = nk } }
  }
  for (key in jobs) {
    split(key, p, "|"); w = p[1]; j = p[2]; wf_ = w; job_ = j
    if (!ran[key "|push"]) continue
    if (!(ran[key "|pull_request"] || ran[key "|merge_group"]))
      viol("push-only", w, j, "-", "runs on push to main but on neither pull_request nor merge_group")
    if ((w in hasreq) && !(key in inclos))
      viol("not-required", w, j, "-", "runs on push to main but no required verdict job (required: lines) depends on it")
    for (e = 2; e <= 3; e++) {
      ev = EVS[e]
      if (!ran[key "|" ev]) continue
      for (ak in rawattr) {
        split(ak, ap, "|")
        if (ap[1] != w || ap[2] != j) continue
        a1 = resolve(rawattr[ak], "push"); a2 = resolve(rawattr[ak], ev)
        if (a1 != a2) viol("attr", w, j, ap[3], ap[3] " differs: push=" a1 " vs " ev "=" a2)
      }
      n = nstep[key]
      for (si = 1; si <= n; si++) {
        sv = stepraw[key, si]; sp = index(sv, "@@"); sif = substr(sv, 1, sp - 1); sname = substr(sv, sp + 2)
        if (ifval(sif, "push") && !ifval(sif, ev)) viol("step", w, j, sname, "step runs on push but not on " ev)
      }
    }
    for (ak in rawattr) {
      split(ak, ap, "|")
      if (ap[1] == w && ap[2] == j && resolve(rawattr[ak], "push") == "UNPARSEABLE")
        viol("attr", w, j, ap[3], ap[3] " expression not understood by the evaluator (fail closed)")
    }
    n = npin[key]
    for (i = 1; i <= n; i++)
      if (pinv[key, i] ~ /github\.event_name/) viol("pin", w, j, "-", "event-dependent tool pin: " pinv[key, i])
  }
  # section SET per event: a fat_driver section that push runs (inside any job) must be run by some job on
  # pull_request or merge_group too, after each job's `if` and each step's `if`/"$EVT" branch are resolved.
  for (key in jobs) {
    split(key, p, "|")
    if (!(key in nsect)) continue
    for (e = 1; e <= 3; e++) { ev = EVS[e]; if (ran[key "|" ev]) sections_of(key, ev) }
  }
  for (sk in secset) {
    split(sk, sp2, "|")
    if (sp2[1] != "push") continue
    sec = substr(sk, 6)
    if (!(("pull_request|" sec) in secset) && !(("merge_group|" sec) in secset)) {
      split(secjob[sk], jp, "|")
      viol("section", jp[1], jp[2], sec, "section runs on push to main (job " jp[2] ") but on neither pull_request nor merge_group")
    }
  }
  for (key in unparse) { split(key, p, "|"); viol("attr", p[1], p[2], "if", "if expression not understood by the evaluator (fail closed): " unparse[key]) }
  # paths: every push path entry must appear in the PR path filter (when the PR has one)
  for (w in wfs) {
    if (!((w "|pull_request") in trig)) continue
    pp = tpaths[w "|push"]; rp = tpaths[w "|pull_request"]
    if (pp == "[]" || pp == "" || rp == "[]" || rp == "") continue
    gsub(/^\[|\]$/, "", pp); n = split(pp, pe, /","|",|,"/)
    for (i = 1; i <= n; i++) { ent = pe[i]; gsub(/"/, "", ent); if (ent != "" && index(rp, "\"" ent "\"") == 0) viol("paths", w, "-", ent, "push path filter entry missing from pull_request paths: " ent) }
  }
  for (key in allowed) if (!(key in used)) { printf "RED stale allow-list line (matches no difference): allow %s\n", key; bad_n++ }
  if (bad_n > 0) { printf "one-gate parity: %d difference(s)\n", bad_n; exit 1 }
  print "one-gate parity: OK (push-to-main job set == PR-side job set, every difference accounted for)"
}
AWK
}

run_check() {
  local dir=$1 al=$2 recs rc
  [ -d "$dir" ] || { echo "one-gate: no workflows dir $dir" >&2; return 2; }
  [ -r "$al" ] || { echo "one-gate: unreadable allow-list $al" >&2; return 2; }
  recs=$(mktemp)
  emit_records "$dir" >"$recs"
  if [ ! -s "$recs" ]; then rm -f "$recs"; echo "one-gate: no push-to-main workflows found in $dir (fail closed)" >&2; return 2; fi
  prog=$(mktemp)
  compare_awk >"$prog"
  rc=0
  awk -F'\t' -f "$prog" "$al" "$recs" || rc=$?
  rm -f "$recs" "$prog"
  return "$rc"
}

if [ "$mode" = check ]; then
  rc=0
  run_check "$wf_dir" "$allow" || rc=$?
  exit "$rc"
fi

# ---------------------------------------------------------------- self-test
st=$(mktemp -d)
[ -n "${st:-}" ] && [ -d "$st" ] || { echo "one-gate: mktemp failed" >&2; exit 2; }
trap '[ -n "${st:-}" ] && rm -rf "${st:?}"' EXIT
fail=0
base_ci() {
  cat <<'YML'
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  merge_group:
jobs:
  build:
    runs-on: [self-hosted, Linux, X64, clean-room]
    if: ${{ github.event.pull_request.draft != true }}
    steps:
      - name: compile
        run: echo hi
  shard:
    needs: [build]
    runs-on: [self-hosted, Linux, X64, clean-room]
    strategy:
      matrix:
        shard: ${{ fromJSON('[1,2,3]') }}
    steps:
      - name: test
        run: echo hi
  gate:
    needs: [build, shard]
    if: always()
    runs-on: [self-hosted, Linux, X64, clean-room]
    steps:
      - name: verdict
        run: echo hi
YML
}
# base_sect: a job that runs fat_driver sections: push runs 4, the queue runs 2 plus the `guards` job's 2,
# a PR runs only the PR-only pair. Spliced in (and made required) when the 5th case_run argument is "sect".
base_sect() {
  cat <<'YML'
  fat:
    needs: [build]
    if: ${{ github.event_name != 'pull_request' }}
    runs-on: [self-hosted, Linux, X64, clean-room]
    steps:
      - name: Sections
        env:
          EVT: ${{ github.event_name }}
        run: |
          if [ "$EVT" = merge_group ]; then
            python3 scripts/ci/fat_driver.py run --sections 'sov.*,determinism[X64]'
          else
            python3 scripts/ci/fat_driver.py run --sections 'sov.*,guard-cargo,determinism[X64]'
          fi
  guards:
    needs: [build]
    if: ${{ github.event_name == 'merge_group' }}
    runs-on: [self-hosted, Linux, X64, clean-room]
    steps:
      - name: Sections
        run: python3 scripts/ci/fat_driver.py run --sections 'guard-cargo'
YML
}
case_run() { # NAME WANT-RC ALLOW-EXTRA SED-MUTATION [sect]
  local name=$1 want=$2 extra=$3 mut=$4 rc
  [ -n "${st:-}" ] && rm -rf "${st:?}/wf"; mkdir -p "$st/wf"
  base_ci >"$st/wf/ci.yml"
  if [ "${5:-}" = sect ]; then
    base_sect >>"$st/wf/ci.yml"
    sed -i 's/^    needs: \[build, shard\]$/    needs: [build, shard, fat, guards]/' "$st/wf/ci.yml"
  fi
  if [ -n "$mut" ]; then sed -i "$mut" "$st/wf/ci.yml"; fi
  printf 'required ci.yml gate\n%s' "$extra" >"$st/allow.txt"
  rc=0; "$0" --workflows "$st/wf" --allow "$st/allow.txt" >"$st/out" 2>&1 || rc=$?
  if [ "$rc" = "$want" ]; then printf 'ok   %s (rc=%s)\n' "$name" "$rc"
  else printf 'FAIL %s: want rc=%s got rc=%s\n' "$name" "$want" "$rc"; sed 's/^/     /' "$st/out"; fail=1; fi
}
Q="'"
case_run "baseline parity holds" 0 "" ""
case_run "M1 job only on push" 1 "" "s/^  shard:\$/  shard:\n    if: github.event_name == ${Q}push${Q}/"
case_run "M2 runs-on label differs on PR" 1 "" "0,/^  shard:\$/! s/^    runs-on: \[self-hosted, Linux, X64, clean-room\]\$/    runs-on: \${{ github.event_name == ${Q}pull_request${Q} \&\& fromJSON(${Q}[\"self-hosted\",\"Linux\",\"clean-room\"]${Q}) || fromJSON(${Q}[\"self-hosted\",\"Linux\",\"X64\",\"clean-room\"]${Q}) }}/"
case_run "M3 shard matrix differs push vs merge_group" 1 "" "s/fromJSON(${Q}\[1,2,3\]${Q})/fromJSON(github.event_name == ${Q}merge_group${Q} \&\& ${Q}[1,2,3]${Q} || ${Q}[1,2,3,4]${Q})/"
case_run "M4 job not under any required verdict" 1 "" 's/^    needs: \[build, shard\]$/    needs: [build]/'
case_run "M5 step only on push" 1 "" "s/^      - name: test\$/      - if: github.event_name == ${Q}push${Q}\n        name: test/"
case_run "M6 event-dependent pin" 1 "" "s/^  build:\$/  build:\n    container: \${{ github.event_name == ${Q}push${Q} \&\& ${Q}img:1${Q} || ${Q}img:2${Q} }}/"
case_run "M7 unparseable if fails closed" 1 "" 's/^    if: always()$/    if: ${{ @@@ }}/'
case_run "M8 env value differs by event" 1 "" "s/^  build:\$/  build:\n    env:\n      X: \${{ github.event_name == ${Q}push${Q} \&\& ${Q}a${Q} || ${Q}b${Q} }}/"
case_run "M9 stale allow line is RED" 1 "allow attr ci.yml build runs-on -- no such difference
" ""
case_run "M10 allow line without reason is RED" 1 "allow push-only ci.yml shard -
" "s/^  shard:\$/  shard:\n    if: github.event_name == ${Q}push${Q}/"
case_run "A1 listed difference is accepted" 0 "allow push-only ci.yml shard - -- planted: reviewed exception
" "s/^  shard:\$/  shard:\n    if: github.event_name == ${Q}push${Q}/"
case_run "S0 section set equal (push sections all run by the queue)" 0 "" "" sect
case_run "S1 section only on push (queue drops guard-cargo)" 1 "" "s/^    if: \${{ github.event_name == ${Q}merge_group${Q} }}\$/    if: \${{ github.event_name == ${Q}push${Q} }}/" sect
case_run "S2 section removed from the queue list and the guards job" 1 "" "s/--sections ${Q}guard-cargo${Q}/--sections ${Q}determinism[X64]${Q}/;s/'sov.\*,determinism\[X64\]'/'sov.*'/" sect
case_run "S3 push-only section accepted when listed with a reason" 0 "allow section ci.yml fat guard-cargo -- planted: reviewed exception
" "s/--sections ${Q}guard-cargo${Q}/--sections ${Q}determinism[X64]${Q}/;s/'sov.\*,determinism\[X64\]'/'sov.*'/" sect
if [ "$fail" -ne 0 ]; then echo "one-gate self-test: FAILED"; exit 1; fi
echo "one-gate self-test: all cases behave (baseline holds, every planted defect is RED)"
