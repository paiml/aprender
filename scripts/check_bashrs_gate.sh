#!/usr/bin/env bash
# check_bashrs_gate.sh — the RELEASE's bashrs gate, run on every pull request.
#
# WHY THIS EXISTS (#3196)
# -----------------------
# scripts/dogfood.sh carries a `bashrs` row that fails the pre-publish dogfood on
# any SEC/DET/IDEM error over the whole shell surface. Nothing ran that filter on
# a PR, so the 0.67.0 train was stopped by it THREE times in one day:
#
#   17:33Z  9 findings  (#3115, #3127)     fixed by #3188
#   20:58Z  2 findings  (#3187)            fixed by #3194
#   22:48Z  2 findings  (#3068)            fixed by #3198
#
# Each fix went round the merge queue while the tag waited. The gate is right;
# its placement was not. This script is the same gate — same enumeration, same
# positive control, same code filter — placed where the finding is cheap: on
# the PR that introduces it.
#
# WHAT IT JUDGES (identical to dogfood.sh, and the self-test asserts that)
# ------------------------------------------------------------------------
#   surface   git ls-files '*.sh' '*.bash' Makefile '*/Makefile' '**/*.sh'
#             enumerated HERE, so a .bashrsignore cannot silently zero it
#   invocation bashrs lint --no-ignore --level error --format json <all> <clean sentinel>
#             one argv, no xargs (xargs remaps exits 1..125 to 123), and the
#             receipt line "Linted N+1 file(s)" must match our own count
#   positive  a sentinel with a known DET002 (timestamp into an artifact name)
#   control   must fire first, or the run is refused — a silent tool is not clean
#   verdict   error-severity diagnostics whose code starts SEC/DET/IDEM are
#             GATING; SC1020/SC1035/SC1140 are the bashrs#226 false-positive
#             class and only reported; every other error is reported, not gating
#
# Exit codes — never read bashrs's own exit as the verdict (0/1/2 are overloaded):
#   0  surface linted, receipt matched, zero gating findings
#   1  gating findings (each printed as  file:line CODE)
#   2  environment or vacuity: bashrs absent, positive control silent, receipt
#      count mismatch, empty surface — a gate that did not run is not a pass
#
# Usage:
#   bash scripts/check_bashrs_gate.sh [--root DIR]      # gate the repo (default: this one)
#   bash scripts/check_bashrs_gate.sh --classify F.json # print "gating soft other rules" + findings
#   bash scripts/check_bashrs_gate.sh --self-test       # case table, incl. a must-RED fixture repo
set -euo pipefail

SELF=$(readlink -f "$0")
SOFT_CODES="('SC1020', 'SC1035', 'SC1140')"
GATING_PREFIXES="('SEC', 'DET', 'IDEM')"

strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }

# classify <json-file>: first line "GATING SOFT OTHER RULE..." then one "file:line CODE" per gating finding
classify() {
  python3 - "$1" <<'PY'
import json, sys
raw = open(sys.argv[1]).read()
i = raw.find('{')
raw = raw[i:] if i >= 0 else ''
dec, pos, gating, soft, other, rules, hits = json.JSONDecoder(), 0, 0, 0, 0, set(), []
while pos < len(raw):
    while pos < len(raw) and raw[pos] in ' \t\r\n': pos += 1
    if pos >= len(raw): break
    obj, pos = dec.raw_decode(raw, pos)
    f = obj.get('file', '?')
    for d in obj.get('diagnostics', []):
        if d.get('severity') != 'error': continue
        c = d.get('code', '')
        if c.startswith(('SEC', 'DET', 'IDEM')):
            gating += 1; rules.add(c)
            hits.append(f"{f}:{(d.get('span') or {}).get('start_line', '?')} {c}")
        elif c in ('SC1020', 'SC1035', 'SC1140'): soft += 1
        else: other += 1
print(f"{gating} {soft} {other} {' '.join(sorted(rules))}".rstrip())
for h in hits: print(h)
PY
}

mk_tmp() {
  local d; d=$(mktemp -d)
  if [ -z "$d" ] || [ ! -d "$d" ] || [[ "$d" == *..* ]] || [[ "$d" != /* ]]; then
    echo "refused: mktemp gave an unusable directory" >&2; exit 2
  fi
  printf '%s' "$d"
}

# positive_control <dir>: the dirty sentinel MUST fire DET002 with exit 2. Returns 0 if it did.
positive_control() {
  local d="$1" rc=0
  # bashrs >= 6.67 flags a timestamp only when it reaches a REPRODUCIBLE sink (an artifact
  # name); one that merely echoes is correctly silent, so the sentinel names an artifact.
  printf '#!/bin/sh\nSTAMP=$(date +%%s%%N)\ncp build.log "out/report_$STAMP.log"\n' > "$d/dirty-sentinel.sh"
  printf '#!/bin/sh\necho ok\n' > "$d/clean-sentinel.sh"
  bashrs lint --no-ignore --level error --format json "$d/dirty-sentinel.sh" "$d/clean-sentinel.sh" > "$d/pc.json" 2> "$d/pc.err" || rc=$?
  [ "$rc" -eq 2 ] && grep -q '"DET002"' "$d/pc.json"
}

# gate <root>: enumerate, lint, classify. Prints findings and a summary. Exit 0/1/2 as documented.
gate() {
  local root="$1" d n=0 rc=0 expect summary gating soft other rules
  command -v bashrs >/dev/null 2>&1 || { echo "bashrs-gate: ENV bashrs is not on PATH - the gate did not run" >&2; return 2; }
  d=$(mk_tmp)
  if ! positive_control "$d"; then
    echo "bashrs-gate: ENV POSITIVE CONTROL SILENT - a sentinel with a known DET002 did not fire (see $d/pc.err); do not read any result from this tool as clean" >&2
    return 2
  fi
  local -a args=()
  while IFS= read -r -d '' f; do args+=("$root/$f"); n=$((n + 1)); done \
    < <(git -C "$root" ls-files -z '*.sh' '*.bash' 'Makefile' '*/Makefile' '**/*.sh')
  if [ "$n" -eq 0 ]; then
    local hidden
    hidden=$(git -C "$root" ls-files --others --ignored --exclude-standard -- '*.sh' 2>/dev/null | head -3 | tr '\n' ' ')
    echo "bashrs-gate: ENV empty surface from git ls-files${hidden:+ - shell scripts exist but are gitignored: $hidden}" >&2
    return 2
  fi
  expect=$((n + 1))
  bashrs lint --no-ignore --level error --format json "${args[@]}" "$d/clean-sentinel.sh" > "$d/out.json" 2> "$d/out.err" || rc=$?
  if ! strip_ansi < "$d/out.err" | grep -q "Linted $expect file(s)"; then
    echo "bashrs-gate: ENV receipt mismatch - expected 'Linted $expect file(s)', bashrs said: $(strip_ansi < "$d/out.err" | grep -m1 Linted || echo '<no receipt>') (exit=$rc)" >&2
    return 2
  fi
  classify "$d/out.json" > "$d/class.txt"
  summary=$(head -n1 "$d/class.txt")
  read -r gating soft other rules <<< "$summary"
  rules=${rules:-}
  if [ "$gating" -gt 0 ]; then
    tail -n +2 "$d/class.txt" | sed "s|^$root/||"
    echo "bashrs-gate: FAIL $gating SEC/DET/IDEM error(s) over $n file(s): ${rules} - real findings, not the #226 class (soft=$soft other=$other)"
    rm -rf "${d:?}"
    return 1
  fi
  echo "bashrs-gate: PASS $n file(s) linted, receipt matched, 0 SEC/DET/IDEM errors ($soft SC10xx suppressed - bashrs#226; $other other)"
  rm -rf "${d:?}"
  return 0
}

self_test() {
  local fail=0 d fx
  ok()   { echo "OK   $*"; }
  bad()  { echo "FAIL $*"; fail=1; }
  d=$(mk_tmp)

  # 1. drift gate: the two rule tuples must be byte-identical to dogfood.sh's bashrs row,
  #    or the PR gate and the release gate will disagree — which is the defect this fixes.
  local df; df=$(dirname "$SELF")/dogfood.sh
  if [ -f "$df" ]; then
    grep -qF "c.startswith($GATING_PREFIXES)" "$df" && ok "dogfood.sh gates the same prefixes $GATING_PREFIXES" || bad "dogfood.sh gating prefixes differ from $GATING_PREFIXES"
    grep -qF "c in $SOFT_CODES" "$df" && ok "dogfood.sh suppresses the same soft codes $SOFT_CODES" || bad "dogfood.sh soft codes differ from $SOFT_CODES"
    grep -qF "c.startswith($GATING_PREFIXES)" "$SELF" && grep -qF "c in $SOFT_CODES" "$SELF" && ok "this script's classifier carries both tuples verbatim" || bad "this script's classifier drifted from its own header"
  else
    bad "dogfood.sh not found beside this script - drift gate cannot run"
  fi

  # 2. positive control fires; a clean sentinel alone does not
  positive_control "$d" && ok "positive control: DET002 sentinel fires (exit 2)" || bad "positive control silent"
  local rc=0; bashrs lint --no-ignore --level error --format json "$d/clean-sentinel.sh" "$d/clean-sentinel.sh" >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && ok "clean sentinel: exit 0" || bad "clean sentinel exit=$rc"

  # 3. classifier case table on a fixture: 1 gating (SEC010), 1 soft (SC1020), 1 other (BRS0010),
  #    and a warning-severity SEC001 that must NOT count
  cat > "$d/fixture.json" <<'JSON'
{"file":"a.sh","diagnostics":[{"code":"SEC010","severity":"error","span":{"start_line":7}},{"code":"SC1020","severity":"error","span":{"start_line":3}}]}
{"file":"b.sh","diagnostics":[{"code":"BRS0010","severity":"error","span":{"start_line":1}},{"code":"SEC001","severity":"warning","span":{"start_line":2}}]}
JSON
  local got; got=$(classify "$d/fixture.json")
  [ "$(printf '%s\n' "$got" | head -n1)" = "1 1 1 SEC010" ] && ok "classify: '1 1 1 SEC010' (gating soft other rules)" || bad "classify summary: '$(printf '%s\n' "$got" | head -n1)'"
  [ "$(printf '%s\n' "$got" | sed -n 2p)" = "a.sh:7 SEC010" ] && ok "classify: finding printed as file:line CODE" || bad "classify finding line: '$(printf '%s\n' "$got" | sed -n 2p)'"
  local rules_none; rules_none=$(printf '{"file":"c.sh","diagnostics":[]}\n' > "$d/empty.json"; classify "$d/empty.json" | head -n1)
  [ "$rules_none" = "0 0 0" ] && ok "classify: empty diagnostics -> '0 0 0'" || bad "classify empty: '$rules_none'"

  # 4. must-RED, end to end: a fixture repo with one committed SEC010 script fails the gate
  #    with exit 1 and names the file; fixing the script turns it green with exit 0.
  fx="$d/fx"; mkdir -p "$fx/scripts"
  git -C "$fx" init -q 2>/dev/null || git init -q "$fx"
  # the fixture is the incident's exact shape (cuda_rust_fleet_check.sh @ 0fd6fbcb0): a --repo
  # argument captured in a parse loop, then a subshell cd into it. `cd "$1"` alone is NOT flagged
  # (measured), so a sentinel that merely says "cd" would prove nothing.
  printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' 'REPO="${REPO_ROOT:-}"' \
    'while [ $# -gt 0 ]; do case "$1" in --repo) REPO="$2"; shift;; esac; shift; done' \
    'rc=0; ( cd "$REPO" && timeout 60 cargo test --no-run ) > out.log 2>&1 || rc=$?' 'echo "$rc"' > "$fx/scripts/bad.sh"
  printf '#!/bin/sh\necho fine\n' > "$fx/scripts/good.sh"
  git -C "$fx" add -A && git -C "$fx" -c user.name=t -c user.email=t@t -c core.hooksPath=/dev/null commit -qm fixture
  rc=0; out=$(gate "$fx" 2>&1) || rc=$?
  { [ "$rc" -eq 1 ] && printf '%s\n' "$out" | grep -q '^scripts/bad.sh:5 SEC010$'; } && ok "must-RED: fixture repo with a SEC010 -> exit 1, 'scripts/bad.sh:5 SEC010'" || bad "must-RED: exit=$rc out=$(printf '%s' "$out" | tail -n2 | tr '\n' '|')"
  # the fixed twin carries #3198's validation: absolute and free of '..' before first use
  printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' 'REPO="${REPO_ROOT:-}"' \
    'while [ $# -gt 0 ]; do case "$1" in --repo) REPO="$2"; shift;; esac; shift; done' \
    'case "$REPO" in /*) ;; *) echo refused >&2; exit 2;; esac' 'case "$REPO" in *..*) echo refused >&2; exit 2;; esac' \
    'rc=0; ( cd "$REPO" && timeout 60 cargo test --no-run ) > out.log 2>&1 || rc=$?' 'echo "$rc"' > "$fx/scripts/bad.sh"
  git -C "$fx" add -A && git -C "$fx" -c user.name=t -c user.email=t@t -c core.hooksPath=/dev/null commit -qm fixed
  rc=0; out=$(gate "$fx" 2>&1) || rc=$?
  { [ "$rc" -eq 0 ] && printf '%s\n' "$out" | grep -q 'bashrs-gate: PASS 2 file(s)'; } && ok "must-GREEN: the validated fixture -> exit 0 over 2 files" || bad "must-GREEN: exit=$rc out=$(printf '%s' "$out" | tail -n1)"

  # 5. vacuity: an empty surface is ENV (2), never a pass; a gitignored script is named
  fx2="$d/fx2"; mkdir -p "$fx2"; git init -q "$fx2"
  printf 'hidden.sh\n' > "$fx2/.gitignore"; printf '#!/bin/sh\ncd "$1"\n' > "$fx2/hidden.sh"
  git -C "$fx2" add -A && git -C "$fx2" -c user.name=t -c user.email=t@t -c core.hooksPath=/dev/null commit -qm v
  rc=0; out=$(gate "$fx2" 2>&1) || rc=$?
  { [ "$rc" -eq 2 ] && printf '%s' "$out" | grep -q 'gitignored: hidden.sh'; } && ok "vacuity: empty surface with a gitignored script -> exit 2, names it" || bad "vacuity: exit=$rc out=$out"

  rm -rf "${d:?}"
  if [ "$fail" -eq 0 ]; then echo "SELF-TEST PASS"; return 0; fi
  echo "SELF-TEST FAIL"; return 1
}

case "${1:-}" in
  --self-test) self_test ;;
  --classify) [ -n "${2:-}" ] || { echo "usage: --classify FILE.json" >&2; exit 2; }; classify "$2" ;;
  --root) [ -n "${2:-}" ] || { echo "usage: --root DIR" >&2; exit 2; }
          case "$2" in /*) ;; *) echo "refused: --root must be absolute" >&2; exit 2;; esac
          case "$2" in *..*) echo "refused: --root must not contain '..'" >&2; exit 2;; esac
          gate "$2" ;;
  "") gate "$(git rev-parse --show-toplevel)" ;;
  *) echo "usage: check_bashrs_gate.sh [--root DIR | --classify FILE.json | --self-test]" >&2; exit 2 ;;
esac
