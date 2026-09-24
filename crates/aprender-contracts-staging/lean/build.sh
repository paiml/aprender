#!/usr/bin/env bash
# build.sh -- PVL-001 EV-5a (#4122): the Lean build of ProvableContracts as a GATE, scoped to our own tree.
#
#   lake exe cache get   Mathlib's prebuilt oleans (the manifest pins the Mathlib SHA)
#   lake build           the default targets; its log is judged, its exit code read directly (never through a pipe)
# The gate over the log:
#   rc 1  a `warning:` in ProvableContracts.lean or under ProvableContracts/ (our proofs), or the build failed --
#         judged even on a cold cache: a failure is never hidden behind a decline
#   rc 2  `decline: mathlib cache miss -- not a verdict`: an otherwise clean log shows a Mathlib MODULE elaborated
#         (`Built Mathlib.X`). A native object (`Built Mathlib.X:c.o`) for the test executable is not a miss: the
#         cache ships oleans, not objects (measured on lambda 2026-09-24, 3 such lines on a warm cache).
#   Mathlib's own warnings are ignored: they are not ours to fix.
#
#   ./build.sh                  fetch the cache, build, judge   (lambda, warm cache: ~6 min)
#   ./build.sh --gate <log> [--build-rc N]    judge a saved log only
#   ./build.sh --self-test      every fixture under fixtures/build/ lands its want + want-msg, and each rule deleted
#                               in a copy of this script breaks the fixture that names it
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
case "${1:-}" in
  ""|--self-test|--gate) ;;
  *) echo "usage: build.sh [--self-test | --gate <log> [--build-rc N]]   (no argument: fetch, build, judge)" >&2; exit 2 ;;
esac

gate() { # gate <log> <build rc>
  python3 - "$1" "$2" <<'PY'
import re, sys
log, brc = sys.argv[1], int(sys.argv[2])
text = open(log, encoding="utf-8", errors="replace").read()
miss = [m.group(1) for m in re.finditer(r"^\S+ \[\d+/\d+\] Built (Mathlib\.[^\s:(]+)(?=\s|$)", text, re.M)]
fails = []
for m in re.finditer(r"^warning: (\S+?):(\d+):(\d+): (.*)$", text, re.M):
    path = m.group(1)
    rel = re.sub(r"^(?:\./)+", "", path)   # lake 4.29 prints ProvableContracts/..., older lakes ././././ProvableContracts/...
    ours = rel.startswith("ProvableContracts/") or "/lean/ProvableContracts/" in path
    ours = ours or rel == "ProvableContracts.lean" or path.endswith("/lean/ProvableContracts.lean")   # the root module
    if ours:
        fails.append("FAIL  warning in our tree: %s:%s: %s" % (rel, m.group(2), m.group(4)[:100]))
if brc != 0:
    errs = sorted(set(re.findall(r"^error: .*$", text, re.M)))
    fails.append("FAIL  lake build exited %d: %s" % (brc, "; ".join(errs[:3]) or "no error line"))
# a cold cache DECLINES only a log that is otherwise clean: a failure or a warning of ours is judged either way
if miss and not fails:
    print("decline: mathlib cache miss -- not a verdict (%d Mathlib module(s) elaborated, e.g. %s)" % (len(miss), miss[0]))
    sys.exit(2)
for f in fails:
    print(f)
if miss:
    print("note  %d Mathlib module(s) elaborated (cache miss), e.g. %s -- the failure above is judged anyway" % (len(miss), miss[0]))
nw = sum(1 for f in fails if f.startswith("FAIL  warning"))
print("%s lake build rc=%d, %d warning(s) in ProvableContracts/" % ("ok   " if not fails else "RED  ", brc, nw))
sys.exit(1 if fails else 0)
PY
}

if [ "${1:-}" = "--self-test" ]; then
  bad=0
  for d in "$HERE"/fixtures/build/*/; do
    [ -f "$d/want" ] || continue
    out=$(gate "$d/lake.out" "$(cat "$d/build-rc" 2> /dev/null || echo 0)" 2>&1); rc=$?
    ok=1; [ "$rc" = "$(cat "$d/want")" ] || ok=0
    while IFS= read -r needle; do [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out" && ok=0; done < "$d/want-msg"
    if grep -qE '^Traceback|^ *File "<stdin>"|^[A-Za-z]+Error: ' <<< "$out"; then echo "CRASH $(basename "$d") -- $(tr '\n' ' ' <<< "$out" | cut -c1-200)"; bad=1
    elif [ "$ok" = 1 ]; then echo "ok    $(basename "$d")"; else echo "FAIL  $(basename "$d") -- rc $rc: $(tr '\n' ' ' <<< "$out" | cut -c1-200)"; bad=1; fi
  done
  if [ "${BUILD_MUTANTS:-1}" = 1 ] && [ "$bad" = 0 ]; then
    M=$(mktemp -d)
    while IFS='~' read -r label must old new; do
      [ -n "$label" ] || continue
      python3 -c 'import sys
s = open(sys.argv[1]).read(); code, cut, rest = s.partition("\nif [ \"${1:-}\" = \"--self-test\" ]; then")
assert code.count(sys.argv[3]) == 1
open(sys.argv[2], "w").write(code.replace(sys.argv[3], sys.argv[4]) + cut + rest)' "$0" "$M/m.sh" "$old" "$new" 2> /dev/null \
        || { echo "FAIL  mutant $label did not apply"; bad=1; continue; }
      mkdir -p "$M/fixtures"; rm -rf -- "${M:?}/fixtures/build"; cp -r "$HERE/fixtures/build" "$M/fixtures/build"
      mo=$(BUILD_MUTANTS=0 bash "$M/m.sh" --self-test 2>&1)
      coll=$(grep -E '^(FAIL|CRASH) ' <<< "$mo" | awk -v m="$must" '$2 != m {print $2}' | paste -sd, -)
      if [ "${label#crash:}" != "$label" ]; then   # the harness's own control: a crashing copy must be refused, not counted
        if grep -q '^CRASH ' <<< "$mo"; then echo "ok    mutant $label refused as a crash"; else echo "FAIL  mutant $label: a crash was not detected"; bad=1; fi
      elif grep -q '^CRASH ' <<< "$mo"; then echo "FAIL  mutant $label CRASHED -- a crash is not a kill"; bad=1
      elif grep -q "^FAIL  $must " <<< "$mo"; then echo "ok    mutant $label killed by $must${coll:+ (collateral: $coll)}"
      else echo "FAIL  mutant $label SURVIVED $must"; bad=1; fi
    done <<'MUT'
scope-dropped~mathlib-warning-is-ignored~    ours = rel.startswith("ProvableContracts/") or "/lean/ProvableContracts/" in path~    ours = True
root-dropped~root-module-warning-is-red~    ours = ours or rel == "ProvableContracts.lean" or path.endswith("/lean/ProvableContracts.lean")~    ours = ours
our-warning-ignored~our-warning-is-red~        fails.append("FAIL  warning in our tree: %s:%s: %s" % (rel, m.group(2), m.group(4)[:100]))~        pass
cache-miss-ignored~mathlib-elaborated-is-a-cache-miss~if miss and not fails:~if False:
miss-masks-failure~cold-cache-failed-build-is-red~if miss and not fails:~if miss:
native-counted-as-miss~native-object-is-not-a-miss~(Mathlib\.[^\s:(]+)(?=\s|$)~(Mathlib\.[^\s(]+)
build-rc-ignored~failed-build-is-red~if brc != 0:~if False:
single-dot-strip~dot-prefixed-warning-is-red~    rel = re.sub(r"^(?:\./)+", "", path)~    rel = path[2:] if path.startswith("./") else path
crash:syntax-error~clean~if brc != 0:~if brc != 0
MUT
    if [ -d "${M:?}" ]; then rm -rf -- "${M:?}"; fi
  fi
  echo "build.sh self-test: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
  exit "$bad"
fi
if [ "${1:-}" = "--gate" ]; then
  usage() { echo "usage: build.sh --gate <log> [--build-rc N]" >&2; exit 2; }
  [ -f "${2:-}" ] && [ -r "$2" ] || usage
  if [ $# -gt 2 ]; then [ "$#" = 4 ] && [ "$3" = --build-rc ] && [[ "$4" =~ ^[0-9]+$ ]] || usage; fi
  gate "$2" "${4:-0}"; exit $?
fi
cd "$HERE" || exit 2
LOG="${BUILD_LOG:-$HERE/.lake/build.log}"; mkdir -p "$(dirname "$LOG")"
lake exe cache get > "$LOG.cache" 2>&1 || { echo "decline: lake exe cache get failed -- $(tail -1 "$LOG.cache")"; exit 2; }
lake build > "$LOG" 2>&1; brc=$?
gate "$LOG" "$brc"
