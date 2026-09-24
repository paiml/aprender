#!/usr/bin/env bash
# build.sh -- PVL-001 EV-5a (#4122): the Lean build of ProvableContracts as a GATE, scoped to our own tree.
#
#   lake exe cache get   Mathlib's prebuilt oleans (the manifest pins the Mathlib SHA)
#   lake build           the default targets; its log is judged, its exit code read directly (never through a pipe)
# The gate over the log:
#   rc 1  a `warning:` whose path is under ProvableContracts/ (our proofs), or the build failed
#   rc 2  `decline: mathlib cache miss -- not a verdict`: the log shows a Mathlib MODULE being elaborated
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

gate() { # gate <log> <build rc>
  python3 - "$1" "$2" <<'PY'
import re, sys
log, brc = sys.argv[1], int(sys.argv[2])
text = open(log, encoding="utf-8", errors="replace").read()
miss = [m.group(1) for m in re.finditer(r"^\S+ \[\d+/\d+\] Built (Mathlib\.[^\s:(]+)(?=\s|$)", text, re.M)]
if miss:
    print("decline: mathlib cache miss -- not a verdict (%d Mathlib module(s) elaborated, e.g. %s)" % (len(miss), miss[0]))
    sys.exit(2)
ours = []
for m in re.finditer(r"^warning: (\S+?):(\d+):(\d+): (.*)$", text, re.M):
    path = m.group(1)
    rel = re.sub(r"^(?:\./)+", "", path)   # lake 4.29 prints ProvableContracts/..., older lakes ././././ProvableContracts/...
    if rel.startswith("ProvableContracts/") or "/lean/ProvableContracts/" in path:
        ours.append("%s:%s: %s" % (rel, m.group(2), m.group(4)[:100]))
bad = 0
for w in ours:
    print("FAIL  warning in our tree: %s" % w); bad = 1
if brc != 0:
    errs = sorted(set(re.findall(r"^error: .*$", text, re.M)))
    print("FAIL  lake build exited %d: %s" % (brc, "; ".join(errs[:3]) or "no error line")); bad = 1
print("%s lake build rc=%d, %d warning(s) in ProvableContracts/" % ("ok   " if not bad else "RED  ", brc, len(ours)))
sys.exit(bad)
PY
}

if [ "${1:-}" = "--self-test" ]; then
  bad=0
  for d in "$HERE"/fixtures/build/*/; do
    [ -f "$d/want" ] || continue
    out=$(gate "$d/build.log" "$(cat "$d/build-rc" 2> /dev/null || echo 0)" 2>&1); rc=$?
    ok=1; [ "$rc" = "$(cat "$d/want")" ] || ok=0
    while IFS= read -r needle; do [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out" && ok=0; done < "$d/want-msg"
    if [ "$ok" = 1 ]; then echo "ok    $(basename "$d")"; else echo "FAIL  $(basename "$d") -- rc $rc: $(tr '\n' ' ' <<< "$out" | cut -c1-200)"; bad=1; fi
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
      if grep -q "^FAIL  $must " <<< "$mo"; then echo "ok    mutant $label killed by $must"; else echo "FAIL  mutant $label SURVIVED $must"; bad=1; fi
    done <<'MUT'
scope-dropped~mathlib-warning-is-ignored~    if rel.startswith("ProvableContracts/") or "/lean/ProvableContracts/" in path:~    if True:
our-warning-ignored~our-warning-is-red~    print("FAIL  warning in our tree: %s" % w); bad = 1~    pass
cache-miss-ignored~mathlib-elaborated-is-a-cache-miss~if miss:~if False:
native-counted-as-miss~native-object-is-not-a-miss~(Mathlib\.[^\s:(]+)(?=\s|$)~(Mathlib\.[^\s(]+)
build-rc-ignored~failed-build-is-red~if brc != 0:~if False:
single-dot-strip~dot-prefixed-warning-is-red~    rel = re.sub(r"^(?:\./)+", "", path)~    rel = path[2:] if path.startswith("./") else path
MUT
    if [ -d "${M:?}" ]; then rm -rf -- "${M:?}"; fi
  fi
  echo "build.sh self-test: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
  exit "$bad"
fi
if [ "${1:-}" = "--gate" ]; then
  [ -n "${2:-}" ] || { echo "usage: build.sh --gate <log> [--build-rc N]" >&2; exit 2; }
  gate "$2" "${4:-0}"; exit $?
fi
cd "$HERE" || exit 2
LOG="${BUILD_LOG:-$HERE/.lake/build.log}"; mkdir -p "$(dirname "$LOG")"
lake exe cache get > "$LOG.cache" 2>&1 || { echo "decline: lake exe cache get failed -- $(tail -1 "$LOG.cache")"; exit 2; }
lake build > "$LOG" 2>&1; brc=$?
gate "$LOG" "$brc"
