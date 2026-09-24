#!/usr/bin/env bash
# check-orphans.sh -- PVL-001 EV-5a (#4122): every ProvableContracts module is in the ROOT's import cone, or allowlisted.
#
# An ORPHAN is a module under ProvableContracts/** that the transitive import cone of ProvableContracts.lean never
# reaches: `lake build` of the default target never compiles it, so no proof in it is checked. Measured on main:
# 59 of 162 (#4124 Theorems, #4125 Defs). orphan-allowlist.yaml names each one with `ticket:` and `reason:`; EV-5c
# drains it (a module that will not compile keeps its entry and its issue, and is never deleted).
#
#   ./check-orphans.sh [--root <lean dir>]   rc 0 clean . 1 an unlisted orphan, an entry without ticket:/reason:, or a
#                                            STALE entry (no longer an orphan: remove it) . 2 usage / unreadable
#   ./check-orphans.sh --self-test           every fixture under fixtures/orphans/ lands its want + want-msg, and each
#                                            rule deleted in a copy of this script breaks the fixture that names it
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE"; SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    *) echo "usage: check-orphans.sh [--root <lean dir>] | --self-test" >&2; exit 2 ;;
  esac
done

check() { # check <lean dir> -> report lines; rc 0/1/2
  python3 - "$1" <<'PY'
import os, re, sys
root = sys.argv[1]
top = os.path.join(root, "ProvableContracts.lean")
if not os.path.isfile(top):
    print("decline: no ProvableContracts.lean under %s" % root); sys.exit(2)
mods = {}
for dp, _, fs in os.walk(os.path.join(root, "ProvableContracts")):
    for f in fs:
        if f.endswith(".lean"):
            p = os.path.join(dp, f)
            mods[os.path.relpath(p, root)[:-5].replace(os.sep, ".")] = p
def strip_block_comments(src):
    """Lean's /- ... -/ (and /-! -/, /-- -/) comments NEST, and `--` runs to end of line; newlines are kept."""
    out, i, depth = [], 0, 0
    while i < len(src):
        two = src[i:i + 2]
        if two == "--" and not depth:   # a line comment: a `/-` inside it opens nothing
            j = src.find("\n", i); i = len(src) if j < 0 else j
        elif two == "/-":
            depth += 1; i += 2
        elif two == "-/" and depth:
            depth -= 1; i += 2
        else:
            if not depth or src[i] == "\n":
                out.append(src[i])
            i += 1
    return "".join(out)
def imports(path):
    out = []
    text = strip_block_comments(open(path, encoding="utf-8").read())
    for ln in text.splitlines():
        m = re.match(r"\s*(?:(?:public|private|meta)\s+)*import\s+(?:all\s+)?(.+)", ln)
        if m:
            out += m.group(1).split("--")[0].split()
    return out
seen, todo = set(), imports(top)
while todo:
    m = todo.pop()
    if m in seen or m not in mods:
        continue
    seen.add(m)
    todo += imports(mods[m])
orphans = set(mods) - seen
allow_p = os.path.join(root, "orphan-allowlist.yaml")
try:
    import yaml
    entries = yaml.safe_load(open(allow_p)) or [] if os.path.exists(allow_p) else []
except Exception as exc:
    print("decline: %s unreadable: %s" % (allow_p, exc)); sys.exit(2)
if not isinstance(entries, list):
    print("decline: %s is not a list of entries" % allow_p); sys.exit(2)
bad, listed = 0, set()
for e in entries:
    mod = e.get("module") if isinstance(e, dict) else None
    if not mod:
        print("FAIL  an allowlist entry has no module: %r" % (e,)); bad = 1; continue
    listed.add(mod)
    missing = [k for k in ("ticket", "reason") if not str((e or {}).get(k) or "").strip()]
    if missing:
        print("FAIL  allowlist entry %s has no %s -- an orphan is allowlisted against an issue, with a reason" % (mod, "/".join(missing))); bad = 1
    if mod not in orphans:
        print("FAIL  allowlist entry %s is STALE: %s -- remove it (EV-5c drains the list)" % (
            mod, "the module is now in the root's import cone" if mod in mods else "no such module")); bad = 1
for m in sorted(orphans - listed):
    print("FAIL  %s is an ORPHAN: the root ProvableContracts.lean never imports it, so no proof in it is built -- import it or allowlist it with ticket:/reason:" % m); bad = 1
print("%s modules=%d in-cone=%d orphans=%d allowlisted=%d" % ("ok   " if not bad else "RED  ", len(mods), len(seen), len(orphans), len(listed)))
sys.exit(bad)
PY
}

if [ "$SELF_TEST" = 1 ]; then
  bad=0
  for d in "$HERE"/fixtures/orphans/*/; do
    [ -f "$d/want" ] || continue
    out=$(check "$d" 2>&1); rc=$?
    ok=1; [ "$rc" = "$(cat "$d/want")" ] || ok=0
    while IFS= read -r needle; do [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out" && ok=0; done < "$d/want-msg"
    if grep -qE '^Traceback|^ *File "<stdin>"|^[A-Za-z]+Error: ' <<< "$out"; then echo "CRASH $(basename "$d") -- $(tr '\n' ' ' <<< "$out" | cut -c1-200)"; bad=1
    elif [ "$ok" = 1 ]; then echo "ok    $(basename "$d")"; else echo "FAIL  $(basename "$d") -- rc $rc: $(tr '\n' ' ' <<< "$out" | cut -c1-200)"; bad=1; fi
  done
  if [ "${ORPHANS_MUTANTS:-1}" = 1 ] && [ "$bad" = 0 ]; then
    M=$(mktemp -d)
    while IFS='~' read -r label must old new; do
      [ -n "$label" ] || continue
      python3 -c 'import sys
s = open(sys.argv[1]).read(); code, cut, rest = s.partition("\nif [ \"$SELF_TEST\" = 1 ]; then")
assert code.count(sys.argv[3]) == 1
open(sys.argv[2], "w").write(code.replace(sys.argv[3], sys.argv[4]) + cut + rest)' "$0" "$M/m.sh" "$old" "$new" 2> /dev/null \
        || { echo "FAIL  mutant $label did not apply"; bad=1; continue; }
      mkdir -p "$M/fixtures"; rm -rf -- "${M:?}/fixtures/orphans"; cp -r "$HERE/fixtures/orphans" "$M/fixtures/orphans"
      mo=$(ORPHANS_MUTANTS=0 bash "$M/m.sh" --self-test 2>&1)
      coll=$(grep -E '^(FAIL|CRASH) ' <<< "$mo" | awk -v m="$must" '$2 != m {print $2}' | paste -sd, -)
      if [ "${label#crash:}" != "$label" ]; then   # the harness's own control: a crashing copy must be refused, not counted
        if grep -q '^CRASH ' <<< "$mo"; then echo "ok    mutant $label refused as a crash"; else echo "FAIL  mutant $label: a crash was not detected"; bad=1; fi
      elif grep -q '^CRASH ' <<< "$mo"; then echo "FAIL  mutant $label CRASHED -- a crash is not a kill"; bad=1
      elif grep -q "^FAIL  $must " <<< "$mo"; then echo "ok    mutant $label killed by $must${coll:+ (collateral: $coll)}"
      else echo "FAIL  mutant $label SURVIVED $must"; bad=1; fi
    done <<'MUT'
unlisted-ok~unlisted-orphan-is-red~for m in sorted(orphans - listed):~for m in []:
ticket-optional~entry-without-ticket-is-red~    missing = [k for k in ("ticket", "reason") if~    missing = [k for k in ("reason",) if
stale-ok~stale-entry-is-red~    if mod not in orphans:~    if False:
cone-not-transitive~transitive-import-is-in-cone~    todo += imports(mods[m])~    pass
comments-kept~doc-comment-import-is-not-an-import~    text = strip_block_comments(open(path, encoding="utf-8").read())~    text = open(path, encoding="utf-8").read()
line-comment-not-skipped~line-comment-with-slash-dash-is-not-a-block~        if two == "--" and not depth:   # a line comment: a `/-` inside it opens nothing~        if False:
crash:syntax-error~clean~    todo += imports(mods[m])~    todo += imports(mods[m]
MUT
    if [ -d "${M:?}" ]; then rm -rf -- "${M:?}"; fi
  fi
  echo "check-orphans self-test: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
  exit "$bad"
fi
check "$ROOT"
