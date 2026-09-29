#!/usr/bin/env bash
# qm11_join.sh -- QM-11 (#4529) join C: the tier router can only ADD tests to today's tier.
#
# Ruled (aprender-1b, 08:32Z): "take C (B's selective + today's quick set; keep-or-tighten,
# pre-approved). T0 narrowing goes to the operator as a separate ruling, not in this PR."
#
# The build job runs `decide` ONCE, after the shadow wrote qm11-route.json, and ships the
# decision inside ws-lib-archive so all 8 shards apply the same one:
#   today != quick                 -> none  (full already runs every lib test; none/reuse kept)
#   route missing/abstained/broken -> none  (today's quick tier, unchanged)
#   router T0                      -> none  (T0 narrowing is a separate ruling)
#   router selective, valid names  -> selective, filter package(a) | package(b) ...
#   router T3, or selective with no package or a name that is not a crate name -> t3, filter all()
# So `none` is exactly today, and every other value runs today's quick steps PLUS the
# workspace lib tests the filter selects, from the compile-once archive. No value removes one.
#
# usage: qm11_join.sh decide <today-tier> <qm11-route.json> <outdir>...  (writes join, join.filter)
#        qm11_join.sh --self-test
# exit:  0 decided (any value) · 1 a write failed · 64 usage
set -euo pipefail

usage() { sed -n 2,19p "$0" >&2; exit 64; }

decide() {
  [ $# -ge 3 ] || usage
  local today=$1 route=$2; shift 2
  local out
  out=$(python3 - "$today" "$route" <<'PY'
import json, re, sys
today, route = sys.argv[1:3]
def say(join, flt, why):
    print(join); print(flt); print(why); sys.exit(0)
if today != "quick":
    say("none", "", f"today={today}: the join adds only to the quick tier")
try:
    doc = json.load(open(route))
except (OSError, ValueError) as e:
    say("none", "", f"no readable route ({type(e).__name__}): today's quick tier, unchanged")
r = doc.get("router") if isinstance(doc, dict) else None
if not isinstance(r, dict) or doc.get("abstained"):
    say("none", "", f"router abstained: {doc.get('abstained') if isinstance(doc, dict) else 'not an object'}")
tier, pkgs = r.get("tier"), r.get("packages")
if tier == "T0":
    say("none", "", "router T0: T0 narrowing is a separate ruling; today's quick tier runs")
if tier == "selective" and isinstance(pkgs, list) and pkgs \
        and all(isinstance(p, str) and re.fullmatch(r"[A-Za-z0-9_-]+", p) for p in pkgs):
    say("selective", " | ".join(f"package({p})" for p in sorted(set(pkgs))), f"router selective: {len(set(pkgs))} packages")
# T3, an unknown tier, or a selective route this cannot turn into a filter: run MORE, never less.
say("t3", "all()", f"router {tier!r}: every workspace lib test")
PY
) || return 1
  local join flt why
  { read -r join; read -r flt; read -r why; } <<< "$out"
  local d
  for d in "$@"; do
    mkdir -p "$d"
    printf '%s\n' "$join" > "$d/join" || return 1
    printf '%s\n' "$flt" > "$d/join.filter" || return 1
  done
  printf 'qm11-join: today=%s join=%s filter=%s why=%s\n' "$today" "$join" "${flt:--}" "$why"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then printf 'join=%s\n' "$join" >> "$GITHUB_OUTPUT"; fi
}

self_test() {
  local t fails=0 n=0
  t=$(mktemp -d)
  [ -n "$t" ] && [ -d "$t" ] || return 1
  trap 'rm -rf "${t:?}"' RETURN
  route() { # route <file> <json>
    printf '%s\n' "$2" > "$t/$1"
  }
  route sel.json '{"abstained":null,"router":{"tier":"selective","packages":["b-core","a_app","a_app"],"causes":[]}}'
  route t3.json '{"abstained":null,"router":{"tier":"T3","packages":["x"],"causes":["C"]}}'
  route t0.json '{"abstained":null,"router":{"tier":"T0","packages":[],"causes":[]}}'
  route abst.json '{"abstained":"no GH_TOKEN","router":null}'
  route inj.json '{"abstained":null,"router":{"tier":"selective","packages":["a) | all() | package(b"],"causes":[]}}'
  route empty.json '{"abstained":null,"router":{"tier":"selective","packages":[],"causes":[]}}'
  route odd.json '{"abstained":null,"router":{"tier":"T9","packages":["a"],"causes":[]}}'
  printf 'not json' > "$t/bad.json"
  row() { # row <name> <today> <route> <want join> <want filter>
    n=$((n + 1))
    local o="$t/o$n" got flt
    if ! decide "$2" "$3" "$o" > "$t/log$n"; then got="rc!=0"; else got=$(cat "$o/join"); fi
    flt=$(cat "$o/join.filter" 2>/dev/null || true)
    if [ "$got" = "$4" ] && [ "$flt" = "$5" ]; then echo "  ok   $1"
    else echo "  FAIL $1: join=$got filter=$flt (want $4 / $5)"; fails=$((fails + 1)); fi
  }
  row "quick + selective -> sorted, deduped package filter" quick "$t/sel.json" selective "package(a_app) | package(b-core)"
  row "quick + T3 -> every lib test" quick "$t/t3.json" t3 "all()"
  row "quick + T0 -> none (T0 narrowing is not this PR)" quick "$t/t0.json" none ""
  row "quick + abstained -> none (today unchanged)" quick "$t/abst.json" none ""
  row "quick + no route file -> none" quick "$t/missing.json" none ""
  row "quick + unparsable route -> none" quick "$t/bad.json" none ""
  row "quick + a package name that is not a crate name -> t3, never spliced" quick "$t/inj.json" t3 "all()"
  row "quick + selective with no package -> t3" quick "$t/empty.json" t3 "all()"
  row "quick + an unknown router tier -> t3" quick "$t/odd.json" t3 "all()"
  row "full + T3 -> none (full already runs every lib test)" full "$t/t3.json" none ""
  row "none + selective -> none (the none tier is kept)" none "$t/sel.json" none ""
  row "reuse + selective -> none" reuse "$t/sel.json" none ""
  # Both staging dirs get the same decision (the shards read one, the fan-in the other).
  n=$((n + 1))
  decide quick "$t/sel.json" "$t/a" "$t/b" > /dev/null
  if cmp -s "$t/a/join" "$t/b/join" && cmp -s "$t/a/join.filter" "$t/b/join.filter"; then echo "  ok   two outdirs, one decision"
  else echo "  FAIL two outdirs disagree"; fails=$((fails + 1)); fi
  # GITHUB_OUTPUT carries the value the compile step's `if:` reads.
  n=$((n + 1))
  : > "$t/gho"
  GITHUB_OUTPUT="$t/gho" decide quick "$t/t3.json" "$t/c" > /dev/null
  if [ "$(cat "$t/gho")" = "join=t3" ]; then echo "  ok   GITHUB_OUTPUT join=t3"
  else echo "  FAIL GITHUB_OUTPUT: $(cat "$t/gho")"; fails=$((fails + 1)); fi
  echo "qm11_join self-test: $((n - fails))/$n"
  [ "$fails" -eq 0 ]
}

case "${1:-}" in
  decide) shift; decide "$@" ;;
  --self-test) self_test ;;
  *) usage ;;
esac
