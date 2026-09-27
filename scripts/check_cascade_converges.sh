#!/usr/bin/env bash
# check_cascade_converges.sh — the crates.io cascade must be able to FINISH, proven
# BEFORE anything is uploaded.
#
# WHY THIS AND NOT A BUDGET CHECK. #3892 replaced the single retry round with a drain
# loop that repeats until no forward progress and compares the deferred set BY CONTENT,
# so there is no round cap left to exceed. What survives is worse and quieter: the drain
# loop's `STUCK` exit happens AFTER the successful part of the cascade has already
# uploaded. crates.io is append-only. A graph that cannot converge therefore costs a
# half-published release at the moment it is discovered, and the discovery is at publish
# time by construction.
#
# So this guard answers the same question statically: given TIERS[] and the real
# dependency graph, does the drain terminate with everything published? It runs in CI,
# reads no network, and uploads nothing.
#
# A CYCLE IS NOT A DEEP GRAPH, and the distinction is the whole point. A check that only
# counted passes would report "needs N" for a graph that needs infinity — two crates each
# waiting on the other leave the deferred set the SAME SIZE every round while nothing
# publishes. Progress is therefore compared as a SET, exactly as cascade-publish.sh does
# it, and a repeat is reported as a cycle naming its members.
#
# RENAMES ARE RESOLVED BY PACKAGE, NOT BY KEY. `provable-contracts = { package =
# "aprender-contracts" }` is an alias for an in-tree crate, not an edge to the T14 facade
# of the same name. Reading the key invents a facade->source->facade cycle that does not
# exist; `cargo metadata` resolves it correctly and this guard uses that.
#
# Usage:
#   bash scripts/check_cascade_converges.sh              # enforce
#   bash scripts/check_cascade_converges.sh --self-test  # the case table
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

analyse() { # analyse <repo-root> -> verdict lines; 0 converges, 1 cannot
  python3 - "$1" <<'PY'
import json, re, subprocess, sys, os
root = sys.argv[1]
def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=root).stdout

uni = {}
for line in sh('python3', os.path.join(root,'scripts/lib/cascade_universe.py'), root).split('\n'):
    p = line.split('\t')
    if len(p) >= 3 and p[0]:
        uni[p[0]] = p[2]
src = open(os.path.join(root, 'scripts/cascade-publish.sh')).read()
tier = {c: int(m.group(1))
        for m in re.finditer(r'^TIERS\[(\d+)\]="([^"]*)"', src, re.M)
        for c in m.group(2).split()}

# ANTI-VACUITY, before any verdict. An enumeration that saw nothing would converge
# trivially and report a clean pass forever -- the shape this fleet keeps finding.
if len(uni) < 10 or len(tier) < 10:
    print(f"FAIL cascade_converges: universe={len(uni)} tiers={len(tier)} — too few to be a real cascade. "
          f"A convergence proved over nothing is not a proof; the enumeration broke.")
    sys.exit(1)

metas = [sh('cargo','metadata','--no-deps','--format-version','1')]
for name, man in uni.items():
    if os.path.dirname(man).startswith(os.path.join(root, 'crates/facades')):
        metas.append(sh('cargo','metadata','--no-deps','--format-version','1','--manifest-path',man))
deps = {n: set() for n in uni}
for raw in metas:
    if not raw.strip(): continue
    for p in json.loads(raw)['packages']:
        if p['name'] in uni:
            deps[p['name']] |= {d['name'] for d in p.get('dependencies', [])
                                if d['name'] in uni and d.get('kind') != 'dev'}

order = sorted(deps, key=lambda c: (tier.get(c, 10**6), c))
published, deferred, rows = set(), [], []
for c in order:
    (published.add(c) if deps[c] <= published else deferred.append(c))
rows.append(('tier walk', len(published), len(deferred)))
passes = 1
while deferred:
    before = set(deferred)
    # An explicit loop, NOT a comprehension. `deps[c] <= published or published.add(c)`
    # short-circuits: when the deps ARE satisfied the `or` never evaluates the add, so
    # `published` never grows and every graph reports STUCK after two passes. Measured:
    # it held `published` at 58 and called a converging DAG a cycle.
    still = []
    for c in deferred:
        if deps[c] <= published:
            published.add(c)
        else:
            still.append(c)
    passes += 1
    rows.append((f'drain {passes-1}', len(published), len(still)))
    if set(still) == before:
        for lbl, pub, d in rows: print(f"  {lbl:<10} published {pub:>3}   deferred {d:>3}")
        waits = {c: sorted(deps[c] & set(still)) for c in sorted(still)}
        print(f"FAIL cascade_converges: STUCK after {passes} pass(es) — a drain round published nothing, "
              f"so the remaining {len(still)} crate(s) can never publish. This is a CYCLE, not a deep graph: "
              f"raising any round count loops forever. cascade-publish.sh would hit this AFTER uploading "
              f"{len(published)} crate(s) to an append-only registry.")
        for c, w in list(waits.items())[:6]:
            print(f"    {c} waits on {', '.join(w) or '<a crate outside TIERS[]>'}")
        sys.exit(1)
    deferred = still
for lbl, pub, d in rows: print(f"  {lbl:<10} published {pub:>3}   deferred {d:>3}")
print(f"PASS cascade_converges: all {len(uni)} crate(s) publish; the drain terminates in {passes} pass(es)")
PY
}

if [ "${1:-}" = "--self-test" ]; then
  TD=$(mktemp -d)
  # SEC011: guarded delete — an empty or non-temp TD is left alone. Same idiom as
  # scripts/dogfood.sh::_rm_worklog; an unguarded `rm -rf "$VAR"` in a trap is one typo
  # from deleting the tree it was meant to clean up after.
  _rm() {
    local v="${TD:-}"
    case "$v" in /tmp/?*|/var/folders/?*) ;; *) return 0 ;; esac
    [ -n "$v" ] && [ "$v" != "/" ] && rm -rf -- "$v" || :
  }
  trap _rm EXIT
  n=0; bad=0
  fixture() { # fixture <dir> <universe rows "name;dep,dep"> <tiers "name:tier,...">
    local d="$1" u="$2" t="$3"
    mkdir -p "$d/scripts/lib" "$d/crates"
    : > "$d/universe.tsv"
    local pkgs="" first=1
    for row in ${u//|/ }; do
      local nm="${row%%;*}" dl="${row#*;}"; [ "$dl" = "$row" ] && dl=""
      printf '%s\t0.0.1\t%s/crates/%s/Cargo.toml\t%s\n' "$nm" "$d" "$nm" "$d" >> "$d/universe.tsv"
      local deps="" dfirst=1
      for dep in ${dl//,/ }; do
        [ -n "$dep" ] || continue
        [ $dfirst = 1 ] || deps="$deps,"; dfirst=0
        deps="$deps{\"name\":\"$dep\",\"kind\":null}"
      done
      [ $first = 1 ] || pkgs="$pkgs,"; first=0
      pkgs="$pkgs{\"name\":\"$nm\",\"dependencies\":[$deps]}"
    done
    printf '{"packages":[%s]}' "$pkgs" > "$d/meta.json"
    { echo '#!/usr/bin/env python3'; echo "import sys; sys.stdout.write(open('$d/universe.tsv').read())"; } > "$d/scripts/lib/cascade_universe.py"
    { local i=1; for spec in ${t//,/ }; do echo "TIERS[${spec#*:}]=\"${spec%%:*}\""; done; } > "$d/scripts/cascade-publish.sh"
    cat > "$d/cargo" <<EOS
#!/usr/bin/env bash
cat "$d/meta.json"
EOS
    chmod +x "$d/cargo"
  }
  row() { # row <want rc> <label> <dir>
    n=$((n+1)); local want=$1 label=$2 dir=$3 rc=0
    ( export PATH="$dir:$PATH"; analyse "$dir" ) > "$TD/out.$n" 2>&1 || rc=$?
    if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$TD/out.$n"; bad=1; fi
  }
  # 12 crates so the anti-vacuity floor (>=10) is cleared and the graph is the subject.
  CH=""; TI=""
  for i in 1 2 3 4 5 6 7 8 9 10; do CH="$CH|c$i;"; TI="$TI,c$i:1"; done
  # 1. DEEP BUT FINITE: a chain that needs many drain passes but terminates.
  fixture "$TD/deep" "${CH#|}|d1;d2|d2;d3|d3;" "${TI#,},d1:1,d2:9,d3:9"
  row 0 "deep-but-finite chain: converges" "$TD/deep"
  # 2. THE PLANTED CYCLE: two crates each waiting on the other. Depth is undefined;
  #    a pass-counter would loop. Must be named as a cycle.
  fixture "$TD/cycle" "${CH#|}|a;b|b;a" "${TI#,},a:1,b:1"
  row 1 "planted CYCLE a<->b: refused, not counted" "$TD/cycle"
  grep -q 'CYCLE' "$TD/out.2" || { echo "FAIL  row 2 did not name it a cycle"; bad=1; }
  # 3. ANTI-VACUITY: an enumeration that saw almost nothing must not read as convergence.
  fixture "$TD/empty" "x;" "x:1"
  row 1 "near-empty universe: refused as an instrument failure" "$TD/empty"
  grep -q 'not a proof' "$TD/out.3" || { echo "FAIL  row 3 did not refuse as vacuity"; bad=1; }
  # 4. the real tree
  row 0 "the REAL tree converges" "$REPO_ROOT"
  # THE MUTANT. Rows 1-4 prove the cases read the guard; they do not prove the STUCK
  # branch is what refuses the cycle. Delete that branch in a copy and run it against the
  # cycle fixture: without it the drain can never terminate, so the copy must NOT exit 0.
  # A guard whose refusal survives its own deletion is decoration.
  MUT="$TD/mutant.sh"
  sed 's/    if set(still) == before:/    if False:/' "$0" > "$MUT"
  if cmp -s "$0" "$MUT"; then
    echo "FAIL  mutant did not apply — the cycle rows prove nothing"; bad=1
  else
    mrc=0
    ( export PATH="$TD/cycle:$PATH"; timeout 10 bash "$MUT" --mutant-analyse "$TD/cycle" ) >/dev/null 2>&1 || mrc=$?
    if [ "$mrc" = 0 ]; then
      echo "FAIL  mutant SURVIVED: the cycle fixture still exits 0 with the STUCK branch deleted"; bad=1
    else
      printf 'ok    mutant  STUCK branch deleted -> cycle fixture no longer passes (rc=%s)\n' "$mrc"
    fi
  fi
  printf '%s/%s rows\n' "$((n - bad))" "$n"
  [ "$bad" = 0 ] || exit 1
  exit 0
fi

# Used only by the mutant harness above, so a copy of this script can be pointed at one
# fixture without re-running the whole table.
if [ "${1:-}" = "--mutant-analyse" ]; then analyse "$2"; exit $?; fi

analyse "$REPO_ROOT"
