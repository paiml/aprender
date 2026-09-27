#!/usr/bin/env bash
# check_lean_modules_reachable.sh -- every Lean module is built, because the root imports it (#4244)
#
#   bash scripts/check_lean_modules_reachable.sh              # judge the staging Lean tree
#   bash scripts/check_lean_modules_reachable.sh --self-test  # mutation rows against copies
#
# `lake build ProvableContracts` and `leanchecker ProvableContracts` see only the modules
# reachable by import from ProvableContracts.lean. 59 of 162 were not: nothing compiled
# them, so 17 had stopped elaborating (renamed Mathlib lemmas, missing imports) while
# their theorems still counted as proved and contracts still bound them. Importing them
# also exposed two duplicate theorem names that broke the root build outright.
#
# This guard is static (no Lean toolchain): it walks `import ProvableContracts.*` lines
# from the root and fails on any .lean file under ProvableContracts/ the walk never
# reaches, and on an import that names a module with no file.
#
# EXIT 0 every module reachable · 1 an orphan or a dangling import · 2 the tree moved.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LEAN="$ROOT/crates/aprender-contracts-staging/lean"

# reach DIR -> prints `orphan <Module>` / `dangling <Module>` lines; rc 0 clean, 1 not, 2 no root
reach() {
    [ -f "$1/ProvableContracts.lean" ] && [ -d "$1/ProvableContracts" ] || { echo "no root in $1"; return 2; }
    python3 - "$1" <<'EOF'
import os, re, sys
d = sys.argv[1]
imp = re.compile(r'^\s*import\s+(.+)')
def imports(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = imp.match(line)
            if m:
                out += [x for x in m.group(1).split() if x.startswith("ProvableContracts.")]
    return out
mods = {}
for dp, _, fs in os.walk(os.path.join(d, "ProvableContracts")):
    for f in fs:
        if f.endswith(".lean"):
            p = os.path.join(dp, f)
            mods[os.path.relpath(p, d)[:-5].replace(os.sep, ".")] = p
seen, dangling, todo = set(), set(), imports(os.path.join(d, "ProvableContracts.lean"))
while todo:
    m = todo.pop()
    if m in seen:
        continue
    if m not in mods:
        dangling.add(m)
        continue
    seen.add(m)
    todo += imports(mods[m])
bad = [f"orphan {m}" for m in sorted(set(mods) - seen)] + [f"dangling {m}" for m in sorted(dangling)]
print("\n".join(bad) if bad else f"reachable {len(seen)}/{len(mods)}")
sys.exit(1 if bad else 0)
EOF
}

judge() {
    local out rc
    out=$(reach "$LEAN"); rc=$?
    if [ "$rc" -eq 0 ]; then echo "PASS  every Lean module is imported by ProvableContracts.lean ($out) (#4244)"; return 0; fi
    echo "FAIL  modules lake build never compiles (import them from ProvableContracts.lean):"
    sed 's/^/        /' <<< "$out"
    return "$rc"
}

self_test() {
    local d fail=0 row want name rc
    d=$(mktemp -d) || return 2
    # copies of the real tree: a mutation proves the guard sees THIS tree's shape
    cp -r -- "$LEAN/ProvableContracts" "$LEAN/ProvableContracts.lean" "$d/" 2>/dev/null || { rmdir -- "$d"; return 2; }
    mk() { rm -rf -- "${d:?}/$1"; mkdir -p -- "$d/$1"; cp -r -- "$d/ProvableContracts" "$d/ProvableContracts.lean" "$d/$1/"; }
    # unimported: Image.Canny is a leaf (only the root imports it); Defs.GPU would stay reachable
    # through Theorems.GPU.DimensionIndependence, which is the `transitive` row.
    mk real
    mk unimported; grep -v '^import ProvableContracts.Theorems.Image.Canny$' "$d/ProvableContracts.lean" > "$d/unimported/ProvableContracts.lean"
    mk newfile;    printf 'theorem t : True := trivial\n' > "$d/newfile/ProvableContracts/Orphan.lean"
    mk dangling;   printf 'import ProvableContracts.NoSuchModule\n' >> "$d/dangling/ProvableContracts.lean"
    mk transitive; printf 'import ProvableContracts.Defs.GPU\n' > "$d/transitive/ProvableContracts/Hub.lean"
    grep -v '^import ProvableContracts.Defs.GPU$' "$d/ProvableContracts.lean" > "$d/transitive/ProvableContracts.lean"
    printf 'import ProvableContracts.Hub\n' >> "$d/transitive/ProvableContracts.lean"
    mk noroot;     rm -f -- "$d/noroot/ProvableContracts.lean"
    for row in "0 real" "1 unimported" "1 newfile" "1 dangling" "0 transitive" "2 noroot"; do
        want=${row%% *}; name=${row#* }
        reach "$d/$name" > /dev/null; rc=$?
        if [ "$rc" = "$want" ]; then echo "  ok   $name -> rc $rc"; else echo "  FAIL $name: wanted rc $want, got $rc"; fail=1; fi
    done
    rm -rf -- "${d:?}"
    [ "$fail" -eq 0 ] && { echo "check_lean_modules_reachable self-test: PASS"; return 0; }
    echo "check_lean_modules_reachable self-test: FAIL"; return 1
}

case "${1:-}" in
    --self-test) self_test ;;
    '') judge ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
