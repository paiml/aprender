#!/usr/bin/env bash
# check_mutants_feature_groups.sh -- red/green proof for the survivor table's feature groups (#4587 A', M1 FALSE-RED).
#
# mutants_table_shard.sh runs each ci/mutants-feature-groups.txt row as its own feature-enabled cargo-mutants run
# and merges the outcomes. This proves, with a fake cargo (the MUTANTS_GATE_CARGO seam) driving the REAL shard
# script and the REAL checker (scripts/ci/mutants_survivor_table.py check), that the change measures feature-gated
# mutants and weakens nothing:
#   green   every mutant caught (default and gated)                  -> checker GREEN
#   red     a planted survivor in a feature-gated file               -> checker RED
#   red     a planted survivor in a default file                     -> checker RED
#   red     a group run that tests nothing (empty outcomes)          -> checker RED (MISSING)
#   red     no group rows at all: gated mutants run default, MISSED  -> checker RED (the old false-red stays RED)
#   red     a group run that writes no outcomes.json                 -> the shard is dead (rc 1)
#   red     a cuda row                                               -> the shard refuses it (rc 1)
# The fake cargo also dies on any run without --copy-vcs, so dropping it from either call turns all-caught DEAD.
# It also checks the real groups file parses and names only files that exist.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
d=$(mktemp -d "${TMPDIR:-/tmp}/mutants-fg.XXXXXX") || exit 2
rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "${1:?}" ;; esac; return 0; }
trap 'rmtree "${d:-}"' EXIT
cat > "$d/cargo" <<'FAKE'
#!/usr/bin/env python3
# fake `cargo mutants`: universe = 3 default mutants + 2 in gated.rs; --shard k/n takes index % n == k.
import json, os, sys
a = sys.argv[2:]
U = ["src/a.rs:1:1: replace a", "src/a.rs:2:1: replace b", "src/b.rs:1:1: replace c",
     "src/gated.rs:1:1: replace g1", "src/gated.rs:2:1: replace g2"]
def opt(name):
    return a[a.index(name) + 1] if name in a else None
excl = [a[i + 1] for i, x in enumerate(a) if x == "--exclude"]
grp = "-p" in a
sel = [m for m in U if (m.startswith(opt("--file")) if grp else not any(m.startswith(e) for e in excl))]
if opt("--shard"):
    k, n = map(int, opt("--shard").split("/"))
    sel = [m for m in sel if U.index(m) % n == k]
if "--list" not in a and "--copy-vcs" not in a:
    sys.exit(4)   # a run without .git in its scratch copy: the real baseline fails (run 36538406766)
if "--list" in a:
    print("\n".join(U if not grp and not opt("--shard") else sel)); sys.exit(0)
mode = os.environ.get("FAKE_MODE", "")
if grp and mode == "group-no-outcomes":
    sys.exit(3)
if grp and mode == "group-empty":
    sel = []
def summ(m):
    if os.environ.get("FAKE_MISS") and m.startswith(os.environ["FAKE_MISS"]): return "MissedMutant"
    if not grp and m.startswith("src/gated.rs"): return "MissedMutant"   # never compiled without the feature
    return "CaughtMutant"
o = os.path.join(opt("--output"), "mutants.out"); os.makedirs(o, exist_ok=True)
json.dump({"outcomes": [{"scenario": "Baseline", "summary": "Success"}] +
           [{"scenario": {"Mutant": {"name": m}}, "summary": summ(m)} for m in sel]},
          open(os.path.join(o, "outcomes.json"), "w"))
FAKE
chmod +x "$d/cargo"
printf 'diff --git a/x b/x\n' > "$d/pr.diff"
printf '# test rows\npkg|feat|src/gated.rs|--lib\n' > "$d/groups.txt"
printf '# none\n' > "$d/nogroups.txt"
printf 'pkg|cuda|src/gated.rs|--lib\n' > "$d/cuda.txt"
head=$(git -C "$ROOT" rev-parse HEAD) || exit 2
bad=0
# case <name> <groups file> <want: GREEN|RED|DEAD> [env...]
case_() {
    local name=$1 gf=$2 want=$3 got; shift 3
    rmtree "$d/t"; mkdir -p "$d/t"
    for k in 1 2; do
        if ! (cd "$ROOT" && env MUTANTS_GATE_CARGO="$d/cargo" MUTANTS_FEATURE_GROUPS="$gf" "$@" \
              bash scripts/mutants_table_shard.sh "$d/pr.diff" "$k" 2 1 "$d/t/mutants-shard-$k") > "$d/t/log" 2>&1; then
            got=DEAD; break
        fi
        got=""
    done
    if [ -z "$got" ]; then
        if python3 "$ROOT/scripts/ci/mutants_survivor_table.py" check --dir "$d/t" --shards 2 --head-sha "$head" \
             --diff "$d/pr.diff" --repo "$ROOT" > "$d/t/chk" 2>&1; then got=GREEN; else got=RED; fi
    fi
    if [ "$got" = "$want" ]; then printf 'ok    %-26s -> %s\n' "$name" "$got"
    else printf 'FAIL  %-26s wanted %s, got %s\n' "$name" "$want" "$got"; sed 's/^/      /' "$d/t/log" "$d/t/chk" 2>/dev/null | tail -6; bad=1; fi
}
echo "=== survivor-table feature groups: red/green over the real shard script + checker ==="
case_ all-caught          "$d/groups.txt"   GREEN
case_ gated-survivor      "$d/groups.txt"   RED  FAKE_MISS=src/gated.rs:2
case_ default-survivor    "$d/groups.txt"   RED  FAKE_MISS=src/b.rs
case_ group-tests-nothing "$d/groups.txt"   RED  FAKE_MODE=group-empty
case_ no-group-rows       "$d/nogroups.txt" RED
case_ group-no-outcomes   "$d/groups.txt"   DEAD FAKE_MODE=group-no-outcomes
case_ cuda-row            "$d/cuda.txt"     DEAD
echo "=== the real groups file: every row well-formed, every file present, no cuda ==="
while IFS='|' read -r p f file t; do
    case "$p" in ''|'#'*) continue ;; esac
    if [ -n "$f" ] && [ -f "$ROOT/$file" ] && [ -n "$t" ] && [[ "$f" != *cuda* ]]; then printf 'ok    %s --features %s %s\n' "$p" "$f" "$file"
    else printf 'FAIL  bad row %s|%s|%s|%s\n' "$p" "$f" "$file" "$t"; bad=1; fi
done < "$ROOT/ci/mutants-feature-groups.txt"
[ "$bad" = 0 ] && { echo PASS; exit 0; }
echo FAIL >&2; exit 1
