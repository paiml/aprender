#!/usr/bin/env bash
# check_bookkeeping_autofix.sh -- the case table for scripts/release/bookkeeping_autofix.sh + autofix_invariants.py
# (#4045 M8), every mutant killed by its NAMED row. Each row runs the SHIPPED fixer in a scratch git repo whose
# regeneration commands are seams (AUTOFIX_CMD_<FIXER>), so what is proven is the fixer's own logic: a fix is a
# commit on its own branch, the start branch and the tree are left as they were, a claim literal may only MOVE,
# a complexity number may only SHRINK, and a refused or failed fix restores the tree.
set -uo pipefail
case "${1:-}" in -h|--help) echo "usage: bash scripts/check_bookkeeping_autofix.sh   (the case table; no arguments)"; exit 0 ;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
T=$(mktemp -d) || exit 2
cleanup() { if [ -n "$T" ] && [ "$T" != "/" ] && [ -d "$T" ]; then rm -rf -- "$T"; fi; }
trap cleanup EXIT
g() { git -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t "$@"; }

scratch() { # scratch <fixer script> <invariants lib> -> a repo path: baseline written at c1, the literal drifted in c2
  local r; r=$(mktemp -d -p "$T" repo-XXXXXX)
  mkdir -p "$r/scripts/release" "$r/scripts/lib" "$r/src" "$r/contracts"
  cp "$1" "$r/scripts/release/bookkeeping_autofix.sh"; cp "$2" "$r/scripts/lib/autofix_invariants.py"
  printf 'fn a() {}\n// Performance: 800+ tok/s (2.8x Ollama)\n' > "$r/src/s.rs"
  printf '# claim literals\nsrc/s.rs:2\n' > "$r/scripts/claim_literal_baseline.txt"
  printf '# complexity\nsrc/s.rs::a 40 30\n' > "$r/scripts/complexity_baseline.txt"
  echo '{"n_files": 1}' > "$r/contracts/census.json"
  g -C "$r" init -q -b main; g -C "$r" add -A; g -C "$r" commit -qm c1
  printf '// header\nfn a() {}\n// Performance: 800+ tok/s (2.8x Ollama)\n' > "$r/src/s.rs"   # the literal drifts to :3
  g -C "$r" commit -qam c2
  printf '%s' "$r"
}
run() { # run <repo> <fixer> [ENV=VAL...] -> sets OUT, RC
  local r=$1 fx=$2; shift 2
  OUT=$(cd "$r" && env "$@" bash scripts/release/bookkeeping_autofix.sh "$fx" 2>&1); RC=$?
}
clean_on_main() { [ -z "$(git -C "$1" status --porcelain --untracked-files=no)" ] && [ "$(git -C "$1" symbolic-ref -q --short HEAD)" = main ]; }

table() { # table <fixer script> <lib> -> ok/FAIL lines
  local f=$1 l=$2 r
  r=$(scratch "$f" "$l"); run "$r" census 'AUTOFIX_CMD_CENSUS=echo {\"n_files\": 2} > contracts/census.json'
  if [ "$RC" = 0 ] && grep -q '^AUTOFIX census: proposed .* on autofix/census-' <<< "$OUT" && clean_on_main "$r" \
     && [ "$(g -C "$r" rev-list --count main)" = 2 ] && g -C "$r" show "autofix/census-$(g -C "$r" rev-parse --short=9 main):contracts/census.json" 2> /dev/null | grep -q '"n_files": 2'; then
    echo "ok    census-proposed-on-its-own-branch"; else echo "FAIL  census-proposed-on-its-own-branch -- rc $RC: $OUT"; fi
  r=$(scratch "$f" "$l"); run "$r" readme 'AUTOFIX_CMD_README=true'
  if [ "$RC" = 0 ] && grep -q '^AUTOFIX readme: nothing to fix' <<< "$OUT"; then echo "ok    nothing-to-fix"; else echo "FAIL  nothing-to-fix -- $OUT"; fi
  r=$(scratch "$f" "$l"); run "$r" claims 'AUTOFIX_CMD_CLAIMS=printf "# claim literals\nsrc/s.rs:3\n" > scripts/claim_literal_baseline.txt'
  if [ "$RC" = 0 ] && grep -q '^AUTOFIX claims: proposed' <<< "$OUT" && clean_on_main "$r"; then echo "ok    claims-pure-drift-proposed"
  else echo "FAIL  claims-pure-drift-proposed -- rc $RC: $OUT"; fi
  r=$(scratch "$f" "$l"); run "$r" claims 'AUTOFIX_CMD_CLAIMS=printf "// NEW: 10x faster than llama.cpp\n" >> src/s.rs && printf "# claim literals\nsrc/s.rs:3\nsrc/s.rs:4\n" > scripts/claim_literal_baseline.txt'
  if [ "$RC" = 1 ] && grep -q '^AUTOFIX claims: REFUSED -- claim literal ADDED in src/s.rs' <<< "$OUT" && clean_on_main "$r" \
     && ! g -C "$r" rev-parse -q --verify "refs/heads/autofix/claims-$(g -C "$r" rev-parse --short=9 main)" > /dev/null; then
    echo "ok    claims-added-refused-and-restored"; else echo "FAIL  claims-added-refused-and-restored -- rc $RC: $OUT"; fi
  r=$(scratch "$f" "$l"); run "$r" complexity 'AUTOFIX_CMD_COMPLEXITY=printf "# complexity\nsrc/s.rs::a 35 30\n" > scripts/complexity_baseline.txt'
  if [ "$RC" = 0 ] && grep -q '^AUTOFIX complexity: proposed' <<< "$OUT"; then echo "ok    complexity-shrink-proposed"; else echo "FAIL  complexity-shrink-proposed -- $OUT"; fi
  r=$(scratch "$f" "$l"); run "$r" complexity 'AUTOFIX_CMD_COMPLEXITY=printf "# complexity\nsrc/s.rs::a 45 30\n" > scripts/complexity_baseline.txt'
  if [ "$RC" = 1 ] && grep -q 'complexity baseline RAISES src/s.rs::a' <<< "$OUT" && clean_on_main "$r"; then echo "ok    complexity-raise-refused"
  else echo "FAIL  complexity-raise-refused -- $OUT"; fi
  r=$(scratch "$f" "$l"); run "$r" complexity 'AUTOFIX_CMD_COMPLEXITY=printf "# complexity\nsrc/s.rs::a 40 30\nsrc/s.rs::b 50 50\n" > scripts/complexity_baseline.txt'
  if [ "$RC" = 1 ] && grep -q 'complexity baseline ADDS src/s.rs::b' <<< "$OUT"; then echo "ok    complexity-add-refused"
  else echo "FAIL  complexity-add-refused -- $OUT"; fi
  r=$(scratch "$f" "$l"); run "$r" census 'AUTOFIX_CMD_CENSUS=echo broken > contracts/census.json; exit 1'
  if [ "$RC" = 1 ] && grep -q '^AUTOFIX census: FAILED' <<< "$OUT" && clean_on_main "$r"; then echo "ok    failed-regen-restored"
  else echo "FAIL  failed-regen-restored -- $OUT"; fi
  r=$(scratch "$f" "$l"); echo dirty >> "$r/src/s.rs"; run "$r" census 'AUTOFIX_CMD_CENSUS=true'
  if [ "$RC" = 2 ] && grep -q 'must start clean' <<< "$OUT"; then echo "ok    dirty-tree-refused"; else echo "FAIL  dirty-tree-refused -- $OUT"; fi
}
bad=0
F="$ROOT/scripts/release/bookkeeping_autofix.sh"; L="$ROOT/scripts/lib/autofix_invariants.py"
out=$(table "$F" "$L"); printf '%s\n' "$out"; grep -q '^FAIL' <<< "$out" && bad=1
mutant() { # mutant <label> <row> <file f|l> <old> <new>
  local src m mo; if [ "$3" = f ]; then src=$F; else src=$L; fi
  m="$T/mut-$1.$( [ "$3" = f ] && echo sh || echo py)"
  python3 -c 'import sys; s=open(sys.argv[1]).read(); assert s.count(sys.argv[3])==1; open(sys.argv[2],"w").write(s.replace(sys.argv[3],sys.argv[4]))' \
    "$src" "$m" "$4" "$5" 2> /dev/null || { echo "FAIL  mutant $1 did not apply"; bad=1; return; }
  if [ "$3" = f ]; then mo=$(table "$m" "$L"); else mo=$(table "$F" "$m"); fi
  if grep -q "^FAIL  $2 " <<< "$mo"; then echo "ok    mutant $1 killed by $2"; else echo "FAIL  mutant $1 SURVIVED $2"; bad=1; fi
}
mutant invariant-skipped   claims-added-refused-and-restored f '  if ! why=$(invariant "$fx"); then' '  if false; then'
mutant restore-skipped     claims-added-refused-and-restored f '    git checkout -q -- . ; continue' '    :'  # bashrs disable-line=SC2242
mutant own-branch-skipped  census-proposed-on-its-own-branch f '  git checkout -q -b "$br" && git add -u' '  git checkout -q main && git add -u'
mutant drift-read-at-head  claims-pure-drift-proposed f 'capture_output=True, text=True).stdout.strip() or base' 'capture_output=True, text=True).stdout.strip() and base or base'
mutant move-only-off       claims-added-refused-and-restored l '        extra = c - by_old.get(p, Counter())' '        extra = Counter()'
mutant raise-ok            complexity-raise-refused l '            if c > oc or g > og:' '            if False:'
mutant add-ok              complexity-add-refused l '        if k not in old:' '        if False:'
echo "check_bookkeeping_autofix: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
exit "$bad"
