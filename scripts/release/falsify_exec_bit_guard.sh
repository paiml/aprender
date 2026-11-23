#!/usr/bin/env bash
# falsify_exec_bit_guard.sh -- case table for scripts/release/exec_bit_guard.sh (PMAT-6729, aprender#6774)
#   bash scripts/release/falsify_exec_bit_guard.sh            # exit 0 = every row held
#   bash scripts/release/falsify_exec_bit_guard.sh --mutants  # exit 0 = every guard mutant is caught
# One row per FORM VARIANT of a direct call, each in a throwaway git repo with the target at a given mode.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SRC="${EXEC_GUARD_SRC:-$HERE/exec_bit_guard.sh}"
if [ "${1:-}" = --mutants ]; then
  M=$(mktemp -d) || exit 2; trap 'rm -rf -- "${M:?}"' EXIT; miss=0
  mut() { sed "$2" "$HERE/exec_bit_guard.sh" > "$M/m.sh"; cmp -s "$HERE/exec_bit_guard.sh" "$M/m.sh" && { echo "MUTANT $1 did not apply"; miss=$((miss+1)); return; }
    EXEC_GUARD_SRC="$M/m.sh" bash "$0" > /dev/null 2>&1 && { echo "MUTANT $1 SURVIVED"; miss=$((miss+1)); }; }
  mut mode-check-dropped 's/^    \[ "\$mode" = 100755 \] || {/    false || {/'
  mut default-form-dropped 's/|:-'"'"'$/'"'"'/'
  mut zero-calls-passes 's/^\[ "\$calls" -gt 0 \] || {.*$/:/'
  mut final-verdict-ignored 's/^\[ "\$bad" = 0 \]$/true/'
  mut comments-scanned "s/grep -vE '\^\[\[:space:\]\]\*#' \"\$f\" |/cat \"\$f\" |/"
  [ "$miss" = 0 ] && { echo "all mutants caught"; exit 0; }; exit 1
fi
[ -f "$SRC" ] || { echo "FAIL guard missing at $SRC"; exit 1; }
T=$(mktemp -d) || exit 2; trap 'rm -rf -- "${T:?}"' EXIT
fails=0
# row NAME WANT_RC FILE_PATH FILE_BODY TARGET_MODE   (a 100755 sibling called directly keeps "calls scanned" > 0 unless NAME=blind)
row() {
  local name=$1 want=$2 path=$3 body=$4 mode=$5 r="$T/$1"
  mkdir -p "$r/scripts" "$r/.github/workflows" && git -C "$r" init -q || { echo "FAIL $name: fixture"; fails=$((fails+1)); return; }
  printf '#!/usr/bin/env bash\ntrue\n' > "$r/scripts/t.sh"; cp "$r/scripts/t.sh" "$r/scripts/ok.sh"
  [ "$name" = blind ] || printf '#!/usr/bin/env bash\ntrue && scripts/ok.sh\n' > "$r/scripts/base.sh"
  printf '%b' "$body" > "$r/$path"
  git -C "$r" add -A && git -C "$r" update-index -q --chmod=+x scripts/ok.sh scripts/base.sh 2> /dev/null
  [ "$mode" = 755 ] && git -C "$r" update-index -q --chmod=+x scripts/t.sh || git -C "$r" update-index -q --chmod=-x scripts/t.sh
  bash "$SRC" "$r" > "$T/out" 2>&1; local rc=$?
  [ "$rc" = "$want" ] && echo "ok   $name" || { echo "FAIL $name: rc=$rc want $want"; fails=$((fails+1)); }
}
W=.github/workflows/w.yml
row wf-run-inline        1 $W 'jobs:\n  a:\n    steps:\n      - run: scripts/t.sh\n' 644
row wf-block-line        1 $W 'jobs:\n  a:\n    steps:\n      - run: |\n          set -e\n          scripts/t.sh --x\n' 644
row wf-dot-slash         1 $W '      - run: ./scripts/t.sh\n' 644
row sh-default-value     1 scripts/c.sh 'BUMP="${ONE_CMD_BUMP:-scripts/t.sh}"\n' 644
row sh-and-chain         1 scripts/c.sh 'true && scripts/t.sh 1\n' 644
row sh-cmd-subst         1 scripts/c.sh 'x=$(scripts/t.sh)\n' 644
row sh-semicolon         1 scripts/c.sh 'true; scripts/t.sh\n' 644
row sh-pipe              1 scripts/c.sh 'echo | scripts/t.sh\n' 644
row wf-bash-prefix-ok    0 $W '      - run: bash scripts/t.sh\n' 644
row sh-bash-default-ok   0 scripts/c.sh 'BUMP="${ONE_CMD_BUMP:-bash scripts/t.sh}"\n' 644
row wf-direct-exec-ok    0 $W '      - run: scripts/t.sh\n' 755
row sh-case-pattern-ok   0 scripts/c.sh 'case "$f" in\nscripts/t.sh) :;;\nesac\n' 644
row wf-paths-list-ok     0 $W "    paths:\n      - 'scripts/t.sh'\n" 644
row sh-comment-ok        0 scripts/c.sh '# true && scripts/t.sh\n' 644
row blind                2 scripts/c.sh 'true\n' 644
[ "$fails" = 0 ] && { echo "all rows held"; exit 0; }
echo "$fails row(s) failed"; exit 1
