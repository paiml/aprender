#!/usr/bin/env bash
# falsify_one_command.sh -- case table for scripts/release/one_command.sh (KZ-001 b, aprender#6729)
#   bash scripts/release/falsify_one_command.sh     # exit 0 = every row held
# one_command.sh runs: bump -> clean-room on the bump sha -> tag -> publish. All of git, gh, bump and
# publish are stubs on PATH that append their argv to $CALLS; nothing real is touched.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SRC="${ONE_CMD_SRC:-$HERE/one_command.sh}"
if [ "${1:-}" = --mutants ]; then
  M=$(mktemp -d) || exit 2; trap 'rm -rf -- "${M:?}"' EXIT; miss=0
  mut() { sed "$2" "$HERE/one_command.sh" > "$M/m.sh"; cmp -s "$HERE/one_command.sh" "$M/m.sh" && { echo "MUTANT $1 did not apply"; miss=$((miss+1)); return; }
    ONE_CMD_SRC="$M/m.sh" bash "$0" > /dev/null 2>&1 && { echo "MUTANT $1 SURVIVED"; miss=$((miss+1)); }; }
  mut sha-check-dropped 's/^\[ "\$tested" = "\$SHA" \] .*$/:/'
  mut success-check-dropped 's/^\[ "\$c" = success \] .*$/:/'
  mut dry-run-ignored 's/^if \[ "\$DRY" = 1 \]; then$/if false; then/'
  mut tag-before-verdict 's/^git push -q origin "\$BR".*$/git tag -a "$T" -m x "$SHA"/'
  [ "$miss" = 0 ] && { echo "all mutants caught"; exit 0; }; exit 1
fi
[ -f "$SRC" ] || { echo "FAIL one_command.sh missing at $SRC"; exit 1; }
T=$(mktemp -d) || exit 2; trap 'rm -rf -- "${T:?}"' EXIT
BIN=$T/bin; mkdir -p "$BIN"
SHA=1111111111111111111111111111111111111111
mkstub() { printf '#!/usr/bin/env bash\necho "%s $*" >> "$CALLS"\n%s\n' "$1" "$2" > "$BIN/$1"; chmod +x "$BIN/$1"; }
fails=0
row() { # name want_rc want_calls_regex forbid_calls_regex dry cr_conclusion cr_sha
  local name=$1 wrc=$2 want=$3 forbid=$4 dry=$5 concl=$6 crsha=$7
  export CALLS=$T/calls; : > "$CALLS"
  mkstub git 'case "$1" in rev-parse) echo '$SHA';; esac; exit 0'
  mkstub bump 'exit 0'
  mkstub publish 'exit 0'
  mkstub gh 'case "$*" in *"run list"*) echo 42;; *"--log"*) echo "tested-sha: '"$crsha"'";; *"run view"*) echo "'"$concl"'";; esac; exit 0'
  PATH=$BIN:$PATH ONE_CMD_BUMP=bump ONE_CMD_PUBLISH=publish ONE_CMD_POLL=0 ONE_CMD_ARM=true bash "$SRC" 0.73.0 "$dry" > "$T/out" 2>&1
  local rc=$?; [ -n "${DBG:-}" ] && cat "$T/out"
  [ "$rc" = "$wrc" ] || { echo "FAIL $name: rc=$rc want $wrc"; fails=$((fails+1)); return; }
  [ -z "$want" ] || grep -Eq -- "$want" "$CALLS" || { echo "FAIL $name: no call /$want/"; fails=$((fails+1)); return; }
  [ -z "$forbid" ] || ! grep -Eq -- "$forbid" "$CALLS" || { echo "FAIL $name: forbidden call /$forbid/"; fails=$((fails+1)); return; }
  echo "ok   $name"
}
row green-tags-then-publishes  0 '^git tag .*v0\.73\.0 '$SHA 'ZZZ'          0 success $SHA
row dry-run-never-tags         0 '^bump'                      '^(git tag|publish)' 1 success $SHA
row red-cleanroom-no-tag       1 ''                           '^(git tag|publish)' 0 failure $SHA
row wrong-sha-no-tag           1 ''                           '^(git tag|publish)' 0 success deadbeef
row absent-verdict-no-tag      1 ''                           '^(git tag|publish)' 0 "" ""
[ "$fails" = 0 ] && { echo "all rows held"; exit 0; }
echo "$fails row(s) failed"; exit 1
