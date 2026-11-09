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
  mut first-match-unanchored "s/^tested=.*\$/tested=\$(gh run view \"\$id\" --repo \"\$INFRA\" --log | sed -n 's\/.*tested-sha: *\/\/p' | head -1)/"
  mut success-check-dropped 's/^\[ "\$c" = success \] .*$/:/'
  mut dry-run-ignored 's/^if \[ "\$DRY" = 1 \]; then$/if false; then/'
  mut tag-before-verdict 's/^git push -q origin "\$BR".*$/git tag -a "$T" -m x "$SHA"/'
  # T5: built from the handed nightly sha, never main HEAD; next-dev bump opened after the tag, and its failure is red
  mut base-ignored 's/^git checkout -q -b "\$BR" "\$BASE"/git checkout -q -b "$BR"/'
  mut base-shape-unchecked 's/^\[\[ \$BASE =~ .*$/:/'
  mut base-off-main-allowed 's/^git merge-base --is-ancestor .*$/:/'
  mut next-dev-skipped 's/^late=0; \$NEXTDEV "\$V" || /late=0; : || /'
  mut next-dev-red-swallowed 's/^\[ "\$late" = 0 \] || die .*$/:/'
  mut next-dev-before-tag 's/^\$ARM "\$T" 120 || die/$NEXTDEV "$V"; &/'
  [ "$miss" = 0 ] && { echo "all mutants caught"; exit 0; }; exit 1
fi
[ -f "$SRC" ] || { echo "FAIL one_command.sh missing at $SRC"; exit 1; }
T=$(mktemp -d) || exit 2; trap 'rm -rf -- "${T:?}"' EXIT
BIN=$T/bin; mkdir -p "$BIN"
SHA=1111111111111111111111111111111111111111
BASE=2222222222222222222222222222222222222222   # the green nightly sha the release is handed (T5)
mkstub() { printf '#!/usr/bin/env bash\necho "%s $*" >> "$CALLS"\n%s\n' "$1" "$2" > "$BIN/$1"; chmod +x "$BIN/$1"; }
fails=0
row() { # name want_rc want_calls_regex forbid_calls_regex dry cr_conclusion cr_sha
  local name=$1 wrc=$2 want=$3 forbid=$4 dry=$5 concl=$6 crsha=$7
  export CALLS=$T/calls; : > "$CALLS"
  mkstub git 'case "$1" in rev-parse) echo '$SHA';; merge-base) [ -z "${NOTMAIN:-}" ] || exit 1;; esac; exit 0'
  mkstub nextdev '[ -z "${NEXTDEV_FAIL:-}" ] || exit 1; exit 0'
  mkstub bump 'exit 0'
  mkstub publish 'exit 0'
  mkstub gh 'case "$*" in *"run list"*) echo 42;; *"--log"*) [ -z "${DECOY:-}" ] || echo "2026-11-05T00:00:00Z   claim step summary carries \"tested-sha: <40hex>\""; echo "2026-11-05T00:00:01Z     tested-sha: '"$crsha"'";; *"run view"*) echo "'"$concl"'";; esac; exit 0'
  local a=(0.73.0 "$dry"); [ -n "${NOBASE:-}" ] || a+=("${BASEARG:-$BASE}")
  PATH=$BIN:$PATH ONE_CMD_BUMP=bump ONE_CMD_PUBLISH=publish ONE_CMD_NEXTDEV=nextdev ONE_CMD_POLL=0 ONE_CMD_ARM=true bash "$SRC" "${a[@]}" > "$T/out" 2>&1
  local rc=$?; [ -n "${DBG:-}" ] && cat "$T/out"
  [ "$rc" = "$wrc" ] || { echo "FAIL $name: rc=$rc want $wrc"; fails=$((fails+1)); return; }
  [ -z "$want" ] || grep -Eq -- "$want" "$CALLS" || { echo "FAIL $name: no call /$want/"; fails=$((fails+1)); return; }
  [ -z "$forbid" ] || ! grep -Eq -- "$forbid" "$CALLS" || { echo "FAIL $name: forbidden call /$forbid/"; fails=$((fails+1)); return; }
  echo "ok   $name"
}
row green-tags-then-publishes  0 '^git tag .*v0\.73\.0 '$SHA 'ZZZ'          0 success $SHA
row dry-run-never-tags         0 '^bump'                      '^(git tag|publish|nextdev)' 1 success $SHA
row red-cleanroom-no-tag       1 ''                           '^(git tag|publish|nextdev)' 0 failure $SHA
# T5 (APR-RELEASE-TRAIN): the release branch starts at the green nightly sha, never main HEAD
row builds-from-nightly-sha    0 '^git checkout -q -b release/0\.73\.0 '$BASE'$' 'ZZZ'  0 success $SHA
NOBASE=1 row no-sha-never-builds 2 ''                         '^(git checkout|bump|git tag|publish|nextdev)' 0 success $SHA
BASEARG=main row branch-name-is-not-a-sha 2 ''                '^(git checkout|bump|git tag|publish|nextdev)' 0 success $SHA
NOTMAIN=1 row sha-off-main-stops 1 ''                         '^(git checkout|bump|git tag|publish|nextdev)' 0 success $SHA
# T5: the next-dev bump opens the moment the tag is pushed; a failure there does not stop publish, but the run is red
row green-opens-next-dev       0 '^nextdev 0\.73\.0$'         'ZZZ'                0 success $SHA
tag_at=$(grep -nE '^git push origin v0\.73\.0$' "$CALLS" | cut -d: -f1); dev_at=$(grep -nE '^nextdev ' "$CALLS" | cut -d: -f1)
if [ -n "$tag_at" ] && [ -n "$dev_at" ] && [ "$dev_at" -gt "$tag_at" ]; then echo "ok   next-dev-after-tag-push"
else echo "FAIL next-dev-after-tag-push: tag push line ${tag_at:-none}, nextdev line ${dev_at:-none}"; fails=$((fails+1)); fi
NEXTDEV_FAIL=1 row next-dev-red-still-publishes 1 '^publish' 'ZZZ'                 0 success $SHA
DECOY=1 row decoy-line-first-still-tags 0 '^git tag .*v0\.73\.0 '$SHA 'ZZZ' 0 success $SHA
row wrong-sha-no-tag           1 ''                           '^(git tag|publish)' 0 success deadbeef
row absent-verdict-no-tag      1 ''                           '^(git tag|publish)' 0 "" ""
[ "$fails" = 0 ] && { echo "all rows held"; exit 0; }
echo "$fails row(s) failed"; exit 1
