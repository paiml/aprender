#!/usr/bin/env bash
# falsify_publish_rehearsal_mutants.sh -- planted defects for scripts/release/publish_rehearsal.sh (PMAT-6914)
#   bash scripts/release/falsify_publish_rehearsal_mutants.sh   # exit 0 = every planted defect caught (also: falsify_publish_rehearsal.sh --mutants)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
SCRATCH_BASE="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
TABLE="$HERE/falsify_publish_rehearsal.sh"
M=$(mktemp -d -p "$SCRATCH_BASE" falsify-pubreh-mut.XXXXXX) || exit 2; trap 'rm -rf -- "${M:?}"' EXIT; miss=0; total=0
mut() { # name sed-expression: the mutated script must turn at least one row red
  total=$((total+1)); sed "$2" "$HERE/publish_rehearsal.sh" > "$M/m.sh"
  cmp -s "$HERE/publish_rehearsal.sh" "$M/m.sh" && { echo "MUTANT $1 did not apply"; miss=$((miss+1)); return; }
  PUB_REH_SRC="$M/m.sh" PUB_REH_SCRIPT_ROWS_ONLY=1 bash "$TABLE" > /dev/null 2>&1 && { echo "MUTANT $1 SURVIVED"; miss=$((miss+1)); }; return 0; }
mut run-number-unchecked        's/^  \[\[ \$N =~ .*$/  :/'
mut run-number-leading-zero     's/\^\[1-9\]\[0-9\]\*\$ \]\] || { echo/^[0-9][0-9]*$ ]] || { echo/'
mut dry-arg-unchecked           's/^  case "\$DRY" in 0|1) ;; .*$/  :/'
mut version-prefix-wrong        's/V="0\.0\.\$N"/V="0.1.$N"/'
mut tag-filter-removed          "s/ | grep -E '[^']*'//"
mut oldest-tag-picked           's/--sort=-creatordate/--sort=creatordate/'
mut no-tag-allowed              's/^  \[ -n "\$TAG" \] || die .*$/  :/'
mut gate-refusal-swallowed      's/^  \[ "\$grc" = 0 \] || die .*$/  :/'
mut gate-skipped                's/^  ( unset CARGO_REGISTRY_TOKEN; run_gate .*$/  grc=0/'
mut gate-sees-the-token         's/unset CARGO_REGISTRY_TOKEN; //'
mut gate-function-unchecked     's/^    grep -q .*fns.*$/    :/'
mut gate-gets-wrong-tag         's/clean_room_gate "\$1" "\$2"/clean_room_gate "$1" "v0.0.0"/'
mut cascade-top-level-run       's/^    \. "\$fns" || exit 1$/    . "$cascade" || exit 1/'
mut dry-run-ignored             's/^  if \[ "\$DRY" = 1 \]; then$/  if false; then/'
mut empty-token-allowed         's/^  \[ -n "\${CARGO_REGISTRY_TOKEN:-}" \] || die .*$/  :/'
mut version-not-set             's/^  sed -i .*$/  :/'
mut version-set-unchecked       's/^  grep -q -F -x -e "version = .*$/  :/'
mut version-set-in-the-tree     's/"\$TMP\/crate\/Cargo.toml"/"$CRATE_DIR\/Cargo.toml"/g'
mut publish-runs-in-the-tree    's/( cd "\$TMP\/crate" \&\& publish_cmd )/( cd "$CRATE_DIR" \&\& publish_cmd )/'
mut upload-refusal-swallowed    's/^    die "upload: refused or failed.*$/    :/'
mut already-uploaded-not-green  's/^  if \[ "\$prc" != 0 \] \&\& already_uploaded .*; then$/  if false; then/'
mut any-failure-is-uploaded     's/^  if \[ "\$prc" != 0 \] \&\& already_uploaded .*; then$/  if [ "$prc" != 0 ]; then/'
mut other-version-is-uploaded   's/^    \&\& grep -q -E -e "(.*$/    \&\& true/'
mut longer-version-is-uploaded  's/(\[^0-9\]|\\\$)" "\$1"$/" "$1"/'
mut index-unchecked             's/^    \$INDEX "\$CRATE" "\$V" > .dev.null 2>&1 \&\& { seen=1; break; }$/    seen=1; break/'
mut index-asked-for-wrong-version 's/\$INDEX "\$CRATE" "\$V"/$INDEX "$CRATE" "0.0.0"/'
mut index-retry-off-by-five     's/^  while \[ "\$i" -lt "\$TRIES" \]; do$/  while [ "$i" -lt $((TRIES+5)) ]; do/'
mut index-never-retried         's/^    \[ "\$i" -lt "\$TRIES" \] \&\& sleep.*$/    break/'
mut index-miss-swallowed        's/^  \[ "\$seen" = 1 \] || die .*$/  :/'
mut output-not-scrubbed         's/^    \[ -z "\$tok" \] || line=.*$/    :/'
mut scratch-left-behind         's/^  trap .*EXIT$/  :/'
[ "$miss" = 0 ] && { echo "mutants: $total of $total killed"; exit 0; }
echo "mutants: $((total-miss)) of $total killed"; exit 1
