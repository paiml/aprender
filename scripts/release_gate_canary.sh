#!/usr/bin/env bash
# release_gate_canary.sh — BUILD-SYNC-001 canary for the vendored release gate (.ops/release/).
#
# 1. The vendored release.mk / release-lib.sh match .ops/release/SHA256SUMS (the copy is the
#    upstream file at .ops/release/PIN, byte for byte; a local edit is refused).
# 2. `make release-gate` decides GO / NO-GO correctly on planted receipts: one GO row and five
#    NO-GO rows. A gate that says GO on any NO-GO row fails this script.
# Exit 0 = every row as expected, 1 = a row disagreed or a hash differs.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DIR="$ROOT/.ops/release"
fail=0

# SEC010: validate the path before cd/cat: absolute, no '..' (it is the git toplevel plus a fixed suffix).
case "$DIR" in
  *..*) echo "FAIL: release gate dir contains '..': $DIR" >&2; exit 1 ;;
  /*) ;;
  *) echo "FAIL: release gate dir is not absolute: $DIR" >&2; exit 1 ;;
esac

if ! (cd "$DIR" && sha256sum --check --quiet SHA256SUMS); then
  echo "FAIL: vendored release gate differs from SHA256SUMS (pin: $(cat "$DIR/PIN"))" >&2
  exit 1
fi

TD="$(mktemp -d "${TMPDIR:-/tmp}/relgate.XXXXXX")"
trap 'rm -rf "${TD:?}"' EXIT
SHA=0123456789abcdef0123456789abcdef01234567
OTHER=fedcba9876543210fedcba9876543210fedcba98

# row <name> <want-exit> <repo> <sha> <cleanroom "sha verdict" | -> <dogfood "sha verdict" | ->
row() {
  local name="$1" want="$2" repo="$3" sha="$4" cr="$5" df="$6" d="$TD/$1" rc
  mkdir -p "$d"
  local a b
  if [ "$cr" != "-" ]; then read -r a b <<< "$cr"; printf 'sha=%s\nverdict=%s\n' "$a" "$b" > "$d/cleanroom-$sha.receipt"; fi
  if [ "$df" != "-" ]; then read -r a b <<< "$df"; printf 'sha=%s\nverdict=%s\n' "$a" "$b" > "$d/dogfood-$sha.receipt"; fi
  make -s -C "$ROOT" release-gate REPO="$repo" SHA="$sha" REL_DIR="$d" > "$d/out" 2> "$d/err"
  rc=$?
  if [ "$want" = GO ] && [ "$rc" -eq 0 ] && grep -q "^GO $repo v" "$d/out"; then
    echo "ok   $name: GO"
  elif [ "$want" = NO-GO ] && [ "$rc" -ne 0 ] && grep -q '^NO-GO' "$d/err"; then
    echo "ok   $name: NO-GO ($(head -n 1 "$d/err"))"
  else
    echo "FAIL $name: want $want, got exit $rc: $(cat "$d/out" "$d/err" | head -n 2)" >&2
    fail=1
  fi
}

set +e
row go               GO    paiml/aprender "$SHA" "$SHA PASS"   "$SHA GO"
row no-cleanroom     NO-GO paiml/aprender "$SHA" -             "$SHA GO"
row cleanroom-other  NO-GO paiml/aprender "$SHA" "$OTHER PASS" "$SHA GO"
row cleanroom-red    NO-GO paiml/aprender "$SHA" "$SHA FAIL"   "$SHA GO"
row dogfood-nogo     NO-GO paiml/aprender "$SHA" "$SHA PASS"   "$SHA NO-GO"
row unpinned-repo    NO-GO paiml/other    "$SHA" "$SHA PASS"   "$SHA GO"
set -e

[ "$fail" -eq 0 ] && echo "PASS: release gate canary, 6/6 rows"
exit "$fail"
