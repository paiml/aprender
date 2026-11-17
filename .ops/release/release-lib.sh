#!/usr/bin/env bash
# release-lib.sh v1 - BUILD-SYNC-001 release gate. WRAP, DON'T REWRITE:
# it only decides GO/NO-GO; the repo's existing publish driver does the publish,
# and the session runs that driver. Order is fixed: clean-room on the TAG SHA
# (hard gate), then dogfood go/no-go, then the printed driver command.
# Receipts: <dir>/cleanroom-<sha>.receipt (sha=, verdict=PASS) and
#           <dir>/dogfood-<sha>.receipt   (sha=, verdict=GO).
# Exit 0 = GO, 1 = NO-GO (reason on stderr), 2 = usage.
set -u

rel_field() { sed -n "s/^$2=//p" "$1" | head -n 1; }

rel_check_receipt() { # file want_sha want_verdict label
  local f="$1" sha="$2" want="$3" label="$4"
  [ -f "$f" ] || { echo "NO-GO: $label receipt missing: $f" >&2; return 1; }
  [ "$(rel_field "$f" sha)" = "$sha" ] || { echo "NO-GO: $label receipt is not for tag sha $sha" >&2; return 1; }
  [ "$(rel_field "$f" verdict)" = "$want" ] || { echo "NO-GO: $label verdict is not $want" >&2; return 1; }
}

rel_pin() { # repo -> "tag driver"
  awk -F'\t' -v r="$1" '$1==r {print $2, $3}' "$2"
}

# rel_gate <repo> <tag-sha> <receipt-dir> <pins.tsv>
rel_gate() {
  [ $# -eq 4 ] || { echo "usage: rel_gate repo sha dir pins" >&2; return 2; }
  local repo="$1" sha="$2" dir="$3" pins="$4" pin
  case "$sha" in *[!0-9a-f]*|"") echo "NO-GO: bad sha" >&2; return 1 ;; esac
  pin="$(rel_pin "$repo" "$pins")"
  [ -n "$pin" ] || { echo "NO-GO: $repo not pinned in $pins" >&2; return 1; }
  rel_check_receipt "$dir/cleanroom-$sha.receipt" "$sha" PASS clean-room || return 1
  rel_check_receipt "$dir/dogfood-$sha.receipt" "$sha" GO dogfood || return 1
  echo "GO $repo ${pin%% *} $sha"
  echo "RUN: ${pin#* } (session runs the driver)"
}
