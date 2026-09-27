#!/usr/bin/env bash
# lean_cache_seed.sh — seed and measure one provable-ladder .lake cache entry (PVL-001 EV-9, #4083).
#
#   lean_cache_seed.sh plan    --tree DIR [--donor-tree DIR] [--root DIR]
#   lean_cache_seed.sh seed    --tree DIR  --donor-tree DIR  [--root DIR] [--receipt FILE]
#   lean_cache_seed.sh measure --tree DIR [--ref SHA] [--root DIR] [--receipt FILE] [--budget-s N]
#
# --tree is an export (git archive) of the commit to warm, e.g. main. The cache key is the ladder's,
# byte-for-byte (ci/sections.yml provable-ladder): sha256(lean-toolchain ++ lake-manifest.json)[:16].
#
# seed copies a WARM donor entry to the tree's key instead of building Mathlib from source. It is
# admissible only when the donor was built from the same inputs: the donor tree's lean-toolchain is
# byte-equal to ours, its manifest pins the same Mathlib rev, and the donor entry's checked-out
# Mathlib HEAD is that rev. Anything else refuses (exit 3); it never guesses. An entry that already
# exists is never overwritten (exit 3): the ladder may be reading it.
#
# measure links <tree>/.lake to the entry and times `lake build` twice (cached, then no-op). PASS
# iff both exit 0, zero Mathlib modules were rebuilt, and the cached run is within --budget-s
# (default 180: the operator's "main x86 cache <=3 min"). The receipt is one JSON line,
# schema lean-cache-seed-v1, appended to --receipt; the host's load average is in it, because a
# timing taken on a loaded host is not the number the gate is about.
#
# Exit: 0 PASS / plan printed · 1 FAIL (measured, over budget or Mathlib rebuilt) · 2 precondition
# missing (no elan, no cache root, missing input file) · 3 refused (donor mismatch, entry exists)
# · 64 usage.
set -uo pipefail

usage() { sed -n '3,6p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 64; }
die() { echo "lean_cache_seed: $2" >&2; exit "$1"; }

[ $# -ge 1 ] || usage
mode=$1; shift
case "$mode" in plan|seed|measure) ;; *) usage ;; esac
ref="" tree="" donor="" root=/mnt/nvme-raid0/lean-cache receipt="" budget=180
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) ref=${2:-}; shift 2 ;;
    --tree) tree=${2:-}; shift 2 ;;
    --donor-tree) donor=${2:-}; shift 2 ;;
    --root) root=${2:-}; shift 2 ;;
    --receipt) receipt=${2:-}; shift 2 ;;
    --budget-s) budget=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$tree" ] || usage
case "$budget" in ''|*[!0-9]*) usage ;; esac
[ "$mode" != seed ] || [ -n "$donor" ] || usage

LEAN_REL=crates/aprender-contracts-staging/lean

# inputs <tree>: dies here, at top level. A die inside $(key …) would only leave the subshell.
inputs() { local f; for f in lean-toolchain lake-manifest.json; do [ -f "$1/$LEAN_REL/$f" ] || die 2 "missing $1/$LEAN_REL/$f"; done; }
# key <tree> — the ladder's formula, unchanged.
key() {
  local l="$1/$LEAN_REL"
  cat "$l/lean-toolchain" "$l/lake-manifest.json" | sha256sum | cut -c1-16
}
mathlib_rev() { jq -r '[.packages[] | select(.name == "mathlib") | .rev][0] // "none"' "$1/$LEAN_REL/lake-manifest.json"; }
toolchain() { cat "$1/$LEAN_REL/lean-toolchain"; }
warm() { [ -d "$1/build" ] || [ -d "$1/packages" ]; }

inputs "$tree"
k=$(key "$tree"); rev=$(mathlib_rev "$tree"); tc=$(toolchain "$tree")
entry="$root/$k"
echo "target: key=$k toolchain=$tc mathlib=$rev entry=$entry $(warm "$entry" && echo warm || echo COLD)"

check_donor() {
  inputs "$donor"
  dk=$(key "$donor"); drev=$(mathlib_rev "$donor"); dentry="$root/$dk"
  echo "donor:  key=$dk toolchain=$(toolchain "$donor") mathlib=$drev entry=$dentry $(warm "$dentry" && echo warm || echo COLD)"
  [ "$dk" != "$k" ] || die 3 "donor key equals the target key: nothing to seed"
  cmp -s "$tree/$LEAN_REL/lean-toolchain" "$donor/$LEAN_REL/lean-toolchain" || die 3 "lean-toolchain differs from the donor's: its oleans are not ours"
  [ "$rev" = "$drev" ] || die 3 "Mathlib rev $rev differs from the donor's $drev"
  warm "$dentry" || die 3 "donor entry $dentry is not warm"
  head=$(git -C "$dentry/packages/mathlib" rev-parse HEAD 2>/dev/null) || die 3 "donor entry has no checked-out packages/mathlib"
  [ "$head" = "$rev" ] || die 3 "donor entry's Mathlib HEAD $head is not the pinned $rev"
}

load() { cut -d' ' -f1-3 /proc/loadavg; }
emit() { # emit <verdict> <json fields object>
  local line
  line=$(jq -cn --arg v "$1" --argjson f "$2" --arg host "$(hostname)" --arg at "$(date -u +%FT%TZ)" \
    --arg ref "$ref" --arg key "$k" --arg tc "$tc" --arg rev "$rev" --arg load "$(load)" --arg nproc "$(nproc)" \
    '{schema: "lean-cache-seed-v1", at: $at, host: $host, ref: $ref, key: $key, toolchain: $tc, mathlib_rev: $rev,
      loadavg: $load, nproc: ($nproc | tonumber)} + $f + {verdict: $v}')
  echo "$line"
  [ -z "$receipt" ] || printf '%s\n' "$line" >> "$receipt"
}

case "$mode" in
  plan)
    [ -z "$donor" ] || check_donor
    [ -d "$root" ] || echo "note: no $root on $(hostname)"
    echo "plan only: nothing written" ;;
  seed)
    [ -d "$root" ] || die 2 "no $root on $(hostname)"
    check_donor
    [ ! -e "$entry" ] || die 3 "$entry exists: never overwritten (the ladder may be reading it)"
    s=$(date +%s)
    # Hold the donor's lock so no ladder run unpacks into it mid-copy; the target's so no run reads a
    # half-copied entry. Copy to a temp name and rename: the entry appears whole or not at all.
    tmp="$root/.$k.seed.$$"
    flock "$dentry.lock" flock "$entry.lock" cp -a "$dentry/." "$tmp/" || { rm -rf "${tmp:?}"; die 1 "copy failed"; }
    mv -T "$tmp" "$entry" || { rm -rf "${tmp:?}"; die 1 "rename to $entry failed"; }
    emit seeded "$(jq -cn --arg dk "$dk" --arg ws "$(( $(date +%s) - s ))" --arg sz "$(du -sh "$entry" | cut -f1)" \
      '{phase: "seed", donor_key: $dk, seed_method: "copy", seed_wall_s: ($ws | tonumber), size: $sz}')" ;;
  measure)
    ELAN_HOME="${ELAN_HOME:-$HOME/.elan}"; export PATH="$ELAN_HOME/bin:$PATH"
    command -v elan >/dev/null || die 2 "no elan in $ELAN_HOME"
    warm "$entry" || die 2 "$entry is not warm: seed it first"
    l="$tree/$LEAN_REL"
    [ ! -e "$l/.lake" ] || [ -L "$l/.lake" ] || die 2 "$l/.lake is a real directory, not the cache link"
    ln -sfn "$entry" "$l/.lake"
    elan toolchain install "$tc" > /dev/null 2>&1 || die 2 "elan toolchain install $tc failed"
    out=$(mktemp -d)
    run() { local s rc; s=$(date +%s); (cd "$l" && flock "$entry.lock" lake build) > "$out/$1.log" 2>&1; rc=$?
            echo "$rc $(( $(date +%s) - s ))"; }
    read -r rc1 w1 < <(run cached); read -r rc2 w2 < <(run noop)
    ml=$(command grep -c 'Built Mathlib\.' "$out/cached.log"); all=$(command grep -c 'Built ' "$out/cached.log")
    v=PASS
    { [ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ] && [ "$ml" -eq 0 ] && [ "$w1" -le "$budget" ]; } || v=FAIL
    emit "$v" "$(jq -cn --arg r1 "$rc1" --arg w1 "$w1" --arg r2 "$rc2" --arg w2 "$w2" --arg ml "$ml" --arg all "$all" \
      --arg b "$budget" --arg logs "$out" \
      '{phase: "measure", cached: {rc: ($r1|tonumber), wall_s: ($w1|tonumber)}, noop: {rc: ($r2|tonumber), wall_s: ($w2|tonumber)},
        built_jobs: ($all|tonumber), mathlib_rebuilt: ($ml|tonumber), budget_s: ($b|tonumber), logs: $logs}')"
    [ "$v" = PASS ] || exit 1 ;;
esac
exit 0
