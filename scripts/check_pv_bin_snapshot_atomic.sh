#!/usr/bin/env bash
# check_pv_bin_snapshot_atomic.sh - the pv snapshot dir built by scripts/pv_bin.sh must
# appear whole or not at all (#5388). A reader loop watches the snapshot path while many
# builders race to make it; the reader must never find the dir without a runnable pv that
# prints its version, and every builder must end up with the same whole dir.
#
# pv_bin.sh resolves (cargo build) when sourced, so the snapshot functions are cut out of
# the file by name and loaded alone; no cargo, no pv from this tree is needed.
#
# Falsifier: FALSIFY=1 builds the dir in place the pre-#5388 way and this check must go RED.
set -euo pipefail

cd "$(dirname "$0")/.."
SRC=scripts/pv_bin.sh
WORK=$(mktemp -d)
trap 'rm -rf -- "${WORK:?}"' EXIT

extract() {
    awk -v fn="$1" '$0 ~ "^"fn"\\(\\) \\{" {p=1} p {print} p && /^}/ {exit}' "$SRC"
}
for f in pv_bin_snapshot_complete pv_bin_snapshot_dir pv_bin_snapshot_locked; do
    body=$(extract "$f")
    [ -n "$body" ] || { echo "FAIL: $f not found in $SRC"; exit 1; }
    printf '%s\n' "$body"
done > "$WORK/lib.sh"
pv_bin_origin() { printf 'own\n'; }
# shellcheck disable=SC1091
. "$WORK/lib.sh"

if [ "${FALSIFY:-0}" = 1 ]; then
    # The defect: mkdir the final path, then fill it in place.
    pv_bin_snapshot_dir() {
        mkdir -p "$1" || return 1
        sleep 0.05
        ln -f -- "$2" "$1/pv" && { [ "$4" = none ] || ln -f -- "$3" "$1/pv-sat"; }
    }
fi

prof="$WORK/target/debug"
mkdir -p "$prof"
printf '#!/bin/sh\necho "pv 9.9.9"\n' > "$prof/pv"
chmod +x "$prof/pv"

snapdir_glob="$WORK/target/.pv-snapshot/*/debug"
bad="$WORK/bad"
: > "$bad"
(
    end=$((SECONDS + 4))
    while [ "$SECONDS" -lt "$end" ]; do
        for d in $snapdir_glob; do
            [ -d "$d" ] || continue
            v=$("$d/pv" --version 2>/dev/null || true)
            [ "$v" = "pv 9.9.9" ] || printf 'partial snapshot dir %s read %q\n' "$d" "$v" >> "$bad"
        done
    done
) &
reader=$!

outs="$WORK/outs"
: > "$outs"
for i in 1 2 3 4 5 6 7 8; do
    ( pv_bin_snapshot_locked "$prof/pv" 2>/dev/null | tail -n 1 >> "$outs" ) &
done
wait "${reader}" || true
wait

if [ -s "$bad" ]; then
    echo "RED: a reader saw a partial snapshot dir:"
    head -n 3 "$bad"
    exit 1
fi
n=$(awk '{print $2}' "$outs" | sort -u | wc -l)
if [ "$(wc -l < "$outs")" -ne 8 ] || [ "$n" -ne 1 ]; then
    echo "RED: builders disagree or failed ($(wc -l < "$outs") results, $n distinct paths)"
    exit 1
fi
out=$(awk '{print $2; exit}' "$outs")
case "$out" in
    "$WORK"/target/.pv-snapshot/*/debug/pv) ;;
    *) echo "RED: result is not a snapshot path: $out"; exit 1 ;;
esac
[ "$("$out" --version)" = "pv 9.9.9" ] || { echo "RED: snapshot pv does not run"; exit 1; }
stray=$(find "$WORK/target/.pv-snapshot" -name '.build.*' | wc -l)
[ "$stray" -eq 0 ] || { echo "RED: $stray leftover .build.* temp dirs"; exit 1; }
echo "ok: check_pv_bin_snapshot_atomic (8 racing builders, 1 whole dir, reader never saw a partial)"
