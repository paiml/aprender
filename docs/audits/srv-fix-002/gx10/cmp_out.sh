#!/usr/bin/env bash
# cmp_out.sh <dir>: compare the generated text (between "Output:" and "Completed in") of flash vs f32
set -uo pipefail
D=$1; same=0; tot=0
gen() { awk "/^Output:/{on=1;next} /^Completed in/{on=0} on" "$1"; }
for f in "$D"/*.flash.out; do
  n=$(basename "$f" .flash.out); tot=$((tot+1))
  [ -s "$f" ] && gen "$f" | grep -q . || { echo "$n EMPTY-OUTPUT"; continue; }
  if cmp -s <(gen "$f") <(gen "$D/$n.f32.out"); then same=$((same+1)); echo "$n IDENTICAL"
  else echo "$n DIFFER at byte $(cmp <(gen "$f") <(gen "$D/$n.f32.out") | awk "{print \$5}")"; fi
done
echo "parity=$same/$tot"
