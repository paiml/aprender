#!/usr/bin/env bash
# prompts.sh <worktree @3b8fc87675-based> <outdir>: 5 prompts from tree files, by char budget
set -euo pipefail
W=$1; O=$2; mkdir -p "$O"
tail='Summarise the code above in one sentence.'
mk() { # name chars
  local n=$1 c=$2
  { git -C "$W" ls-files 'crates/aprender-gpu/src/*.rs' | sort | while read -r f; do cat "$W/$f"; done | head -c "$c" > "$O/$n.txt"; } || true
  printf '\n\n%s\n' "$tail" >> "$O/$n.txt"
}
printf 'In Rust, what does the ? operator do? %s\n' "$tail" > "$O/p1_short.txt"
mk p2_2k 7000; mk p3_8k 28000; mk p4_16k 56000; mk p5_30k 104000
sha256sum "$O"/*.txt
