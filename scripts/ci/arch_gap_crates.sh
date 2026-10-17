#!/usr/bin/env bash
# Prints the `-p <crate>` arguments of every workspace crate whose src/ gates code on an
# x86 target_arch. Those are the crates whose registered tests differ between X64 and
# ARM64 (#5437): the tests behind cfg(target_arch = "x86_64") do not compile on ARM, and
# the aarch64 ones do not compile on X64. The list is derived from the source, never
# hand-written, so a new arch-gated test lands in the gap step by itself.
# The crates excluded from the workspace lib run are excluded here too.
set -euo pipefail
cd "$(dirname "$0")/../.."
grep -rlE 'target_arch *= *"(x86|aarch64)' --include='*.rs' crates/*/src 2>/dev/null |
  cut -d/ -f2 | sort -u |
  while read -r d; do sed -n 's/^name *= *"\(.*\)"/\1/p' "crates/$d/Cargo.toml" | head -n 1; done |
  grep -vxE 'aprender-gpu|aprender-cuda-edge|aprender-compute' | sort -u |
  sed 's/^/-p /'
