#!/usr/bin/env bash
# Print one sha256 hex identifying the build toolchain: rust-toolchain.toml (nothing when
# absent), the full `rustc -vV`, then the literal line "image=none". Same bytes, same order
# as the batcher's signing side, so a receipt's toolchain field compares equal here.
# A missing rustc exits 1 (the verifier fails closed).
set -u
root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
rustc_v=$(rustc -vV 2>/dev/null) || exit 1
[ -n "$rustc_v" ] || exit 1
{
  if [ -f "$root/rust-toolchain.toml" ]; then cat "$root/rust-toolchain.toml"; fi
  printf '%s\n' "$rustc_v"
  printf 'image=none\n'
} | sha256sum | cut -d' ' -f1
