#!/usr/bin/env bash
# Print one sha256 hex identifying the build toolchain: rust-toolchain.toml (nothing when
# absent), the full `rustc -vV`, then the literal line "image=none". Same bytes, same order
# as the batcher's signing side, so a receipt's toolchain field compares equal here.
# A missing rustc exits 1 (the verifier fails closed).
# Hardened (#5798): the toml is read from HEAD's tree (git show), not the working tree; a
# `path =` key (an override that makes rustc run an arbitrary binary) is refused; and rustc runs
# in $RECEIPT_RUSTC_CWD when the judge sets it (a dir holding the BASE's toml), never in the PR cwd.
set -u
root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
toml=""
if git rev-parse --verify -q 'HEAD^{tree}' >/dev/null 2>&1; then
  toml=$(git show HEAD:rust-toolchain.toml 2>/dev/null; printf x); toml=${toml%x}
elif [ -f "$root/rust-toolchain.toml" ]; then
  toml=$(cat "$root/rust-toolchain.toml"; printf x); toml=${toml%x}
fi
if printf '%s\n' "$toml" | grep -q -E '^[[:space:]]*path[[:space:]]*='; then exit 1; fi
rustc_v=$(cd "${RECEIPT_RUSTC_CWD:-.}" && rustc -vV 2>/dev/null) || exit 1
[ -n "$rustc_v" ] || exit 1
{
  printf '%s' "$toml"
  printf '%s\n' "$rustc_v"
  printf 'image=none\n'
} | sha256sum | cut -d' ' -f1
