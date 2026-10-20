#!/usr/bin/env bash
# FLOW-010 U(t, M): crates touched by a diff plus their reverse dependencies.
# Usage: scripts/test_affected.sh [BASE [HEAD]]   (default origin/main HEAD)
# Prints one package name per line; "ALL" when a workspace-level file changed.
set -euo pipefail
base="${1:-origin/main}"
head="${2:-HEAD}"
meta="$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null || cargo metadata --no-deps --format-version 1)"
root="$(git rev-parse --show-toplevel)"
files="$(git diff --name-only "$base...$head")"
[ -n "$files" ] || exit 0
if printf '%s\n' "$files" | grep -qE '^(Cargo\.(toml|lock)|rust-toolchain\.toml|\.cargo/|\.github/|ci/|Makefile$)'; then
  echo ALL
  exit 0
fi
# package -> manifest dir (relative), and package -> workspace deps
touched="$(jq -r --arg root "$root/" '.packages[] | [.name, (.manifest_path | sub($root;"") | sub("/?Cargo.toml$";""))] | @tsv' <<<"$meta" |
  while IFS=$'\t' read -r name dir; do
    [ -n "$dir" ] || continue
    printf '%s\n' "$files" | grep -q "^${dir}/" && echo "$name"
  done || true)"
[ -n "$touched" ] || exit 0
jq -r --argjson t "$(printf '%s\n' "$touched" | jq -R . | jq -s .)" '
  (.packages | map({name, deps: [.dependencies[].name]})) as $p
  | def grow($s): ($s + [$p[] | select(any(.deps[]; . as $d | $s | index($d))) | .name] | unique);
    [limit(200; $t | recurse(grow(.); (grow(.) != .)))] | last | .[]' <<<"$meta"
