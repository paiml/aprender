#!/usr/bin/env bash
# FLOW-010 U(t, M): crates touched by a diff plus their reverse dependencies.
# Usage: scripts/test_affected.sh [BASE [HEAD]]   (default origin/main HEAD)
# Prints one package name per line; "ALL" when a workspace-level file changed
# in a way that cannot be narrowed to named packages.
#
# Measured: 10 of the last 30 main commits hit ALL under the blanket rule
# (ci/sections.yml 5, Cargo.lock 3, ci/explicit-test-commands.d 2,
# ci/shard-step-costs.tsv 1, Cargo.toml 1). Narrowing:
#   ci/sections.yml, ci/shard-step-costs.tsv, ci/explicit-test-commands.d/*
#       schedule and cost data, no crate code: contribute nothing.
#   Cargo.lock   packages whose [[package]] block changed seed the closure.
#   Cargo.toml   a diff of only `members` entries and path entries of
#       [workspace.dependencies] seeds those packages; any other line is ALL.
#   rust-toolchain.toml, .cargo/, .github/, other ci/, Makefile: ALL.
set -euo pipefail
base="${1:-origin/main}"
head="${2:-HEAD}"
meta="$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null || cargo metadata --no-deps --format-version 1)"
root="$(git rev-parse --show-toplevel)"
files="$(git diff --name-only "$base...$head")"
[ -n "$files" ] || exit 0

# Workspace-level files that cannot be narrowed.
if printf '%s\n' "$files" |
  grep -vE '^(ci/(sections\.yml|shard-step-costs\.tsv|explicit-test-commands\.d/)|Cargo\.(toml|lock)$)' |
  grep -qE '^(rust-toolchain\.toml|\.cargo/|\.github/|ci/|Makefile$)'; then
  echo ALL
  exit 0
fi

mergebase="$(git merge-base "$base" "$head")"
seeds=""

# One line per [[package]] block: "name<TAB>whole block with newlines as |".
lock_pkgs() {
  git show "$1:Cargo.lock" 2>/dev/null | awk -v RS='\\[\\[package\\]\\]\n' '
    match($0, /name = "[^"]+"/) { n = substr($0, RSTART + 8, RLENGTH - 9); gsub(/\n/, "|"); print n "\t" $0 }'
}
if printf '%s\n' "$files" | grep -qx 'Cargo.lock'; then
  changed="$(comm -13 <(lock_pkgs "$mergebase" | sort -u) <(lock_pkgs "$head" | sort -u) | cut -f1 | sort -u)"
  # Reverse closure over the whole lock graph (a changed transitive crate reaches
  # the members through other external crates); members are the packages in $meta.
  edges="$(lock_pkgs "$head" | awk -F'\t' '{ n = split($2, a, "|"); for (i = 1; i <= n; i++) if (a[i] ~ /^ "/) { d = a[i]; gsub(/^ *"|"?,?$/, "", d); sub(/ .*/, "", d); print d "\t" $1 } }')"
  seeds="$seeds"$'\n'"$(jq -rn --argjson c "$(printf '%s\n' "$changed" | jq -R . | jq -s .)" \
    --argjson e "$(printf '%s\n' "$edges" | jq -R 'split("\t")' | jq -s .)" '
    def grow($s): ($s + [$e[] | select(.[0] as $d | $s | index($d)) | .[1]] | unique);
    [limit(500; $c | recurse(grow(.); (grow(.) != .)))] | last | .[]')"
fi

if printf '%s\n' "$files" | grep -qx 'Cargo.toml'; then
  lines="$(git diff -U0 "$mergebase" "$head" -- Cargo.toml | grep -E '^[+-][^+-]' | sed -E 's/^[+-]//' || true)"
  if printf '%s\n' "$lines" | grep -vE '^\s*($|#|"crates/[^"]+",?$|[A-Za-z0-9_-]+ *= *\{ *path *=)' | grep -q .; then
    echo ALL
    exit 0
  fi
  seeds="$seeds"$'\n'"$(printf '%s\n' "$lines" | sed -nE 's/^\s*([A-Za-z0-9_-]+) *= *\{ *path.*/\1/p')"
fi

# package -> manifest dir (relative); a package is touched when a changed file is under it.
touched="$(jq -r --arg root "$root/" '.packages[] | [.name, (.manifest_path | sub($root;"") | sub("/?Cargo.toml$";""))] | @tsv' <<<"$meta" |
  while IFS=$'\t' read -r name dir; do
    [ -n "$dir" ] || continue
    printf '%s\n' "$files" | grep -q "^${dir}/" && echo "$name"
  done || true)"
touched="$(printf '%s\n%s\n' "$touched" "$seeds" | grep -v '^$' | sort -u || true)"
[ -n "$touched" ] || exit 0
jq -r --argjson t "$(printf '%s\n' "$touched" | jq -R . | jq -s .)" '
  (.packages | map({name, deps: [.dependencies[].name]})) as $p
  | def grow($s): ($s + [$p[] | select(any(.deps[]; . as $d | $s | index($d))) | .name] | unique);
    (.packages | map(.name)) as $m | [limit(200; $t | recurse(grow(.); (grow(.) != .)))] | last | .[] | select(. as $n | $m | index($n))' <<<"$meta"
