#!/usr/bin/env bash
# FLOW-010 E5: the diff -> the set of workspace packages whose tests it can change.
#
#   scripts/affected_set.sh [--filter] [--diff-from FILE] [BASE [HEAD]]
#
# Prints one package per line, or the single word ALL. With --filter it prints the
# nextest filterset instead (`package(a) | package(b)`, or `all()` for ALL).
# Fail-safe: anything this tool cannot place in a package is ALL.
#   crates/<pkg>/...        the package + its reverse-dependency closure (cargo metadata)
#   a CORE crate            ALL (its closure is the workspace; naming it would only hide that)
#   docs/, *.md, ci/sections.yml, ci/shard-step-costs.tsv, ci/explicit-test-commands.d/
#                           contribute nothing (no crate code)
#   Cargo.lock, Cargo.toml, rust-toolchain.toml, .cargo/, .github/, Makefile, any other path
#                           ALL
# An empty diff prints nothing. --diff-from reads the changed paths from FILE (tests).
set -euo pipefail
CORE="${AFFECTED_CORE:-aprender-core aprender-compute aprender-contracts}"
filter=0 from=""
while [ $# -gt 0 ]; do
  case "$1" in
    --filter) filter=1; shift ;;
    --diff-from) from="$2"; shift 2 ;;
    *) break ;;
  esac
done
if [ -n "$from" ]; then files="$(cat "$from")"; else files="$(git diff --name-only "${1:-origin/main}...${2:-HEAD}")"; fi
emit() { # $@ = package names, or ALL
  if [ "${1:-}" = ALL ]; then [ "$filter" = 1 ] && echo 'all()' || echo ALL; return; fi
  if [ "$filter" = 1 ]; then printf '%s\n' "$@" | sed 's/.*/package(&)/' | paste -sd'|' | sed 's/|/ | /g'; else printf '%s\n' "$@"; fi
}
[ -n "$files" ] || exit 0
meta="$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null || cargo metadata --no-deps --format-version 1)"
root="$(git rev-parse --show-toplevel)"
dirs="$(jq -r --arg root "$root/" '.packages[] | [.name, (.manifest_path | sub($root;"") | sub("/?Cargo.toml$";""))] | @tsv' <<<"$meta")"
touched=""
while IFS= read -r f; do
  case "$f" in
    '') continue ;;
    docs/* | *.md | ci/sections.yml | ci/shard-step-costs.tsv | ci/explicit-test-commands.d/*) continue ;;
  esac
  # longest manifest dir that prefixes the path
  pkg="$(awk -F'\t' -v f="$f" 'index(f, $2 "/") == 1 && length($2) > best { best = length($2); n = $1 } END { print n }' <<<"$dirs")"
  [ -n "$pkg" ] || { emit ALL; exit 0; }
  case " $CORE " in *" $pkg "*) emit ALL; exit 0 ;; esac
  touched="$touched"$'\n'"$pkg"
done <<<"$files"
touched="$(printf '%s\n' "$touched" | grep -v '^$' | sort -u || true)"
[ -n "$touched" ] || exit 0
mapfile -t out < <(jq -r --argjson t "$(printf '%s\n' "$touched" | jq -R . | jq -s .)" '
  (.packages | map({name, deps: [.dependencies[].name]})) as $p
  | def grow($s): ($s + [$p[] | select(any(.deps[]; . as $d | $s | index($d))) | .name] | unique);
    (.packages | map(.name)) as $m
    | ($t | until(grow(.) == .; grow(.))) | .[] | select(. as $n | $m | index($n))' <<<"$meta")
emit "${out[@]}"
