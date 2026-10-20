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
# An empty diff, or a diff git cannot compute (bad base: exit 128), is ALL: nothing was
# measured, so nothing may be skipped. A diff of only no-crate-code paths prints nothing.
# git diff runs with --no-renames: a moved file is its old path AND its new path (a rename
# out of a crate would otherwise show only the destination and the source crate would be skipped).
# A docs/*.md file that a crate embeds with include_str!/include_bytes!("literal") counts as that crate.
# --diff-from reads the changed paths from FILE (tests). AFFECTED_META / AFFECTED_ROOT
# replace `cargo metadata` and the checkout root (tests; synthetic graphs).
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
if [ -n "$from" ]; then files="$(cat "$from")"
else files="$(git diff --name-only --no-renames "${1:-origin/main}...${2:-HEAD}")" || files=""; fi
emit() { # $@ = package names, or ALL
  if [ "${1:-}" = ALL ]; then [ "$filter" = 1 ] && echo 'all()' || echo ALL; return; fi
  if [ "$filter" = 1 ]; then printf '%s\n' "$@" | sed 's/.*/package(&)/' | paste -sd'|' | sed 's/|/ | /g'; else printf '%s\n' "$@"; fi
}
[ -n "$files" ] || { emit ALL; exit 0; }
if [ -n "${AFFECTED_META:-}" ]; then meta="$(cat "$AFFECTED_META")"
else meta="$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null || cargo metadata --no-deps --format-version 1)"; fi
root="${AFFECTED_ROOT:-$(git rev-parse --show-toplevel)}"
dirs="$(jq -r --arg root "$root/" '.packages[] | [.name, (.manifest_path | sub($root;"") | sub("/?Cargo.toml$";""))] | @tsv' <<<"$meta")"
# "<embedded path>\t<embedding .rs file>" for every include_str!/include_bytes!("literal") under crates/,
# the literal resolved against the embedding file's directory. Non-literal (concat!/macro) args are not resolved.
INCL_PL='use File::Basename; for $f (@ARGV) { open F, "<", $f or next; local $/; $s = <F>; close F;
  while ($s =~ /include_(?:str|bytes)!\s*\(\s*"([^"]*)"/g) { @o = ();
    for (split m{/}, dirname($f) . "/" . $1) { next if $_ eq "" || $_ eq "."; if ($_ eq "..") { pop @o } else { push @o, $_ } }
    print join("/", @o) . "\t" . $f . "\n" } }'
incl="$( (cd "$root" 2>/dev/null && find crates -name "*.rs" -type f -print0 2>/dev/null | xargs -0 -r perl -0777 -e "$INCL_PL") || true )"
touched=""
while IFS= read -r f; do
  case "$f" in
    '') continue ;;
    docs/* | *.md | ci/sections.yml | ci/shard-step-costs.tsv | ci/explicit-test-commands.d/*)
      # skipped as "no crate code" unless a crate embeds it: then it is that crate's input
      while IFS=$'\t' read -r _ src; do
        [ -n "$src" ] || continue
        pkg="$(awk -F'\t' -v f="$src" 'index(f, $2 "/") == 1 && length($2) > best { best = length($2); n = $1 } END { print n }' <<<"$dirs")"
        [ -n "$pkg" ] || { emit ALL; exit 0; }
        case " $CORE " in *" $pkg "*) emit ALL; exit 0 ;; esac
        touched="$touched"$'\n'"$pkg"
      done < <(awk -F'\t' -v f="$f" '$1 == f' <<<"$incl")
      continue ;;
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
