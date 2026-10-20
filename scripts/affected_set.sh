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
#   Cargo.lock              (git mode only) the packages whose [[package]] block changed, closed over the
#                           lock graph; unreadable lock / no changed block / --diff-from -> ALL
#   Cargo.toml, rust-toolchain.toml, .cargo/, .github/, Makefile, any other path
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
if [ -n "$from" ]; then files="$(cat "$from")"; else files="$(git diff --name-only "${1:-origin/main}...${2:-HEAD}" 2>/dev/null || git diff --name-only "${1:-origin/main}" "${2:-HEAD}")"; fi
emit() { # $@ = package names, or ALL
  if [ "${1:-}" = ALL ]; then [ "$filter" = 1 ] && echo 'all()' || echo ALL; return; fi
  if [ "$filter" = 1 ]; then printf '%s\n' "$@" | sed 's/.*/package(&)/' | paste -sd'|' | sed 's/|/ | /g'; else printf '%s\n' "$@"; fi
}
[ -n "$files" ] || exit 0
meta="$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null || cargo metadata --no-deps --format-version 1)"
root="$(git rev-parse --show-toplevel)"
dirs="$(jq -r --arg root "$root/" '.packages[] | [.name, (.manifest_path | sub($root;"") | sub("/?Cargo.toml$";""))] | @tsv' <<<"$meta")"
touched="" lock=0
while IFS= read -r f; do
  case "$f" in
    '') continue ;;
    # FLOW-010 P3: a lock-only change is narrowed by its [[package]] diff, but only when
    # git can answer (no --diff-from: a bare path list has no content, so it stays ALL).
    Cargo.lock) if [ -z "$from" ]; then lock=1; continue; fi ;;
    docs/* | *.md | ci/sections.yml | ci/shard-step-costs.tsv | ci/explicit-test-commands.d/*) continue ;;
  esac
  # longest manifest dir that prefixes the path
  pkg="$(awk -F'\t' -v f="$f" 'index(f, $2 "/") == 1 && length($2) > best { best = length($2); n = $1 } END { print n }' <<<"$dirs")"
  [ -n "$pkg" ] || { emit ALL; exit 0; }
  case " $CORE " in *" $pkg "*) emit ALL; exit 0 ;; esac
  touched="$touched"$'\n'"$pkg"
done <<<"$files"
if [ "$lock" = 1 ]; then
  b="${1:-origin/main}"; h="${2:-HEAD}"
  mb="$(git merge-base "$b" "$h" 2>/dev/null || echo "$b")"
  # one line per [[package]] block: name<TAB>block with newlines as |
  lock_pkgs() { git show "$1:Cargo.lock" 2>/dev/null | awk -v RS='\\[\\[package\\]\\]\n' 'match($0, /name = "[^"]+"/) { n = substr($0, RSTART + 8, RLENGTH - 9); gsub(/\n/, "|"); print n "\t" $0 }'; }
  old="$(lock_pkgs "$mb" | sort -u)"; new="$(lock_pkgs "$h" | sort -u)"
  # unreadable lock on either side, or no changed block: cannot place the change -> ALL
  { [ -n "$old" ] && [ -n "$new" ]; } || { emit ALL; exit 0; }
  changed="$(comm -13 <(printf '%s\n' "$old") <(printf '%s\n' "$new") | cut -f1 | sort -u)"
  [ -n "$changed" ] || { emit ALL; exit 0; }
  # reverse closure over the lock graph (a changed external crate reaches members through other externals)
  seeds="$(printf '%s\n' "$new" | awk -F'\t' -v chg="$changed" '
    BEGIN { n = split(chg, c, "\n"); for (i = 1; i <= n; i++) hit[c[i]] = 1 }
    { split($2, a, "|"); for (i = 1; i <= length(a); i++) if (a[i] ~ /^ "/) { d = a[i]; gsub(/^ *"|"?,?$/, "", d); sub(/ .*/, "", d); dep[++m] = d; who[m] = $1 } }
    END { do { g = 0; for (i = 1; i <= m; i++) if ((dep[i] in hit) && !(who[i] in hit)) { hit[who[i]] = 1; g = 1 } } while (g)
          for (k in hit) print k }')"
  touched="$touched"$'\n'"$(jq -r --argjson s "$(printf '%s\n' "$seeds" | jq -R . | jq -s .)" '.packages[].name | select(. as $n | $s | index($n))' <<<"$meta")"
fi
touched="$(printf '%s\n' "$touched" | grep -v '^$' | sort -u || true)"
[ -n "$touched" ] || { [ "$lock" = 1 ] && emit ALL; exit 0; }
mapfile -t out < <(jq -r --argjson t "$(printf '%s\n' "$touched" | jq -R . | jq -s .)" '
  (.packages | map({name, deps: [.dependencies[].name]})) as $p
  | def grow($s): ($s + [$p[] | select(any(.deps[]; . as $d | $s | index($d))) | .name] | unique);
    (.packages | map(.name)) as $m
    | ($t | until(grow(.) == .; grow(.))) | .[] | select(. as $n | $m | index($n))' <<<"$meta")
emit "${out[@]}"
