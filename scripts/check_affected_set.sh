#!/usr/bin/env bash
# FLOW-010 E5 RED rows for scripts/affected_set.sh. Oracle for the closure is
# `cargo tree -i` (a different code path from the tool's cargo-metadata jq).
set -euo pipefail
A=scripts/affected_set.sh
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
fail=0
row() { # name want got
  if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1"; echo "  want: $2" | head -c 300; echo; echo "  got:  $3" | head -c 300; echo; fail=1; fi
}
run() { printf '%s\n' "$@" >"$t/d"; $A --diff-from "$t/d"; }
core_list=" aprender-core aprender-compute aprender-contracts "
meta=$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null)
members=$(jq -r '.packages[].name' <<<"$meta" | sort)
# a non-core crate that has dependents, and a leaf with none
withdep="" leaf=""
for p in $members; do
  case "$core_list" in *" $p "*) continue ;; esac
  n=$(cargo tree --offline -i "$p" --workspace -e normal,dev,build --prefix none 2>/dev/null | sed 's/ v.*//' | sort -u | grep -vc "^$p$" || true)
  dir=$(jq -r --arg p "$p" '.packages[]|select(.name==$p)|.manifest_path' <<<"$meta" | sed "s|$(git rev-parse --show-toplevel)/||; s|/Cargo.toml||")
  [ -z "$withdep" ] && [ "$n" -gt 0 ] && [ "$n" -lt 10 ] && withdep="$p:$dir"
  [ -z "$leaf" ] && [ "$n" -eq 0 ] && leaf="$p:$dir"
  [ -n "$withdep" ] && [ -n "$leaf" ] && break
done
[ -n "$withdep" ] && [ -n "$leaf" ] || { echo "FAIL could not find a leaf and a crate with dependents"; exit 1; }
wp=${withdep%%:*}; wd=${withdep#*:}; lp=${leaf%%:*}; ld=${leaf#*:}
want=$(cargo tree --offline -i "$wp" --workspace -e normal,dev,build --prefix none 2>/dev/null | sed 's/ v.*//' | sort -u)
row "crate with dependents ($wp) -> itself + dependents" "$want" "$(run "$wd/src/lib.rs" | sort -u)"
row "leaf crate ($lp) -> itself only" "$lp" "$(run "$ld/src/lib.rs")"
row "core crate -> ALL" ALL "$(run crates/aprender-core/src/lib.rs)"
row "Cargo.lock -> ALL" ALL "$(run Cargo.lock)"
row "unknown path -> ALL" ALL "$(run some/unknown/path.txt)"
row "unknown path next to a leaf -> ALL" ALL "$(run "$ld/src/lib.rs" scripts/whatever.sh)"
row "docs only -> nothing" "" "$(run docs/x.md)"
row "--filter names the package" "package($lp)" "$($A --filter --diff-from <(echo "$ld/src/lib.rs"))"
row "--filter ALL -> all()" "all()" "$($A --filter --diff-from <(echo Cargo.lock))"
exit "$fail"
