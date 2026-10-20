#!/usr/bin/env bash
# FLOW-010 E5 RED rows for scripts/affected_set.sh. Oracle for the closure is
# `tree -i` of the package manager (a different code path from the tool's cargo-metadata jq).
set -euo pipefail
# metadata and tree only read the manifests (no build), so this stays in the guard-tree no-build population.
CG=${CARGO:-cargo}
A=scripts/affected_set.sh
t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
fail=0
row() { # name want got
  if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1"; echo "  want: $2" | head -c 300; echo; echo "  got:  $3" | head -c 300; echo; fail=1; fi
}
run() { printf '%s\n' "$@" >"$t/d"; $A --diff-from "$t/d"; }
core_list=" aprender-core aprender-compute aprender-contracts "
meta=$("$CG" metadata --no-deps --format-version 1 --offline 2>/dev/null)
members=$(jq -r '.packages[].name' <<<"$meta" | sort)
# a non-core crate that has dependents, and a leaf with none
withdep="" leaf=""
for p in $members; do
  case "$core_list" in *" $p "*) continue ;; esac
  n=$("$CG" tree --offline -i "$p" --workspace -e normal,dev,build --prefix none 2>/dev/null | sed 's/ v.*//' | sort -u | grep -vc "^$p$" || true)
  dir=$(jq -r --arg p "$p" '.packages[]|select(.name==$p)|.manifest_path' <<<"$meta" | sed "s|$(git rev-parse --show-toplevel)/||; s|/Cargo.toml||")
  [ -z "$withdep" ] && [ "$n" -gt 0 ] && [ "$n" -lt 10 ] && withdep="$p:$dir"
  [ -z "$leaf" ] && [ "$n" -eq 0 ] && leaf="$p:$dir"
  [ -n "$withdep" ] && [ -n "$leaf" ] && break
done
[ -n "$withdep" ] && [ -n "$leaf" ] || { echo "FAIL could not find a leaf and a crate with dependents"; exit 1; }
wp=${withdep%%:*}; wd=${withdep#*:}; lp=${leaf%%:*}; ld=${leaf#*:}
want=$("$CG" tree --offline -i "$wp" --workspace -e normal,dev,build --prefix none 2>/dev/null | sed 's/ v.*//' | sort -u)
row "crate with dependents ($wp) -> itself + dependents" "$want" "$(run "$wd/src/lib.rs" | sort -u)"
row "leaf crate ($lp) -> itself only" "$lp" "$(run "$ld/src/lib.rs")"
row "core crate -> ALL" ALL "$(run crates/aprender-core/src/lib.rs)"
row "Cargo.lock -> ALL" ALL "$(run Cargo.lock)"
row "unknown path -> ALL" ALL "$(run some/unknown/path.txt)"
row "unknown path next to a leaf -> ALL" ALL "$(run "$ld/src/lib.rs" scripts/whatever.sh)"
row "docs only -> nothing" "" "$(run docs/x.md)"
row "--filter names the package" "package($lp)" "$($A --filter --diff-from <(echo "$ld/src/lib.rs"))"
row "--filter ALL -> all()" "all()" "$($A --filter --diff-from <(echo Cargo.lock))"
# FLOW-010 P3: git-mode Cargo.lock narrowing, on a throwaway a <- b, c workspace (b depends on a).
lk="$t/lk"; abs="$PWD/$A"
mkdir -p "$lk/a/src" "$lk/b/src" "$lk/c/src"
: >"$lk/a/src/lib.rs"; : >"$lk/b/src/lib.rs"; : >"$lk/c/src/lib.rs"
printf '[workspace]\nresolver = "2"\nmembers = ["a", "b", "c"]\n' >"$lk/Cargo.toml"
printf '[package]\nname = "a"\nversion = "0.1.0"\nedition = "2021"\n' >"$lk/a/Cargo.toml"
printf '[package]\nname = "b"\nversion = "0.1.0"\nedition = "2021"\n[dependencies]\na = { path = "../a" }\n' >"$lk/b/Cargo.toml"
printf '[package]\nname = "c"\nversion = "0.1.0"\nedition = "2021"\n' >"$lk/c/Cargo.toml"
printf 'version = 3\n\n[[package]]\nname = "a"\nversion = "0.1.0"\n\n[[package]]\nname = "b"\nversion = "0.1.0"\ndependencies = [\n "a",\n]\n\n[[package]]\nname = "c"\nversion = "0.1.0"\n' >"$lk/Cargo.lock"
lkgit() { git -C "$lk" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t "$@"; }
lkgit init -q -b main && lkgit add -A && lkgit commit -q -m base
sed -i '0,/version = "0.1.0"/s//version = "0.1.1"/' "$lk/Cargo.lock"; lkgit commit -q -am "bump a"
row "git mode: a lock-only bump of a -> a + its dependent b, not ALL" "a
b" "$(cd "$lk" && bash "$abs" HEAD~1 HEAD | sort)"
row "git mode: the same diff via --diff-from has no content -> ALL" ALL "$(cd "$lk" && printf 'Cargo.lock\n' >"$t/dl" && bash "$abs" --diff-from "$t/dl")"
lkgit commit -q --allow-empty -m noop
row "git mode: a lock that did not change (empty diff) -> nothing" "" "$(cd "$lk" && bash "$abs" HEAD~1 HEAD)"
lkgit rm -q Cargo.lock; lkgit commit -q -m "drop the lock"
row "git mode: a lock unreadable at HEAD -> ALL (cannot place it)" ALL "$(cd "$lk" && bash "$abs" HEAD~1 HEAD)"
exit "$fail"
