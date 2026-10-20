#!/usr/bin/env bash
# FLOW-010 E5 RED rows for scripts/affected_set.sh. Oracle for the closure is
# `cargo tree -i` (a different code path from the tool's cargo-metadata jq).
set -euo pipefail
A=scripts/affected_set.sh
t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT
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
# empty diff and a bad base are ALL (nothing measured, nothing skipped)
: >"$t/empty"
row "empty diff -> ALL" ALL "$($A --diff-from "$t/empty")"
row "bad base (git exit 128) -> ALL" ALL "$($A no-such-base-ref-5802 HEAD 2>/dev/null)"
# dev-dep counting: b depends on a ONLY as a dev-dependency; touching a must reach b.
cat >"$t/meta.json" <<'JSON'
{"packages":[
 {"name":"a","manifest_path":"/r/crates/a/Cargo.toml","dependencies":[]},
 {"name":"b","manifest_path":"/r/crates/b/Cargo.toml","dependencies":[{"name":"a","kind":"dev"}]},
 {"name":"c","manifest_path":"/r/crates/c/Cargo.toml","dependencies":[]}]}
JSON
row "dev-only edge is counted: touching a reaches b" "a
b" "$(AFFECTED_META="$t/meta.json" AFFECTED_ROOT=/r run crates/a/src/lib.rs | sort -u)"
row "an unrelated crate is not reached" "c" "$(AFFECTED_META="$t/meta.json" AFFECTED_ROOT=/r run crates/c/src/lib.rs)"
# mutant: a tool that drops dev edges misses b, and the dev-dep row must see it
sed 's/\[\.dependencies\[\]\.name\]/[.dependencies[]|select(.kind!="dev")|.name]/' "$A" >"$t/mut.sh"
if cmp -s "$A" "$t/mut.sh"; then echo "FAIL mutant did not apply"; fail=1
else
  got=$(AFFECTED_META="$t/meta.json" AFFECTED_ROOT=/r bash "$t/mut.sh" --diff-from <(echo crates/a/src/lib.rs) | sort -u)
  row "MUTANT: ignoring dev edges is caught by the dev-dep row" "a" "$got"
fi
# --no-renames: moving a file out of crate a into crate c touches BOTH (git's rename detection would show only c).
rt="$t/repo"; mkdir -p "$rt/crates/a/src" "$rt/crates/b" "$rt/crates/c/src"; abs="$(cd "$(dirname "$A")" && pwd)/$(basename "$A")"
sed "s|/r/|$rt/|" "$t/meta.json" >"$t/meta-rt.json"
printf 'pub fn moved() -> u32 { 1 }\n// padding so git sees a rename, not a delete+add\n// padding\n' >"$rt/crates/a/src/x.rs"
# plumbing (write-tree/commit-tree) so the scratch repo runs no commit hooks
(cd "$rt" && git init -q && git add -A && b=$(GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t git commit-tree "$(git write-tree)" -m base) && git mv crates/a/src/x.rs crates/c/src/x.rs && git update-ref HEAD "$b" && git add -A && GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t git update-ref HEAD "$(GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t git commit-tree "$(git write-tree)" -p "$b" -m move)")
mv_run() { (cd "$rt" && AFFECTED_META="$t/meta-rt.json" AFFECTED_ROOT="$rt" bash "$1" HEAD~1 HEAD | sort -u); }
row "rename a->c touches both ends (a, its dependent b, and c)" "a
b
c" "$(mv_run "$abs")"
sed 's/--name-only --no-renames/--name-only/' "$A" >"$t/mut-rn.sh"
if cmp -s "$A" "$t/mut-rn.sh"; then echo "FAIL mutant did not apply"; fail=1
else row "MUTANT: letting git detect renames is caught by the rename row" "c" "$(mv_run "$t/mut-rn.sh")"; fi
# include_str!: a docs file a crate embeds is that crate's input (and reaches its dependents); an unembedded one is not.
it="$t/inc"; mkdir -p "$it/crates/a/src" "$it/docs"
printf 'pub const D: &str = include_str!("../../../docs/embedded.md");\n' >"$it/crates/a/src/lib.rs"
sed "s|/r/|$it/|" "$t/meta.json" >"$t/meta-inc.json"
inc_run() { printf '%s\n' "$2" >"$t/dinc"; AFFECTED_META="$t/meta-inc.json" AFFECTED_ROOT="$it" bash "$1" --diff-from "$t/dinc" | sort -u; }
row "docs file embedded by crate a -> a and its dependent b" "a
b" "$(inc_run "$A" docs/embedded.md)"
row "docs file nobody embeds -> nothing" "" "$(inc_run "$A" docs/other.md)"
sed "s/'\\\$1 == f'/'0'/" "$A" >"$t/mut-inc.sh"
if cmp -s "$A" "$t/mut-inc.sh"; then echo "FAIL include mutant did not apply"; fail=1
else row "MUTANT: ignoring include_str! is caught by the embedded-doc row" "" "$(inc_run "$t/mut-inc.sh" docs/embedded.md)"; fi
real="$($A --diff-from <(echo docs/audits/prm-001/prereg.lock) | sort -u)"
case "
$real
" in *"
aprender-review-experiment
"*) echo "ok   real tree: prereg.lock reaches aprender-review-experiment" ;; *) echo "FAIL real tree: prereg.lock -> '$real'"; fail=1 ;; esac
exit "$fail"
