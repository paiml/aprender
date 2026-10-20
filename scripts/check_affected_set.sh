#!/usr/bin/env bash
# FLOW-010 E5 RED rows for scripts/affected_set.sh. Oracle for the closure is
# `cargo tree -i` (a different code path from the tool's cargo-metadata jq).
set -euo pipefail
# the CI container runs as root over a host-owned checkout: git would refuse it (dubious ownership)
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
A=scripts/affected_set.sh
t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' EXIT
fail=0
row() { # name want got
  if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1"; echo "  want: $2" | head -c 300; echo; echo "  got:  $3" | head -c 300; echo; fail=1; fi
}
run() { printf '%s\n' "$@" >"$t/d"; $A --diff-from "$t/d"; }
core_list=" aprender-core aprender-compute aprender-contracts apr-cli aprender-quant "
readers="$(sed -n 's/^READERS="${AFFECTED_READERS:-\(.*\)}"$/\1/p' "$A")"
meta=$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null)
members=$(jq -r '.packages[].name' <<<"$meta" | sort)
# a READERS entry that is a directory and not a package (e.g. facades) is dropped by the tool; drop it from the oracle too
readers="$(for r in $readers; do if grep -qx "$r" <<<"$members"; then echo "$r"; fi; done)"
# a non-core crate that has dependents, and a leaf with none
withdep="" leaf=""
for p in $members; do
  case "$core_list" in *" $p "*) continue ;; esac
  n=$(cargo tree --offline -i "$p" --workspace -e normal,dev,build --prefix none 2>/dev/null | sed 's/ v.*//' | sort -u | grep -vc "^$p$" || true)
  dir=$(jq -r --arg p "$p" '.packages[]|select(.name==$p)|.manifest_path' <<<"$meta" | sed "s|$(git rev-parse --show-toplevel)/||; s|/Cargo.toml||")
  case "$dir" in crates/*) ;; *) continue ;; esac
  [ -z "$withdep" ] && [ "$n" -gt 0 ] && [ "$n" -lt 10 ] && withdep="$p:$dir"
  [ -z "$leaf" ] && [ "$n" -eq 0 ] && leaf="$p:$dir"
  [ -n "$withdep" ] && [ -n "$leaf" ] && break
done
[ -n "$withdep" ] && [ -n "$leaf" ] || { echo "FAIL could not find a leaf and a crate with dependents"; exit 1; }
wp=${withdep%%:*}; wd=${withdep#*:}; lp=${leaf%%:*}; ld=${leaf#*:}
# every non-ALL set also carries aprender-core (its falsification_spec_v10 test scans every crate's src)
want=$( { cargo tree --offline -i "$wp" --workspace -e normal,dev,build --prefix none 2>/dev/null | sed 's/ v.*//'; printf '%s\n' $readers; } | sort -u)
lc="$(printf '%s\n' "$lp" $readers | sort -u)"
row "crate with dependents ($wp) -> itself + dependents + aprender-core" "$want" "$(run "$wd/src/lib.rs" | sort -u)"
row "leaf crate ($lp) -> itself + aprender-core" "$lc" "$(run "$ld/src/lib.rs" | sort -u)"
row "core crate -> ALL" ALL "$(run crates/aprender-core/src/lib.rs)"
row "aprender-quant source -> ALL (aprender-serve capability.rs reads aprender-quant/src/ggml_type.rs)" ALL "$(run crates/aprender-quant/src/ggml_type.rs)"
row "a crate's build.rs -> ALL (contracts build_helper.rs reads crates/*/build.rs)" ALL "$(run "$ld/build.rs")"
row "apr-cli source -> ALL (aprender-core tests read apr-cli/src/commands_enum.rs)" ALL "$(run crates/apr-cli/src/commands_enum.rs)"
row "--diff-from with .. is refused (SEC010)" "2" "$($A --diff-from ../x >/dev/null 2>&1; echo $?)"
row "Cargo.lock -> ALL" ALL "$(run Cargo.lock)"
row "unknown path -> ALL" ALL "$(run some/unknown/path.txt)"
row "unknown path next to a leaf -> ALL" ALL "$(run "$ld/src/lib.rs" scripts/whatever.sh)"
row "docs file -> ALL (tests read docs/ at run time)" ALL "$(run docs/x.md)"
row "root README.md -> ALL" ALL "$(run README.md)"
row "root CLAUDE.md -> ALL" ALL "$(run CLAUDE.md)"
row "ci/sections.yml -> ALL" ALL "$(run ci/sections.yml)"
row "ci/explicit-test-commands.d fragment -> ALL" ALL "$(run ci/explicit-test-commands.d/815-affected-set-rows.cmd)"
row "a crate's own Cargo.toml -> ALL" ALL "$(run "$ld/Cargo.toml")"
row "a .md inside a leaf crate -> ALL (aprender-core reads crates/*/README.md)" ALL "$(run "$ld/README.md")"
row "a .json inside a leaf crate -> ALL (kernel-registry.json is read by aprender-contracts)" ALL "$(run "$ld/kernel-registry.json")"
row "a .rs inside a leaf crate stays scoped (itself + aprender-core)" "$lc" "$(run "$ld/src/other.rs" | sort -u)"
row "--filter names the package and aprender-core" "$(printf 'package(%s)\n' $lc | paste -sd'|' | sed 's/|/ | /g')" "$($A --filter --diff-from <(echo "$ld/src/lib.rs"))"
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
sed '/^    crates\/\*\/build.rs)/d' "$A" >"$t/mut-build.sh"
if cmp -s "$A" "$t/mut-build.sh"; then echo "FAIL build.rs mutant did not apply"; fail=1
else printf '%s\n' "$ld/build.rs" >"$t/db"; row "MUTANT: dropping the build.rs arm is caught by the build.rs row" "$lc" "$(bash "$t/mut-build.sh" --diff-from "$t/db" | sort -u || true)"; fi
# mutants: the skip list that used to exist (docs/*, *.md) must be caught by the ALL rows above.
sed 's#^    crates/\*.rs) ;;#    docs/* | *.md) continue ;;\n    crates/*.rs) ;;#' "$A" >"$t/mut-docs.sh"
if cmp -s "$A" "$t/mut-docs.sh"; then echo "FAIL docs-skip mutant did not apply"; fail=1
else printf '%s\n%s\n' docs/x.md "$ld/src/lib.rs" >"$t/dd"; row "MUTANT: skipping docs/*.md is caught by the docs-ALL row" "$lc" "$(bash "$t/mut-docs.sh" --diff-from "$t/dd" | sort -u || true)"; fi
sed 's#^    crates/\*) emit ALL.*#    crates/*) continue ;;#' "$A" >"$t/mut-toml.sh"
if cmp -s "$A" "$t/mut-toml.sh"; then echo "FAIL manifest mutant did not apply"; fail=1
else printf '%s\n%s\n' "$ld/Cargo.toml" "$ld/src/lib.rs" >"$t/dt"; row "MUTANT: skipping non-.rs crate files is caught by the manifest row" "$lc" "$(bash "$t/mut-toml.sh" --diff-from "$t/dt" | sort -u || true)"; fi
# CORE crates: each must give ALL, and dropping each from CORE must be seen (its dependents alone are not the workspace)
for c in aprender-compute aprender-contracts aprender-quant; do
  cd_=$(jq -r --arg p "$c" '.packages[]|select(.name==$p)|.manifest_path' <<<"$meta" | sed "s|$(git rev-parse --show-toplevel)/||; s|/Cargo.toml||")
  row "CORE crate $c -> ALL" ALL "$(run "$cd_/src/lib.rs")"
  sed "s/ $c//" "$A" >"$t/mut-core.sh"
  if cmp -s "$A" "$t/mut-core.sh"; then echo "FAIL core mutant ($c) did not apply"; fail=1
  else printf '%s\n' "$cd_/src/lib.rs" >"$t/dcore"; row "MUTANT: dropping $c from CORE is caught" "not-ALL" "$([ "$(bash "$t/mut-core.sh" --diff-from "$t/dcore" | head -n1)" != ALL ] && echo not-ALL || echo ALL)"; fi
done
# a 2-hop chain (c -> b -> a) and the multi-package --filter format, on a synthetic graph
cat >"$t/chain.json" <<'JSON'
{"packages":[
 {"name":"a","manifest_path":"/r/crates/a/Cargo.toml","dependencies":[]},
 {"name":"b","manifest_path":"/r/crates/b/Cargo.toml","dependencies":[{"name":"a","kind":null}]},
 {"name":"c","manifest_path":"/r/crates/c/Cargo.toml","dependencies":[{"name":"b","kind":null}]},
 {"name":"aprender-core","manifest_path":"/r/crates/core/Cargo.toml","dependencies":[]}]}
JSON
chain() { AFFECTED_META="$t/chain.json" AFFECTED_ROOT=/r AFFECTED_CORE=none AFFECTED_READERS=aprender-core bash "$1" --diff-from <(echo crates/a/src/lib.rs) | sort -u; }
row "2-hop chain a<-b<-c: touching a reaches b and c (+ aprender-core)" "a
aprender-core
b
c" "$(chain "$A")"
sed 's/until(grow(.) == .; grow(.))/grow(.)/' "$A" >"$t/mut-hop.sh"
if cmp -s "$A" "$t/mut-hop.sh"; then echo "FAIL 1-hop mutant did not apply"; fail=1
else row "MUTANT: a 1-hop closure is caught by the 2-hop row" "a
aprender-core
b" "$(chain "$t/mut-hop.sh")"; fi
want_f="package(a) | package(aprender-core) | package(b) | package(c)"
row "multi-package --filter is ' | ' joined" "$want_f" "$(AFFECTED_META="$t/chain.json" AFFECTED_ROOT=/r AFFECTED_CORE=none AFFECTED_READERS=aprender-core bash "$A" --filter --diff-from <(echo crates/a/src/lib.rs))"
sed "s/ | sed 's\/|\/ | \/g'//" "$A" >"$t/mut-fmt.sh"
if cmp -s "$A" "$t/mut-fmt.sh"; then echo "FAIL filter-format mutant did not apply"; fail=1
else row "MUTANT: an unspaced multi-package filter is caught" "package(a)|package(aprender-core)|package(b)|package(c)" "$(AFFECTED_META="$t/chain.json" AFFECTED_ROOT=/r AFFECTED_CORE=none AFFECTED_READERS=aprender-core bash "$t/mut-fmt.sh" --filter --diff-from <(echo crates/a/src/lib.rs))"; fi
# an unreadable cargo metadata is ALL, not an empty answer
echo 'not json' >"$t/bad.json"
row "unreadable metadata -> ALL" ALL "$(AFFECTED_META="$t/bad.json" bash "$A" --diff-from <(echo crates/a/src/lib.rs) 2>/dev/null)"
sed 's/ || { emit ALL; exit 0; }$//' "$A" >"$t/mut-jq.sh"
if cmp -s "$A" "$t/mut-jq.sh"; then echo "FAIL jq-failure mutant did not apply"; fail=1
else row "MUTANT: dropping the jq-failure fallback is caught" "" "$(AFFECTED_META="$t/bad.json" bash "$t/mut-jq.sh" --diff-from <(echo crates/a/src/lib.rs) 2>/dev/null || true)"; fi
# READERS: each reader is in every non-ALL set (real tree), and dropping the READERS list is caught
for r in $readers; do
  case " $(run "$ld/src/lib.rs" | tr '\n' ' ')" in *" $r "*) echo "ok   reader $r is in a non-ALL set" ;; *) echo "FAIL reader $r missing from a non-ALL set"; fail=1 ;; esac
done
sed 's#^READERS=.*#READERS="aprender-core"#' "$A" >"$t/mut-readers.sh"
if cmp -s "$A" "$t/mut-readers.sh"; then echo "FAIL readers mutant did not apply"; fail=1
else row "MUTANT: an emptied READERS list is caught" "$(printf '%s\n' aprender-core "$lp" | sort -u)" "$(printf '%s\n' "$ld/src/lib.rs" >"$t/drd"; bash "$t/mut-readers.sh" --diff-from "$t/drd" | sort -u)"; fi
# the res= jq fallback ALONE: metadata with no `dependencies` key makes that jq fail -> ALL
echo '{"packages":[{"name":"a","manifest_path":"/r/crates/a/Cargo.toml"}]}' >"$t/nodeps.json"
row "no dependencies key -> ALL (res= fallback)" ALL "$(AFFECTED_META="$t/nodeps.json" AFFECTED_ROOT=/r bash "$A" --diff-from <(echo crates/a/src/lib.rs) 2>/dev/null)"
sed '/<<<"$meta")" || { emit ALL; exit 0; }$/s/ || { emit ALL; exit 0; }$//' "$A" >"$t/mut-res.sh"
if cmp -s "$A" "$t/mut-res.sh"; then echo "FAIL res-fallback mutant did not apply"; fail=1
else row "MUTANT: dropping only the res= fallback is caught" "" "$(AFFECTED_META="$t/nodeps.json" AFFECTED_ROOT=/r bash "$t/mut-res.sh" --diff-from <(echo crates/a/src/lib.rs) 2>/dev/null || true)"; fi
exit "$fail"
