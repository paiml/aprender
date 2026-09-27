#!/usr/bin/env bash
# Case table for lean_cache_seed.sh: plan/seed refusals and a seed on fake trees (no Lean needed).
# Each row: name, expected exit, command. The Mathlib-rev check and the donor-HEAD check overlap by design,
# so the planted mutant deletes BOTH; it must turn
# row s4 RED, so s4 is live.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
SUT=${SUT:-$here/lean_cache_seed.sh}
T=$(mktemp -d); trap 'rm -rf "${T:?}"' EXIT
L=crates/aprender-contracts-staging/lean
fail=0 n=0

mk_tree() { # mk_tree <dir> <toolchain> <mathlib rev> <inputRev>
  mkdir -p "$1/$L"; printf '%s\n' "$2" > "$1/$L/lean-toolchain"
  printf '{"version":"1.1.0","packages":[{"name":"mathlib","rev":"%s","inputRev":"%s"}]}\n' "$3" "$4" > "$1/$L/lake-manifest.json"
}
key_of() { cat "$1/$L/lean-toolchain" "$1/$L/lake-manifest.json" | sha256sum | cut -c1-16; }
mk_entry() { # mk_entry <root> <key> <mathlib HEAD>: a warm entry whose packages/mathlib is a git repo at a commit
  local m="$1/$2/packages/mathlib"; mkdir -p "$m" "$1/$2/build/lib"; : > "$1/$2/build/lib/X.olean"
  git -C "$m" init -q && git -C "$m" -c user.email=t@t -c user.name=t commit -q --allow-empty -m x
  : > "$1/$2.lock"
}
row() { local name=$1 want=$2; shift 2; n=$((n + 1)); "$@" > "$T/out" 2>&1; local got=$?
  if [ "$got" -eq "$want" ]; then echo "ok   $name"; else echo "FAIL $name: exit $got, want $want"; sed 's/^/     /' "$T/out"; fail=1; fi; }

R="$T/root"; mkdir -p "$R"
mk_tree "$T/donor" "leanprover/lean4:v4.29.0-rc4" PLACEHOLDER PLACEHOLDER
dk=$(key_of "$T/donor"); mk_entry "$R" "$dk"
rev=$(git -C "$R/$dk/packages/mathlib" rev-parse HEAD)
mk_tree "$T/donor" "leanprover/lean4:v4.29.0-rc4" "$rev" "$rev"          # the donor pins the SHA
dk2=$(key_of "$T/donor"); mv "$R/$dk" "$R/$dk2"; mv "$R/$dk.lock" "$R/$dk2.lock"
mk_tree "$T/main" "leanprover/lean4:v4.29.0-rc4" "$rev" master           # main: same rev, inputRev master
mk_tree "$T/tc"   "leanprover/lean4:v4.30.0" "$rev" master               # another toolchain
mk_tree "$T/ml"   "leanprover/lean4:v4.29.0-rc4" 0000000000000000000000000000000000000000 master
mk=$(key_of "$T/main")

row "u1 no arguments is usage"                 64 bash "$SUT"
row "u2 seed without a donor is usage"         64 bash "$SUT" seed --tree "$T/main" --root "$R"
row "u3 a non-numeric budget is usage"         64 bash "$SUT" measure --tree "$T/main" --budget-s x
row "p1 plan with a matching donor"             0 bash "$SUT" plan --tree "$T/main" --donor-tree "$T/donor" --root "$R"
row "p2 plan writes nothing"                    0 test ! -e "$R/$mk"
row "i1 a tree without the lean inputs"         2 bash "$SUT" plan --tree "$T/nope" --root "$R"
row "s1 donor == target is refused"             3 bash "$SUT" seed --tree "$T/donor" --donor-tree "$T/donor" --root "$R"
row "s2 another toolchain is refused"           3 bash "$SUT" seed --tree "$T/tc" --donor-tree "$T/donor" --root "$R"
row "s4 another Mathlib rev is refused"         3 bash "$SUT" seed --tree "$T/ml" --donor-tree "$T/donor" --root "$R"
row "s5 no cache root is a precondition"        2 bash "$SUT" seed --tree "$T/main" --donor-tree "$T/donor" --root "$T/none"
row "s6 seed copies the donor to main's key"    0 bash "$SUT" seed --tree "$T/main" --donor-tree "$T/donor" --root "$R" --receipt "$T/r.jsonl"
row "s7 the entry is whole at main's key"       0 test -f "$R/$mk/build/lib/X.olean"
row "s8 the receipt row is lean-cache-seed-v1"  0 jq -e --arg k "$mk" --arg d "$dk2" 'select(.schema == "lean-cache-seed-v1" and .key == $k and .donor_key == $d and .verdict == "seeded")' "$T/r.jsonl"
row "s9 an existing entry is never overwritten" 3 bash "$SUT" seed --tree "$T/main" --donor-tree "$T/donor" --root "$R"
row "s10 no temp dir is left behind"            0 bash -c "! ls -A '$R' | grep -q '^\.'"
rm -rf "${R:?}/${dk2:?}/packages/mathlib/.git"
mk_tree "$T/main2" "leanprover/lean4:v4.29.0-rc4" "$rev" v1
row "s3 a donor without a Mathlib checkout"     3 bash "$SUT" seed --tree "$T/main2" --donor-tree "$T/donor" --root "$R"
if [ -z "${MUTANT:-}" ]; then
  sed -e '/\[ "\$rev" = "\$drev" \]/d' -e '/\[ "\$head" = "\$rev" \]/d' "$SUT" > "$T/mut.sh"
  if cmp -s "$SUT" "$T/mut.sh"; then echo "FAIL m0 the mutation did not apply"; fail=1
  else MUTANT=1 SUT="$T/mut.sh" bash "$0" > "$T/mut.out" 2>&1
    if command grep -q '^FAIL s4' "$T/mut.out"; then echo "ok   m1 the planted mutant turns s4 RED"
    else echo "FAIL m1 s4 survived the planted mutant"; fail=1; fi
  fi
fi
[ "$fail" -eq 0 ] && echo "GREEN lean_cache_seed_test: $n rows" || echo "RED lean_cache_seed_test"
exit "$fail"
