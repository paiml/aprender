#!/usr/bin/env bash
# check_zram_gpu_cut.sh -- aprender-zram-core keeps no dependency on aprender-gpu (FLOW-010 cut 2),
# and the `cuda` feature still refuses to build with ONE named error rather than a stray one.
#   1 the manifest names no aprender-gpu outside comments
#   2 no non-comment line under zram-core/src or zram/src uses the aprender_gpu crate
#   3 lib.rs holds the compile_error! under cfg(all(feature="cuda", not(feature="__cuda-linked")))
#   4 `pub mod gpu;` is gated on both cuda and __cuda-linked
# The 408 lib test names were diffed before/after by hand (identical); this guard keeps the
# edge from coming back. `--self-test` runs the case table with mutants that must turn it RED.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
M=crates/aprender-zram-core/Cargo.toml
L=crates/aprender-zram-core/src/lib.rs
verdict() {
  local r=$1 bad=0
  grep -v '^[[:space:]]*#' "$r/$M" | grep -q 'aprender-gpu' && { echo "manifest names aprender-gpu"; bad=1; }
  if grep -rn --include='*.rs' -E 'aprender_gpu' "$r/crates/aprender-zram-core/src" "$r/crates/aprender-zram/src" 2>/dev/null \
    | grep -v -E '^[^:]+:[0-9]+:[[:space:]]*//' | grep -q .; then echo "source names aprender_gpu"; bad=1; fi
  tr -d '\n' <"$r/$L" | grep -q -E 'cfg\(all\(feature = "cuda", not\(feature = "__cuda-linked"\)\)\)\]\s*compile_error!' \
    || { echo "lib.rs lost the named cuda refusal"; bad=1; }
  tr -d '\n' <"$r/$L" | grep -q -E 'cfg\(all\(feature = "cuda", feature = "__cuda-linked"\)\)\]\s*pub mod gpu;' \
    || { echo "pub mod gpu is not gated on __cuda-linked"; bad=1; }
  return $bad
}
if [ "${1:-}" = "--self-test" ]; then
  t=$(mktemp -d) || exit 2; trap 'rm -rf "${t:?}"' EXIT
  mk() { mkdir -p "$t/$1/crates/aprender-zram-core/src" "$t/$1/crates/aprender-zram/src"
         cp "$ROOT/$M" "$t/$1/$M"; cp "$ROOT/$L" "$t/$1/$L"; }
  n=0; f=0
  case_() { ( verdict "$t/$3" >/dev/null ); rc=$?
    if { [ "$2" = 0 ] && [ $rc -eq 0 ]; } || { [ "$2" != 0 ] && [ $rc -ne 0 ]; }; then n=$((n+1)); echo "ok   $1"; else f=$((f+1)); echo "FAIL $1 (rc=$rc)"; fi; }
  mk g;  case_ "Z1 current tree holds" 0 g
  mk a;  printf '[dependencies]\naprender-gpu = { workspace = true, optional = true }\n' >>"$t/a/$M"; case_ "Z2 MUTANT: manifest dep back" 1 a
  mk b;  echo 'use aprender_gpu::Foo;' >"$t/b/crates/aprender-zram/src/x.rs"; case_ "Z3 MUTANT: source uses aprender_gpu" 1 b
  mk c;  sed -i 's/not(feature = "__cuda-linked")/feature = "x"/' "$t/c/$L"; case_ "Z4 MUTANT: refusal cfg changed" 1 c
  mk d;  sed -i 's/cfg(all(feature = "cuda", feature = "__cuda-linked"))\]/cfg(feature = "cuda")]/' "$t/d/$L"; case_ "Z5 MUTANT: gpu mod ungated" 1 d
  echo "zram gpu cut rows: $n passed, $f failed"; [ $f -eq 0 ]; exit
fi
out=$(verdict "${1:-$ROOT}") && { echo "zram gpu cut: no aprender-gpu edge; cuda refusal intact"; exit 0; }
printf '%s\n' "$out"; exit 1
