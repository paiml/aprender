#!/usr/bin/env bash
# check_tuner_test_move.sh -- the F041-F060 training-data-quality tests moved from aprender-compute to
# aprender-core (FLOW-010, #5803) must still be COMPILED, LISTED and NOT IGNORED by the test harness.
# BEHAVIOUR, not text: it asks cargo what each target would run, so no source-level re-gating trick
# (#[cfg], #[ignore], macro_rules! shadowing, #[path] mods, code joined onto a // line) can hide a test.
#   1. `cargo test -p aprender-core --test tuner_training_data_quality -- --list` plus
#      `cargo test -p aprender-compute --test tuner_falsification -- --list` (only training_data_quality::)
#      must name exactly the 11 pre-move tests, each once; a cargo failure or empty list is RED (fail closed).
#   2. the same two targets with `--list --ignored` must name none of them.
#   3. compute's Cargo.toml names no aprender-core and no compute test/example calls `aprender::`.
#   4. content pin, VERBATIM bytes (no whitespace normalising -- that let code be joined onto a `//` line): a digest of
#      each moved test fn with its doc/attr lines, a digest of each file from its first item on, and the header before
#      it may hold only blank/`//!`/`#![allow]`/single `use` lines (no macro_rules!). The moved bodies equal the
#      merge-base bodies up to rustfmt and the dropped `#[cfg(feature = "ml-tuner")]` lines (proved in the PR body).
#      Text cannot prove a test runs; parts 1-2 do. Editing a moved test means updating its digest here in the same PR.
# `--self-test` feeds synthetic cargo listings, so it needs no compile; the mutants must turn it RED.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CF=crates/aprender-compute/tests/tuner_falsification/training_data_quality.rs
KF=crates/aprender-core/tests/tuner_training_data_quality.rs
WANT="f041_empty_training_errors f042_single_sample_graceful f043_nan_labels_error f044_negative_labels_handled f045_heuristic_no_training f046_training_improves f047_large_training_no_oom f048_classifier_training f049_training_data_variance f050_feature_correlation f051_to_f060_reserved"

DIGESTS="
f041_empty_training_errors 070237018993271c
f042_single_sample_graceful bc88d040de7f6792
f043_nan_labels_error 9fd06ac9384bfcaa
f044_negative_labels_handled fbb01296c746a127
f045_heuristic_no_training 8cabd670f956b0d0
f046_training_improves f221030467215901
f047_large_training_no_oom 140c3b8e5cb1d85b
f048_classifier_training 72c5d41797dc1285
f049_training_data_variance 1f2fd8cbb55e4919
f050_feature_correlation 0a494cbb1416adde
f051_to_f060_reserved fbb8c9ddc4be9944
fit_throughput_rf 7da5ef57b0d6baeb
@tail:crates/aprender-core/tests/tuner_training_data_quality.rs e599e682001e6f49
@tail:crates/aprender-compute/tests/tuner_falsification/training_data_quality.rs 4268c362cc9890b0
"

# body FILE NAME -> VERBATIM text of `fn NAME(` with its leading `///` and `#[..]` lines, up to its closing `}` at column 0.
# No whitespace is stripped: stripping newlines is what let code be joined onto a `//` line and still hash equal.
body() {
  awk -v n="$2" '/^(\/\/\/|#\[)/ { pend = pend $0 "\n"; next }
      $0 ~ "^fn "n"\\(" { printf "%s", pend; p = 1 }
      p { print } p && /^}/ { exit }
      { pend = "" }' "$1"
}
# tail FILE -> everything from the first item (`///`, `#[test]` or `fn`) on; the header before it is checked, not hashed
tailof() { awk '!t && /^(\/\/\/|#\[test\]|fn )/ { t = 1 } t { print }' "$1"; }
# header_ok FILE -> rc 0 only when every line before the first item is blank, `//!`, `#![allow(..)]`, `#[allow(unused_imports)]`
# or ONE single-statement `use trueno|aprender::..;` -- so no macro_rules!, mod, #[path] or fn can hide in the header
header_ok() { awk '/^(\/\/\/|#\[test\]|fn )/ { exit } { print }' "$1" | /usr/bin/grep -v -E '^$|^//!|^#!\[allow\(|^#\[allow\(unused_imports\)\]$|^use (trueno|aprender)::[^;]*;$' | { ! /usr/bin/grep -q .; }; }
# listed FILE KIND -> sorted names of the moved tests in a `--list` output (KIND core: every test; comp: training_data_quality:: only)
listed() {
  awk -v kind="$2" '/: test$/ { n = $0; sub(/: test$/, "", n)
      if (kind == "comp") { if (n !~ /^training_data_quality::/) next; sub(/^training_data_quality::/, "", n) }
      print n }' "$1" 2> /dev/null | sort
}
# verdict_lists CORE COMP CORE_IGN COMP_IGN -> problems; rc 0 only when none
verdict_lists() {
  local bad=0 got want x
  want=$(printf '%s\n' $WANT | sort)
  for x in "$1" "$2"; do [ -s "$x" ] || { echo "empty or missing cargo listing ($x): fail closed"; bad=1; }; done
  got=$({ listed "$1" core; listed "$2" comp; } | sort)
  [ "$got" = "$want" ] || { echo "listed tests differ from the 11 pre-move names: got: $(echo $got)"; bad=1; }
  got=$({ listed "$3" core; listed "$4" comp; } | sort)
  [ -z "$got" ] || { echo "ignored moved tests: $(echo $got)"; bad=1; }
  return $bad
}
# verdict_static ROOT -> Cargo dependency, aprender:: calls, content pin
verdict_static() {
  local r=$1 bad=0 n d b
  while read -r n d; do
    [ -n "$n" ] || continue
    case $n in
      @tail:*) b=$(tailof "$r/${n#@tail:}") ;;
      *) b=$(body "$r/$CF" "$n"); [ -n "$b" ] || b=$(body "$r/$KF" "$n") ;;
    esac
    [ "$(printf '%s\n' "$b" | sha256sum | cut -c1-16)" = "$d" ] || { echo "$n differs from its pinned verbatim digest (empty or edited)"; bad=1; }
  done <<< "$DIGESTS"
  for b in "$CF" "$KF"; do header_ok "$r/$b" || { echo "$b: header before the first item holds more than blank, //!, #![allow], use lines"; bad=1; }; done
  if grep -v '^[[:space:]]*#' "$r/crates/aprender-compute/Cargo.toml" | grep -q 'aprender-core'; then echo "compute Cargo.toml names aprender-core"; bad=1; fi
  if grep -rn --include='*.rs' -E '\baprender::' "$r/crates/aprender-compute/tests" "$r/crates/aprender-compute/examples" 2>/dev/null | grep -v -E '^[^:]+:[0-9]+:[[:space:]]*//' | grep -q .; then
    echo "compute tests/examples call aprender::"; bad=1
  fi
  return $bad
}
# cargo_lists ROOT DIR -> writes DIR/{core,comp,core_ign,comp_ign}.txt from cargo; a cargo failure leaves the file empty (RED)
cargo_lists() {
  local r=$1 d=$2
  ( cd "$r" && cargo test --quiet -p aprender-core --test tuner_training_data_quality -- --list 2> /dev/null ) > "$d/core.txt" || : > "$d/core.txt"
  ( cd "$r" && cargo test --quiet -p aprender-compute --test tuner_falsification -- --list 2> /dev/null ) > "$d/comp.txt" || : > "$d/comp.txt"
  ( cd "$r" && cargo test --quiet -p aprender-core --test tuner_training_data_quality -- --list --ignored 2> /dev/null ) > "$d/core_ign.txt" || echo FAILED > "$d/core_ign.txt"
  ( cd "$r" && cargo test --quiet -p aprender-compute --test tuner_falsification -- --list --ignored 2> /dev/null ) > "$d/comp_ign.txt" || echo FAILED > "$d/comp_ign.txt"
}
verdict() { # ROOT -> problems; rc 0 only when none
  local r=$1 d bad=0
  d=$(mktemp -d) || return 2
  cargo_lists "$r" "$d"
  [ "$(cat "$d/core_ign.txt")" = FAILED ] || [ "$(cat "$d/comp_ign.txt")" = FAILED ] && { echo "cargo --list --ignored failed: fail closed"; bad=1; }
  verdict_lists "$d/core.txt" "$d/comp.txt" "$d/core_ign.txt" "$d/comp_ign.txt" || bad=1
  verdict_static "$r" || bad=1
  rm -rf "${d:?}"
  return $bad
}
if [ "${1:-}" = "--self-test" ]; then
  t=$(mktemp -d) || exit 2; trap 'rm -rf "${t:?}"' EXIT
  n=0; f=0
  mkl() { # DIR -> a good set of listings
    mkdir -p "$t/$1"; : > "$t/$1/core.txt"; : > "$t/$1/comp.txt"; : > "$t/$1/ci.txt"; : > "$t/$1/pi.txt"
    for w in $WANT; do case $w in f041*|f042*|f043*|f044*|f046*|f047*|f048*) echo "$w: test" >> "$t/$1/core.txt" ;; *) echo "training_data_quality::$w: test" >> "$t/$1/comp.txt" ;; esac; done
    echo "other::unrelated: test" >> "$t/$1/comp.txt"; echo "0 tests, 0 benchmarks" >> "$t/$1/ci.txt"; echo "0 tests, 0 benchmarks" >> "$t/$1/pi.txt"; }
  lcase() { # NAME WANT_RC DIR
    ( verdict_lists "$t/$3/core.txt" "$t/$3/comp.txt" "$t/$3/ci.txt" "$t/$3/pi.txt" > /dev/null ); rc=$?
    if { [ "$2" = 0 ] && [ $rc -eq 0 ]; } || { [ "$2" != 0 ] && [ $rc -ne 0 ]; }; then n=$((n+1)); echo "ok   $1"; else f=$((f+1)); echo "FAIL $1 (rc=$rc)"; fi; }
  mkl g;                                                                                 lcase "L1 good listing: the 11 once each, none ignored" 0 g
  mkl a; sed -i '/f041_/d' "$t/a/core.txt";                                              lcase "L2 MUTANT: a moved test not listed (cfg'd out / shadowed / #[path]'d)" 1 a
  mkl b; echo "f041_empty_training_errors: test" >> "$t/b/core.txt";                     lcase "L3 MUTANT: a test listed twice" 1 b
  mkl c; echo "f046_training_improves: test" > "$t/c/ci.txt";                            lcase "L4 MUTANT: a moved test is #[ignore]d (in --ignored)" 1 c
  mkl d; : > "$t/d/comp.txt";                                                            lcase "L5 MUTANT: compute listing empty (cargo failed): fail closed" 1 d
  mkl e; sed -i '/f049_/d' "$t/e/comp.txt";                                              lcase "L6 MUTANT: compute target lost its module (CI step off / mod removed)" 1 e
  mkl h; echo "training_data_quality::f052_extra: test" >> "$t/h/comp.txt";              lcase "L7 MUTANT: an unaccounted test in the moved module" 1 h
  mkl i; echo "training_data_quality::f045_heuristic_no_training: test" > "$t/i/pi.txt"; lcase "L8 MUTANT: a compute test is ignored" 1 i
  # static rows run on copies of the real files
  mk() { mkdir -p "$t/$1/crates/aprender-compute/tests/tuner_falsification" "$t/$1/crates/aprender-compute/examples" "$t/$1/crates/aprender-core/tests"
         cp "$ROOT/$CF" "$t/$1/$CF"; cp "$ROOT/$KF" "$t/$1/$KF"; cp "$ROOT/crates/aprender-compute/Cargo.toml" "$t/$1/crates/aprender-compute/Cargo.toml"; }
  scase() { ( verdict_static "$t/$3" > /dev/null ); rc=$?
    if { [ "$2" = 0 ] && [ $rc -eq 0 ]; } || { [ "$2" != 0 ] && [ $rc -ne 0 ]; }; then n=$((n+1)); echo "ok   $1"; else f=$((f+1)); echo "FAIL $1 (rc=$rc)"; fi; }
  mk s0;                                                                                 scase "S1 current tree: pins hold, compute has no aprender-core" 0 s0
  mk s1; sed -i 's/F046 FALSIFIED/F046 weakened/' "$t/s1/$KF";                           scase "S2 MUTANT: a moved test body edited" 1 s1
  mk s2; sed -i '/fn f044_negative_labels_handled/,/^}/{s/\.build()/.build(); let _ = 0/}' "$t/s2/$KF"; scase "S3 MUTANT: a moved test body changed in logic" 1 s2
  mk s3; printf '[dev-dependencies]\naprender = { path = "../aprender-core", package = "aprender-core" }\n' >> "$t/s3/crates/aprender-compute/Cargo.toml"; scase "S4 MUTANT: dev-dep on aprender-core back" 1 s3
  mk s4; printf 'use aprender::tree::RandomForestRegressor;\n' > "$t/s4/crates/aprender-compute/examples/x.rs"; scase "S5 MUTANT: compute example calls aprender::" 1 s4
  mk s5; sed -i 's/< 10/< 11/' "$t/s5/$KF";                                              scase "S6 MUTANT: helper fit_throughput_rf guard changed" 1 s5
  mk s6; sed -i '/Should error (insufficient), not panic/{N;s/\n */ /}' "$t/s6/$KF";   scase "S7 MUTANT: code joined onto a // line (the r3/r4 hole)" 1 s6
  mk s7; sed -i '/^use trueno::tuner::TunerFeatures;/a macro_rules! assert { ($($t:tt)*) => {}; }' "$t/s7/$KF"; scase "S8 MUTANT: macro_rules! assert shadow after the use line" 1 s7
  mk s8; sed -i '/^use trueno::tuner::TunerFeatures;/a #[path = "elsewhere.rs"] mod shadow;' "$t/s8/$KF"; scase "S9 MUTANT: #[path] mod in the header" 1 s8
  mk s9; sed -i 's/F045 FALSIFIED/F045 weakened/' "$t/s9/$CF";                           scase "S10 MUTANT: a compute-resident test body edited" 1 s9
  mk s10; printf '\n#[test]\nfn f099_extra() {}\n' >> "$t/s10/$KF";                    scase "S11 MUTANT: an extra test appended to the moved file (tail digest)" 1 s10
  echo "tuner test move rows: $n passed, $f failed"; [ $f -eq 0 ]; exit
fi
out=$(verdict "${1:-$ROOT}") && { echo "tuner test move: cargo lists the 11 moved tests once each, none ignored; compute has no aprender-core"; exit 0; }
printf '%s\n' "$out"; exit 1
