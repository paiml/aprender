#!/usr/bin/env bash
# check_tuner_test_move.sh -- the F041-F060 training-data-quality tests moved from
# aprender-compute to aprender-core (FLOW-010, #5803) keep their names, count AND bodies
# (pinned digests), and
# aprender-compute keeps no dependency on aprender-core, not even a dev one.
#   union(compute file, core file) == the 11 names below, each exactly once,
#   compute's Cargo.toml names no aprender-core, and no compute file calls `aprender::`.
# `--self-test` runs the case table, including mutants that must turn it RED.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CF=crates/aprender-compute/tests/tuner_falsification/training_data_quality.rs
KF=crates/aprender-core/tests/tuner_training_data_quality.rs
WANT="f041_empty_training_errors f042_single_sample_graceful f043_nan_labels_error f044_negative_labels_handled f045_heuristic_no_training f046_training_improves f047_large_training_no_oom f048_classifier_training f049_training_data_variance f050_feature_correlation f051_to_f060_reserved"

# DIGESTS: sha256[:16] of each test body with whitespace and trailing commas before )]} removed (rustfmt-proof).
# A moved test whose body changed is RED; editing a test means updating its digest here, in the same PR.
DIGESTS="
f041_empty_training_errors 47de75e437653c3e
f042_single_sample_graceful cbabb3faeb2409c9
f043_nan_labels_error bb9d7a655d14a6ca
f044_negative_labels_handled 91ad5915598c736b
f045_heuristic_no_training f0117334ccf4dd41
f046_training_improves 71148d3ae804eadf
f047_large_training_no_oom 580958506c220314
f048_classifier_training bfbae842f8408223
f049_training_data_variance 7fae71f475e2a6c6
f050_feature_correlation c302d2faf46ee11a
f051_to_f060_reserved 4281b7880278d002
"
# body FILE NAME -> normalized body of `fn NAME(` up to its closing `}` at column 0
body() {
  awk -v n="$2" '$0 ~ "^fn "n"\\(" {p=1} p{print} p&&/^}/{exit}' "$1" | tr -d ' \t\n' | sed -E 's/,([])}])/\1/g'
}
# names ROOT -> sorted test fn names of both files, one per line (duplicates kept)
names() {
  local f
  for f in "$1/$CF" "$1/$KF"; do
    [ -f "$f" ] && awk '/^#\[test\]/{t=1;next} t&&/^fn /{sub(/^fn /,"");sub(/\(.*/,"");print;t=0}' "$f"
  done | sort
}
# verdict ROOT -> prints problems; rc 0 only when none
verdict() {
  local r=$1 bad=0 got want
  got=$(names "$r"); want=$(printf '%s\n' $WANT | sort)
  [ "$got" = "$want" ] || { echo "names differ: want 11 once each, got: $(echo $got)"; bad=1; }
  local n d b
  while read -r n d; do
    [ -n "$n" ] || continue
    b=$(body "$r/$CF" "$n"); [ -n "$b" ] || b=$(body "$r/$KF" "$n")
    [ "$(printf %s "$b" | sha256sum | cut -c1-16)" = "$d" ] || { echo "body of $n differs from its pinned digest (empty or edited)"; bad=1; }
  done <<< "$DIGESTS"
  if grep -q 'aprender-core' "$r/crates/aprender-compute/Cargo.toml"; then
    grep -v '^[[:space:]]*#' "$r/crates/aprender-compute/Cargo.toml" | grep -q 'aprender-core' && { echo "compute Cargo.toml names aprender-core"; bad=1; }
  fi
  if grep -rn --include='*.rs' -E '\baprender::' "$r/crates/aprender-compute/tests" "$r/crates/aprender-compute/examples" 2>/dev/null | grep -v -E '^[^:]+:[0-9]+:[[:space:]]*//' | grep -q .; then
    echo "compute tests/examples call aprender::"; bad=1
  fi
  return $bad
}
if [ "${1:-}" = "--self-test" ]; then
  t=$(mktemp -d) || exit 2; trap 'rm -rf "${t:?}"' EXIT
  mk() { mkdir -p "$t/$1/crates/aprender-compute/tests/tuner_falsification" "$t/$1/crates/aprender-compute/examples" "$t/$1/crates/aprender-core/tests"
         cp "$ROOT/$CF" "$t/$1/$CF"; cp "$ROOT/$KF" "$t/$1/$KF"; cp "$ROOT/crates/aprender-compute/Cargo.toml" "$t/$1/crates/aprender-compute/Cargo.toml"; }
  n=0; f=0
  case_() { # NAME WANT_RC DIR
    ( verdict "$t/$3" >/dev/null ); rc=$?
    if { [ "$2" = 0 ] && [ $rc -eq 0 ]; } || { [ "$2" != 0 ] && [ $rc -ne 0 ]; }; then n=$((n+1)); echo "ok   $1"; else f=$((f+1)); echo "FAIL $1 (rc=$rc)"; fi; }
  mk good;                                   case_ "T1 current tree holds" 0 good
  mk m1; sed -i '/fn f046_training_improves/,/^}/d' "$t/m1/$KF";   case_ "T2 MUTANT: a moved test dropped" 1 m1
  mk m2; cat "$ROOT/$KF" >>"$t/m2/$CF";      case_ "T3 MUTANT: a test duplicated in both files" 1 m2
  mk m3; printf '[dev-dependencies]\naprender = { path = "../aprender-core", package = "aprender-core" }\n' >>"$t/m3/crates/aprender-compute/Cargo.toml"; case_ "T4 MUTANT: dev-dep on aprender-core back" 1 m3
  mk m4; printf 'use aprender::tree::RandomForestRegressor;\n' >"$t/m4/crates/aprender-compute/examples/x.rs"; case_ "T5 MUTANT: compute example calls aprender::" 1 m4
  mk m5; sed -i 's/F046 FALSIFIED/F046 weakened/' "$t/m5/$KF"; case_ "T6 MUTANT: a moved test body edited (assert message)" 1 m5
  mk m6; sed -i '/fn f044_negative_labels_handled/,/^}/{s/\.build()/.build(); let _ = 0/}' "$t/m6/$KF"; case_ "T7 MUTANT: a moved test body changed in logic" 1 m6
  echo "tuner test move rows: $n passed, $f failed"; [ $f -eq 0 ]; exit
fi
out=$(verdict "${1:-$ROOT}") && { echo "tuner test move: 11 names, once each; compute has no aprender-core"; exit 0; }
printf '%s\n' "$out"; exit 1
