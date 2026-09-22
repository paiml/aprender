#!/usr/bin/env bash
# check_unwired_capabilities.sh — "the corrected replacement has zero callers"
# (aprender#3686), built as TWO rules because one does not reach.
#
# THE DEFECT CLASS. Someone diagnoses a bug, writes the fix, and never wires it
# in. The correct code ships next to the wrong behaviour, compiled and tested,
# called by nothing.
#
# RULE 2 (PRIMARY) — a declared-unsupported CAPABILITY that is already
# implemented. `gpu_supported_ops()` says which ops the CUDA backend supports.
# The ops it omits are refused at the capability layer, LOUD, at load. For each
# omitted op, ask: does an implementation already exist with no production
# caller? If yes, the model is being refused for a capability the tree HAS.
#   Measured on 5593c2037: LayerNorm -> `layer_norm_gpu`, 0 production callers.
#   That is aprender#3075 (Phi-2/Phi-3 GGUF never GPU-eligible). The kernel is
#   compiled and unit-tested; the only references outside its own file are 11
#   test sites and one `include!`. The gate's first run names an open issue.
#
# RULE 1 (NARROWED) — #3686 as literally specified: an EXPORTED function whose
# own doc comment asserts that live behaviour elsewhere is wrong, with zero
# references outside its defining file. Narrowed to exported because "no
# external reference" is what `private` means: unnarrowed, 45 of 49 hits in
# this tree were private fns and most of the rest were `#[test]` functions.
#
# WHY RULE 2 IS PRIMARY AND RULE 1 IS NOT. #3686's premise is that the finder
# "already wrote it down, in the tree, in a comment". For #3075 they did not —
# `layer_norm_gpu`'s doc comment says "PAR-014: Apply LayerNorm on GPU" and
# asserts no defect at all; the diagnosis lives in the issue. Rule 1 cannot
# reach it, and neither can its caller condition (12 external refs). Rule 2
# starts from a registry that already exists instead of from phrase-guessing.
#
# WHY THE UNSUPPORTED SET IS DERIVED, NOT LISTED. capability.rs carries a prose
# comment naming the unsupported ops. Reading that comment would make the gate
# a second representation of the list, free to drift from the code. Instead:
#   unsupported = <RequiredOp variants> MINUS <ops.insert(...) in gpu_supported_ops()>
# Both parsed from the source. The comment is never read.
#
#
# MEASURED: RULE 1 WOULD HAVE REPORTED ONE QUARTER OF A FOUR-INSTANCE CLUSTER.
# crates/aprender-core/src/format/layout_contract_enforce.rs holds six contract
# enforcers. FOUR have zero production callers:
#   enforce_import_contract            9 production callers   wired
#   enforce_architecture_completeness  4                      wired
#   enforce_load_contract              0
#   enforce_embedding_contract         0   doc: "MANDATORY ... prevents garbage
#                                          inference output"; all 10 external
#                                          references are tests
#   enforce_matmul_contract            0   doc: "MANDATORY"
#   validate_ffn_shape_symmetry        0
# Rule 1 surfaced exactly ONE of the four -- `enforce_embedding_contract` -- and
# only because its doc happens to contain the words "layout is wrong". The other
# three are the identical defect with doc comments that assert nothing. That is
# the strongest available argument for rule 2: a phrase rule finds the instances
# whose author happened to narrate them, which is not the same population as the
# instances that exist.
#
# DELIBERATE DEVIATION FROM #3686 AS WRITTEN. The ticket asks for "zero callers
# outside its own module". This gate requires ZERO PRODUCTION CALLERS ANYWHERE.
# The ticket's criterion flags correctly-wired code: measured here, it flagged
# `cpu_forward_handles` (crates/apr-cli/src/commands/qa_capability.rs:237),
# which `hybrid_ssm_verdict` calls twenty lines below it in the same file. A
# module-private helper called by its own module's public entry point has no
# external caller BY DESIGN. It would equally flag `qwen35_route_notice` today,
# now that #3595 is fixed and its caller sits in the same file -- i.e. the
# ticket's own flagship instance would stay RED after being repaired. Recording
# this as a deviation rather than quietly satisfying the done_when.
# Bare: judge the tree.  --self-test: the case table.  --update: restamp rule 1.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="scripts/check_unwired_capabilities.sh"
CAP="$ROOT/crates/aprender-serve/src/capability.rs"
MAP="$ROOT/scripts/capability_op_impl_map.txt"
ACK="$ROOT/scripts/unwired_capabilities_acknowledged.txt"
BASELINE="$ROOT/scripts/unwired_doc_assert_baseline.txt"

# Doc-comment phrases that ASSERT a defect rather than describe behaviour.
DOC_ASSERT='is now false|no longer|is wrong|used to .{0,80}now|(is|isn.t) *not? *wired|never (called|wired|used)|nothing (calls|uses)'
FNDEF='^[[:space:]]*pub([[:space:]]*\([^)]*\))?[[:space:]]+(const[[:space:]]+|async[[:space:]]+|unsafe[[:space:]]+)*fn[[:space:]]+'

# ------------------------------------------------------------------ helpers
# A reference that lives in test code, not in what ships.
is_test_path() {
  case "$1" in
    */tests/*|tests/*) return 0 ;;
    *test*|*falsify*|*bench*) return 0 ;;
  esac
  return 1
}

# Line number where a file's `#[cfg(test)]` section starts (huge if none), so
# references below it are test references even in a non-test-named file.
cfg_test_line() {
  local n
  n=$(grep -nE '^[[:space:]]*#\[cfg\(test\)\]' "$1" 2>/dev/null | head -1 | cut -d: -f1)
  printf '%s\n' "${n:-999999999}"
}

# Production callers of $1: references that are not the definition, not in test
# code, and not an `include!` directive naming the file.
production_callers() {
  local sym=$1 f ln text n=0
  while IFS=: read -r f ln text; do
    [ -n "$f" ] || continue
    local rel=${f#"$ROOT"/}
    is_test_path "$rel" && continue
    case "$text" in *'include!'*) continue ;; esac
    grep -qE "${FNDEF}${sym}\b" <<<"$text" && continue      # the definition itself
    grep -qE "^[[:space:]]*(///|//!|//)" <<<"$text" && continue  # a mention in prose
    [ "$ln" -ge "$(cfg_test_line "$f")" ] && continue
    n=$((n + 1))
  done < <(grep -rnw --include='*.rs' -- "$sym" "$ROOT/crates" 2>/dev/null)
  printf '%s\n' "$n"
}

# --------------------------------------------------- rule 2: derive the sets
enum_variants() {
  awk '/^pub enum RequiredOp \{/{f=1;next} f&&/^\}/{exit}
       f&&/^[[:space:]]*[A-Z][A-Za-z0-9]*,[[:space:]]*$/{gsub(/[ \t,]/,"");print}' "$CAP"
}
supported_ops() {
  awk '/^pub fn gpu_supported_ops\(\)/{f=1} f&&/ops\.insert\(RequiredOp::/{
         line=$0; sub(/.*RequiredOp::/,"",line); sub(/\).*/,"",line); print line }
       f&&/^\}/{exit}' "$CAP"
}
unsupported_ops() { comm -23 <(enum_variants | sort) <(supported_ops | sort); }

map_symbol() { awk -F'\t' -v o="$1" '$1==o{print $2; exit}' "$MAP"; }
ack_issue()  { awk -F'\t' -v o="$1" '$1==o{print $2; exit}' "$ACK"; }

rule2() {
  local findings=$1 op sym callers rc=0 nvar nmapped
  : > "$findings"
  nvar=$(enum_variants | wc -l)
  # VACUITY FLOOR: a parse that found no variants measured nothing.
  if [ "$nvar" -lt 5 ]; then
    echo "VACUOUS: parsed $nvar RequiredOp variants from $CAP; the enum is larger" >&2
    return 2
  fi
  # The map must cover the enum, or it rots silently as the enum grows.
  while IFS= read -r op; do
    [ -n "$op" ] || continue
    if ! grep -qE "^${op}$(printf '\t')" "$MAP"; then
      echo "  MAP INCOMPLETE: RequiredOp::$op is not in scripts/capability_op_impl_map.txt" >> "$findings"
      rc=1
    fi
  done < <(enum_variants)

  while IFS= read -r op; do
    [ -n "$op" ] || continue
    sym=$(map_symbol "$op")
    [ "$sym" = "-" ] && continue          # unimplemented, not unwired
    [ -n "$sym" ] || continue
    callers=$(production_callers "$sym")
    if [ "$callers" -eq 0 ]; then
      local iss; iss=$(ack_issue "$op")
      if [ -n "$iss" ]; then
        echo "  ACKNOWLEDGED: RequiredOp::$op -- \`$sym\` exists with 0 production callers (open issue #$iss)" >> "$findings"
      else
        echo "  UNWIRED CAPABILITY: RequiredOp::$op is declared unsupported, but \`$sym\` exists with 0 production callers" >> "$findings"
        echo "                      No acknowledgement row. Wire it, or add one naming an open issue:" >> "$findings"
        echo "                      scripts/unwired_capabilities_acknowledged.txt" >> "$findings"
        rc=1
      fi
    fi
  done < <(unsupported_ops)
  return $rc
}

# --------------------------------------- rule 1: doc asserts a defect, unused
rule1_scan() {
  # Candidates: an EXPORTED fn whose doc comment ASSERTS a defect. Then the
  # discriminator -- zero PRODUCTION callers anywhere, the same primitive rule 2
  # uses.
  #
  # WHY NOT "no caller outside its own module", which is what #3686 says. That
  # criterion flags correctly-wired code: a module-private helper called by its
  # own module's public entry point has no external caller BY DESIGN. Measured
  # here: it flagged `cpu_forward_handles` (qa_capability.rs:237), which
  # `hybrid_ssm_verdict` calls 20 lines below it. It would equally flag
  # `qwen35_route_notice` TODAY, now that #3595 is fixed and its caller
  # `run_qwen35_generate_dispatch` sits in the same file. "Nothing calls this at
  # all" is the defect; "nothing OUTSIDE calls this" is ordinary structure.
  local findings=$1 total=0 files=0 f rel cands
  : > "$findings"
  cands=$(mktemp) || return 2
  trap 'rm -f "$cands"' RETURN
  while IFS= read -r rel; do
    f="$ROOT/$rel"
    [ -r "$f" ] || continue
    is_test_path "$rel" && continue
    files=$((files + 1))
    awk -v rx="$DOC_ASSERT" -v rel="$rel" \
        '/^[[:space:]]*\/\/\//   { doc = doc " " $0; next }
         /^[[:space:]]*#\[/      { next }
         /^[[:space:]]*$/        { next }
         {
           if (doc != "" && $0 ~ /^[[:space:]]*pub([[:space:]]*\([^)]*\))?[[:space:]]+([a-z]+[[:space:]]+)*fn[[:space:]]+/) {
             low = tolower(doc)
             if (low ~ rx) { n = $0; sub(/.*fn[[:space:]]+/, "", n); sub(/[^A-Za-z0-9_].*/, "", n); print rel "|" NR "|" n }
           }
           doc = ""
         }' "$f" >> "$cands"
  done < <(cd "$ROOT" && git ls-files 'crates/*.rs' 2>/dev/null)

  if [ "$files" -lt 100 ]; then
    echo "VACUOUS: rule 1 reached $files source files; this tree has thousands" >&2
    rm -f "$cands"; return 2
  fi
  # VACUITY FLOOR 2: the phrase set must match SOMETHING, or it has rotted.
  local ncand; ncand=$(wc -l < "$cands")
  if [ "$ncand" -lt 1 ]; then
    echo "VACUOUS: the doc-assert phrase set matched 0 functions in $files files" >&2
    rm -f "$cands"; return 2
  fi

  local crel cln cname callers
  while IFS='|' read -r crel cln cname; do
    [ -n "$cname" ] || continue
    callers=$(production_callers "$cname")
    if [ "$callers" -eq 0 ]; then
      echo "  $crel:$cln  $cname  (doc asserts a defect; ZERO production callers)" >> "$findings"
      total=$((total + 1))
    fi
  done < "$cands"
  rm -f "$cands"
  printf '%s\n' "$total"
}

read_baseline() {
  [ -f "$BASELINE" ] || { echo "FAIL: no baseline at $BASELINE" >&2; return 2; }
  local v
  v=$(grep -vE '^[[:space:]]*(#|$)' "$BASELINE" | head -1 | tr -d '[:space:]')
  case "$v" in ''|*[!0-9]*) echo "FAIL: baseline is not a count: '${v}'" >&2; return 2 ;; esac
  printf '%s\n' "$v"
}

# ---------------------------------------------------------------- case table
# Every row puts a decision to the SHIPPED code, both polarities. A gate tuned
# only on its known instances is either vacuous or noisy and you cannot tell
# which from reading it.
CHECKS=0; FAILED=0
ok()  { CHECKS=$((CHECKS+1)); printf 'ok    %s\n' "$1"; }
bad() { CHECKS=$((CHECKS+1)); FAILED=$((FAILED+1)); printf 'FAIL  %s\n' "$1"; }
doc_match()    { if grep -qiE "$DOC_ASSERT" <<<"$2"; then ok "$1"; else bad "$1 -- wanted a MATCH on: $2"; fi; }
doc_no_match() { if grep -qiE "$DOC_ASSERT" <<<"$2"; then bad "$1 -- wanted NO match on: $2"; else ok "$1"; fi; }

self_test() {
  echo "=== rule 2: the unsupported set is DERIVED from code, never from the comment ==="
  local nv ns nu
  nv=$(enum_variants | wc -l); ns=$(supported_ops | wc -l); nu=$(unsupported_ops | wc -l)
  if [ "$nv" -gt 5 ] && [ "$ns" -gt 3 ] && [ "$nu" -gt 0 ] && [ $((ns + nu)) -eq "$nv" ]; then
    ok "D1  $nv variants = $ns supported + $nu unsupported (partition holds)"
  else
    bad "D1  partition broken: $nv variants, $ns supported, $nu unsupported"
  fi
  if unsupported_ops | grep -qx 'LayerNorm'; then ok 'D2  LayerNorm derives as UNSUPPORTED'
  else bad 'D2  LayerNorm did not derive as unsupported -- the parse missed it'; fi
  if supported_ops | grep -qx 'RMSNorm'; then ok 'D3  RMSNorm derives as SUPPORTED'
  else bad 'D3  RMSNorm did not derive as supported'; fi

  echo "=== rule 2 MUST-MATCH: the known instance (aprender#3075) ==="
  local c; c=$(production_callers layer_norm_gpu)
  if [ "$c" -eq 0 ]; then ok "M1  layer_norm_gpu has 0 production callers -- #3075 is flagged"
  else bad "M1  layer_norm_gpu now has $c production callers; if it was WIRED, delete this row and the map entry"; fi

  echo "=== rule 2 MUST-NOT-MATCH: the substring trap that fooled the first draft ==="
  # GeluMlp is declared unsupported, but the symbol sharing its prefix is the
  # ACTIVATION kernel and it is wired. Mapping GeluMlp -> gelu_gpu would have
  # reported a phantom defect forever.
  local g; g=$(production_callers gelu_gpu)
  if [ "$g" -gt 0 ]; then ok "N1  op_whose_impl_shares_a_prefix_with_an_unrelated_wired_kernel: gelu_gpu has $g production callers, so it is NOT unwired"
  else bad "N1  gelu_gpu shows 0 production callers; either it regressed or production_callers() is broken"; fi
  if [ "$(map_symbol GeluMlp)" = "-" ]; then ok 'N2  GeluMlp maps to "-" (unimplemented), not to the activation kernel'
  else bad 'N2  GeluMlp is mapped to a symbol -- re-read the map header before changing this'; fi

  echo "=== rule 2: unimplemented is NOT unwired ==="
  local u ok3=1
  for u in AbsolutePos AttnFinalSoftcap PostAttnFfnNorm; do
    [ "$(map_symbol "$u")" = "-" ] || ok3=0
  done
  if [ "$ok3" -eq 1 ]; then ok 'U1  the three PMAT-824 ops map to "-" and cannot be reported as unwired'
  else bad 'U1  a PMAT-824 op gained a symbol; if real, it is now a rule-2 candidate'; fi

  echo "=== rule 2 MUTATION: is the production/test distinction load bearing? ==="
  # layer_norm_gpu has 11 test references. A production_callers() that counted
  # test references would report >0 and the gate would MISS #3075 entirely.
  local all; all=$(grep -rnw --include='*.rs' -- layer_norm_gpu "$ROOT/crates" 2>/dev/null | wc -l)
  if [ "$all" -gt 5 ] && [ "$(production_callers layer_norm_gpu)" -eq 0 ]; then
    ok "M2  layer_norm_gpu: $all total references, 0 production -- counting tests would hide #3075, so the filter is load bearing"
  else
    bad "M2  mutation row inconclusive: $all total references"
  fi

  echo "=== rule 1: the doc-assert phrases, both polarities ==="
  doc_match    'A1  the #3595 shape'   '/// #3477: the GPU case used to print X -- which is now false: the hybrid runs on the GPU.'
  doc_match    'A2  "no longer"'       '/// PMAT-1: the loop no longer re-reads the header.'
  doc_match    'A3  explicit "not wired"' '/// The kernel exists but is not wired into the forward loop.'
  doc_no_match 'A4  should_use_matched_because_its_doc_says_should_be_used  -- the phrase-artifact FP' '/// Check if multi-queue mode should be used'
  doc_no_match 'A5  verdict_from_cache_matched_exists_but_inside_cache_exists -- the phrase-artifact FP' '/// Inputs: - `cache_exists`: was the teacher_logits dir populated'
  doc_no_match 'A6  a plain behaviour description asserts nothing' '/// PAR-014: Apply LayerNorm on GPU. Performs: output = (input - mean) / sqrt(var + eps) * gamma + beta'

  echo "=== rule 1 MUTATION: does dropping the phrase requirement flood? ==="
  # A6 is layer_norm_gpu's real doc comment. If the phrase set matched it, rule
  # 1 would claim the #3075 instance it demonstrably cannot reach.
  if grep -qiE 'layernorm|apply' <<<'/// PAR-014: Apply LayerNorm on GPU' &&
     ! grep -qiE "$DOC_ASSERT" <<<'/// PAR-014: Apply LayerNorm on GPU'; then
    ok 'M3  a descriptive doc is matchable by a LOOSER pattern but not by the shipped one -- the assertion requirement is load bearing'
  else
    bad 'M3  the shipped phrase set matches a purely descriptive doc comment'
  fi

  printf '\n%s checks, %s failed\n' "$CHECKS" "$FAILED"
  [ "$FAILED" -eq 0 ]
}

# -------------------------------------------------------------------- driver
main() {
  case "${1:-}" in
    --self-test) self_test; return $? ;;
  esac

  local f2 f1 count base rc=0
  f2=$(mktemp) || return 2
  f1=$(mktemp) || { rm -f "$f2"; return 2; }
  trap 'rm -f "$f1" "$f2"' RETURN

  echo "=== rule 2: declared-unsupported capabilities that are already implemented ==="
  echo "  RequiredOp variants : $(enum_variants | wc -l)"
  echo "  supported (derived) : $(supported_ops | tr '\n' ' ')"
  echo "  UNsupported (derived): $(unsupported_ops | tr '\n' ' ')"
  rule2 "$f2" || rc=1
  if [ -s "$f2" ]; then echo; cat "$f2"; fi

  count=$(rule1_scan "$f1") || { rm -f "$f1" "$f2"; return 2; }
  base=$(read_baseline)    || { rm -f "$f1" "$f2"; return 2; }

  case "${1:-}" in
    --update)
      if [ "$count" -gt "$base" ]; then
        echo "REFUSED: --update may only lower the rule-1 baseline ($base -> $count is a RISE)"
        cat "$f1"; rm -f "$f1" "$f2"; return 1
      fi
      printf '# tool_version=none (measured by awk/grep over tracked crates/**/*.rs)\n# Rule 1 (#3686): EXPORTED fns whose doc asserts a defect and that nothing\n# outside their own file references. This number may only FALL.\n# Re-stamp with: bash %s --update\n%s\n' "$SELF" "$count" > "$BASELINE"
      echo "rule-1 baseline restamped: $base -> $count"; rm -f "$f1" "$f2"; return 0 ;;
  esac

  echo
  echo "=== rule 1: exported fns whose doc asserts a defect, referenced nowhere else ==="
  echo "  baseline $base   measured $count"
  if [ "$count" -gt "$base" ]; then
    echo
    echo "FAIL: $count sites, ceiling $base. A doc comment that says the live"
    echo "      behaviour is wrong, on a function nothing outside its file calls,"
    echo "      is a fix that was written and never wired in."
    cat "$f1"; rc=1
  elif [ "$count" -lt "$base" ]; then
    echo "  PASS -- $((base - count)) below the ceiling. Lower it: bash $SELF --update"
  else
    echo "  PASS"
  fi

  rm -f "$f1" "$f2"
  [ "$rc" -eq 0 ] || echo
  [ "$rc" -eq 0 ] || echo "FAIL: see findings above (aprender#3686)."
  return $rc
}

main "$@"
