#!/usr/bin/env bash
# kani_parity_chain.sh — run the decode-path Kani harnesses, BY NAME, under a budget
# (PMAT-3140, PVL-001 F4).
#
# WHY THIS EXISTS
# ---------------
# `crates/aprender-contracts/src/kernels/kani_proofs*.rs` carries bounded model-checking
# harnesses, and no gate ran any of them: CI's only kani step is
# `cargo test --test kani_harness_generation`, which checks that harnesses are
# GENERATED, never that one is PROVEN. When this list was first run (2026-09-25),
# 3 of 11 decode-path harnesses did not verify: one unwind bound was too small, one
# claim was false (RMSNorm finiteness with unbounded gamma), and one was refuted by
# its own nondeterministic exp stub. A harness that no gate runs can say anything.
#
# NOT LISTED: verify_softmax_normalization, which ran past 45 min under -j 8 and
# past 15 min alone.
# NOT LISTED: verify_swiglu_fused_equivalence. Fixed, it asks CBMC to prove two
# symbolic f32 division circuits bit-identical, and it timed out at 600 s on 2 and
# on 4 elements. The same property is FALSIFY-SG-002 (proptest,
# crates/aprender-contracts/tests/includes/falsify_actgate_swi_ce_rope.rs).
#
# The list is the decode-path kernels: RMSNorm, softmax, SiLU/SwiGLU, the
# quantized dot, RoPE, attention and GQA. It is NAMED, not globbed, so a renamed or deleted harness is a
# failure (Kani matches nothing) rather than a silently shorter run.
#
#   bash scripts/kani_parity_chain.sh             # 0 all verified · 1 a harness failed · 2 ENV
#   bash scripts/kani_parity_chain.sh --list      # the harness names, one per line
#
# EXIT CODES: 2 IS NEVER A PASS
#   0  every named harness reported VERIFICATION:- SUCCESSFUL
#   1  at least one failed, timed out, or matched no harness (each one named)
#   2  ENV: cargo-kani is not installed. A runner that cannot run Kani must not
#      report ok.
#
# BUDGET: KANI_HARNESS_TIMEOUT seconds per harness (default 600). The whole list
# took about 9 min on lambda-vector (Threadripper 7960X), 2026-09-25;
# verify_online_softmax_2tiles alone is about 5.5 min of it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=kani_parity_chain
TIMEOUT="${KANI_HARNESS_TIMEOUT:-600}"

HARNESSES=(
    verify_rms_positive
    verify_rmsnorm_finiteness
    verify_softmax_positivity
    verify_online_softmax_2tiles
    verify_silu_zero
    verify_silu_lower_bound
    verify_silu_positive_monotonicity
    verify_swiglu_zero_preservation
    verify_swiglu_silu_lower_bound
    verify_quantized_dot_bounded
    # Attention and RoPE (aprender-cb, #3140). They needed stub_exp(0) = 1,
    # stub_sqrt bracketed in [min(x,1), max(x,1)] and ACT_BOUND inputs.
    verify_softmax_bounded
    verify_log_softmax_upper_bound
    verify_rope_norm_preservation
    verify_attention_weights_normalize
    verify_gqa_convex_bound
    verify_gqa_mha_equivalence
    verify_gqa_weight_normalization
)

if [ "${1:-}" = "--list" ]; then
    printf '%s\n' "${HARNESSES[@]}"
    exit 0
fi

command -v cargo-kani > /dev/null 2>&1 || {
    printf '%s: ENV — cargo-kani is not on PATH; %s harnesses NOT run (this is NOT a pass)\n' \
        "$PROG" "${#HARNESSES[@]}" >&2
    exit 2
}

cd "$ROOT" || exit 2
log_dir="${KANI_LOG_DIR:-$(mktemp -d)}"
mkdir -p "$log_dir" || exit 2
fail=0
for h in "${HARNESSES[@]}"; do
    log="$log_dir/$h.log"
    t0=$(date +%s)  # bashrs disable-line=DET002
    timeout "$TIMEOUT" cargo kani -Z stubbing -p aprender-contracts --lib \
        --harness "kernels::kani_proofs::$h" --exact > "$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    # Kani exits 0 on success, but also demand the verdict line: rc alone cannot tell
    # "verified" from "no harness matched" on every Kani version.
    if [ "$rc" -eq 0 ] && grep -q 'VERIFICATION:- SUCCESSFUL' "$log"; then
        printf 'ok      %-40s %4ss\n' "$h" "$dt"
    elif [ "$rc" -eq 124 ]; then
        printf 'TIMEOUT %-40s %4ss (budget %ss)  log: %s\n' "$h" "$dt" "$TIMEOUT" "$log"
        fail=$((fail + 1))
    else
        printf 'FAIL    %-40s %4ss rc=%s  log: %s\n' "$h" "$dt" "$rc" "$log"
        grep -m3 -E 'Failed Checks:|Check [0-9]+:.*FAILURE|no harnesses matched|error(\[|:)' "$log" | sed 's/^/        /'
        fail=$((fail + 1))
    fi
done

if [ "$fail" -eq 0 ]; then
    printf '%s: %s/%s decode-path harnesses verified\n' "$PROG" "${#HARNESSES[@]}" "${#HARNESSES[@]}"
    exit 0
fi
printf '%s: %s of %s harnesses did NOT verify\n' "$PROG" "$fail" "${#HARNESSES[@]}" >&2
exit 1
