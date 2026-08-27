
// ============================================================================
// Additive attention mask builder (plan 01-01)
// Contract: setfit-encoder-conformance-v1, equation `additive_attention_mask`
// ============================================================================

/// Additive penalty applied to padded key positions.
///
/// A large **finite** negative constant, deliberately not `f32::MIN` and not
/// `f32::NEG_INFINITY`. All-padding rows are rejected before this value is ever
/// used, so every softmax row keeps at least one valid key; `exp(-1e9 - max)`
/// then underflows to exactly `0.0` in f32, giving parity with torch on valid
/// positions without any `-inf` or `NaN` arithmetic to reason about.
pub const NEG_MASK: f32 = -1e9;

/// Build the `[B, 1, 1, S]` additive attention mask for a `[B, S]` binary mask.
///
/// Kept positions get `0.0`; padded positions get [`NEG_MASK`]. The rank-4
/// shape is what lets the mask broadcast over `[B, heads, T, S]` attention
/// scores.
///
/// # This op is deliberately a CONSTANT
///
/// It takes no differentiable input, so the result has `requires_grad == false`
/// and records **no** `grad_fn`. That is not an oversight and not a severed
/// graph: `contracts/setfit-encoder-conformance-v1.yaml` carves this equation
/// out of the general graph-connectivity invariant for exactly this reason.
///
/// The graph-connectivity guarantee for masking lives on `apply_additive_mask`
/// (plan 01-09) — the op that ADDS this constant to the attention scores. That
/// is where a severed edge would actually cost gradient, and that is where the
/// contract puts the obligation.
///
/// # Errors
///
/// * [`OpError::ZeroDimension`] — `batch` or `seq` is 0.
/// * [`OpError::ShapeOverflow`] — `batch * seq` overflows `usize`.
/// * [`OpError::LengthMismatch`] — `mask.len()` is not `batch * seq`.
/// * [`OpError::NonBinaryMaskValue`] — an entry is neither `0` nor `1`. A `2` is
///   never silently treated as "keep".
/// * [`OpError::AllPaddingRow`] — a row has no kept position, which would make
///   the whole softmax row `-1e9` and its denominator meaningless.
#[provable_contracts_macros::contract(
    "setfit-encoder-conformance-v1",
    equation = "additive_attention_mask"
)]
pub fn additive_attention_mask(
    mask: &[u8],
    batch: usize,
    seq: usize,
) -> Result<Tensor, OpError> {
    // ---- 1. Shape --------------------------------------------------------
    if batch == 0 {
        return Err(OpError::ZeroDimension { which: "batch" });
    }
    if seq == 0 {
        return Err(OpError::ZeroDimension { which: "seq" });
    }
    let positions = batch
        .checked_mul(seq)
        .ok_or(OpError::ShapeOverflow {
            dims: vec![batch, seq],
        })?;

    if mask.len() != positions {
        return Err(OpError::LengthMismatch {
            ids: positions,
            mask: mask.len(),
        });
    }

    // ---- 2. Values -------------------------------------------------------
    // Malformed values are reported BEFORE all-padding rows: a row of all `2`s
    // is a broken mask, not a padded row, and saying "AllPaddingRow" would send
    // the caller after the wrong defect.
    for (position, &v) in mask.iter().enumerate() {
        if v > 1 {
            return Err(OpError::NonBinaryMaskValue { value: v, position });
        }
    }

    // ---- 3. Per-row validity --------------------------------------------
    // Checked before building anything: an all-padding row makes the whole
    // softmax row NEG_MASK, so its denominator carries no information.
    for row in 0..batch {
        let base = row * seq;
        if !mask[base..base + seq].iter().any(|&v| v == 1) {
            return Err(OpError::AllPaddingRow { row });
        }
    }

    contract_pre_additive_attention_mask!(mask);

    // ---- 4. Build the constant (row-major, LAYOUT-001) -------------------
    // Rank 4 so it broadcasts over [B, heads, T, S] attention scores. No
    // `grad_fn` is recorded and `requires_grad` stays false: this is a
    // CONSTANT by contract, not a severed graph.
    let data: Vec<f32> = mask
        .iter()
        .map(|&keep| if keep == 1 { 0.0 } else { NEG_MASK })
        .collect();
    let result = Tensor::from_vec(data, &[batch, 1, 1, seq]);

    contract_post_additive_attention_mask!(result.data());
    Ok(result)
}
