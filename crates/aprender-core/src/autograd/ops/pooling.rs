
// ============================================================================
// Masked mean pooling (plan 01-01)
// Contract: setfit-encoder-conformance-v1, equation `masked_mean_pool`
// ============================================================================

/// Reduce a `[B, S, H]` token batch to `[B, H]` sentence embeddings by averaging
/// only the VALID positions of each row.
///
/// `out[b][h] = (Σ_s mask[b*S + s] · hidden[b][s][h]) / n_b`, where
/// `n_b = Σ_s mask[b*S + s]`.
///
/// Row-major throughout (LAYOUT-001): `hidden` is indexed
/// `b * S * H + s * H + h`.
///
/// # Why the denominator is checked
///
/// This is the D-03 checked denominator. A row with no valid position would
/// divide by zero and emit `NaN`, which then silently poisons every parameter
/// gradient downstream — the failure surfaces nowhere near its cause. The row is
/// rejected with a typed error BEFORE the pool is computed, so no `NaN` output
/// is reachable.
///
/// The divisor is also **per row**, not shared. A single batch-wide denominator
/// is invisible on a uniform-length batch and wrong on every mixed-length one.
///
/// # Gradient
///
/// When `hidden` requires grad, the result carries a `MaskedMeanPoolBackward`
/// edge that routes `grad_output[b][h] / n_b` to every valid position and
/// exactly `0.0` to every padded position.
///
/// # Not checked: finiteness of `hidden`
///
/// Unlike `embedding_gather` — whose weight table is loaded from an untrusted
/// model file — `hidden` is a computed graph intermediate produced by the
/// encoder itself. Rejecting a non-finite activation mid-graph would convert a
/// training-dynamics signal into a hard error at an arbitrary layer. Gradient
/// finiteness is asserted where it is meaningful: at the ENC-04 gate.
///
/// # Errors
///
/// * [`OpError::ShapeMismatch`] — `hidden` is not 3-D.
/// * [`OpError::ZeroDimension`] — `batch`, `seq` or `hidden` is 0.
/// * [`OpError::ShapeOverflow`] — `batch * seq` overflows `usize`.
/// * [`OpError::LengthMismatch`] — `mask.len()` is not `batch * seq`.
/// * [`OpError::NonBinaryMaskValue`] — a mask entry is neither `0` nor `1`.
/// * [`OpError::AllPaddingRow`] — a row has no valid position (the checked
///   denominator).
#[provable_contracts_macros::contract(
    "setfit-encoder-conformance-v1",
    equation = "masked_mean_pool"
)]
pub fn masked_mean_pool(hidden: &Tensor, mask: &[u8]) -> Result<Tensor, OpError> {
    // ---- 1. Shape --------------------------------------------------------
    let shape = hidden.shape();
    if shape.len() != 3 {
        return Err(OpError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: shape.to_vec(),
        });
    }
    let batch = shape[0];
    let seq = shape[1];
    let hidden_size = shape[2];
    if batch == 0 {
        return Err(OpError::ZeroDimension { which: "batch" });
    }
    if seq == 0 {
        return Err(OpError::ZeroDimension { which: "seq" });
    }
    if hidden_size == 0 {
        return Err(OpError::ZeroDimension { which: "hidden" });
    }

    let positions = batch.checked_mul(seq).ok_or(OpError::ShapeOverflow {
        dims: vec![batch, seq, hidden_size],
    })?;
    let total = positions
        .checked_mul(hidden_size)
        .ok_or(OpError::ShapeOverflow {
            dims: vec![batch, seq, hidden_size],
        })?;

    if mask.len() != positions {
        return Err(OpError::LengthMismatch {
            ids: positions,
            mask: mask.len(),
        });
    }

    // ---- 2. Mask values --------------------------------------------------
    for (position, &v) in mask.iter().enumerate() {
        if v > 1 {
            return Err(OpError::NonBinaryMaskValue { value: v, position });
        }
    }

    // ---- 3. Checked denominators (D-03) ----------------------------------
    // Every row count is computed and validated BEFORE any pooling arithmetic
    // runs, so a zero denominator can never reach a division and no NaN output
    // is reachable. `total` above is deliberately unused until this point.
    let mut counts = Vec::with_capacity(batch);
    for row in 0..batch {
        let base = row * seq;
        let count = mask[base..base + seq]
            .iter()
            .fold(0usize, |acc, &m| acc + usize::from(m == 1));
        if count == 0 {
            return Err(OpError::AllPaddingRow { row });
        }
        counts.push(count);
    }

    contract_pre_masked_mean_pool!(mask);

    // ---- 4. Forward (row-major, LAYOUT-001) ------------------------------
    debug_assert_eq!(hidden.numel(), total, "shape product must match numel");
    let x = hidden.data();
    let mut out = vec![0.0f32; batch * hidden_size];

    for (row, &count) in counts.iter().enumerate() {
        let base = row * seq;
        let out_off = row * hidden_size;
        for pos in 0..seq {
            if mask[base + pos] != 1 {
                continue;
            }
            let src = base * hidden_size + pos * hidden_size;
            for j in 0..hidden_size {
                out[out_off + j] += x[src + j];
            }
        }
        // PER-ROW divisor — a shared one is invisible on a uniform-length batch
        // and wrong on every mixed-length one.
        let inv = 1.0 / count as f32;
        for j in 0..hidden_size {
            out[out_off + j] *= inv;
        }
    }

    let mut result = Tensor::from_vec(out, &[batch, hidden_size]);

    // ---- 5. Record the graph edge ----------------------------------------
    if is_grad_enabled() && hidden.requires_grad_enabled() {
        result.requires_grad_(true);
        let grad_fn = Arc::new(MaskedMeanPoolBackward {
            mask: mask.to_vec(),
            batch,
            seq,
            hidden: hidden_size,
        });
        result.set_grad_fn(grad_fn.clone());

        with_graph(|graph| {
            graph.register_tensor(hidden.clone());
            graph.record(result.id(), grad_fn, vec![hidden.id()]);
        });
    }

    contract_post_masked_mean_pool!(result.data());
    Ok(result)
}
