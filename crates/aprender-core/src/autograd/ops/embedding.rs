
// ============================================================================
// Batched embedding gather (plan 01-01)
// Contract: setfit-encoder-conformance-v1, equation `embedding_gather`
// ============================================================================

/// Gather a `[B, S, H]` batch of token embeddings from a `[V, H]` weight table.
///
/// Row-major throughout (LAYOUT-001): the output element `(b, s, h)` lives at
/// `b * S * H + s * H + h`, and weight row `id` starts at `id * H`.
///
/// # Gradient
///
/// When `weight` requires grad, the result carries the existing
/// [`EmbeddingBackward`] edge with the **flattened** `B*S` index list. That
/// backward SCATTER-ADDs `grad_output[i]` into `dW[ids[i]]`, so a token id that
/// appears `k` times in the batch accumulates `k` gradient rows into its weight
/// row. Overwriting instead of accumulating would silently drop gradient for
/// every repeated token — the single most common way a gather backward is wrong.
///
/// The token ids are integers and carry no gradient.
///
/// # Errors
///
/// Fails **closed** — never a zero-filled row, never a panic:
///
/// * [`OpError::ShapeMismatch`] — `weight` is not 2-D, or `ids.len()` is not
///   `batch * seq`.
/// * [`OpError::ZeroDimension`] — `batch`, `seq`, `hidden` or `vocab_size` is 0.
///   An empty tensor is refused because it silently no-ops downstream.
/// * [`OpError::ShapeOverflow`] — `batch * seq * hidden` overflows `usize`.
///   Checked with `checked_mul` BEFORE the output buffer is allocated.
/// * [`OpError::NonFiniteInput`] — `weight` contains `NaN` or `±Inf`.
/// * [`OpError::OutOfVocabulary`] — an id is at or beyond `vocab_size`.
///   Deliberately NOT the `aprender-train` zero-fill nor the `qwen2` N-09
///   warn-and-emit-zeros escape: both hide the defect until it shows up as
///   unexplained accuracy loss.
#[provable_contracts_macros::contract(
    "setfit-encoder-conformance-v1",
    equation = "embedding_gather"
)]
pub fn embedding_gather(
    weight: &Tensor,
    ids: &[u32],
    batch: usize,
    seq: usize,
) -> Result<Tensor, OpError> {
    // ---- 1. Shape of the table -------------------------------------------
    let w_shape = weight.shape();
    if w_shape.len() != 2 {
        return Err(OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: w_shape.to_vec(),
        });
    }
    let vocab_size = w_shape[0];
    let hidden = w_shape[1];
    if vocab_size == 0 {
        return Err(OpError::ZeroDimension {
            which: "vocab_size",
        });
    }
    if hidden == 0 {
        return Err(OpError::ZeroDimension { which: "hidden" });
    }

    // ---- 2. Shape of the request -----------------------------------------
    if batch == 0 {
        return Err(OpError::ZeroDimension { which: "batch" });
    }
    if seq == 0 {
        return Err(OpError::ZeroDimension { which: "seq" });
    }

    // Checked BEFORE allocating: a wrapping element count would produce an
    // under-sized buffer and turn every later index into silent corruption.
    let overflow = || OpError::ShapeOverflow {
        dims: vec![batch, seq, hidden],
    };
    let positions = batch.checked_mul(seq).ok_or_else(overflow)?;
    let total = positions.checked_mul(hidden).ok_or_else(overflow)?;

    if ids.len() != positions {
        return Err(OpError::ShapeMismatch {
            expected: vec![batch, seq],
            got: vec![ids.len()],
        });
    }

    // ---- 3. Value-level guards -------------------------------------------
    let w = weight.data();
    if let Some(position) = w.iter().position(|v| !v.is_finite()) {
        return Err(OpError::NonFiniteInput { position });
    }

    // Fail CLOSED on out-of-vocabulary ids. Validated up front, over the whole
    // id slice, so the reported position is the FIRST offender rather than
    // whichever one happened to be reached mid-copy.
    for (position, &id) in ids.iter().enumerate() {
        if id as usize >= vocab_size {
            return Err(OpError::OutOfVocabulary {
                id,
                vocab_size,
                position,
            });
        }
    }

    // Domain is now proven by the guards above, so the contract precondition
    // cannot fire. It is asserted HERE rather than at entry deliberately: at
    // entry a `debug_assert!` would turn a fail-closed typed error into a debug
    // panic on exactly the hostile inputs this op exists to reject.
    contract_pre_embedding_gather!(ids);

    // ---- 4. Forward (row-major, LAYOUT-001) ------------------------------
    let mut out = vec![0.0f32; total];
    for (i, &id) in ids.iter().enumerate() {
        let src = (id as usize) * hidden;
        let dst = i * hidden;
        out[dst..dst + hidden].copy_from_slice(&w[src..src + hidden]);
    }
    let mut result = Tensor::from_vec(out, &[batch, seq, hidden]);

    // ---- 5. Record the graph edge ----------------------------------------
    // Reuses the existing `EmbeddingBackward` as-is: it is index-list based and
    // therefore batch-agnostic, so the flattened B*S id list is exactly what it
    // wants. A bespoke gather backward here would duplicate — and eventually
    // diverge from — an already-proven scatter-add.
    if is_grad_enabled() && weight.requires_grad_enabled() {
        result.requires_grad_(true);
        let grad_fn = Arc::new(EmbeddingBackward {
            indices: ids.to_vec(),
            vocab_size,
            hidden_size: hidden,
        });
        result.set_grad_fn(grad_fn.clone());

        with_graph(|graph| {
            graph.register_tensor(weight.clone());
            graph.record(result.id(), grad_fn, vec![weight.id()]);
        });
    }

    contract_post_embedding_gather!(result.data());
    Ok(result)
}
