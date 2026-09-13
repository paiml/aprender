
// ============================================================================
// Row-wise cosine similarity + tensor-valued MSE reduction (plan 01-03)
// Contract: setfit-encoder-conformance-v1, equations `cosine_similarity_rows`
// and `mse_loss`
// ============================================================================

/// Row-wise cosine similarity between two `[B, H]` matrices, with an explicit
/// epsilon floor on each norm.
///
/// `out[b] = <a[b], b[b]> / (max(||a[b]||_2, eps) * max(||b[b]||_2, eps))`
///
/// Result shape is `[B]` — one similarity per pair of rows. Row-major
/// throughout (LAYOUT-001).
///
/// # Each factor is clamped INDEPENDENTLY
///
/// The denominator is `max(n_a, eps) * max(n_b, eps)`, not
/// `max(n_a * n_b, eps)`. The two agree everywhere both norms exceed `eps`
/// — i.e. across the entire non-degenerate domain — and differ only when at
/// least one row is degenerate. Per-factor clamping is what makes the
/// derivative decompose into two independent branch decisions, one per input,
/// which is what the FD tests exercise. It also matches
/// `torch.nn.functional.cosine_similarity`, whose clamp is applied to each norm
/// before the product.
///
/// The invariant `|out| <= 1` survives both branches, because
/// `|<a,b>| <= n_a * n_b <= max(n_a, eps) * max(n_b, eps)`.
///
/// # Gradient — PIECEWISE, and independently so on each side
///
/// With `n_a = ||a_row||`, `d_a = max(n_a, eps)` (likewise for `b`), and
/// `s = out[row]`:
///
/// ```text
/// n_a >  eps :  ds/da_i = ( b_i/d_b - s * a_i/n_a ) / n_a
/// n_a <= eps :  ds/da_i =   b_i / (eps * d_b)
/// ```
///
/// and symmetrically for `b`. Below the clamp `d_a` is the literal constant
/// `eps`, so the term that differentiates the denominator — the `s * a_i/n_a`
/// projection — does not exist. Four branch combinations are therefore
/// reachable, and the gradient of each input follows only its OWN branch: a
/// clamped `a` does not change how `b`'s gradient is computed.
///
/// # Errors
///
/// * [`OpError::ShapeMismatch`] — either input is not 2-D, or the two shapes
///   differ.
/// * [`OpError::ZeroDimension`] — `batch` or `hidden` is 0.
/// * [`OpError::InvalidEpsilon`] — `eps` is not finite, or is `<= 0`.
/// * [`OpError::NonFiniteInput`] — either input contains `NaN` or `±Inf`.
///   `a` is scanned first, and `position` is the flattened index within
///   whichever tensor tripped the guard.
#[provable_contracts_macros::contract(
    "setfit-encoder-conformance-v1",
    equation = "cosine_similarity_rows"
)]
pub fn cosine_similarity_rows(a: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor, OpError> {
    // ---- 1. Shape --------------------------------------------------------
    if a.shape().len() != 2 {
        return Err(OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: a.shape().to_vec(),
        });
    }
    if b.shape().len() != 2 {
        return Err(OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: b.shape().to_vec(),
        });
    }
    if a.shape() != b.shape() {
        return Err(OpError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    let batch = a.shape()[0];
    let hidden = a.shape()[1];
    if batch == 0 {
        return Err(OpError::ZeroDimension { which: "batch" });
    }
    if hidden == 0 {
        return Err(OpError::ZeroDimension { which: "hidden" });
    }

    // ---- 2. The epsilon floor --------------------------------------------
    // `eps <= 0.0` is FALSE for NaN, so the finiteness test is not redundant.
    if !(eps.is_finite() && eps > 0.0) {
        return Err(OpError::invalid_epsilon(eps));
    }

    // ---- 3. Value-level guards -------------------------------------------
    let ad = a.data();
    let bd = b.data();
    if let Some(position) = ad.iter().position(|v| !v.is_finite()) {
        return Err(OpError::NonFiniteInput { position });
    }
    if let Some(position) = bd.iter().position(|v| !v.is_finite()) {
        return Err(OpError::NonFiniteInput { position });
    }

    // Domain proven by the guards above; asserted here rather than at entry so a
    // `debug_assert!` cannot turn a fail-closed error into a debug panic.
    contract_pre_cosine_similarity_rows!(ad);

    // ---- 4. Forward (row-major, LAYOUT-001) ------------------------------
    // Norms and the dot product accumulate in f64 for the same reason
    // `l2_normalize_rows` does: a 1e-20 component squares to a subnormal in f32
    // and would silently round the norm to zero, deciding the clamp branch on a
    // fabricated value.
    let mut out = vec![0.0f32; batch];
    let mut norms_a = Vec::with_capacity(batch);
    let mut norms_b = Vec::with_capacity(batch);

    for row in 0..batch {
        let base = row * hidden;
        let (mut sa, mut sb, mut dot) = (0.0f64, 0.0f64, 0.0f64);
        for j in 0..hidden {
            let (x, y) = (f64::from(ad[base + j]), f64::from(bd[base + j]));
            sa += x * x;
            sb += y * y;
            dot += x * y;
        }
        let na = sa.sqrt() as f32;
        let nb = sb.sqrt() as f32;

        // Each factor is clamped INDEPENDENTLY. `n > eps` selects the projected
        // branch for that input; `n == eps` is assigned to the constant branch.
        let da = if na > eps { na } else { eps };
        let db = if nb > eps { nb } else { eps };

        out[row] = (dot / (f64::from(da) * f64::from(db))) as f32;
        norms_a.push(na);
        norms_b.push(nb);
    }

    let mut result = Tensor::from_vec(out, &[batch]);

    // ---- 5. Record the graph edge ----------------------------------------
    if is_grad_enabled() && (a.requires_grad_enabled() || b.requires_grad_enabled()) {
        result.requires_grad_(true);
        let grad_fn = Arc::new(CosineSimilarityBackward {
            a: a.clone(),
            b: b.clone(),
            similarity: result.clone(),
            norms_a,
            norms_b,
            eps,
            batch,
            hidden,
        });
        result.set_grad_fn(grad_fn.clone());

        with_graph(|graph| {
            graph.register_tensor(a.clone());
            graph.register_tensor(b.clone());
            graph.record(result.id(), grad_fn, vec![a.id(), b.id()]);
        });
    }

    contract_post_cosine_similarity_rows!(result.data());
    Ok(result)
}

/// Mean squared error between a graph-connected `[B]` prediction and a detached
/// target slice, reduced to a **graph-connected `[1]` tensor**.
///
/// `L = (1/n) * Σ_i (pred[i] - target[i])²`
///
/// # Why this is not `nn::loss` / `nn::self_supervised`
///
/// Those helpers return `f32`. An `f32` cannot carry a `grad_fn`, so composing
/// them into a training step produces a loss that decreases on paper while the
/// encoder never moves — PF-001, the trap this whole phase exists to close.
/// This op returns a `Tensor` of shape `[1]` with an `MseBackward` edge, exactly
/// like [`Tensor::mean`]. Nothing here calls into, or re-exports from, either
/// of those modules.
///
/// # Gradient
///
/// `dL/dpred[i] = 2 * (pred[i] - target[i]) / n`. The target is detached data,
/// not a tensor, so it cannot receive gradient by construction rather than by
/// convention.
///
/// # Errors
///
/// * [`OpError::ShapeMismatch`] — `pred` is not 1-D `[B]`.
/// * [`OpError::ZeroDimension`] — `pred` is empty (the mean's denominator).
/// * [`OpError::LengthMismatch`] — `target.len()` is not `pred.numel()`.
/// * [`OpError::NonFiniteInput`] — a target value is `NaN` or `±Inf`. The
///   target is caller-supplied label data, so it is untrusted; `pred` is a
///   computed graph intermediate and is deliberately NOT scanned, following the
///   same rule `masked_mean_pool` documents.
#[provable_contracts_macros::contract("setfit-encoder-conformance-v1", equation = "mse_loss")]
pub fn mse_loss(pred: &Tensor, target: &[f32]) -> Result<Tensor, OpError> {
    // ---- 1. Shape --------------------------------------------------------
    if pred.shape().len() != 1 {
        return Err(OpError::ShapeMismatch {
            expected: vec![0],
            got: pred.shape().to_vec(),
        });
    }
    let n = pred.shape()[0];
    if n == 0 {
        return Err(OpError::ZeroDimension { which: "batch" });
    }
    if target.len() != n {
        return Err(OpError::LengthMismatch {
            ids: n,
            mask: target.len(),
        });
    }

    // ---- 2. Value-level guards -------------------------------------------
    // The TARGET is scanned: labels are caller-supplied and untrusted, and a
    // NaN label would poison every parameter gradient with a NaN traceable to
    // nothing. `pred` is a computed graph intermediate and is deliberately NOT
    // scanned — the same rule `masked_mean_pool` documents.
    if let Some(position) = target.iter().position(|v| !v.is_finite()) {
        return Err(OpError::NonFiniteInput { position });
    }

    contract_pre_mse_loss!(target);

    // ---- 3. Forward (reduction to [1], the `Tensor::mean` shape) ---------
    let p = pred.data();
    let inv_n = 1.0f64 / n as f64;
    let mut acc = 0.0f64;
    for i in 0..n {
        let d = f64::from(p[i]) - f64::from(target[i]);
        acc += d * d;
    }
    let mut result = Tensor::from_vec(vec![(acc * inv_n) as f32], &[1]);

    // ---- 4. Record the graph edge ----------------------------------------
    // Returning a Tensor rather than an f32 is the entire point (PF-001): an
    // f32 cannot carry this edge, and a training loop built on one reports a
    // falling loss while every upstream weight stays frozen.
    if is_grad_enabled() && pred.requires_grad_enabled() {
        result.requires_grad_(true);
        let grad_fn = Arc::new(MseBackward {
            pred: pred.clone(),
            target: target.to_vec(),
        });
        result.set_grad_fn(grad_fn.clone());

        with_graph(|graph| {
            graph.register_tensor(pred.clone());
            graph.record(result.id(), grad_fn, vec![pred.id()]);
        });
    }

    contract_inv_mse_loss!(result.data());
    Ok(result)
}
