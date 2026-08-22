
// ============================================================================
// Row-wise L2 normalization (plan 01-03)
// Contract: setfit-encoder-conformance-v1, equation `l2_normalize_rows`
// ============================================================================

/// Normalize every row of a `[B, H]` matrix to unit L2 length, with an
/// **explicit** epsilon floor on the denominator.
///
/// `y[b][h] = x[b][h] / max(||x[b]||_2, eps)`
///
/// Row-major throughout (LAYOUT-001): element `(b, h)` lives at `b * H + h`.
///
/// # The epsilon is a parameter, not a hidden default
///
/// The floor decides which of two *different functions* is evaluated (see the
/// derivative note below), so it cannot be an implementation detail. It is
/// validated up front: a zero, negative, `NaN` or infinite `eps` is rejected
/// with [`OpError::InvalidEpsilon`] rather than silently reintroducing the
/// division-by-zero the floor exists to prevent.
///
/// # Gradient — the derivative is PIECEWISE
///
/// With `n = ||x_row||_2` and `d = max(n, eps)`:
///
/// ```text
/// n >  eps :  dy/dx = (I - y yᵀ) / n     # d depends on x — projected form
/// n <= eps :  dy/dx = I / eps            # d is a CONSTANT — no projection term
/// ```
///
/// Below the clamp the denominator does not depend on `x` at all, so the
/// chain-rule term that produces the `y yᵀ` projection simply does not exist:
/// the map is the plain linear scaling `x ↦ x / eps`. Applying the projected
/// form there is not an approximation, it is the derivative of a different
/// function — and it would pass a naive finite-difference test as long as that
/// test never visits the clamped branch.
///
/// The boundary `n == eps` is assigned to the **clamped** branch (the condition
/// for the projected form is the strict `n > eps`). Both branches agree in the
/// limit only in value, not in derivative, so the choice is documented rather
/// than left to whichever comparison the code happened to use.
///
/// `L2NormalizeRowsBackward` therefore captures the RAW per-row norm and the
/// epsilon, and re-takes the identical `n > eps` decision. It deliberately does
/// not try to infer the branch from the clamped output, which carries no record
/// of which side it came from.
///
/// # Errors
///
/// Fails **closed** — never a `NaN`, never a panic:
///
/// * [`OpError::ShapeMismatch`] — `x` is not 2-D.
/// * [`OpError::ZeroDimension`] — `batch` or `hidden` is 0.
/// * [`OpError::InvalidEpsilon`] — `eps` is not finite, or is `<= 0`.
/// * [`OpError::NonFiniteInput`] — `x` contains `NaN` or `±Inf`.
#[provable_contracts_macros::contract(
    "setfit-encoder-conformance-v1",
    equation = "l2_normalize_rows"
)]
pub fn l2_normalize_rows(x: &Tensor, eps: f32) -> Result<Tensor, OpError> {
    // ---- 1. Shape --------------------------------------------------------
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(OpError::ShapeMismatch {
            expected: vec![0, 0],
            got: shape.to_vec(),
        });
    }
    let batch = shape[0];
    let hidden = shape[1];
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
    let xd = x.data();
    if let Some(position) = xd.iter().position(|v| !v.is_finite()) {
        return Err(OpError::NonFiniteInput { position });
    }

    // Domain is proven by the guards above, so the contract precondition cannot
    // fire. Asserted HERE rather than at entry deliberately: at entry a
    // `debug_assert!` would turn a fail-closed typed error into a debug panic on
    // exactly the hostile inputs this op exists to reject.
    contract_pre_l2_normalize_rows!(xd);

    // ---- 4. Forward (row-major, LAYOUT-001) ------------------------------
    // The sum of squares accumulates in f64. In f32 a legitimate embedding of
    // magnitude 1e-20 squares to 1e-40, which is subnormal and rounds toward
    // zero — the row's norm would collapse to 0 and the clamp decision would be
    // made on a fabricated value. f64 has ~300 decades of headroom here.
    let mut out = vec![0.0f32; batch * hidden];
    let mut norms = Vec::with_capacity(batch);

    for row in 0..batch {
        let base = row * hidden;
        let sumsq: f64 = xd[base..base + hidden]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        let n = sumsq.sqrt() as f32;

        // The clamp. `n > eps` selects the projected branch; `n == eps` is
        // assigned to the constant branch. The backward re-takes this exact
        // comparison from the stored raw norm.
        let d = if n > eps { n } else { eps };
        let inv = 1.0 / f64::from(d);
        for j in 0..hidden {
            out[base + j] = (f64::from(xd[base + j]) * inv) as f32;
        }
        norms.push(n);
    }

    let mut result = Tensor::from_vec(out, &[batch, hidden]);

    // ---- 5. Record the graph edge ----------------------------------------
    if is_grad_enabled() && x.requires_grad_enabled() {
        result.requires_grad_(true);
        let grad_fn = Arc::new(L2NormalizeRowsBackward {
            output: result.clone(),
            norms,
            eps,
            batch,
            hidden,
        });
        result.set_grad_fn(grad_fn.clone());

        with_graph(|graph| {
            graph.register_tensor(x.clone());
            graph.record(result.id(), grad_fn, vec![x.id()]);
        });
    }

    contract_post_l2_normalize_rows!(result.data());
    Ok(result)
}
