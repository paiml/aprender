//! The MAP fit: core's `LbfgsF64` driven over the Prophet objective.
//!
//! Ported from `sources/004-forecast-mcp-thin-server/src/lib.rs:130-172` (D-08). EVERY
//! piece of the D-09 recipe is load-bearing and was measured against Python Prophet
//! 1.4.0 — objective / T, the non-finite guard, exact L1 on delta,
//! `LbfgsF64::new(MAX_ITERS_PER_ROUND, 1e-7, 20)`, `Stalled` accepted as success,
//! restarts from the stall point until the relative improvement falls below 1e-6 or
//! `max_rounds` is spent, and the x-keyed value+grad cache. Changing any of it
//! re-opens the parity ladder.

use std::cell::RefCell;
use std::time::Instant;

use aprender::optim::{ConvergenceStatus, LbfgsF64};
use aprender::primitives::Vector;

use crate::prophet::{Design, Model, Params};

/// Objective + gradient evaluated ONCE per distinct point; L-BFGS's line search asks for both
/// at the same x, and re-asks at the accepted point next iteration.
struct Cached<'a> {
    model: Model<'a>,
    last: RefCell<Option<(Vec<f64>, f64, Vec<f64>)>>,
    evals: RefCell<usize>,
}

impl Cached<'_> {
    fn ensure(&self, x: &[f64]) {
        let hit = self
            .last
            .borrow()
            .as_ref()
            .is_some_and(|(k, _, _)| k.as_slice() == x);
        if !hit {
            *self.evals.borrow_mut() += 1;
            let (f, g) = self.model.value_and_grad(x);
            *self.last.borrow_mut() = Some((x.to_vec(), f, g));
        }
    }
    fn f(&self, x: &[f64]) -> f64 {
        self.ensure(x);
        self.last.borrow().as_ref().expect("cached").1
    }
    fn g(&self, x: &[f64]) -> Vec<f64> {
        self.ensure(x);
        self.last.borrow().as_ref().expect("cached").2.clone()
    }
}

/// What the fit did, surfaced as `diagnostics.lbfgs` on the wire.
#[derive(Debug, Clone)]
pub struct FitInfo {
    pub rounds: usize,
    pub iterations: usize,
    pub evals: usize,
    pub objective: f64,
    pub status: String,
    pub budget_hit: bool,
}

/// Per-round L-BFGS iteration cap. This, [`crate::types::MAX_POINTS`] and
/// [`crate::types::MAX_HORIZON`] are the HARD bounds on the work a request can buy.
pub const MAX_ITERS_PER_ROUND: usize = 2_000;

/// COOPERATIVE fit budget, in seconds — **not** a hard wall-clock cap (REVIEW-06-U1).
///
/// The budget is cooperative: [`fit_prophet`] inspects the elapsed wall clock exactly
/// once per COMPLETED L-BFGS round, after that round's improvement check. A single round
/// of up to [`MAX_ITERS_PER_ROUND`] iterations may therefore overrun 15 s, and
/// `budget_hit` reports only that a round boundary was crossed late — never that the fit
/// was cut off at 15 s. The HARD bounds
/// are [`MAX_ITERS_PER_ROUND`], [`crate::types::MAX_POINTS`] and
/// [`crate::types::MAX_HORIZON`]. Do NOT add a mid-iteration budget check: that would
/// change the D-09 recipe the parity ladder was measured under.
pub const FIT_BUDGET_SECS: f64 = 15.0;

/// Spike 001 config + spike 003 restarts (D-09, verbatim).
#[must_use]
pub fn fit_prophet(design: &Design, max_rounds: usize) -> (Params, FitInfo) {
    let model = Model::new(design);
    let init = model.init();
    let mut x = Vector::from_vec(model.pack(&init));
    let cache = Cached {
        model,
        last: RefCell::new(None),
        evals: RefCell::new(0),
    };
    let mut best_f = cache.f(x.as_slice());
    let mut opt = LbfgsF64::new(MAX_ITERS_PER_ROUND, 1e-7, 20);
    let mut rounds = 0;
    let mut iters = 0;
    let mut status = String::new();
    let mut budget_hit = false;
    let t0 = Instant::now();
    for _ in 0..max_rounds {
        let r = opt.minimize(
            |v: &Vector<f64>| cache.f(v.as_slice()),
            |v: &Vector<f64>| Vector::from_vec(cache.g(v.as_slice())),
            &x,
        );
        rounds += 1;
        iters += r.iterations;
        status = format!("{:?}", r.status);
        let improved = r.objective_value < best_f - 1e-6 * best_f.abs().max(1.0);
        if improved {
            best_f = r.objective_value;
            x = r.solution;
        }
        if !improved || r.status == ConvergenceStatus::Converged {
            break;
        }
        if t0.elapsed().as_secs_f64() > FIT_BUDGET_SECS {
            budget_hit = true;
            break;
        }
    }
    let p = cache.model.unpack(x.as_slice());
    let evals = *cache.evals.borrow();
    (
        p,
        FitInfo {
            rounds,
            iterations: iters,
            evals,
            objective: best_f / cache.model.scale,
            status,
            budget_hit,
        },
    )
}
