//! Spike-004 Prophet fit: exact L1, objective ÷ T, guard, cached value+gradient, restarts, caps.
use crate::prophet::*;
use aprender::optim::{ConvergenceStatus, LbfgsF64};
use aprender::primitives::Vector;
use std::cell::RefCell;
use std::time::Instant;

struct Cached<'a> { model: Model<'a>, last: RefCell<Option<(Vec<f64>, f64, Vec<f64>)>> }
impl<'a> Cached<'a> {
    fn ensure(&self, x: &[f64]) { let hit = self.last.borrow().as_ref().map_or(false, |(k, _, _)| k.as_slice() == x); if !hit { let (f, g) = self.model.value_and_grad(x); *self.last.borrow_mut() = Some((x.to_vec(), f, g)); } }
    fn f(&self, x: &[f64]) -> f64 { self.ensure(x); self.last.borrow().as_ref().expect("c").1 }
    fn g(&self, x: &[f64]) -> Vec<f64> { self.ensure(x); self.last.borrow().as_ref().expect("c").2.clone() }
}
pub fn fit_prophet(design: &Design) -> (Params, f64) {
    let model = Model::new(design);
    let init = model.init();
    let mut x = Vector::from_vec(model.pack(&init));
    let cache = Cached { model, last: RefCell::new(None) };
    let mut best_f = cache.f(x.as_slice());
    let mut opt = LbfgsF64::new(2_000, 1e-7, 20);
    let t0 = Instant::now();
    for _ in 0..8 {
        let r = opt.minimize(|v: &Vector<f64>| cache.f(v.as_slice()), |v: &Vector<f64>| Vector::from_vec(cache.g(v.as_slice())), &x);
        let improved = r.objective_value < best_f - 1e-6 * best_f.abs().max(1.0);
        if improved { best_f = r.objective_value; x = r.solution; }
        if !improved || r.status == ConvergenceStatus::Converged || t0.elapsed().as_secs_f64() > 15.0 { break; }
    }
    (cache.model.unpack(x.as_slice()), t0.elapsed().as_secs_f64())
}
