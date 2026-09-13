//! Limited-memory BFGS (L-BFGS) optimizer.
//!
//! L-BFGS is a quasi-Newton method for large-scale optimization that approximates
//! the inverse Hessian using a limited history of gradient information.
//!
//! # Precision
//!
//! The algorithm lives ONCE, in the private generic core [`LbfgsImpl<T>`], and is
//! exposed through two NON-generic public entry points:
//!
//! - [`LBFGS`] — f32, byte-compatible with the pre-widening API. It gains no
//!   generic parameter and no default type parameter, because either of those
//!   would be a semver break for downstream `use` sites in turbofish or
//!   trait-bound position.
//! - [`LbfgsF64`] — f64, for objectives whose gradient norm approaches f32
//!   epsilon near the optimum (softmax-NLL is one). An f32 Wolfe line search
//!   can stall there and report progress it did not make.
//!
//! The f32 path is proven unchanged by a frozen bit-pattern golden trajectory
//! (`lbfgs_f32_golden_trajectory_is_unchanged` in `lbfgs_tests.rs`), captured
//! against the pre-widening implementation.

use crate::primitives::Vector;

use super::line_search::WolfeLineSearch;
use super::{ConvergenceStatus, OptimizationResult, OptimizationResultF64, Optimizer};

// =========================================================================
// Private float abstraction
// =========================================================================

/// The float operations the L-BFGS core needs, and nothing else.
///
/// Deliberately NOT `num_traits::Float`: this phase adds zero external
/// dependencies. Every tuning constant is an associated const so each width
/// uses the nearest representable value in ITS OWN precision — that is what
/// keeps the f32 path bit-identical to the pre-widening implementation rather
/// than merely "close".
pub(crate) trait LbfgsFloat:
    Copy
    + PartialOrd
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity, and the initial line-search step.
    const ONE: Self;
    /// Step-growth factor when the upper bracket is still infinite.
    const TWO: Self;
    /// Positive infinity (the initial upper bracket).
    ///
    /// Named `INF`, not `INFINITY`, so it cannot shadow — or cycle with — the
    /// inherent `f32::INFINITY` / `f64::INFINITY` it is defined from.
    const INF: Self;
    /// Not-a-number, used ONLY as the line search's "non-finite observed" signal.
    const NAN_SIGNAL: Self;
    /// Wolfe Armijo constant c1.
    const WOLFE_C1: Self;
    /// Wolfe curvature constant c2.
    const WOLFE_C2: Self;
    /// Threshold on `y^T s` above which a correction pair is stored.
    const CURVATURE_EPS: Self;
    /// Step size below which progress is declared stalled.
    const STALL_EPS: Self;

    /// Square root.
    fn sqrt(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// True when neither NaN nor infinite.
    fn is_finite(self) -> bool;
    /// Midpoint, with the same overflow behaviour the width's `midpoint` has.
    fn midpoint(self, other: Self) -> Self;
}

impl LbfgsFloat for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const INF: Self = f32::INFINITY;
    const NAN_SIGNAL: Self = f32::NAN;
    const WOLFE_C1: Self = 1e-4;
    const WOLFE_C2: Self = 0.9;
    const CURVATURE_EPS: Self = 1e-10;
    const STALL_EPS: Self = 1e-12;

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    fn midpoint(self, other: Self) -> Self {
        f32::midpoint(self, other)
    }
}

impl LbfgsFloat for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const INF: Self = f64::INFINITY;
    const NAN_SIGNAL: Self = f64::NAN;
    const WOLFE_C1: Self = 1e-4;
    const WOLFE_C2: Self = 0.9;
    const CURVATURE_EPS: Self = 1e-10;
    const STALL_EPS: Self = 1e-12;

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    fn midpoint(self, other: Self) -> Self {
        f64::midpoint(self, other)
    }
}

/// Allocates a zero vector of the given width.
///
/// `Vector::zeros` exists only for f32; `Vector::from_vec` is generic over
/// `T: Copy`, so the core needs no widening of the shared `Vector` surface.
fn zeros<T: LbfgsFloat>(n: usize) -> Vector<T> {
    Vector::from_vec(vec![T::ZERO; n])
}

/// True when every element is finite.
fn is_all_finite<T: LbfgsFloat>(v: &Vector<T>) -> bool {
    (0..v.len()).all(|i| v[i].is_finite())
}

// =========================================================================
// Private generic Wolfe line search
// =========================================================================

/// Wolfe line search, generic over the float width.
///
/// Structurally identical to [`super::line_search::WolfeLineSearch`] (which
/// stays f32 and public), plus the non-finite guard the public one does not
/// need: every `>` and `<=` below is FALSE for NaN, so without the guard a
/// poisoned trial point is silently "accepted".
#[derive(Debug, Clone)]
struct WolfeSearch<T: LbfgsFloat> {
    c1: T,
    c2: T,
    max_iter: usize,
}

impl<T: LbfgsFloat> WolfeSearch<T> {
    fn new(c1: T, c2: T, max_iter: usize) -> Self {
        Self { c1, c2, max_iter }
    }

    /// Returns the accepted step size, or `T::NAN` when a non-finite value was
    /// observed.
    ///
    /// A NaN return is a SIGNAL, not a step: [`LbfgsImpl::minimize`] maps it to
    /// [`ConvergenceStatus::NumericalError`]. This is the line search's OWN
    /// guard — it fires for objectives and gradients that are finite at `x` but
    /// non-finite at a trial point, which the entry-point check on `x0` cannot
    /// see.
    fn search<F, G>(&self, f: &F, grad: &G, x: &Vector<T>, d: &Vector<T>) -> T
    where
        F: Fn(&Vector<T>) -> T,
        G: Fn(&Vector<T>) -> Vector<T>,
    {
        let fx = f(x);
        let grad_x = grad(x);

        // Directional derivative: ∇f(x)ᵀd
        let mut dir_deriv = T::ZERO;
        for i in 0..x.len() {
            dir_deriv += grad_x[i] * d[i];
        }

        if !fx.is_finite() || !dir_deriv.is_finite() {
            return T::NAN_SIGNAL;
        }

        let mut alpha = T::ONE;
        let mut alpha_lo = T::ZERO;
        let mut alpha_hi = T::INF;

        let mut x_new = zeros::<T>(x.len());
        for _ in 0..self.max_iter {
            for i in 0..x.len() {
                x_new[i] = x[i] + alpha * d[i];
            }

            let fx_new = f(&x_new);
            let grad_new = grad(&x_new);

            let mut dir_deriv_new = T::ZERO;
            for i in 0..x.len() {
                dir_deriv_new += grad_new[i] * d[i];
            }

            if !fx_new.is_finite() || !dir_deriv_new.is_finite() || !is_all_finite(&x_new) {
                return T::NAN_SIGNAL;
            }

            // Armijo: f(x + α*d) ≤ f(x) + c₁*α*∇f(x)ᵀd
            if fx_new > fx + self.c1 * alpha * dir_deriv {
                alpha_hi = alpha;
                alpha = alpha_lo.midpoint(alpha_hi);
                continue;
            }

            // Curvature: |∇f(x + α*d)ᵀd| ≤ c₂*|∇f(x)ᵀd|
            if dir_deriv_new.abs() <= self.c2 * dir_deriv.abs() {
                return alpha;
            }

            if dir_deriv_new > T::ZERO {
                alpha_hi = alpha;
            } else {
                alpha_lo = alpha;
            }

            if alpha_hi.is_finite() {
                alpha = alpha_lo.midpoint(alpha_hi);
            } else {
                alpha *= T::TWO;
            }
        }

        alpha
    }
}

// =========================================================================
// Private generic core
// =========================================================================

/// One completed run of the generic core, before it is mapped onto the
/// width-specific public result type.
#[derive(Debug, Clone)]
struct LbfgsOutcome<T: LbfgsFloat> {
    solution: Vector<T>,
    objective_value: T,
    iterations: usize,
    status: ConvergenceStatus,
    gradient_norm: T,
    elapsed_time: std::time::Duration,
}

/// The L-BFGS algorithm, generic over the float width.
///
/// PRIVATE by construction: the public surface is the two non-generic wrappers
/// [`LBFGS`] and [`LbfgsF64`].
#[derive(Debug, Clone)]
struct LbfgsImpl<T: LbfgsFloat> {
    max_iter: usize,
    tol: T,
    m: usize,
    line_search: WolfeSearch<T>,
    s_history: Vec<Vector<T>>,
    y_history: Vec<Vector<T>>,
}

impl<T: LbfgsFloat> LbfgsImpl<T> {
    fn new(max_iter: usize, tol: T, m: usize, line_search: WolfeSearch<T>) -> Self {
        Self {
            max_iter,
            tol,
            m,
            line_search,
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
        }
    }

    /// Two-loop recursion to compute the search direction.
    ///
    /// Approximates H^(-1) * grad where H is the Hessian, using the stored
    /// history of s (position diff) and y (gradient diff).
    #[provable_contracts_macros::contract("lbfgs-kernel-v1", equation = "two_loop_recursion")]
    fn compute_direction(&self, grad: &Vector<T>) -> Vector<T> {
        let n = grad.len();
        let k = self.s_history.len();

        if k == 0 {
            // No history: use steepest descent
            let mut d = zeros::<T>(n);
            for i in 0..n {
                d[i] = -grad[i];
            }
            return d;
        }

        // Initialize q = -grad
        let mut q = zeros::<T>(n);
        for i in 0..n {
            q[i] = -grad[i];
        }

        let mut alpha = vec![T::ZERO; k];
        let mut rho = vec![T::ZERO; k];

        // First loop: backward pass
        for i in (0..k).rev() {
            let s = &self.s_history[i];
            let y = &self.y_history[i];

            // rho_i = 1 / (y_i^T s_i)
            let mut y_dot_s = T::ZERO;
            for j in 0..n {
                y_dot_s += y[j] * s[j];
            }
            rho[i] = T::ONE / y_dot_s;

            // alpha_i = rho_i * s_i^T * q
            let mut s_dot_q = T::ZERO;
            for j in 0..n {
                s_dot_q += s[j] * q[j];
            }
            alpha[i] = rho[i] * s_dot_q;

            // q = q - alpha_i * y_i
            for j in 0..n {
                q[j] -= alpha[i] * y[j];
            }
        }

        // Scale by H_0 = (s^T y) / (y^T y) from most recent update
        let s_last = &self.s_history[k - 1];
        let y_last = &self.y_history[k - 1];

        let mut s_dot_y = T::ZERO;
        let mut y_dot_y = T::ZERO;
        for i in 0..n {
            s_dot_y += s_last[i] * y_last[i];
            y_dot_y += y_last[i] * y_last[i];
        }
        let gamma = s_dot_y / y_dot_y;

        // r = H_0 * q = gamma * q
        let mut r = zeros::<T>(n);
        for i in 0..n {
            r[i] = gamma * q[i];
        }

        // Second loop: forward pass
        for i in 0..k {
            let s = &self.s_history[i];
            let y = &self.y_history[i];

            // beta = rho_i * y_i^T * r
            let mut y_dot_r = T::ZERO;
            for j in 0..n {
                y_dot_r += y[j] * r[j];
            }
            let beta = rho[i] * y_dot_r;

            // r = r + s_i * (alpha_i - beta)
            for j in 0..n {
                r[j] += s[j] * (alpha[i] - beta);
            }
        }

        r
    }

    /// Computes the L2 norm of a vector.
    fn norm(v: &Vector<T>) -> T {
        let mut sum = T::ZERO;
        for i in 0..v.len() {
            sum += v[i] * v[i];
        }
        sum.sqrt()
    }

    fn outcome(
        solution: Vector<T>,
        objective_value: T,
        iterations: usize,
        status: ConvergenceStatus,
        gradient_norm: T,
        elapsed_time: std::time::Duration,
    ) -> LbfgsOutcome<T> {
        LbfgsOutcome {
            solution,
            objective_value,
            iterations,
            status,
            gradient_norm,
            elapsed_time,
        }
    }

    /// Minimizes `objective` from `x0`.
    ///
    /// Non-finite input from ANY channel — `x0`, the objective's return, the
    /// gradient's return, or a line-search trial point — yields
    /// [`ConvergenceStatus::NumericalError`]. It never panics and never
    /// returns a finite-looking result derived from a poisoned step.
    #[provable_contracts_macros::contract("lbfgs-kernel-v1", equation = "nonfinite_input_status")]
    fn minimize<F, G>(&mut self, objective: F, gradient: G, x0: Vector<T>) -> LbfgsOutcome<T>
    where
        F: Fn(&Vector<T>) -> T,
        G: Fn(&Vector<T>) -> Vector<T>,
    {
        let start_time = std::time::Instant::now();
        let n = x0.len();

        // Clear history from previous runs
        self.s_history.clear();
        self.y_history.clear();

        let mut x = x0;
        let mut fx = objective(&x);
        let mut grad = gradient(&x);
        let mut grad_norm = Self::norm(&grad);

        // Channels (i)-(iii): non-finite x0, objective, or gradient AT x0.
        // Without this the poisoned comparison `grad_norm < tol` is simply
        // false and the solver would iterate on garbage.
        if !is_all_finite(&x) || !fx.is_finite() || !is_all_finite(&grad) || !grad_norm.is_finite()
        {
            return Self::outcome(
                x,
                fx,
                0,
                ConvergenceStatus::NumericalError,
                grad_norm,
                start_time.elapsed(),
            );
        }

        for iter in 0..self.max_iter {
            // Check convergence
            if grad_norm < self.tol {
                return Self::outcome(
                    x,
                    fx,
                    iter,
                    ConvergenceStatus::Converged,
                    grad_norm,
                    start_time.elapsed(),
                );
            }

            // Compute search direction
            let d = self.compute_direction(&grad);

            // Line search
            let alpha = self.line_search.search(&objective, &gradient, &x, &d);

            // Channel (iv): the line search saw a non-finite value. This check
            // MUST precede the stall check — `alpha < STALL_EPS` is false for
            // NaN, so a poisoned search would otherwise be reported as a
            // successful tiny step.
            if !alpha.is_finite() {
                return Self::outcome(
                    x,
                    fx,
                    iter,
                    ConvergenceStatus::NumericalError,
                    grad_norm,
                    start_time.elapsed(),
                );
            }

            // Check for stalled progress
            if alpha < T::STALL_EPS {
                return Self::outcome(
                    x,
                    fx,
                    iter,
                    ConvergenceStatus::Stalled,
                    grad_norm,
                    start_time.elapsed(),
                );
            }

            // Update position: x_new = x + alpha * d
            let mut x_new = zeros::<T>(n);
            for i in 0..n {
                x_new[i] = x[i] + alpha * d[i];
            }

            // Compute new objective and gradient
            let fx_new = objective(&x_new);
            let grad_new = gradient(&x_new);

            // Check for numerical errors (objective, gradient, and the point itself)
            if !fx_new.is_finite() || !is_all_finite(&grad_new) || !is_all_finite(&x_new) {
                return Self::outcome(
                    x,
                    fx,
                    iter,
                    ConvergenceStatus::NumericalError,
                    grad_norm,
                    start_time.elapsed(),
                );
            }

            // Compute s_k = x_new - x and y_k = grad_new - grad
            let mut s_k = zeros::<T>(n);
            let mut y_k = zeros::<T>(n);
            for i in 0..n {
                s_k[i] = x_new[i] - x[i];
                y_k[i] = grad_new[i] - grad[i];
            }

            // Check curvature condition: y^T s > 0
            let mut y_dot_s = T::ZERO;
            for i in 0..n {
                y_dot_s += y_k[i] * s_k[i];
            }

            if y_dot_s > T::CURVATURE_EPS {
                // Store in history
                if self.s_history.len() >= self.m {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                }
                self.s_history.push(s_k);
                self.y_history.push(y_k);
            }

            // Update for next iteration
            x = x_new;
            fx = fx_new;
            grad = grad_new;
            grad_norm = Self::norm(&grad);
        }

        // Max iterations reached
        Self::outcome(
            x,
            fx,
            self.max_iter,
            ConvergenceStatus::MaxIterations,
            grad_norm,
            start_time.elapsed(),
        )
    }
}

// =========================================================================
// Public f32 entry point (API-unchanged)
// =========================================================================

/// Limited-memory BFGS (L-BFGS) optimizer.
///
/// L-BFGS is a quasi-Newton method that approximates the inverse Hessian using
/// a limited history of gradient information. It's efficient for large-scale
/// optimization problems where storing the full Hessian is infeasible.
///
/// # Algorithm
///
/// 1. Compute gradient `g_k` = ∇`f(x_k)`
/// 2. Compute search direction `d_k` using two-loop recursion (approximates H^(-1) * `g_k`)
/// 3. Find step size `α_k` via line search (Wolfe conditions)
/// 4. Update: x_{k+1} = `x_k` - `α_k` * `d_k`
/// 5. Store gradient and position differences for next iteration
///
/// # Parameters
///
/// - **`max_iter`**: Maximum number of iterations
/// - **tol**: Convergence tolerance (gradient norm)
/// - **m**: History size (typically 5-20, tradeoff between memory and convergence)
///
/// # Precision
///
/// This type is f32 and NON-generic — deliberately, because adding a generic
/// or default type parameter would break downstream `use` sites. For f64 (for
/// example a softmax-NLL objective whose gradient norm reaches f32 epsilon
/// near the optimum) use [`LbfgsF64`].
///
/// # Example
///
/// ```
/// use aprender::optim::{LBFGS, Optimizer};
/// use aprender::primitives::Vector;
///
/// let mut optimizer = LBFGS::new(100, 1e-5, 10);
///
/// // Define Rosenbrock function and its gradient
/// let f = |x: &Vector<f32>| {
///     let a = x[0];
///     let b = x[1];
///     (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
/// };
///
/// let grad = |x: &Vector<f32>| {
///     let a = x[0];
///     let b = x[1];
///     Vector::from_slice(&[
///         -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
///         200.0 * (b - a * a),
///     ])
/// };
///
/// let x0 = Vector::from_slice(&[0.0, 0.0]);
/// let result = optimizer.minimize(f, grad, x0);
///
/// // Should converge to (1, 1)
/// assert_eq!(result.status, aprender::optim::ConvergenceStatus::Converged);
/// ```
#[derive(Debug, Clone)]
pub struct LBFGS {
    /// Maximum number of iterations
    pub(crate) max_iter: usize,
    /// Convergence tolerance (gradient norm)
    pub(crate) tol: f32,
    /// History size (number of correction pairs to store)
    pub(crate) m: usize,
    /// Line search strategy
    line_search: WolfeLineSearch,
    /// Position differences: `s_k` = x_{k+1} - `x_k`
    pub(crate) s_history: Vec<Vector<f32>>,
    /// Gradient differences: `y_k` = g_{k+1} - `g_k`
    pub(crate) y_history: Vec<Vector<f32>>,
}

impl LBFGS {
    /// Creates a new L-BFGS optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of iterations (typical: 100-1000)
    /// * `tol` - Convergence tolerance for gradient norm (typical: 1e-5)
    /// * `m` - History size (typical: 5-20)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::LBFGS;
    ///
    /// let optimizer = LBFGS::new(100, 1e-5, 10);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, tol: f32, m: usize) -> Self {
        Self {
            max_iter,
            tol,
            m,
            line_search: WolfeLineSearch::new(1e-4, 0.9, 50),
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
        }
    }

    /// Builds the f32 core from this wrapper's configuration.
    ///
    /// The Wolfe constants are READ OFF the `line_search` field rather than
    /// re-typed here, so the wrapper and the core cannot drift apart.
    fn core(&self) -> LbfgsImpl<f32> {
        LbfgsImpl::new(
            self.max_iter,
            self.tol,
            self.m,
            WolfeSearch::new(
                self.line_search.c1,
                self.line_search.c2,
                self.line_search.max_iter,
            ),
        )
    }
}

/// Test-visibility shims.
///
/// The pre-widening `LBFGS` carried `compute_direction` and `norm` as private
/// inherent items and this module's tests exercise them directly. The
/// implementations now live on the generic core, so these delegate; they are
/// `cfg(test)` because production code reaches them through [`LbfgsImpl`].
#[cfg(test)]
impl LBFGS {
    fn compute_direction(&self, grad: &Vector<f32>) -> Vector<f32> {
        let mut core = self.core();
        core.s_history.clone_from(&self.s_history);
        core.y_history.clone_from(&self.y_history);
        core.compute_direction(grad)
    }

    fn norm(v: &Vector<f32>) -> f32 {
        LbfgsImpl::<f32>::norm(v)
    }
}

impl Optimizer for LBFGS {
    fn step(&mut self, _params: &mut Vector<f32>, _gradients: &Vector<f32>) {
        panic!(
            "L-BFGS does not support stochastic updates (step). Use minimize() for batch optimization."
        )
    }

    fn minimize<F, G>(&mut self, objective: F, gradient: G, x0: Vector<f32>) -> OptimizationResult
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let mut core = self.core();
        let outcome = core.minimize(objective, gradient, x0);
        self.s_history = core.s_history;
        self.y_history = core.y_history;

        OptimizationResult {
            solution: outcome.solution,
            objective_value: outcome.objective_value,
            iterations: outcome.iterations,
            status: outcome.status,
            gradient_norm: outcome.gradient_norm,
            constraint_violation: 0.0,
            elapsed_time: outcome.elapsed_time,
        }
    }

    fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
    }
}

// =========================================================================
// Public f64 entry point
// =========================================================================

/// Double-precision L-BFGS.
///
/// Same algorithm, same Wolfe constants and same iteration budget as [`LBFGS`],
/// carried out in f64. Use it when the objective's gradient norm approaches
/// f32 epsilon (~1.2e-7) near the optimum — a regularized softmax-NLL head is
/// the motivating case. At that scale an f32 Wolfe line search can stall and
/// report progress it did not make, so a tolerance like 1e-10 is unreachable
/// in f32 no matter how many iterations are spent.
///
/// This is a SEPARATE non-generic type rather than a generic parameter on
/// [`LBFGS`], so no existing f32 call site changes.
///
/// # Example
///
/// ```
/// use aprender::optim::{ConvergenceStatus, LbfgsF64};
/// use aprender::primitives::Vector;
///
/// let mut optimizer = LbfgsF64::new(200, 1e-10, 10);
///
/// let f = |x: &Vector<f64>| (x[0] - 5.0) * (x[0] - 5.0);
/// let grad = |x: &Vector<f64>| Vector::from_vec(vec![2.0 * (x[0] - 5.0)]);
///
/// let result = optimizer.minimize(f, grad, &Vector::from_vec(vec![0.0]));
///
/// assert_eq!(result.status, ConvergenceStatus::Converged);
/// assert!(result.gradient_norm < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct LbfgsF64 {
    /// Maximum number of iterations
    pub(crate) max_iter: usize,
    /// Convergence tolerance (gradient norm)
    pub(crate) tol: f64,
    /// History size (number of correction pairs to store)
    pub(crate) m: usize,
    /// Position differences: `s_k` = x_{k+1} - `x_k`
    pub(crate) s_history: Vec<Vector<f64>>,
    /// Gradient differences: `y_k` = g_{k+1} - `g_k`
    pub(crate) y_history: Vec<Vector<f64>>,
}

impl LbfgsF64 {
    /// Creates a new double-precision L-BFGS optimizer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum number of iterations (typical: 100-1000)
    /// * `tol` - Convergence tolerance for gradient norm (typical: 1e-10)
    /// * `m` - History size (typical: 5-20)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::LbfgsF64;
    ///
    /// let optimizer = LbfgsF64::new(200, 1e-10, 10);
    /// ```
    #[must_use]
    pub fn new(max_iter: usize, tol: f64, m: usize) -> Self {
        Self {
            max_iter,
            tol,
            m,
            s_history: Vec::with_capacity(m),
            y_history: Vec::with_capacity(m),
        }
    }

    /// Minimizes `objective` from `x0` in double precision.
    ///
    /// Non-finite input from ANY channel — `x0`, the objective's return, the
    /// gradient's return, or a line-search trial point — yields
    /// [`ConvergenceStatus::NumericalError`] rather than a panic.
    pub fn minimize<F, G>(
        &mut self,
        objective: F,
        gradient: G,
        x0: &Vector<f64>,
    ) -> OptimizationResultF64
    where
        F: Fn(&Vector<f64>) -> f64,
        G: Fn(&Vector<f64>) -> Vector<f64>,
    {
        // Wolfe constants and budget are the f64 twins of `LBFGS::new`'s.
        let mut core = LbfgsImpl::new(
            self.max_iter,
            self.tol,
            self.m,
            WolfeSearch::new(f64::WOLFE_C1, f64::WOLFE_C2, 50),
        );
        let outcome = core.minimize(objective, gradient, x0.clone());
        self.s_history = core.s_history;
        self.y_history = core.y_history;

        // `outcome.elapsed_time` is deliberately DROPPED: wall-clock time is a
        // semantic-hash poison (it makes any record containing it
        // irreproducible), and the f64 path exists to serve a deterministic,
        // hashable head fit.
        OptimizationResultF64 {
            solution: outcome.solution,
            objective_value: outcome.objective_value,
            iterations: outcome.iterations,
            status: outcome.status,
            gradient_norm: outcome.gradient_norm,
        }
    }

    /// Resets the optimizer state (correction-pair history).
    pub fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
    }
}

#[cfg(test)]
#[path = "lbfgs_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_lbfgs_contract.rs"]
mod tests_lbfgs_contract;
