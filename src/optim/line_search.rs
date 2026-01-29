//! Line search algorithms for optimization.
//!
//! Line search methods find a suitable step size along a given search direction,
//! ensuring sufficient decrease in the objective function.
//!
//! # Available Line Searches
//!
//! - [`BacktrackingLineSearch`] - Simple Armijo condition (sufficient decrease)
//! - [`WolfeLineSearch`] - Armijo + curvature conditions (for quasi-Newton methods)

use crate::primitives::Vector;

/// Trait for line search strategies.
///
/// Given a function f, current point x, and search direction d, finds a step
/// size α > 0 such that x + α*d satisfies certain decrease conditions.
///
/// # Example
///
/// ```
/// use aprender::optim::{LineSearch, BacktrackingLineSearch};
/// use aprender::primitives::Vector;
///
/// let ls = BacktrackingLineSearch::default();
/// let f = |x: &Vector<f32>| x[0] * x[0];
/// let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
///
/// let x = Vector::from_slice(&[1.0]);
/// let d = Vector::from_slice(&[-2.0]); // Descent direction
/// let alpha = ls.search(&f, &grad, &x, &d);
/// assert!(alpha > 0.0);
/// ```
///
/// Different strategies enforce different conditions on the step size.
pub trait LineSearch {
    /// Finds a suitable step size along the search direction.
    ///
    /// # Arguments
    ///
    /// * `f` - Objective function f: ℝⁿ → ℝ
    /// * `grad` - Gradient function ∇f: ℝⁿ → ℝⁿ
    /// * `x` - Current point
    /// * `d` - Search direction (typically descent direction, ∇f(x)·d < 0)
    ///
    /// # Returns
    ///
    /// Step size α > 0 satisfying the line search conditions
    fn search<F, G>(&self, f: &F, grad: &G, x: &Vector<f32>, d: &Vector<f32>) -> f32
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>;
}

/// Backtracking line search with Armijo condition.
///
/// Starts with step size α = 1 and repeatedly shrinks it by factor ρ until
/// the Armijo condition is satisfied:
///
/// ```text
/// f(x + α*d) ≤ f(x) + c₁*α*∇f(x)ᵀd
/// ```
///
/// This ensures sufficient decrease in the objective function.
///
/// # Parameters
///
/// - **c1**: Armijo constant (typical: 1e-4), controls acceptable decrease
/// - **rho**: Backtracking factor (typical: 0.5), shrinkage rate for α
/// - **`max_iter`**: Maximum backtracking iterations (safety limit)
///
/// # Example
///
/// ```
/// use aprender::optim::{BacktrackingLineSearch, LineSearch};
/// use aprender::primitives::Vector;
///
/// let line_search = BacktrackingLineSearch::new(1e-4, 0.5, 50);
///
/// // Define a simple quadratic function
/// let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
/// let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);
///
/// let x = Vector::from_slice(&[1.0, 1.0]);
/// let d = Vector::from_slice(&[-2.0, -2.0]); // Descent direction
///
/// let alpha = line_search.search(&f, &grad, &x, &d);
/// assert!(alpha > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct BacktrackingLineSearch {
    /// Armijo constant (c₁ ∈ (0, 1), typical: 1e-4)
    pub(crate) c1: f32,
    /// Backtracking factor (ρ ∈ (0, 1), typical: 0.5)
    pub(crate) rho: f32,
    /// Maximum backtracking iterations
    pub(crate) max_iter: usize,
}

impl BacktrackingLineSearch {
    /// Creates a new backtracking line search.
    ///
    /// # Arguments
    ///
    /// * `c1` - Armijo constant (typical: 1e-4)
    /// * `rho` - Backtracking factor (typical: 0.5)
    /// * `max_iter` - Maximum iterations (typical: 50)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::BacktrackingLineSearch;
    ///
    /// let line_search = BacktrackingLineSearch::new(1e-4, 0.5, 50);
    /// ```
    #[must_use]
    pub fn new(c1: f32, rho: f32, max_iter: usize) -> Self {
        Self { c1, rho, max_iter }
    }
}

impl Default for BacktrackingLineSearch {
    /// Creates a backtracking line search with default parameters.
    ///
    /// Defaults: c1=1e-4, rho=0.5, `max_iter=50`
    fn default() -> Self {
        Self::new(1e-4, 0.5, 50)
    }
}

impl LineSearch for BacktrackingLineSearch {
    fn search<F, G>(&self, f: &F, grad: &G, x: &Vector<f32>, d: &Vector<f32>) -> f32
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let mut alpha = 1.0;
        let fx = f(x);
        let grad_x = grad(x);

        // Compute directional derivative: ∇f(x)ᵀd
        let mut dir_deriv = 0.0;
        for i in 0..x.len() {
            dir_deriv += grad_x[i] * d[i];
        }

        // Backtracking loop
        for _ in 0..self.max_iter {
            // Compute x_new = x + alpha * d
            let mut x_new = Vector::zeros(x.len());
            for i in 0..x.len() {
                x_new[i] = x[i] + alpha * d[i];
            }

            let fx_new = f(&x_new);

            // Check Armijo condition: f(x + α*d) ≤ f(x) + c₁*α*∇f(x)ᵀd
            if fx_new <= fx + self.c1 * alpha * dir_deriv {
                return alpha;
            }

            // Shrink step size
            alpha *= self.rho;
        }

        // Return the last alpha if max iterations reached
        alpha
    }
}

/// Wolfe line search with Armijo and curvature conditions.
///
/// Enforces both the Armijo condition (sufficient decrease) and the curvature
/// condition (sufficient curvature):
///
/// ```text
/// Armijo:    f(x + α*d) ≤ f(x) + c₁*α*∇f(x)ᵀd
/// Curvature: |∇f(x + α*d)ᵀd| ≤ c₂*|∇f(x)ᵀd|
/// ```
///
/// The curvature condition ensures the step size is not too small by requiring
/// that the gradient has decreased sufficiently along the search direction.
///
/// # Parameters
///
/// - **c1**: Armijo constant (typical: 1e-4), c₁ ∈ (0, c₂)
/// - **c2**: Curvature constant (typical: 0.9), c₂ ∈ (c₁, 1)
/// - **`max_iter`**: Maximum line search iterations
///
/// # Example
///
/// ```
/// use aprender::optim::{WolfeLineSearch, LineSearch};
/// use aprender::primitives::Vector;
///
/// let line_search = WolfeLineSearch::new(1e-4, 0.9, 50);
///
/// let f = |x: &Vector<f32>| x[0] * x[0];
/// let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);
///
/// let x = Vector::from_slice(&[1.0]);
/// let d = Vector::from_slice(&[-2.0]);
///
/// let alpha = line_search.search(&f, &grad, &x, &d);
/// assert!(alpha > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct WolfeLineSearch {
    /// Armijo constant (c₁ ∈ (0, c₂), typical: 1e-4)
    pub(crate) c1: f32,
    /// Curvature constant (c₂ ∈ (c₁, 1), typical: 0.9)
    pub(crate) c2: f32,
    /// Maximum line search iterations
    pub(crate) max_iter: usize,
}

impl WolfeLineSearch {
    /// Creates a new Wolfe line search.
    ///
    /// # Arguments
    ///
    /// * `c1` - Armijo constant (typical: 1e-4)
    /// * `c2` - Curvature constant (typical: 0.9)
    /// * `max_iter` - Maximum iterations (typical: 50)
    ///
    /// # Panics
    ///
    /// Panics if c1 >= c2 or values are outside (0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::WolfeLineSearch;
    ///
    /// let line_search = WolfeLineSearch::new(1e-4, 0.9, 50);
    /// ```
    #[must_use]
    pub fn new(c1: f32, c2: f32, max_iter: usize) -> Self {
        assert!(
            c1 < c2 && c1 > 0.0 && c2 < 1.0,
            "Wolfe conditions require 0 < c1 < c2 < 1"
        );
        Self { c1, c2, max_iter }
    }
}

impl Default for WolfeLineSearch {
    /// Creates a Wolfe line search with default parameters.
    ///
    /// Defaults: c1=1e-4, c2=0.9, `max_iter=50`
    fn default() -> Self {
        Self::new(1e-4, 0.9, 50)
    }
}

impl LineSearch for WolfeLineSearch {
    fn search<F, G>(&self, f: &F, grad: &G, x: &Vector<f32>, d: &Vector<f32>) -> f32
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let fx = f(x);
        let grad_x = grad(x);

        // Compute directional derivative: ∇f(x)ᵀd
        let mut dir_deriv = 0.0;
        for i in 0..x.len() {
            dir_deriv += grad_x[i] * d[i];
        }

        // Start with alpha = 1.0
        let mut alpha = 1.0;
        let mut alpha_lo = 0.0;
        let mut alpha_hi = f32::INFINITY;

        for _ in 0..self.max_iter {
            // Compute x_new = x + alpha * d
            let mut x_new = Vector::zeros(x.len());
            for i in 0..x.len() {
                x_new[i] = x[i] + alpha * d[i];
            }

            let fx_new = f(&x_new);
            let grad_new = grad(&x_new);

            // Compute new directional derivative
            let mut dir_deriv_new = 0.0;
            for i in 0..x.len() {
                dir_deriv_new += grad_new[i] * d[i];
            }

            // Check Armijo condition
            if fx_new > fx + self.c1 * alpha * dir_deriv {
                // Armijo fails - alpha too large
                alpha_hi = alpha;
                alpha = (alpha_lo + alpha_hi) / 2.0;
                continue;
            }

            // Check curvature condition: |∇f(x + α*d)ᵀd| ≤ c₂*|∇f(x)ᵀd|
            if dir_deriv_new.abs() <= self.c2 * dir_deriv.abs() {
                // Both conditions satisfied
                return alpha;
            }

            // Curvature condition fails
            if dir_deriv_new > 0.0 {
                // Gradient sign changed - reduce alpha
                alpha_hi = alpha;
            } else {
                // Gradient still negative - increase alpha
                alpha_lo = alpha;
            }

            // Update alpha
            if alpha_hi.is_finite() {
                alpha = (alpha_lo + alpha_hi) / 2.0;
            } else {
                alpha *= 2.0;
            }
        }

        // Return the last alpha if max iterations reached
        alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtracking_quadratic() {
        let ls = BacktrackingLineSearch::default();
        let f = |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 2.0 * x[1]]);

        let x = Vector::from_slice(&[1.0, 1.0]);
        let d = Vector::from_slice(&[-2.0, -2.0]); // Descent direction

        let alpha = ls.search(&f, &grad, &x, &d);
        assert!(alpha > 0.0);
        assert!(alpha <= 1.0);
    }

    #[test]
    fn test_wolfe_quadratic() {
        let ls = WolfeLineSearch::default();
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let x = Vector::from_slice(&[1.0]);
        let d = Vector::from_slice(&[-2.0]);

        let alpha = ls.search(&f, &grad, &x, &d);
        assert!(alpha > 0.0);
    }

    #[test]
    fn test_backtracking_ensures_decrease() {
        let ls = BacktrackingLineSearch::new(1e-4, 0.5, 100);
        let f = |x: &Vector<f32>| x[0] * x[0];
        let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

        let x = Vector::from_slice(&[5.0]);
        let g = grad(&x);
        let d = Vector::from_slice(&[-g[0]]); // Steepest descent

        let alpha = ls.search(&f, &grad, &x, &d);

        // f(x + alpha*d) should be less than f(x)
        let mut x_new = Vector::zeros(1);
        x_new[0] = x[0] + alpha * d[0];
        assert!(f(&x_new) < f(&x));
    }
}
