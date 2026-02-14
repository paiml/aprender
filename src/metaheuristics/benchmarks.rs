//! CEC 2013 Benchmark Functions for Metaheuristic Evaluation
//!
//! Standard test functions from the CEC 2013 competition for
//! real-parameter single objective optimization.
//!
//! Reference: Liang et al. (2013) "Problem Definitions and Evaluation
//! Criteria for the CEC 2013 Special Session on Real-Parameter Optimization"

use std::f64::consts::PI;

/// Sphere function (f1) - Unimodal, separable
///
/// Global minimum: f(0, 0, ..., 0) = 0
/// Search domain: [-100, 100]^D
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::sphere;
/// let x = vec![0.0, 0.0, 0.0];
/// assert!((sphere(&x) - 0.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

/// Rosenbrock function (f2) - Unimodal, non-separable
///
/// Global minimum: f(1, 1, ..., 1) = 0
/// Search domain: [-30, 30]^D
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::rosenbrock;
/// let x = vec![1.0, 1.0, 1.0];
/// assert!((rosenbrock(&x) - 0.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| {
            let a = w[1] - w[0] * w[0];
            let b = 1.0 - w[0];
            100.0 * a * a + b * b
        })
        .sum()
}

/// Rastrigin function (f3) - Multimodal, separable
///
/// Global minimum: f(0, 0, ..., 0) = 0
/// Search domain: [-5.12, 5.12]^D
/// Many local minima arranged in a regular lattice.
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::rastrigin;
/// let x = vec![0.0, 0.0, 0.0];
/// assert!((rastrigin(&x) - 0.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    10.0 * n
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
            .sum::<f64>()
}

/// Ackley function (f4) - Multimodal, non-separable
///
/// Global minimum: f(0, 0, ..., 0) = 0
/// Search domain: [-32, 32]^D
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::ackley;
/// let x = vec![0.0, 0.0, 0.0];
/// assert!(ackley(&x).abs() < 1e-10);
/// ```
#[must_use]
pub fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|xi| (2.0 * PI * xi).cos()).sum();

    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
}

/// Schwefel function (f5) - Multimodal, separable
///
/// Global minimum: f(420.9687, ..., 420.9687) ≈ 0
/// Search domain: [-500, 500]^D
/// Deceptive: global minimum far from next best local minima.
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::schwefel;
/// let x = vec![420.9687, 420.9687, 420.9687];
/// assert!(schwefel(&x).abs() < 1.0);
/// ```
#[must_use]
pub fn schwefel(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    418.9829 * n - x.iter().map(|xi| xi * (xi.abs().sqrt()).sin()).sum::<f64>()
}

/// Griewank function (f6) - Multimodal, non-separable
///
/// Global minimum: f(0, 0, ..., 0) = 0
/// Search domain: [-600, 600]^D
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::griewank;
/// let x = vec![0.0, 0.0, 0.0];
/// assert!(griewank(&x).abs() < 1e-10);
/// ```
#[must_use]
pub fn griewank(x: &[f64]) -> f64 {
    let sum: f64 = x.iter().map(|xi| xi * xi).sum::<f64>() / 4000.0;
    let prod: f64 = x
        .iter()
        .enumerate()
        .map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos())
        .product();
    sum - prod + 1.0
}

/// Levy function (f7) - Multimodal, non-separable
///
/// Global minimum: f(1, 1, ..., 1) = 0
/// Search domain: [-10, 10]^D
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::levy;
/// let x = vec![1.0, 1.0, 1.0];
/// assert!(levy(&x).abs() < 1e-10);
/// ```
#[must_use]
pub fn levy(x: &[f64]) -> f64 {
    let w: Vec<f64> = x.iter().map(|xi| 1.0 + (xi - 1.0) / 4.0).collect();
    let n = w.len();

    let term1 = (PI * w[0]).sin().powi(2);

    let term2: f64 = w[..n - 1]
        .iter()
        .map(|wi| (wi - 1.0).powi(2) * (1.0 + 10.0 * (PI * wi + 1.0).sin().powi(2)))
        .sum();

    let term3 = (w[n - 1] - 1.0).powi(2) * (1.0 + (2.0 * PI * w[n - 1]).sin().powi(2));

    term1 + term2 + term3
}

/// Michalewicz function (f8) - Multimodal, separable
///
/// Global minimum depends on dimension D.
/// Search domain: [0, π]^D
/// Steepness parameter m=10 (default).
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::michalewicz;
/// // Optimal for D=2 is approximately -1.8013
/// ```
#[must_use]
pub fn michalewicz(x: &[f64]) -> f64 {
    let m = 10.0;
    -x.iter()
        .enumerate()
        .map(|(i, xi)| xi.sin() * ((i + 1) as f64 * xi * xi / PI).sin().powi(2 * m as i32))
        .sum::<f64>()
}

/// Zakharov function (f9) - Unimodal, non-separable
///
/// Global minimum: f(0, 0, ..., 0) = 0
/// Search domain: [-5, 10]^D
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::zakharov;
/// let x = vec![0.0, 0.0, 0.0];
/// assert!(zakharov(&x).abs() < 1e-10);
/// ```
#[must_use]
pub fn zakharov(x: &[f64]) -> f64 {
    let sum1: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum2: f64 = x
        .iter()
        .enumerate()
        .map(|(i, xi)| 0.5 * (i + 1) as f64 * xi)
        .sum();
    sum1 + sum2.powi(2) + sum2.powi(4)
}

/// Dixon-Price function (f10) - Unimodal, non-separable
///
/// Global minimum: f(x*) = 0 where `x_i` = 2^(-(2^i - 2) / 2^i)
/// Search domain: [-10, 10]^D
///
/// # Example
/// ```
/// use aprender::metaheuristics::benchmarks::dixon_price;
/// // Optimal solution is dimension-dependent
/// ```
#[must_use]
pub fn dixon_price(x: &[f64]) -> f64 {
    let term1 = (x[0] - 1.0).powi(2);
    let term2: f64 = x
        .windows(2)
        .enumerate()
        .map(|(i, w)| (i + 2) as f64 * (2.0 * w[1] * w[1] - w[0]).powi(2))
        .sum();
    term1 + term2
}

/// Benchmark function metadata
#[derive(Debug, Clone)]
pub struct BenchmarkInfo {
    /// Function name
    pub name: &'static str,
    /// Function ID (CEC numbering)
    pub id: u8,
    /// Is multimodal (multiple local minima)
    pub multimodal: bool,
    /// Is separable (can optimize each dimension independently)
    pub separable: bool,
    /// Recommended search bounds [lower, upper]
    pub bounds: (f64, f64),
    /// Global optimum value
    pub optimum: f64,
}

/// Get metadata for all benchmark functions
#[must_use]
pub fn all_benchmarks() -> Vec<BenchmarkInfo> {
    vec![
        BenchmarkInfo {
            name: "Sphere",
            id: 1,
            multimodal: false,
            separable: true,
            bounds: (-100.0, 100.0),
            optimum: 0.0,
        },
        BenchmarkInfo {
            name: "Rosenbrock",
            id: 2,
            multimodal: false,
            separable: false,
            bounds: (-30.0, 30.0),
            optimum: 0.0,
        },
        BenchmarkInfo {
            name: "Rastrigin",
            id: 3,
            multimodal: true,
            separable: true,
            bounds: (-5.12, 5.12),
            optimum: 0.0,
        },
        BenchmarkInfo {
            name: "Ackley",
            id: 4,
            multimodal: true,
            separable: false,
            bounds: (-32.0, 32.0),
            optimum: 0.0,
        },
        BenchmarkInfo {
            name: "Schwefel",
            id: 5,
            multimodal: true,
            separable: true,
            bounds: (-500.0, 500.0),
            optimum: 0.0,
        },
        BenchmarkInfo {
            name: "Griewank",
            id: 6,
            multimodal: true,
            separable: false,
            bounds: (-600.0, 600.0),
            optimum: 0.0,
        },
        BenchmarkInfo {
            name: "Levy",
            id: 7,
            multimodal: true,
            separable: false,
            bounds: (-10.0, 10.0),
            optimum: 0.0,
        },
        BenchmarkInfo {
            name: "Michalewicz",
            id: 8,
            multimodal: true,
            separable: true,
            bounds: (0.0, PI),
            optimum: f64::NEG_INFINITY, // Dimension-dependent
        },
        BenchmarkInfo {
            name: "Zakharov",
            id: 9,
            multimodal: false,
            separable: false,
            bounds: (-5.0, 10.0),
            optimum: 0.0,
        },
        BenchmarkInfo {
            name: "Dixon-Price",
            id: 10,
            multimodal: false,
            separable: false,
            bounds: (-10.0, 10.0),
            optimum: 0.0,
        },
    ]
}

#[cfg(test)]
#[path = "benchmarks_tests.rs"]
mod tests;
