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
mod tests {
    use super::*;

    // ---- Sphere ----

    #[test]
    fn test_sphere_optimum() {
        let x = vec![0.0; 10];
        assert!((sphere(&x)).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_known_value() {
        // sphere([1,2,3]) = 1 + 4 + 9 = 14
        assert!((sphere(&[1.0, 2.0, 3.0]) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_single_dim() {
        assert!((sphere(&[3.0]) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_empty() {
        assert!((sphere(&[])).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_symmetry() {
        // f(x) == f(-x) for all x
        let x = vec![1.5, -2.3, 4.7];
        let neg_x: Vec<f64> = x.iter().map(|xi| -xi).collect();
        assert!((sphere(&x) - sphere(&neg_x)).abs() < 1e-10);
    }

    // ---- Rosenbrock ----

    #[test]
    fn test_rosenbrock_optimum() {
        let x = vec![1.0; 5];
        assert!((rosenbrock(&x)).abs() < 1e-10);
    }

    #[test]
    fn test_rosenbrock_known_value() {
        // rosenbrock([0,0]) = 100*(0-0)^2 + (1-0)^2 = 1
        assert!((rosenbrock(&[0.0, 0.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rosenbrock_another_known_value() {
        // rosenbrock([-1,1]) = 100*(1-1)^2 + (1-(-1))^2 = 0 + 4 = 4
        assert!((rosenbrock(&[-1.0, 1.0]) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rosenbrock_single_dim() {
        // windows(2) on single element = empty iterator, sum = 0
        assert!((rosenbrock(&[5.0])).abs() < 1e-10);
    }

    #[test]
    fn test_rosenbrock_empty() {
        assert!((rosenbrock(&[])).abs() < 1e-10);
    }

    // ---- Rastrigin ----

    #[test]
    fn test_rastrigin_optimum() {
        let x = vec![0.0; 10];
        assert!((rastrigin(&x)).abs() < 1e-10);
    }

    #[test]
    fn test_rastrigin_single_dim() {
        // rastrigin([1.0]) = 10*1 + (1 - 10*cos(2*pi*1))
        //                   = 10 + 1 - 10*1 = 1
        assert!((rastrigin(&[1.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rastrigin_at_half() {
        // rastrigin([0.5]) = 10 + 0.25 - 10*cos(pi) = 10 + 0.25 + 10 = 20.25
        assert!((rastrigin(&[0.5]) - 20.25).abs() < 1e-10);
    }

    #[test]
    fn test_rastrigin_empty() {
        assert!((rastrigin(&[])).abs() < 1e-10);
    }

    #[test]
    fn test_rastrigin_always_nonnegative() {
        // Rastrigin >= 0 for all inputs
        let inputs: Vec<Vec<f64>> = vec![vec![1.0, -1.0], vec![5.12, -5.12], vec![0.5, 0.5, 0.5]];
        for x in &inputs {
            assert!(
                rastrigin(x) >= -1e-10,
                "rastrigin({x:?}) = {} should be >= 0",
                rastrigin(x)
            );
        }
    }

    // ---- Ackley ----

    #[test]
    fn test_ackley_optimum() {
        let x = vec![0.0; 5];
        assert!(ackley(&x).abs() < 1e-10);
    }

    #[test]
    fn test_ackley_known_value() {
        // ackley([1,1]) = -20*exp(-0.2*sqrt(1)) - exp(cos(2*pi)) + 20 + e
        // cos(2*pi*1) = 1, so sum_cos/n = 1
        // = -20*exp(-0.2) - exp(1) + 20 + e
        // = -20*exp(-0.2) - e + 20 + e = 20 - 20*exp(-0.2)
        let expected = 20.0 - 20.0 * (-0.2_f64).exp();
        assert!((ackley(&[1.0, 1.0]) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ackley_single_dim() {
        // ackley([0]) = -20*exp(0) - exp(1) + 20 + e = -20 - e + 20 + e = 0
        assert!(ackley(&[0.0]).abs() < 1e-10);
    }

    #[test]
    fn test_ackley_always_nonnegative() {
        // Ackley >= 0 for all inputs
        let inputs: Vec<Vec<f64>> = vec![vec![1.0, -1.0, 2.5], vec![32.0, -32.0], vec![0.001]];
        for x in &inputs {
            assert!(
                ackley(x) >= -1e-10,
                "ackley({x:?}) = {} should be >= 0",
                ackley(x)
            );
        }
    }

    // ---- Schwefel ----

    #[test]
    fn test_schwefel_near_optimum() {
        let x = vec![420.9687; 3];
        assert!(schwefel(&x).abs() < 1.0);
    }

    #[test]
    fn test_schwefel_single_dim_at_optimum() {
        // Near the global optimum for 1D
        let val = schwefel(&[420.9687]);
        assert!(val.abs() < 0.5);
    }

    #[test]
    fn test_schwefel_at_zero() {
        // schwefel([0]) = 418.9829 - 0*sin(0) = 418.9829
        assert!((schwefel(&[0.0]) - 418.9829).abs() < 1e-4);
    }

    #[test]
    fn test_schwefel_scales_with_dimension() {
        // At zero, schwefel = 418.9829 * n
        let val_1d = schwefel(&[0.0]);
        let val_3d = schwefel(&[0.0, 0.0, 0.0]);
        assert!((val_3d - 3.0 * val_1d).abs() < 1e-10);
    }

    // ---- Griewank ----

    #[test]
    fn test_griewank_optimum() {
        let x = vec![0.0; 10];
        assert!(griewank(&x).abs() < 1e-10);
    }

    #[test]
    fn test_griewank_single_dim_at_origin() {
        // griewank([0]) = 0/4000 - cos(0/1) + 1 = 0 - 1 + 1 = 0
        assert!(griewank(&[0.0]).abs() < 1e-10);
    }

    #[test]
    fn test_griewank_known_value() {
        // griewank([PI]) = PI^2/4000 - cos(PI/1) + 1
        //                = PI^2/4000 - (-1) + 1 = PI^2/4000 + 2
        let expected = PI * PI / 4000.0 + 2.0;
        assert!((griewank(&[PI]) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_griewank_divisor_index() {
        // Verify the (i+1) indexing in product term:
        // griewank([1, 1]) = (1+1)/4000 - cos(1/sqrt(1))*cos(1/sqrt(2)) + 1
        let sum_part = 2.0 / 4000.0;
        let prod_part = (1.0_f64).cos() * (1.0 / 2.0_f64.sqrt()).cos();
        let expected = sum_part - prod_part + 1.0;
        assert!((griewank(&[1.0, 1.0]) - expected).abs() < 1e-10);
    }

    // ---- Levy ----

    #[test]
    fn test_levy_optimum() {
        let x = vec![1.0; 5];
        assert!(levy(&x).abs() < 1e-10);
    }

    #[test]
    fn test_levy_known_value_2d() {
        // levy([0, 0]): w = [1 + (0-1)/4, 1 + (0-1)/4] = [0.75, 0.75]
        // term1 = sin(pi*0.75)^2 = sin(3*pi/4)^2 = (sqrt(2)/2)^2 = 0.5
        // term2 (i=0 only, w[..n-1] = [0.75]):
        //   (0.75-1)^2 * (1 + 10*sin(pi*0.75+1)^2)
        //   = 0.0625 * (1 + 10*sin(pi*0.75+1)^2)
        // term3: (0.75-1)^2 * (1 + sin(2*pi*0.75)^2)
        //   = 0.0625 * (1 + sin(1.5*pi)^2) = 0.0625 * (1 + 1) = 0.125
        let w0 = 0.75;
        let term1 = (PI * w0).sin().powi(2);
        let term2 = (w0 - 1.0).powi(2) * (1.0 + 10.0 * (PI * w0 + 1.0).sin().powi(2));
        let term3 = (w0 - 1.0).powi(2) * (1.0 + (2.0 * PI * w0).sin().powi(2));
        let expected = term1 + term2 + term3;
        assert!((levy(&[0.0, 0.0]) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_levy_single_dim() {
        // levy([1.0]): w=[1.0], term1=sin(pi)^2=0, term2=sum over empty=0,
        // term3=(1-1)^2*(...)=0. Total = 0.
        assert!(levy(&[1.0]).abs() < 1e-10);
    }

    #[test]
    fn test_levy_single_dim_nonzero() {
        // levy([0.0]): w=[0.75]
        // term1 = sin(0.75*pi)^2 = 0.5
        // term2 = empty sum = 0
        // term3 = (-0.25)^2 * (1 + sin(1.5*pi)^2) = 0.0625 * (1+1) = 0.125
        let expected = 0.5 + 0.125;
        assert!((levy(&[0.0]) - expected).abs() < 1e-10);
    }

    // ---- Michalewicz ----

    #[test]
    fn test_michalewicz_2d_near_known_optimum() {
        // Known 2D optimum ≈ -1.8013 at approximately (2.20, 1.57)
        let val = michalewicz(&[2.20, 1.57]);
        assert!(val < -1.7, "Expected < -1.7, got {val}");
        assert!(val > -2.0, "Expected > -2.0, got {val}");
    }

    #[test]
    fn test_michalewicz_at_zero() {
        // michalewicz([0]) = -sin(0)*(...) = 0
        assert!(michalewicz(&[0.0]).abs() < 1e-10);
    }

    #[test]
    fn test_michalewicz_at_pi_half() {
        // michalewicz([pi/2]) = -sin(pi/2) * sin(1*(pi/2)^2 / pi)^20
        //                     = -1 * sin(pi/4)^20 = -(sqrt(2)/2)^20 = -(2^{-10})
        let expected = -(2.0_f64.powi(-10));
        assert!((michalewicz(&[PI / 2.0]) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_michalewicz_empty() {
        assert!(michalewicz(&[]).abs() < 1e-10);
    }

    // ---- Zakharov ----

    #[test]
    fn test_zakharov_optimum() {
        let x = vec![0.0; 5];
        assert!(zakharov(&x).abs() < 1e-10);
    }

    #[test]
    fn test_zakharov_known_value() {
        // zakharov([1, 2]) = (1+4) + (0.5*1 + 1.0*2)^2 + (0.5*1 + 1.0*2)^4
        //                  = 5 + (2.5)^2 + (2.5)^4 = 5 + 6.25 + 39.0625 = 50.3125
        assert!((zakharov(&[1.0, 2.0]) - 50.3125).abs() < 1e-10);
    }

    #[test]
    fn test_zakharov_single_dim() {
        // zakharov([2]) = 4 + (0.5*1*2)^2 + (0.5*1*2)^4 = 4 + 1 + 1 = 6
        assert!((zakharov(&[2.0]) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_zakharov_empty() {
        assert!(zakharov(&[]).abs() < 1e-10);
    }

    // ---- Dixon-Price ----

    #[test]
    fn test_dixon_price_1d_optimum() {
        // dixon_price([1.0]) = (1-1)^2 + 0 = 0
        assert!(dixon_price(&[1.0]).abs() < 1e-10);
    }

    #[test]
    fn test_dixon_price_2d_optimum() {
        // x_1 = 1.0, x_2 = 2^(-1/2) = 1/sqrt(2)
        // term1 = (1-1)^2 = 0
        // term2 = 2 * (2*(1/sqrt(2))^2 - 1)^2 = 2 * (2*0.5 - 1)^2 = 2 * 0 = 0
        let x2 = 2.0_f64.powf(-0.5);
        assert!(dixon_price(&[1.0, x2]).abs() < 1e-10);
    }

    #[test]
    fn test_dixon_price_known_value() {
        // dixon_price([2, 1]) = (2-1)^2 + 2*(2*1^2 - 2)^2 = 1 + 2*(2-2)^2 = 1 + 0 = 1
        assert!((dixon_price(&[2.0, 1.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dixon_price_known_value_2() {
        // dixon_price([0, 0]) = (0-1)^2 + 2*(2*0 - 0)^2 = 1 + 0 = 1
        assert!((dixon_price(&[0.0, 0.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dixon_price_3d_exact_optimum() {
        // x_1 = 1, x_2 = 2^(-1/2), x_3 = 2^(-3/4)
        let x1 = 1.0;
        let x2 = 2.0_f64.powf(-0.5);
        let x3 = 2.0_f64.powf(-0.75);
        assert!(dixon_price(&[x1, x2, x3]).abs() < 1e-10);
    }

    // ---- all_benchmarks() metadata ----

    #[test]
    fn test_all_benchmarks_count() {
        assert_eq!(all_benchmarks().len(), 10);
    }

    #[test]
    fn test_all_benchmarks_ids_sequential() {
        let benchmarks = all_benchmarks();
        for (i, info) in benchmarks.iter().enumerate() {
            assert_eq!(
                info.id as usize,
                i + 1,
                "Benchmark '{}' has id {} but expected {}",
                info.name,
                info.id,
                i + 1
            );
        }
    }

    #[test]
    fn test_all_benchmarks_valid_bounds() {
        for info in all_benchmarks() {
            assert!(
                info.bounds.0 < info.bounds.1,
                "{} has invalid bounds: ({}, {})",
                info.name,
                info.bounds.0,
                info.bounds.1
            );
        }
    }

    #[test]
    fn test_all_benchmarks_names_unique() {
        let benchmarks = all_benchmarks();
        let names: Vec<&str> = benchmarks.iter().map(|b| b.name).collect();
        for (i, name) in names.iter().enumerate() {
            assert!(
                !names[..i].contains(name),
                "Duplicate benchmark name: {name}"
            );
        }
    }

    #[test]
    fn test_all_benchmarks_expected_names() {
        let benchmarks = all_benchmarks();
        let names: Vec<&str> = benchmarks.iter().map(|b| b.name).collect();
        let expected = [
            "Sphere",
            "Rosenbrock",
            "Rastrigin",
            "Ackley",
            "Schwefel",
            "Griewank",
            "Levy",
            "Michalewicz",
            "Zakharov",
            "Dixon-Price",
        ];
        assert_eq!(names, expected);
    }

    #[test]
    fn test_all_benchmarks_modality_flags() {
        let benchmarks = all_benchmarks();
        // Unimodal: sphere, rosenbrock, zakharov, dixon-price
        assert!(!benchmarks[0].multimodal, "Sphere should be unimodal");
        assert!(!benchmarks[1].multimodal, "Rosenbrock should be unimodal");
        assert!(benchmarks[2].multimodal, "Rastrigin should be multimodal");
        assert!(benchmarks[3].multimodal, "Ackley should be multimodal");
        assert!(!benchmarks[8].multimodal, "Zakharov should be unimodal");
        assert!(!benchmarks[9].multimodal, "Dixon-Price should be unimodal");
    }

    #[test]
    fn test_all_benchmarks_separability_flags() {
        let benchmarks = all_benchmarks();
        // Separable: sphere, rastrigin, schwefel, michalewicz
        assert!(benchmarks[0].separable, "Sphere should be separable");
        assert!(
            !benchmarks[1].separable,
            "Rosenbrock should be non-separable"
        );
        assert!(benchmarks[2].separable, "Rastrigin should be separable");
        assert!(!benchmarks[3].separable, "Ackley should be non-separable");
        assert!(benchmarks[7].separable, "Michalewicz should be separable");
    }

    #[test]
    fn test_michalewicz_optimum_is_neg_infinity() {
        // Michalewicz optimum is dimension-dependent, stored as NEG_INFINITY
        let benchmarks = all_benchmarks();
        assert!(
            benchmarks[7].optimum.is_infinite() && benchmarks[7].optimum < 0.0,
            "Michalewicz optimum should be NEG_INFINITY"
        );
    }

    #[test]
    fn test_all_other_benchmarks_optimum_zero() {
        // All benchmarks except Michalewicz have optimum = 0.0
        for info in all_benchmarks() {
            if info.name != "Michalewicz" {
                assert!(
                    (info.optimum).abs() < 1e-10,
                    "{} should have optimum 0.0 but got {}",
                    info.name,
                    info.optimum
                );
            }
        }
    }
}
