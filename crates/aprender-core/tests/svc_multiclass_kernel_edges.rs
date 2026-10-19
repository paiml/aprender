//! Public-API coverage for `Kernel` and `MultiClassSVC` (coverage ticket, Rule 7).
//! Contract: FALSIFY-SVC-RBF-008 in contracts/svc-rbf-v1.yaml.

use aprender::classification::{Kernel, MultiClassSVC, SVCRbf};
use aprender::primitives::Matrix;

/// Three well-separated 2-D clusters, labels 0/5/9 (non-contiguous on purpose).
fn clusters() -> (Matrix<f32>, Vec<usize>) {
    let centers = [(0.0f32, 0.0f32, 0usize), (6.0, 0.0, 5), (0.0, 6.0, 9)];
    let jitter = [(0.0f32, 0.0f32), (0.3, 0.1), (-0.2, 0.3), (0.1, -0.3)];
    let (mut data, mut y) = (Vec::new(), Vec::new());
    for (cx, cy, label) in centers {
        for (dx, dy) in jitter {
            data.extend([cx + dx, cy + dy]);
            y.push(label);
        }
    }
    (Matrix::from_vec(y.len(), 2, data).expect("matrix"), y)
}

#[test]
fn kernel_eval_rbf_and_poly_values() {
    let a = [1.0f32, 2.0];
    let b = [3.0f32, 5.0];
    let rbf = Kernel::Rbf { gamma: 0.5 };
    assert!((rbf.eval(&a, &a) - 1.0).abs() < 1e-6, "K(a,a)=1");
    // ||a-b||^2 = 13
    assert!((rbf.eval(&a, &b) - (-6.5f32).exp()).abs() < 1e-6);
    let poly = Kernel::Poly {
        gamma: 1.0,
        coef0: 1.0,
        degree: 2,
    };
    // (1*3 + 2*5 + 1)^2 = 14^2
    assert!((poly.eval(&a, &b) - 196.0).abs() < 1e-3);
    // degree 0 is the constant 1 regardless of inputs
    let flat = Kernel::Poly {
        gamma: 2.0,
        coef0: 3.0,
        degree: 0,
    };
    assert!((flat.eval(&a, &b) - 1.0).abs() < 1e-6);
}

#[test]
fn svc_kernel_builders_set_the_kernel() {
    assert_eq!(
        SVCRbf::new()
            .with_kernel(Kernel::Rbf { gamma: 2.0 })
            .kernel(),
        Kernel::Rbf { gamma: 2.0 }
    );
    assert_eq!(
        SVCRbf::new().with_poly(0.5, 1.0, 3).kernel(),
        Kernel::Poly {
            gamma: 0.5,
            coef0: 1.0,
            degree: 3
        }
    );
    assert_eq!(
        SVCRbf::new().with_gamma(0.25).kernel(),
        Kernel::Rbf { gamma: 0.25 }
    );
}

#[test]
fn multiclass_fits_and_predicts_noncontiguous_labels() {
    let (x, y) = clusters();
    let mut m = MultiClassSVC::new()
        .with_gamma(0.5)
        .with_c(10.0)
        .with_max_iter(500);
    m.fit(&x, &y).expect("fit");
    assert_eq!(m.classes(), &[0, 5, 9]);
    assert_eq!(m.predict(&x).expect("predict"), y);
    assert!((m.score(&x, &y).expect("score") - 1.0).abs() < 1e-6);

    let probe = Matrix::from_vec(3, 2, vec![0.1, 0.2, 5.9, 0.1, 0.1, 5.8]).expect("probe");
    assert_eq!(m.predict(&probe).expect("predict"), vec![0, 5, 9]);
}

#[test]
fn multiclass_polynomial_kernel_path_runs() {
    let (x, y) = clusters();
    let mut m = MultiClassSVC::new()
        .with_poly(0.1, 1.0, 2)
        .with_c(10.0)
        .with_max_iter(500);
    m.fit(&x, &y).expect("fit");
    let preds = m.predict(&x).expect("predict");
    assert_eq!(preds.len(), y.len());
    assert!(
        preds.iter().all(|p| [0, 5, 9].contains(p)),
        "labels stay in the training set"
    );
    assert!(m.score(&x, &y).expect("score") > 0.6);
}

#[test]
fn multiclass_is_deterministic() {
    let (x, y) = clusters();
    let run = || {
        let mut m = MultiClassSVC::new()
            .with_kernel(Kernel::Rbf { gamma: 0.5 })
            .with_c(5.0);
        m.fit(&x, &y).expect("fit");
        m.predict(&x).expect("predict")
    };
    assert_eq!(run(), run());
}

#[test]
fn multiclass_rejects_bad_input() {
    let (x, y) = clusters();
    let mut m = MultiClassSVC::new();
    // predict before fit
    assert!(m.predict(&x).is_err());
    // length mismatch
    assert!(m.fit(&x, &y[..y.len() - 1]).is_err());
    // single class
    assert!(m.fit(&x, &vec![3usize; y.len()]).is_err());
    // zero samples
    let empty = Matrix::from_vec(0, 2, vec![]).expect("empty");
    assert!(m.fit(&empty, &[]).is_err());
    // a failed fit leaves it unfitted
    assert!(m.classes().is_empty());
    assert!(m.predict(&x).is_err());
}

#[test]
fn multiclass_score_of_empty_truth_is_zero() {
    let (x, y) = clusters();
    let mut m = MultiClassSVC::new().with_gamma(0.5).with_c(10.0);
    m.fit(&x, &y).expect("fit");
    let none = Matrix::from_vec(0, 2, vec![]).expect("empty");
    assert!(m.score(&none, &[]).expect("score").abs() < f32::EPSILON);
}

#[test]
fn default_constructors_match_new() {
    assert_eq!(
        MultiClassSVC::default().classes(),
        MultiClassSVC::new().classes()
    );
    assert_eq!(SVCRbf::default().kernel(), SVCRbf::new().kernel());
    assert_eq!(SVCRbf::default().n_support_vectors(), 0);
}
