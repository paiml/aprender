// =========================================================================
// FALSIFY-TN: validated-tensor-v1.yaml contract (aprender autograd Tensor)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-TN-* tests for autograd Tensor
//   Why 2: Tensor tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from validated-tensor-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Tensor add/matmul was "obviously correct" (basic linear algebra)
//
// References:
//   - provable-contracts/contracts/validated-tensor-v1.yaml
// =========================================================================

use super::Tensor;

/// FALSIFY-TN-001: Tensor addition is element-wise and commutative
#[test]
fn falsify_tn_001_add_commutative() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[4.0, 5.0, 6.0], &[3]);

    let ab = a.add(&b);
    let ba = b.add(&a);

    for i in 0..3 {
        assert!(
            (ab.data()[i] - ba.data()[i]).abs() < 1e-6,
            "FALSIFIED TN-001: add not commutative at [{i}]: {} != {}",
            ab.data()[i],
            ba.data()[i]
        );
    }
    // Check actual values
    assert!(
        (ab.data()[0] - 5.0).abs() < 1e-6,
        "FALSIFIED TN-001: 1+4={}, expected 5",
        ab.data()[0]
    );
}

/// FALSIFY-TN-002: Tensor shape preserved after add
#[test]
fn falsify_tn_002_add_preserves_shape() {
    let a = Tensor::new(&[1.0; 6], &[2, 3]);
    let b = Tensor::new(&[2.0; 6], &[2, 3]);

    let c = a.add(&b);
    assert_eq!(
        c.shape(),
        &[2, 3],
        "FALSIFIED TN-002: add changed shape to {:?}",
        c.shape()
    );
}

/// FALSIFY-TN-003: Tensor matmul produces correct shape
#[test]
fn falsify_tn_003_matmul_shape() {
    let a = Tensor::new(&[1.0; 6], &[2, 3]); // 2x3
    let b = Tensor::new(&[1.0; 12], &[3, 4]); // 3x4

    let c = a.matmul(&b);
    assert_eq!(
        c.shape(),
        &[2, 4],
        "FALSIFIED TN-003: matmul [2,3]x[3,4] produced shape {:?}, expected [2,4]",
        c.shape()
    );
}

/// FALSIFY-TN-004: Tensor mul_scalar scales all elements
#[test]
fn falsify_tn_004_mul_scalar() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = a.mul_scalar(2.5);

    for (i, &v) in b.data().iter().enumerate() {
        let expected = (i as f32 + 1.0) * 2.5;
        assert!(
            (v - expected).abs() < 1e-6,
            "FALSIFIED TN-004: mul_scalar[{i}]={v}, expected {expected}"
        );
    }
}
