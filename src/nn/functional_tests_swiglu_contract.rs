// =========================================================================
// FALSIFY-SG: swiglu-kernel-v1.yaml contract (aprender functional::swiglu)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest SwiGLU tests but zero inline FALSIFY-SG-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from swiglu-kernel-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: SwiGLU was "obviously correct" (x * SiLU(gate))
//
// References:
//   - provable-contracts/contracts/swiglu-kernel-v1.yaml
//   - Shazeer (2020) "GLU Variants Improve Transformer"
// =========================================================================

use super::*;

/// FALSIFY-SG-001: Zero preservation — SwiGLU(0, gate) = 0 for any gate
///
/// When x = 0, the output must be 0 regardless of gate value.
#[test]
fn falsify_sg_001_zero_x_preservation() {
    let gates = vec![-10.0, -1.0, 0.0, 1.0, 10.0];

    for &g in &gates {
        let y = swiglu_scalar(0.0, g);
        assert!(
            y.abs() < 1e-7,
            "FALSIFIED SG-001: SwiGLU(0, {g}) = {y}, expected 0"
        );
    }

    // Tensor API
    let x = Tensor::new(&[0.0; 5], &[5]);
    let gate = Tensor::new(&gates, &[5]);
    let y = swiglu(&x, &gate);
    for (i, &val) in y.data().iter().enumerate() {
        assert!(
            val.abs() < 1e-7,
            "FALSIFIED SG-001: tensor SwiGLU(0, {})[{i}] = {val}",
            gates[i]
        );
    }
}

/// FALSIFY-SG-002: Gate bounded below — SiLU(gate) > -0.279
///
/// The gate component is SiLU(gate), which has a global minimum > -0.279.
#[test]
fn falsify_sg_002_gate_bounded_below() {
    let gates: Vec<f32> = vec![-100.0, -10.0, -1.278, -1.0, 0.0, 1.0, 100.0];

    for &g in &gates {
        let gate_output = silu_scalar(g);
        assert!(
            gate_output > -0.28,
            "FALSIFIED SG-002: SiLU({g}) = {gate_output}, expected > -0.279"
        );
    }
}

/// FALSIFY-SG-003: Decomposition — SwiGLU(x, gate) = x * SiLU(gate)
///
/// The fused and decomposed computations must agree.
#[test]
fn falsify_sg_003_decomposition() {
    let test_cases: Vec<(f32, f32)> = vec![
        (1.0, 1.0),
        (-2.0, 3.0),
        (5.0, -1.0),
        (0.5, 0.5),
        (100.0, 0.0),
        (-0.5, -0.5),
    ];

    for &(x, g) in &test_cases {
        let fused = swiglu_scalar(x, g);
        let decomposed = x * silu_scalar(g);
        assert!(
            (fused - decomposed).abs() < 1e-5,
            "FALSIFIED SG-003: swiglu({x},{g})={fused} != {x}*silu({g})={decomposed}"
        );
    }
}

/// FALSIFY-SG-004: Finite output — all outputs finite for finite inputs
#[test]
fn falsify_sg_004_finite_output() {
    let x_vals = vec![-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
    let g_vals = vec![-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];

    for &x in &x_vals {
        for &g in &g_vals {
            let y = swiglu_scalar(x, g);
            assert!(
                y.is_finite(),
                "FALSIFIED SG-004: SwiGLU({x}, {g}) = {y} (not finite)"
            );
        }
    }
}

/// FALSIFY-SG-005: Tensor API matches scalar API element-wise
#[test]
fn falsify_sg_005_tensor_scalar_equivalence() {
    let x_data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
    let g_data = vec![0.5, -1.0, 2.0, -3.0, 4.0];

    let x = Tensor::new(&x_data, &[5]);
    let gate = Tensor::new(&g_data, &[5]);
    let y = swiglu(&x, &gate);

    for i in 0..5 {
        let expected = swiglu_scalar(x_data[i], g_data[i]);
        assert!(
            (y.data()[i] - expected).abs() < 1e-6,
            "FALSIFIED SG-005: tensor[{i}]={} != scalar={}",
            y.data()[i],
            expected
        );
    }
}
