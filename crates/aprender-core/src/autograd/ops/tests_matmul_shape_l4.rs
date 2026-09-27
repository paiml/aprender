//! L4 simulation witness for `Tensor::matmul`'s shape path (ONT-10).
//!
//! The Lean model `tensorMatmulShape` / `tensorMatmulDataLen` in
//! `crates/aprender-contracts-staging/lean/ProvableContracts/Theorems/MatMul/Shape.lean`
//! transcribes `activation.rs:274-299`. Its `#eval rustMatmulGolden` output is
//! committed as `matmul_shape_l4_golden.txt`: every pair of shapes of rank 1..3
//! with dims in 1..3. This test runs the Rust fn on each pair and requires the
//! same answer — a panic where the model says `panic`, else output shape `[m, n]`
//! and a buffer of `m * n` elements.

use crate::autograd::Tensor;
use std::panic::{catch_unwind, AssertUnwindSafe};

const GOLDEN: &str = include_str!("matmul_shape_l4_golden.txt");

fn dims(s: &str) -> Vec<usize> {
    s.split(',')
        .map(|d| d.parse().expect("golden dim is a usize"))
        .collect()
}

fn ones(shape: &[usize]) -> Tensor {
    Tensor::from_vec(vec![1.0; shape.iter().product()], shape)
}

#[test]
fn matmul_shape_matches_lean_model_on_exhaustive_domain() {
    let mut rows = 0usize;
    let mut panics = 0usize;
    for line in GOLDEN.lines().filter(|l| !l.starts_with('#')) {
        let mut f = line.split(';');
        let (a, b, want) = (
            dims(f.next().expect("lhs")),
            dims(f.next().expect("rhs")),
            f.next().expect("out"),
        );
        let (x, y) = (ones(&a), ones(&b));
        let got = catch_unwind(AssertUnwindSafe(|| x.matmul(&y)));
        match (want, got) {
            ("panic", Err(_)) => panics += 1,
            ("panic", Ok(t)) => panic!("{a:?} @ {b:?}: model panics, Rust returned {:?}", t.shape()),
            (_, Err(_)) => panic!("{a:?} @ {b:?}: model returns {want}, Rust panicked"),
            (w, Ok(t)) => {
                let want = dims(w);
                assert_eq!(t.shape(), want.as_slice(), "{a:?} @ {b:?}");
                assert_eq!(t.data().len(), want[0] * want[1], "{a:?} @ {b:?}: tensorMatmulDataLen");
            },
        }
        rows += 1;
    }
    // 39 shapes squared; 9 * 3 = 27 pairs agree on k and are 2-D.
    assert_eq!(rows, 39 * 39, "golden table is truncated");
    assert_eq!(rows - panics, 27, "wrong number of returning pairs");
}
