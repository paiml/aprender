//! L4 simulation witness (ONT-10): `transpose_row_major` against the Lean model
//! `transposeRowMajor` in `ProvableContracts/Theorems/TensorLayout/IndexAlgebra.lean`.
//!
//! The golden table is that model's `#eval` on every shape `rows, cols` in `0..=8` with the
//! source `0..rows*cols`. Distinct source values make each output line the full index map for
//! its shape, so agreement here is agreement on every slot of every one of the 81 shapes.

use super::transpose_row_major;

const GOLDEN: &str = include_str!("transpose_row_major_golden.txt");

fn golden_rows() -> Vec<(usize, usize, Vec<usize>)> {
    GOLDEN
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let v: Vec<usize> = l
                .split_whitespace()
                .map(|t| t.parse().expect("golden token is a usize"))
                .collect();
            (v[0], v[1], v[2..].to_vec())
        })
        .collect()
}

#[test]
fn transpose_row_major_matches_lean_model_on_every_shape_to_8x8() {
    let rows = golden_rows();
    assert_eq!(rows.len(), 81, "golden table must cover all 9x9 shapes");
    for (r, c, want) in rows {
        let src: Vec<usize> = (0..r * c).collect();
        assert_eq!(
            transpose_row_major(&src, r, c),
            want,
            "shape rows={r} cols={c}"
        );
    }
}

/// The pre-extraction loop from `transpose_q{4,5,6}k_for_matmul`, kept as the behaviour oracle:
/// the extraction must be output-identical to it, not just to the model.
#[test]
fn transpose_row_major_equals_the_loop_it_replaced() {
    for r in 0..=12 {
        for c in 0..=12 {
            let src: Vec<f32> = (0..r * c).map(|i| i as f32 * 0.5 - 3.0).collect();
            let mut old = vec![0.0f32; r * c];
            for rr in 0..r {
                for cc in 0..c {
                    old[rr * c + cc] = src[cc * r + rr];
                }
            }
            assert_eq!(
                transpose_row_major(&src, r, c),
                old,
                "shape rows={r} cols={c}"
            );
        }
    }
}
