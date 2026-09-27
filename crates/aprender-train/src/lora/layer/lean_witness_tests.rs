//! ONT-10 L4 witness for `ProvableContracts/Theorems/LoRA/Lora_Shape.lean`.
//!
//! That module's `matmul_len` and `merge_len` transcribe the length checks of
//! [`matmul`] and [`LoRALayer::merge`]: which calls panic, and how many elements
//! the result has. The two golden tables are the Lean model's own output
//! (`#eval golden_matmul`, `#eval golden_merge`, verbatim), so these tests run
//! the Rust over the same grid and compare call by call. A drift on either
//! side, in the Rust or in the Lean def, turns them red.

use super::LoRALayer;
use crate::autograd::matmul;
use crate::Tensor;
use std::panic::{catch_unwind, AssertUnwindSafe};

const GOLDEN_MATMUL: &str = include_str!("lean_golden_matmul.txt");
const GOLDEN_MERGE: &str = include_str!("lean_golden_merge.txt");

fn golden(table: &str, want: usize) -> Vec<char> {
    let g: Vec<char> = table.trim_end().chars().collect();
    assert_eq!(g.len(), want, "golden table has the wrong number of cells");
    g
}

#[test]
fn matmul_lengths_match_lean_golden_table() {
    let g = golden(GOLDEN_MATMUL, 2700);
    let mut i = 0;
    let mut bad = Vec::new();
    for m in 1..=3usize {
        for k in 1..=3usize {
            for n in 1..=3usize {
                for a in 0..10usize {
                    for b in 0..10usize {
                        let run = catch_unwind(AssertUnwindSafe(|| {
                            matmul(&Tensor::zeros(a, false), &Tensor::zeros(b, false), m, k, n)
                                .len()
                        }));
                        let got = match run {
                            Err(_) => '.',
                            Ok(c) => u32::try_from(c)
                                .ok()
                                .and_then(|c| char::from_digit(c, 10))
                                .unwrap_or('?'),
                        };
                        if got != g[i] {
                            bad.push(format!(
                                "m={m} k={k} n={n} a={a} b={b}: rust {got}, lean {}",
                                g[i]
                            ));
                        }
                        i += 1;
                    }
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} cells differ from Lean's matmul_len:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn merge_lengths_match_lean_golden_table() {
    let g = golden(GOLDEN_MERGE, 243);
    let mut i = 0;
    let mut bad = Vec::new();
    for d_out in 1..=3usize {
        for d_in in 1..=3usize {
            for rank in 1..=3usize {
                for da in 0..3usize {
                    for db in 0..3usize {
                        let base = d_out * d_in;
                        let mut layer = LoRALayer::new(
                            Tensor::zeros(base, false),
                            d_out,
                            d_in,
                            rank,
                            rank as f32,
                        );
                        *layer.lora_a_mut() = Tensor::zeros(rank * d_in + da - 1, true);
                        *layer.lora_b_mut() = Tensor::zeros(d_out * rank + db - 1, true);
                        let run = catch_unwind(AssertUnwindSafe(move || {
                            layer.merge();
                            layer.base_weight().len()
                        }));
                        let got = match run {
                            Err(_) => '.',
                            Ok(c) if c == base => 'k',
                            Ok(_) => '?',
                        };
                        if got != g[i] {
                            bad.push(format!(
                                "d_out={d_out} d_in={d_in} rank={rank} da={da} db={db}: rust {got}, lean {}",
                                g[i]
                            ));
                        }
                        i += 1;
                    }
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} cells differ from Lean's merge_len:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// `merge_len true` is `some base_len`: a second merge is a no-op on the shape.
#[test]
fn merging_a_merged_layer_keeps_the_base_length() {
    let mut layer = LoRALayer::new(Tensor::zeros(6, false), 2, 3, 2, 2.0);
    layer.merge();
    layer.merge();
    assert!(layer.is_merged());
    assert_eq!(layer.base_weight().len(), 6);
}
