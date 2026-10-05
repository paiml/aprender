//! FALSIFY-LORA_TARGET_SELECTION_V1_012 (R15a C6): a backward GEMM's key and its
//! kernel dims agree with the runtime's, and the LoRA backward GEMMs are those of
//! every target. No device: these check the keys and dims the device code builds from.

use super::{gemm_backward_a, gemm_backward_b, is_backward_a, lora_backward_gemms, BackwardGemm};
use std::collections::BTreeSet;

const S: u32 = 128;
const R: u32 = 8;
// hidden, q_dim, kv_dim and intermediate all differ, so no two dims coincide.
const H: u32 = 64;
const QD: u32 = 96;
const KV: u32 = 32;
const I: u32 = 192;

/// (d_out, d_in) of q, k, v, o, gate, up, down at these dims.
const ALL: [(u32, u32); 7] = [(QD, H), (KV, H), (KV, H), (H, QD), (I, H), (I, H), (H, I)];

fn keys(gemms: &[BackwardGemm]) -> Vec<String> {
    gemms.iter().map(|g| g.key.clone()).collect()
}

#[test]
fn falsify_lora_target_selection_v1_012_key_and_kernel_dims_agree() {
    // The runtime calls gemm_backward_x(m, k, n), looks up gemm_backward_x_{m}_{k}_{n}
    // and builds tiled_unrolled(m, n, k).
    for (m, k, n) in [(4, 8, 32), (S, R, QD), (S, QD, R), (S, H, I), (S, I, H), (7, 7, 7)] {
        let a = gemm_backward_a(m, k, n);
        assert_eq!(a.key, format!("gemm_backward_a_{m}_{k}_{n}"));
        assert_eq!((a.m, a.n, a.k), (m, n, k), "gemm_backward_a({m}, {k}, {n}) kernel dims");
        assert!(is_backward_a(&a));
        let b = gemm_backward_b(m, k, n);
        assert_eq!(b.key, format!("gemm_backward_b_{m}_{k}_{n}"));
        assert_eq!((b.m, b.n, b.k), (m, n, k), "gemm_backward_b({m}, {k}, {n}) kernel dims");
        assert!(!is_backward_a(&b));
    }
}

#[test]
fn falsify_lora_target_selection_v1_012_every_target_gets_its_four_gemms() {
    for (t, &(d_out, d_in)) in ALL.iter().enumerate() {
        let got = lora_backward_gemms(&[(d_out, d_in)], S, R);
        let want: BTreeSet<BackwardGemm> = [
            gemm_backward_b(S, R, d_out),
            gemm_backward_a(S, d_out, R),
            gemm_backward_b(S, d_in, R),
            gemm_backward_a(S, R, d_in),
        ]
        .into_iter()
        .collect();
        assert_eq!(got, want.into_iter().collect::<Vec<_>>(), "target {t}");
    }

    let all = lora_backward_gemms(&ALL, S, R);
    let mut union: BTreeSet<BackwardGemm> = BTreeSet::new();
    for &dims in &ALL {
        union.extend(lora_backward_gemms(&[dims], S, R));
    }
    assert_eq!(all, union.into_iter().collect::<Vec<_>>(), "union over the targets");
    let sorted = keys(&all);
    let mut deduped = sorted.clone();
    deduped.sort();
    deduped.dedup();
    assert_eq!(sorted, deduped, "sorted, no repeat");
    // q, k/v, o, gate/up and down: d_out in {QD, KV, H, I} and d_in in {H, QD, I}
    assert_eq!(all.len(), 2 * 4 + 2 * 3, "{:?}", keys(&all));
}

#[test]
fn falsify_lora_target_selection_v1_012_qv_gives_the_six_keys_warmed_before() {
    let got: BTreeSet<String> =
        keys(&lora_backward_gemms(&[(QD, H), (KV, H)], S, R)).into_iter().collect();
    let (s, r, h, qd, kv) = (S, R, H, QD, KV);
    let before: BTreeSet<String> = [
        format!("gemm_backward_b_{s}_{r}_{qd}"),
        format!("gemm_backward_b_{s}_{r}_{kv}"),
        format!("gemm_backward_b_{s}_{h}_{r}"),
        format!("gemm_backward_a_{s}_{qd}_{r}"),
        format!("gemm_backward_a_{s}_{kv}_{r}"),
        format!("gemm_backward_a_{s}_{r}_{h}"),
    ]
    .into_iter()
    .collect();
    assert_eq!(got, before);

    // kv == q_dim (no GQA) folds to four, as the old `if kv != qd` did
    assert_eq!(lora_backward_gemms(&[(QD, H), (QD, H)], S, R).len(), 4);
    assert!(lora_backward_gemms(&[], S, R).is_empty());
}
