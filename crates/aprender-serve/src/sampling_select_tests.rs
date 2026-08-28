//! PERF-034 equivalence proof for [`super`].
//!
//! Every test here pins the *pre-existing* implementation as a literal reference
//! (`legacy_*`, copied verbatim from the code that shipped before PERF-034) and
//! asserts the optimised routine produces a byte-identical result. Deleting the
//! reference and asserting only "the new code is self-consistent" would prove
//! nothing, so the references stay.

use super::*;

/// The exact `sort_by` + `truncate` the samplers used before PERF-034.
fn legacy_top_k(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    if k < indexed.len() {
        indexed.truncate(k);
    }
    indexed
}

/// Deterministic pseudo-logits: a cheap LCG, so the corpus is reproducible without
/// pulling `rand` into a unit test.
fn pseudo_logits(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to roughly [-8, 8), the range real logits live in.
            ((s >> 33) as f32 / f32::from(u16::MAX)) % 16.0 - 8.0
        })
        .collect()
}

#[test]
fn retain_top_k_matches_legacy_full_sort_on_random_logits() {
    for seed in 0..16u64 {
        let data = pseudo_logits(1024, seed * 7 + 3);
        for k in [1usize, 2, 5, 40, 100, 1023, 1024, 4096] {
            let expected = legacy_top_k(&data, k);
            let mut got: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
            retain_top_k_sorted(&mut got, k);
            assert_eq!(got, expected, "seed={seed} k={k}");
        }
    }
}

#[test]
fn retain_top_k_matches_legacy_when_every_logit_is_tied() {
    // The case an unstable selection is most likely to get wrong: the comparator
    // reports Equal for every pair, so ONLY the stable sort's index order decides.
    let data = vec![1.5f32; 512];
    for k in [1usize, 3, 64, 511, 512] {
        let expected = legacy_top_k(&data, k);
        let mut got: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
        retain_top_k_sorted(&mut got, k);
        assert_eq!(got, expected, "k={k}");
        // And spell out what "identical" means here: the k LOWEST indices, in order.
        let idx: Vec<usize> = got.iter().map(|(i, _)| *i).collect();
        assert_eq!(idx, (0..k.min(512)).collect::<Vec<_>>());
    }
}

#[test]
fn retain_top_k_matches_legacy_with_heavy_partial_ties() {
    // Realistic decode shape: a handful of live logits, the rest masked to -inf by a
    // repetition penalty or a grammar constraint. The -inf block is one giant tie,
    // and a top_k larger than the live set has to cut *inside* it.
    let mut data = vec![f32::NEG_INFINITY; 2048];
    for (i, slot) in data.iter_mut().enumerate().take(2048).step_by(97) {
        *slot = (i % 13) as f32;
    }
    for k in [1usize, 8, 21, 22, 40, 512, 2048] {
        let expected = legacy_top_k(&data, k);
        let mut got: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
        retain_top_k_sorted(&mut got, k);
        assert_eq!(got, expected, "k={k}");
    }
}

#[test]
fn retain_top_k_matches_legacy_on_duplicated_value_blocks() {
    // Only 4 distinct values across 600 slots: ~150-way ties at every rank.
    let data: Vec<f32> = (0..600).map(|i| ((i % 4) as f32) * 0.25).collect();
    for k in [1usize, 4, 150, 151, 300, 599, 600] {
        let expected = legacy_top_k(&data, k);
        let mut got: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
        retain_top_k_sorted(&mut got, k);
        assert_eq!(got, expected, "k={k}");
    }
}

#[test]
fn retain_top_k_zero_clears() {
    let mut got: Vec<(usize, f32)> = vec![(0, 1.0), (1, 2.0)];
    retain_top_k_sorted(&mut got, 0);
    assert!(got.is_empty());
}

#[test]
fn sort_desc_by_index_matches_legacy_stable_sort() {
    for seed in 0..8u64 {
        let data = pseudo_logits(777, seed * 11 + 5);
        let expected = legacy_top_k(&data, usize::MAX);
        let mut got: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
        sort_desc_by_index(&mut got);
        assert_eq!(got, expected, "seed={seed}");
    }
}

#[test]
fn cmp_desc_then_index_is_a_total_order_with_no_ties() {
    // The property the whole equivalence argument rests on: distinct indices mean
    // the comparator never returns Equal, so the sorted permutation is unique and a
    // stable and an unstable sort cannot disagree.
    let pairs = [(0usize, 1.0f32), (1, 1.0), (2, -0.0), (3, 0.0), (4, 5.0)];
    for a in &pairs {
        for b in &pairs {
            let ord = cmp_desc_then_index(a, b);
            assert_eq!(ord == Ordering::Equal, a.0 == b.0, "{a:?} vs {b:?}");
            assert_eq!(ord, cmp_desc_then_index(b, a).reverse(), "antisymmetry");
        }
    }
}

#[test]
fn cmp_treats_negative_and_positive_zero_as_equal_like_partial_cmp() {
    // Guards against a "cleanup" to `total_cmp`, which orders -0.0 < 0.0 and would
    // silently reorder tied logits relative to the shipped behaviour.
    assert_eq!(cmp_desc_then_index(&(0, -0.0), &(1, 0.0)), Ordering::Less);
    assert_eq!(
        cmp_desc_then_index(&(1, -0.0), &(0, 0.0)),
        Ordering::Greater
    );
}

#[test]
fn fill_scaled_matches_a_separate_scaled_vec() {
    let data = pseudo_logits(256, 42);
    for temperature in [0.1f32, 0.7, 1.0, 2.5] {
        let legacy: Vec<f32> = data.iter().map(|&x| x / temperature).collect();
        let expected: Vec<(usize, f32)> = legacy.iter().copied().enumerate().collect();
        let mut got = Vec::new();
        fill_scaled(&mut got, &data, temperature);
        assert_eq!(got, expected, "T={temperature}");
    }
}

#[test]
fn argmax_first_wins_matches_legacy_sort_then_first() {
    for seed in 0..8u64 {
        let data = pseudo_logits(513, seed * 3 + 1);
        let expected = legacy_top_k(&data, 1).first().map(|(i, _)| *i);
        assert_eq!(argmax_first_wins(&data), expected, "seed={seed}");
    }
    // All-tied: lowest index, matching the stable sort.
    assert_eq!(argmax_first_wins(&[2.0, 2.0, 2.0]), Some(0));
    assert_eq!(argmax_first_wins(&[]), None);
}

#[test]
fn scratch_is_reused_across_calls_and_handed_back_empty() {
    with_candidate_scratch(|buf| {
        buf.extend((0..4096).map(|i| (i, i as f32)));
    });
    let cap = with_candidate_scratch(|buf| {
        assert!(buf.is_empty(), "scratch must arrive cleared");
        buf.capacity()
    });
    assert!(
        cap >= 4096,
        "scratch must keep its allocation between tokens, got capacity {cap}"
    );
}
