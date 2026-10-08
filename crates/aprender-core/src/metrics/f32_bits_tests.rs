//! FROZEN `to_bits()` snapshot of the shared f32 metrics (plan 08-27, step 0b).
//!
//! `metrics::classification::f1_score` (every `Average` mode) and
//! `calibration::expected_calibration_error_top_label` feed the SetFit claims-stats phases.
//! Plan 08-27 ADDS an f64, exactly-summed path for the Laya gate beside them; it must not move
//! a single bit of what these two f32 functions return. This table was printed by the
//! pre-plan tree's own functions (`print_f32_snapshot`, ignored) and committed ALONE before
//! any change to classification.rs or calibration.rs. It is never edited afterwards: if a
//! later change turns `f32_metrics_are_bit_identical_to_the_frozen_snapshot` red, the later
//! change is wrong.
//!
//! Inputs: every `(y_pred, y_true)` the classification unit tests pass `f1_score` (directly or
//! through `classification_report` / the evaluator), every case of
//! `scripts/setfit_fixtures/claims_stats/ece_top_label_cases.json` plus the saturated-row
//! calibration test, and 24 seeded random inputs (N 3..600 rows, K 2..6, 15 ECE bins) from
//! the fixed-seed generator below.

use super::classification::{f1_score, Average};
use crate::calibration::expected_calibration_error_top_label;

const ECE_FIXTURES: &str =
    include_str!("../../../../scripts/setfit_fixtures/claims_stats/ece_top_label_cases.json");

/// Seeded random cases in the snapshot.
const N_RANDOM: u64 = 24;
/// ECE bins for the random cases (laya-finetune-gate-v1 `ece_bins`).
const RANDOM_BINS: usize = 15;

/// The fixed `(y_true, y_pred)` inputs the existing unit tests pass `f1_score`.
const FIXED_F1: &[(&str, &[usize], &[usize])] = &[
    ("dense3_perfect", &[0, 1, 2, 0, 1, 2], &[0, 1, 2, 0, 1, 2]),
    ("dense3_doc", &[0, 1, 2, 0, 1, 2], &[0, 2, 1, 0, 0, 1]),
    ("noncontig02_perfect", &[0, 2, 0, 2], &[0, 2, 0, 2]),
    ("noncontig02_swapped", &[0, 0, 2, 2], &[0, 2, 2, 0]),
    (
        "jaccard_fbeta_binary",
        &[0, 0, 1, 1, 1, 0, 1, 0],
        &[0, 1, 1, 1, 0, 0, 1, 1],
    ),
    ("per_class_doc", &[1, 0, 1, 0], &[1, 1, 0, 0]),
    ("report_doc", &[0, 0, 1, 1, 2, 2], &[0, 1, 1, 1, 2, 0]),
    ("report_rotated3", &[0, 1, 2], &[1, 2, 0]),
    ("report_binary_half", &[0, 0, 1, 1], &[0, 1, 1, 0]),
    ("report_skewed", &[0, 0, 0, 1, 1, 2], &[0, 0, 1, 1, 0, 2]),
    (
        "report_imbalanced_perfect",
        &[0, 0, 0, 0, 1, 2],
        &[0, 0, 0, 0, 1, 2],
    ),
    ("report_missing_pred2", &[0, 1, 2], &[0, 1, 0]),
    ("harmonic_micro", &[0, 0, 1, 1, 2, 2], &[0, 1, 1, 0, 2, 2]),
    ("between_p_r", &[0, 0, 1, 1, 2, 2], &[0, 1, 1, 0, 2, 1]),
    ("identity3", &[0, 1, 2], &[0, 1, 2]),
    (
        "nine_rows",
        &[0, 0, 1, 1, 2, 2, 0, 1, 2],
        &[0, 1, 1, 2, 2, 0, 0, 1, 1],
    ),
    ("single_class", &[0, 0, 0], &[0, 0, 0]),
    ("binary_six", &[0, 0, 1, 1, 1, 0], &[0, 1, 1, 1, 0, 0]),
    ("binary_three_a", &[0, 1, 1], &[0, 0, 1]),
    ("binary_three_b", &[0, 1, 1], &[1, 1, 0]),
    ("binary_five", &[0, 0, 1, 1, 1], &[0, 1, 1, 1, 0]),
    (
        "per_class_consistency",
        &[0, 0, 1, 1, 2, 2],
        &[0, 1, 1, 2, 2, 0],
    ),
    (
        "evaluator_perfect",
        &[0, 0, 1, 1, 2, 2],
        &[0, 0, 1, 1, 2, 2],
    ),
];

/// splitmix64: a fixed, platform-independent integer generator.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A uniform f32 in [0, 1) from the top 24 bits (exactly representable).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
}

/// First-max argmax (numpy's tie-break, the one the calibration code uses).
fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (k, &p) in row.iter().enumerate().skip(1) {
        if p > row[best] {
            best = k;
        }
    }
    best
}

/// One seeded random input: row-major probabilities, K, true labels and argmax predictions.
struct RandomCase {
    probs: Vec<f32>,
    k: usize,
    y_true: Vec<usize>,
    y_pred: Vec<usize>,
}

fn random_case(index: u64) -> RandomCase {
    let mut rng = SplitMix(0x0827_F32B_0000_0000 ^ index);
    let n = 3 + rng.below(598) as usize;
    let k = 2 + rng.below(5) as usize;
    // Sharpness: the raw weight is raised to 1, 2 or 4 by repeated multiplication (no powi).
    let sharp = 1 << rng.below(3);
    // How often the true label is forced to the argmax (accuracy varies case to case).
    let agree = rng.below(101);
    let mut probs = Vec::with_capacity(n * k);
    let mut y_true = Vec::with_capacity(n);
    let mut y_pred = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = Vec::with_capacity(k);
        for _ in 0..k {
            let u = rng.unit() + 1.0e-3;
            let mut w = u;
            for _ in 1..sharp {
                w *= u;
            }
            row.push(w);
        }
        let sum: f32 = row.iter().sum();
        for w in &mut row {
            *w /= sum;
        }
        let pred = argmax(&row);
        let label = if rng.below(100) < agree {
            pred
        } else {
            rng.below(k as u64) as usize
        };
        probs.extend_from_slice(&row);
        y_true.push(label);
        y_pred.push(pred);
    }
    RandomCase {
        probs,
        k,
        y_true,
        y_pred,
    }
}

/// Every f1_score input in snapshot order: `(name, y_true, y_pred)`.
fn f1_inputs() -> Vec<(String, Vec<usize>, Vec<usize>)> {
    let mut out: Vec<(String, Vec<usize>, Vec<usize>)> = FIXED_F1
        .iter()
        .map(|(name, t, p)| ((*name).to_string(), t.to_vec(), p.to_vec()))
        .collect();
    let rotated: Vec<usize> = (0..50).map(|i| i % 5).collect();
    let rotated_pred: Vec<usize> = (0..50).map(|i| (i + 1) % 5).collect();
    out.push(("rotated50".to_string(), rotated, rotated_pred));
    let tens: Vec<usize> = (0..100).map(|i| i % 10).collect();
    out.push(("tens100_perfect".to_string(), tens.clone(), tens));
    for i in 0..N_RANDOM {
        let c = random_case(i);
        out.push((format!("rand_{i:02}"), c.y_true, c.y_pred));
    }
    out
}

/// Every top-label ECE input in snapshot order: `(name, probs, k, labels, bins)`.
fn ece_inputs() -> Vec<(String, Vec<f32>, usize, Vec<usize>, usize)> {
    let doc: serde_json::Value = serde_json::from_str(ECE_FIXTURES).expect("fixture JSON parses");
    let mut out = Vec::new();
    for case in doc["cases"].as_array().expect("cases array") {
        let rows = case["probabilities"].as_array().expect("probabilities");
        let k = rows[0].as_array().expect("row").len();
        let probs: Vec<f32> = rows
            .iter()
            .flat_map(|r| r.as_array().expect("row").iter())
            .map(|v| v.as_f64().expect("number") as f32)
            .collect();
        let labels: Vec<usize> = case["labels"]
            .as_array()
            .expect("labels")
            .iter()
            .map(|v| v.as_u64().expect("label") as usize)
            .collect();
        let bins = case["n_bins"].as_u64().expect("n_bins") as usize;
        let id = case["id"].as_str().expect("id");
        out.push((format!("fixture_{id}"), probs, k, labels, bins));
    }
    out.push((
        "saturated_two_rows".to_string(),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        3,
        vec![0, 1],
        10,
    ));
    for i in 0..N_RANDOM {
        let c = random_case(i);
        out.push((format!("rand_{i:02}"), c.probs, c.k, c.y_true, RANDOM_BINS));
    }
    out
}

fn f1_bits(y_true: &[usize], y_pred: &[usize]) -> [u32; 3] {
    [
        f1_score(y_pred, y_true, Average::Macro).to_bits(),
        f1_score(y_pred, y_true, Average::Micro).to_bits(),
        f1_score(y_pred, y_true, Average::Weighted).to_bits(),
    ]
}

/// `(name, [macro, micro, weighted])` as printed by the pre-plan tree.
const F1_SNAPSHOT: &[(&str, [u32; 3])] = &[
    ("dense3_perfect", [0x3f800000, 0x3f800000, 0x3f800000]),
    ("dense3_doc", [0x3e888889, 0x3eaaaaab, 0x3e888889]),
    ("noncontig02_perfect", [0x3f800000, 0x3f800000, 0x3f800000]),
    ("noncontig02_swapped", [0x3f000000, 0x3f000000, 0x3f000000]),
    ("jaccard_fbeta_binary", [0x3f1e79e8, 0x3f200000, 0x3f1e79e8]),
    ("per_class_doc", [0x3f000000, 0x3f000000, 0x3f000000]),
    ("report_doc", [0x3f27d27d, 0x3f2aaaab, 0x3f27d27d]),
    ("report_rotated3", [0x00000000, 0x00000000, 0x00000000]),
    ("report_binary_half", [0x3f000000, 0x3f000000, 0x3f000000]),
    ("report_skewed", [0x3f38e38f, 0x3f2aaaab, 0x3f2aaaab]),
    (
        "report_imbalanced_perfect",
        [0x3f800000, 0x3f800000, 0x3f800000],
    ),
    ("report_missing_pred2", [0x3f0e38e4, 0x3f2aaaab, 0x3f0e38e4]),
    ("harmonic_micro", [0x3f2aaaab, 0x3f2aaaab, 0x3f2aaaab]),
    ("between_p_r", [0x3f05b05b, 0x3f000000, 0x3f05b05b]),
    ("identity3", [0x3f800000, 0x3f800000, 0x3f800000]),
    ("nine_rows", [0x3f0bc8bd, 0x3f0e38e4, 0x3f0bc8bc]),
    ("single_class", [0x3f800000, 0x3f800000, 0x3f800000]),
    ("binary_six", [0x3f2aaaab, 0x3f2aaaab, 0x3f2aaaab]),
    ("binary_three_a", [0x3f2aaaab, 0x3f2aaaab, 0x3f2aaaab]),
    ("binary_three_b", [0x3e800000, 0x3eaaaaab, 0x3eaaaaab]),
    ("binary_five", [0x3f155556, 0x3f19999a, 0x3f19999a]),
    (
        "per_class_consistency",
        [0x3f000000, 0x3f000000, 0x3f000000],
    ),
    ("evaluator_perfect", [0x3f800000, 0x3f800000, 0x3f800000]),
    ("rotated50", [0x00000000, 0x00000000, 0x00000000]),
    ("tens100_perfect", [0x3f800000, 0x3f800000, 0x3f800001]),
    ("rand_00", [0x3e893a3c, 0x3e8a3d71, 0x3e8a80a2]),
    ("rand_01", [0x3f3a202e, 0x3f3a8fe5, 0x3f3a78a4]),
    ("rand_02", [0x3f215d8a, 0x3f232a33, 0x3f23832d]),
    ("rand_03", [0x3f12879e, 0x3f126c9b, 0x3f125981]),
    ("rand_04", [0x3f59f49f, 0x3f59df52, 0x3f5a2ad9]),
    ("rand_05", [0x3f196c7c, 0x3f196370, 0x3f1975de]),
    ("rand_06", [0x3f174c6e, 0x3f17515d, 0x3f174128]),
    ("rand_07", [0x3f569615, 0x3f56b5ad, 0x3f56b9fd]),
    ("rand_08", [0x3f2ed772, 0x3f2edd07, 0x3f2ed94e]),
    ("rand_09", [0x3f0254ab, 0x3f052bf6, 0x3f0509a0]),
    ("rand_10", [0x3ed55556, 0x3e99999a, 0x3e888889]),
    ("rand_11", [0x3f6582fe, 0x3f655556, 0x3f655be4]),
    ("rand_12", [0x3f3b22d1, 0x3f3b45a7, 0x3f3b5410]),
    ("rand_13", [0x3f1269fe, 0x3f12e29f, 0x3f1300c8]),
    ("rand_14", [0x3f142553, 0x3f146fde, 0x3f143f46]),
    ("rand_15", [0x3f41c480, 0x3f41f07c, 0x3f421fa9]),
    ("rand_16", [0x3f6b1aa3, 0x3f6b74f0, 0x3f6b819d]),
    ("rand_17", [0x3f49aa51, 0x3f49d89d, 0x3f4a44a4]),
    ("rand_18", [0x3ea88a76, 0x3ea8fc37, 0x3ea8ad47]),
    ("rand_19", [0x3f610001, 0x3f61642d, 0x3f613549]),
    ("rand_20", [0x3f7002d0, 0x3f7057af, 0x3f707950]),
    ("rand_21", [0x3f3d2c6f, 0x3f3e5be5, 0x3f3da812]),
    ("rand_22", [0x3e5266fb, 0x3e535c04, 0x3e542e18]),
    ("rand_23", [0x3f527d28, 0x3f5b6db7, 0x3f58fd90]),
];

/// `(name, ece)` as printed by the pre-plan tree.
const ECE_SNAPSHOT: &[(&str, u32)] = &[
    ("fixture_underconfident_moderate", 0x3ec5c856),
    ("fixture_overconfident_sharp", 0x3f155d76),
    ("fixture_uniform_uncertain", 0x33000000),
    ("fixture_saturated_top_bin", 0x3eb4ccce),
    ("fixture_mixed_hard", 0x3eb9e63a),
    ("saturated_two_rows", 0x00000000),
    ("rand_00", 0x3e114e4a),
    ("rand_01", 0x3e8e863c),
    ("rand_02", 0x3e8643eb),
    ("rand_03", 0x3e332a6c),
    ("rand_04", 0x3eb7af1e),
    ("rand_05", 0x3e8586d9),
    ("rand_06", 0x3e9a5879),
    ("rand_07", 0x3e1a39d3),
    ("rand_08", 0x3e8a60b2),
    ("rand_09", 0x3e6d2c47),
    ("rand_10", 0x3eddb90e),
    ("rand_11", 0x3e989794),
    ("rand_12", 0x3e3f1284),
    ("rand_13", 0x3e06961a),
    ("rand_14", 0x3e20185d),
    ("rand_15", 0x3e74c43d),
    ("rand_16", 0x3f12cb63),
    ("rand_17", 0x3e7c97fb),
    ("rand_18", 0x3dc72032),
    ("rand_19", 0x3eb40d19),
    ("rand_20", 0x3eafee0c),
    ("rand_21", 0x3e931776),
    ("rand_22", 0x3e48ee9f),
    ("rand_23", 0x3eb52801),
];

#[test]
fn f32_metrics_are_bit_identical_to_the_frozen_snapshot() {
    let f1 = f1_inputs();
    assert_eq!(
        f1.len(),
        F1_SNAPSHOT.len(),
        "f1_score snapshot has {} entries, the input list {}",
        F1_SNAPSHOT.len(),
        f1.len()
    );
    assert!(
        N_RANDOM >= 20,
        "the snapshot must hold at least 20 random inputs"
    );
    for ((name, y_true, y_pred), (want_name, want)) in f1.iter().zip(F1_SNAPSHOT) {
        assert_eq!(name, want_name, "f1_score snapshot order drifted at {name}");
        let got = f1_bits(y_true, y_pred);
        for (mode, (g, w)) in ["Macro", "Micro", "Weighted"]
            .iter()
            .zip(got.iter().zip(want.iter()))
        {
            assert_eq!(
                g,
                w,
                "f1_score({mode}) on '{name}' moved: got {:#010x} ({}), frozen {:#010x} ({})",
                g,
                f32::from_bits(*g),
                w,
                f32::from_bits(*w)
            );
        }
    }

    let ece = ece_inputs();
    assert_eq!(
        ece.len(),
        ECE_SNAPSHOT.len(),
        "top-label ECE snapshot has {} entries, the input list {}",
        ECE_SNAPSHOT.len(),
        ece.len()
    );
    for ((name, probs, k, labels, bins), (want_name, want)) in ece.iter().zip(ECE_SNAPSHOT) {
        assert_eq!(name, want_name, "ECE snapshot order drifted at {name}");
        let got = expected_calibration_error_top_label(probs, *k, labels, *bins).to_bits();
        assert_eq!(
            got,
            *want,
            "expected_calibration_error_top_label on '{name}' moved: got {:#010x} ({}), frozen \
             {:#010x} ({})",
            got,
            f32::from_bits(got),
            want,
            f32::from_bits(*want)
        );
    }
}

/// Prints the table body from the CURRENT tree. Run once, on the pre-plan tree, to freeze it:
/// `cargo test -p aprender-core --lib metrics::f32_bits_tests::print_f32_snapshot -- --ignored --nocapture`.
#[test]
#[ignore = "regenerates the frozen table; only meaningful on the pre-plan tree"]
fn print_f32_snapshot() {
    println!("const F1_SNAPSHOT: &[(&str, [u32; 3])] = &[");
    for (name, y_true, y_pred) in f1_inputs() {
        let [m, u, w] = f1_bits(&y_true, &y_pred);
        println!("    (\"{name}\", [{m:#010x}, {u:#010x}, {w:#010x}]),");
    }
    println!("];");
    println!("const ECE_SNAPSHOT: &[(&str, u32)] = &[");
    for (name, probs, k, labels, bins) in ece_inputs() {
        let bits = expected_calibration_error_top_label(&probs, k, &labels, bins).to_bits();
        println!("    (\"{name}\", {bits:#010x}),");
    }
    println!("];");
}
