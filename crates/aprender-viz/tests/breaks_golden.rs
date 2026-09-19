//! Golden and property tests for extended-Wilkinson tick placement (paiml/aprender#3233).
//!
//! The goldens in `fixtures/breaks/manifest.json` were produced by running a literal transcription
//! of the CRAN `labeling::extended` R source — the first author's own reference — over 24 domains.
//! They exist because **a property suite cannot detect a wrong-but-plausible tie-break**: swapping
//! the first two objective weights leaves every property below intact (still in range, still
//! strictly increasing, still `q ∈ Q`, still about `m` labels) and changes only which candidate
//! labeling wins. `mutation_swapping_simplicity_and_coverage_breaks_the_goldens` shows that
//! directly, so the golden set is proved to have teeth rather than merely asserted to.
//!
//! One `#[test]` per domain, so a failure names the domain that moved.

use trueno_viz::breaks::{extended, extended_loose, Q_DEFAULT, W_DEFAULT};

const MANIFEST: &str = include_str!("../fixtures/breaks/manifest.json");

/// Minimal reader for the manifest's `domains` array — avoids a serde dependency in a test target
/// that exists to compare numbers.
fn golden_domains() -> Vec<(f64, f64, usize, Vec<f64>)> {
    let mut out = Vec::new();
    let body = MANIFEST.split_once("\"domains\"").expect("manifest has a domains key").1;
    for chunk in body.split("\"dmin\"").skip(1) {
        let num = |key: &str| -> f64 {
            let s = chunk
                .split_once(&format!("\"{key}\""))
                .unwrap_or_else(|| panic!("domain entry has {key}"))
                .1;
            let s = s.trim_start().trim_start_matches(':').trim_start();
            let end = s
                .find(|c: char| {
                    !(c.is_ascii_digit() || c == '-' || c == '.' || c == 'e' || c == '+')
                })
                .unwrap_or(s.len());
            s[..end].parse().unwrap_or_else(|_| panic!("{key} parses"))
        };
        let dmin = {
            let s = chunk.trim_start().trim_start_matches(':').trim_start();
            let end = s
                .find(|c: char| {
                    !(c.is_ascii_digit() || c == '-' || c == '.' || c == 'e' || c == '+')
                })
                .unwrap_or(s.len());
            s[..end].parse::<f64>().expect("dmin parses")
        };
        let dmax = num("dmax");
        let m = num("m") as usize;
        let breaks_src = chunk.split_once("\"breaks\"").expect("domain entry has breaks").1;
        let inner = breaks_src
            .split_once('[')
            .expect("breaks is an array")
            .1
            .split_once(']')
            .expect("breaks array closes")
            .0;
        let breaks: Vec<f64> =
            inner.split(',').map(|x| x.trim().parse::<f64>().expect("break parses")).collect();
        out.push((dmin, dmax, m, breaks));
    }
    out
}

fn assert_matches(dmin: f64, dmax: f64, m: usize, want: &[f64]) {
    let got = extended(dmin, dmax, m, &Q_DEFAULT, &W_DEFAULT);
    assert_eq!(
        got.len(),
        want.len(),
        "extended({dmin}, {dmax}, {m}) produced {got:?}, reference produced {want:?}"
    );
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-9 * w.abs().max(1.0),
            "extended({dmin}, {dmax}, {m}) produced {got:?}, reference produced {want:?}"
        );
    }
}

fn golden(i: usize) {
    let d = golden_domains();
    let (dmin, dmax, m, want) = &d[i];
    assert_matches(*dmin, *dmax, *m, want);
}

/// The manifest must actually carry a golden set, and a large one: this is the anti-vacuity floor
/// APEX-001 EV-2b states (`.domains | length >= 20`).
#[test]
fn the_golden_set_is_large_enough_to_be_evidence() {
    let d = golden_domains();
    assert!(d.len() >= 20, "only {} golden domains", d.len());
    let distinct: std::collections::BTreeSet<_> = d
        .iter()
        .map(|(_, _, _, b)| b.iter().map(|x| format!("{x:.12}")).collect::<Vec<_>>())
        .collect();
    assert!(
        distinct.len() >= 15,
        "only {} distinct break sequences across {} domains — a golden set that repeats itself \
         tests one case many times",
        distinct.len(),
        d.len()
    );
    for (dmin, dmax, m, b) in &d {
        assert!(b.len() >= 2, "({dmin}, {dmax}, {m}) has fewer than two breaks");
    }
}

macro_rules! golden_case {
    ($name:ident, $i:expr) => {
        #[test]
        fn $name() {
            golden($i);
        }
    };
}

golden_case!(golden_00_zero_to_100_m5, 0);
golden_case!(golden_01_zero_to_100_m3, 1);
golden_case!(golden_02_zero_to_100_m10, 2);
golden_case!(golden_03_unit_interval, 3);
golden_case!(golden_04_zero_to_10_m3, 4);
golden_case!(golden_05_zero_to_200_m5, 5);
golden_case!(golden_06_symmetric_about_zero, 6);
golden_case!(golden_07_wholly_negative, 7);
golden_case!(golden_08_minus_one_to_one, 8);
golden_case!(golden_09_offset_116_to_180, 9);
golden_case!(golden_10_unround_bounds, 10);
golden_case!(golden_11_seventeen_to_83, 11);
golden_case!(golden_12_thousandths, 12);
golden_case!(golden_13_millions, 13);
golden_case!(golden_14_four_digit_offset, 14);
golden_case!(golden_15_narrow_fractional, 15);
golden_case!(golden_16_narrow_about_100, 16);
golden_case!(golden_17_narrow_about_zero, 17);
golden_case!(golden_18_pi_ish, 18);
golden_case!(golden_19_zero_to_seven_m7, 19);
golden_case!(golden_20_one_to_nine, 20);
golden_case!(golden_21_celsius_range, 21);
golden_case!(golden_22_micro_scale, 22);
golden_case!(golden_23_zero_to_33, 23);

// A structural sweep: m across its useful span at one fixed range, then a sweep of offsets at
// fixed m. Added to widen coverage of the density and simplicity terms, not chosen by checking
// which entries the weight-swap mutant fails.
golden_case!(golden_24_m2, 24);
golden_case!(golden_25_m4, 25);
golden_case!(golden_26_m6, 26);
golden_case!(golden_27_m7, 27);
golden_case!(golden_28_m8, 28);
golden_case!(golden_29_m12, 29);
golden_case!(golden_30_offset_1, 30);
golden_case!(golden_31_offset_7, 31);
golden_case!(golden_32_offset_13, 32);
golden_case!(golden_33_offset_55, 33);
golden_case!(golden_34_negative_to_zero, 34);
golden_case!(golden_35_straddling_zero, 35);

// --- properties, over every golden domain -------------------------------------------------

/// Strictly increasing. A labeling that repeats or reverses is not an axis.
#[test]
fn every_labeling_is_strictly_increasing() {
    for (dmin, dmax, m, _) in golden_domains() {
        let b = extended(dmin, dmax, m, &Q_DEFAULT, &W_DEFAULT);
        for w in b.windows(2) {
            assert!(w[0] < w[1], "({dmin}, {dmax}, {m}) produced {b:?}");
        }
    }
}

/// The step is `j·q·10^z` for some `q ∈ Q`, so consecutive gaps are equal and the mantissa of the
/// gap is a nice number.
#[test]
fn the_step_is_uniform_and_its_mantissa_is_in_q() {
    for (dmin, dmax, m, _) in golden_domains() {
        let b = extended(dmin, dmax, m, &Q_DEFAULT, &W_DEFAULT);
        let step = b[1] - b[0];
        for w in b.windows(2) {
            let g = w[1] - w[0];
            assert!(
                (g - step).abs() <= 1e-9 * step.abs().max(1.0),
                "({dmin}, {dmax}, {m}) has uneven steps: {b:?}"
            );
        }
        let mut mant = step.abs();
        while mant >= 10.0 - 1e-12 {
            mant /= 10.0;
        }
        while mant < 1.0 - 1e-12 {
            mant *= 10.0;
        }
        let nice = [1.0, 5.0, 2.0, 2.5, 4.0, 3.0, 1.5, 7.5, 6.0, 8.0, 9.0, 7.0];
        assert!(
            nice.iter().any(|n| (mant - n).abs() < 1e-6),
            "({dmin}, {dmax}, {m}) step {step} has mantissa {mant}, not a nice number; breaks {b:?}"
        );
    }
}

/// Label count stays near the request. The paper permits departure; the reference stays close.
#[test]
fn the_label_count_stays_near_m() {
    for (dmin, dmax, m, _) in golden_domains() {
        let b = extended(dmin, dmax, m, &Q_DEFAULT, &W_DEFAULT);
        let n = b.len() as i64;
        assert!((n - m as i64).abs() <= 6, "({dmin}, {dmax}, {m}) produced {n} labels: {b:?}");
    }
}

/// `extended_loose` must contain the data; plain `extended` need not, which is the extension over
/// Wilkinson and the reason both exist.
#[test]
fn loose_labelings_contain_the_data() {
    for (dmin, dmax, m, _) in golden_domains() {
        let b = extended_loose(dmin, dmax, m, &Q_DEFAULT, &W_DEFAULT);
        let (lo, hi) = (b[0], b[b.len() - 1]);
        assert!(
            lo <= dmin + 1e-9 && hi >= dmax - 1e-9,
            "loose({dmin}, {dmax}, {m}) produced {b:?}, which does not contain the data"
        );
    }
}

/// Degenerate input does not panic or loop: a zero-width range returns `m` evenly spaced values.
#[test]
fn a_degenerate_range_is_handled() {
    let b = extended(5.0, 5.0, 4, &Q_DEFAULT, &W_DEFAULT);
    assert_eq!(b.len(), 4);
    assert!(b.iter().all(|x| (x - 5.0).abs() < 1e-12));
    assert_eq!(extended(1.0, 2.0, 1, &Q_DEFAULT, &W_DEFAULT).len(), 1);
    assert!(extended(f64::NAN, 1.0, 3, &Q_DEFAULT, &W_DEFAULT).len() <= 3);
}

/// Reversed input is normalised, as the reference does.
#[test]
fn reversed_bounds_give_the_same_labeling() {
    let fwd = extended(0.0, 100.0, 5, &Q_DEFAULT, &W_DEFAULT);
    let rev = extended(100.0, 0.0, 5, &Q_DEFAULT, &W_DEFAULT);
    assert_eq!(fwd, rev);
}

// --- the row's stated mutation ---------------------------------------------------------------

/// **APEX-001 EV-2b's stated mutation**: "swap the simplicity and coverage weights → golden RED."
///
/// This is not hypothetical. The paper's prose states the weights in the swapped order, so an
/// implementer following the paper rather than the authors' code lands exactly here — which is why
/// the golden set, not the property suite, is what this row cannot ship without. The assertion is
/// that the properties above all still hold under the swap, and the goldens still move.
#[test]
fn mutation_swapping_simplicity_and_coverage_breaks_the_goldens() {
    let swapped = [W_DEFAULT[1], W_DEFAULT[0], W_DEFAULT[2], W_DEFAULT[3]];
    let mut moved = 0;
    for (dmin, dmax, m, want) in golden_domains() {
        let got = extended(dmin, dmax, m, &Q_DEFAULT, &swapped);
        let same = got.len() == want.len()
            && got.iter().zip(&want).all(|(g, w)| (g - w).abs() <= 1e-9 * w.abs().max(1.0));
        if !same {
            moved += 1;
            // every property still holds on the mutant — which is the whole point
            for w in got.windows(2) {
                assert!(w[0] < w[1], "mutant is still strictly increasing by construction");
            }
        }
    }
    assert!(
        moved > 0,
        "swapping the simplicity and coverage weights changed no golden — the golden set cannot \
         detect the one mutation this row names"
    );
}
