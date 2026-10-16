use super::*;

/// The pilot's item count: the dev split (~100) plus 30 fixed test items.
const N: usize = 130;

/// Approximate N(0,1) from twelve uniforms, on the frozen generator.
fn gauss(rng: &mut SplitMix64) -> f64 {
    (0..12)
        .map(|_| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
        .sum::<f64>()
        - 6.0
}

/// Item difficulty: log-normal wall time around 8 s with σ = 0.6, which is
/// wide (diff sizes vary by an order of magnitude); fixed across tags.
fn base() -> Vec<f64> {
    let mut rng = SplitMix64::new(1);
    (0..N)
        .map(|_| (8000f64.ln() + 0.6 * gauss(&mut rng)).exp())
        .collect()
}

/// One tag's run: every item × `scale`, with 3 % run-to-run noise.
fn tag(name: &str, seed: u64, scale: f64) -> Entry {
    let mut rng = SplitMix64::new(seed);
    let ids: Vec<String> = (0..N).map(|i| format!("item-{i:03}")).collect();
    let ms: Vec<f64> = base()
        .iter()
        .map(|b| b * scale * (1.0 + 0.03 * gauss(&mut rng)))
        .collect();
    let s: Vec<Sample<'_>> = ids
        .iter()
        .zip(&ms)
        .map(|(id, &wall_ms)| Sample {
            item_id: id,
            apr_sha256: "aa",
            wall_ms,
        })
        .collect();
    record(name, "C4", &s, &[], "rr").expect("records")
}

fn baseline() -> Vec<Entry> {
    vec![
        tag("v0.70.0", 10, 1.0),
        tag("v0.70.1", 11, 1.0),
        tag("v0.70.2", 12, 1.0),
    ]
}

#[test]
fn falsify_rxr_001_a_planted_ten_percent_p95_regression_is_red() {
    let mut e = baseline();
    assert_eq!(
        check(&e).andon,
        Andon::Arming,
        "3 tags only arm the ratchet"
    );
    e.push(tag("v0.70.3", 13, 1.0));
    let same = check(&e);
    assert_eq!(same.andon, Andon::Green, "{same:?}");
    e[3] = tag("v0.70.3", 13, 1.10);
    let v = check(&e);
    assert_eq!(v.andon, Andon::Red, "{v:?}");
    let c = &v.comparisons[0];
    assert!(c.regression && c.p95_ratio_ci[0] > 1.0 && c.paired_items == N);
}

#[test]
fn falsify_rxr_001_a_gain_is_kept_the_reference_is_the_best_tag() {
    let mut e = baseline();
    e[2] = tag("v0.70.2", 12, 0.80);
    e.push(tag("v0.70.3", 13, 1.0));
    let v = check(&e);
    assert_eq!(v.andon, Andon::Red, "giving back a 20 % gain is a rise");
    assert_eq!(v.comparisons[0].reference, "v0.70.2");
}

/// The finding behind the paired rule: at the pilot's n, the two unpaired
/// p95 CIs of a +10 % tag still overlap, so the §4 overlap rule calls it a
/// tie. The ratchet could not meet its own acceptance on that rule.
#[test]
fn falsify_rxr_002_the_unpaired_overlap_rule_misses_ten_percent() {
    let a = tag("v0.70.2", 12, 1.0);
    let b = tag("v0.70.3", 13, 1.10);
    assert!(b.p95_ms > a.p95_ms);
    assert!(
        b.p95_ci_ms[0] <= a.p95_ci_ms[1],
        "unpaired CIs {:?} vs {:?} unexpectedly separate",
        a.p95_ci_ms,
        b.p95_ci_ms
    );
}

#[test]
fn falsify_rxr_003_the_ratchet_fails_closed() {
    let mut dup = baseline();
    dup.push(tag("v0.70.1", 13, 1.0));
    assert_eq!(check(&dup).andon, Andon::Red, "duplicate tag");

    let mut foreign = baseline();
    foreign[1].cell = "C1".into();
    assert_eq!(check(&foreign).andon, Andon::Red, "foreign cell");

    let mut disjoint = baseline();
    let mut t = tag("v0.70.3", 13, 1.0);
    t.items = t
        .items
        .into_iter()
        .map(|(k, v)| (format!("x{k}"), v))
        .collect();
    disjoint.push(t);
    let v = check(&disjoint);
    assert_eq!(v.andon, Andon::Red, "no common items");
    assert!(v.violations.iter().any(|m| m.contains("no common items")));

    let ok = Sample {
        item_id: "a",
        apr_sha256: "aa",
        wall_ms: 1.0,
    };
    assert!(
        record("t", "C4", &[], &[], "r").is_err(),
        "a tag that did not run"
    );
    let other = Sample {
        apr_sha256: "bb",
        ..ok.clone()
    };
    assert!(
        record("t", "C4", &[ok.clone(), other], &[], "r").is_err(),
        "two binaries"
    );
    let nan = Sample {
        wall_ms: f64::NAN,
        ..ok
    };
    assert!(
        record("t", "C4", &[nan], &[], "r").is_err(),
        "a non-finite timing"
    );
}

#[test]
fn falsify_rxr_003_a_growing_llama_cpp_gap_is_noted() {
    let mut e = baseline();
    e[0].llama_cpp_p95_ms = Some(e[0].p95_ms);
    e[1].llama_cpp_p95_ms = Some(e[1].p95_ms / 1.2);
    let v = check(&e);
    assert!(
        v.notes.iter().any(|n| n.contains("gap to llama.cpp grew")),
        "{v:?}"
    );
}

#[test]
fn an_item_timed_twice_counts_its_median() {
    let s = |ms| Sample {
        item_id: "a",
        apr_sha256: "aa",
        wall_ms: ms,
    };
    let e = record("t", "C4", &[s(1.0), s(9.0), s(2.0)], &[], "r").expect("records");
    assert_eq!(e.items["a"], 2.0);
}

#[test]
fn harrell_davis_weights_are_a_distribution_centred_on_the_tail() {
    let w = harrell_davis_weights(N, 0.95);
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(w.iter().all(|x| *x >= 0.0));
    let x: Vec<f64> = (1..=N).map(|i| i as f64).collect();
    let q: f64 = x.iter().zip(&w).map(|(x, w)| x * w).sum();
    assert!(
        (q - 0.95 * (N as f64 + 1.0)).abs() < 1.0,
        "HD p95 of 1..N = {q}"
    );
}

/// Rule 6: one seed is an anecdote. Over 10 independent runs of the planted
/// tag the ratchet catches +10 % every time and seldom alarms on an unchanged
/// tag (40/40 caught and 0/40 alarms when first measured over 40 runs, at the
/// modelled 3 % run-to-run noise per item).
#[test]
fn falsify_rxr_001_power_and_false_alarms_over_ten_runs() {
    let e = baseline();
    let red = |scale: f64| {
        (100..110)
            .filter(|&s| {
                let mut t = e.clone();
                t.push(tag("v0.70.3", s, scale));
                check(&t).andon == Andon::Red
            })
            .count()
    };
    let (caught, false_alarms) = (red(1.10), red(1.0));
    eprintln!("RXR power: +10% caught {caught}/10, unchanged red {false_alarms}/10");
    assert_eq!(caught, 10);
    assert!(false_alarms <= 1, "{false_alarms}/10 false alarms");
}
