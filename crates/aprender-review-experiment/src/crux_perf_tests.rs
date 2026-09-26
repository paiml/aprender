use super::*;

const TAG_SHA: &str = "aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11aa11";

fn cell() -> Cell {
    Cell {
        host: "gx10".to_string(),
        gpu: "GB10".to_string(),
        driver_cuda: "580.95/13.0".to_string(),
        model: "Qwen3.5-4B".to_string(),
        gguf_sha256: "00fe7986".to_string(),
        quant: "Q4_K_M".to_string(),
        ctx: 36864,
        batch: 1,
        prompt_set_sha: "a4910a65".to_string(),
    }
}

/// Samples scaled by `k`: the same shape, so medians scale exactly by `k`.
fn samples(base: f64, k: f64) -> Vec<f64> {
    [0.97, 0.99, 1.0, 1.0, 1.01, 1.02, 1.03]
        .iter()
        .map(|x| x * base * k)
        .collect()
}

fn phases(k_decode: f64) -> Phases {
    Phases::from_samples(
        &samples(1900.0, 1.0),
        &samples(0.26, 1.0),
        &samples(15.7 * k_decode, 1.0),
        7,
    )
    .expect("seven finite positive samples per phase")
}

fn receipt(tag: &str, k_decode: f64) -> CruxPerfReceipt {
    let apr = phases(k_decode);
    let competitor_phases = Phases::from_samples(
        &samples(1900.0, 1.0),
        &samples(0.25, 1.0),
        &samples(15.7, 1.0),
        7,
    )
    .expect("competitor samples");
    CruxPerfReceipt::new(
        tag.to_string(),
        TAG_SHA.to_string(),
        cell(),
        Competitor {
            name: "llama.cpp".to_string(),
            version: Some("d1d3c3396".to_string()),
            binary_sha256: "bb22".to_string(),
        },
        apr,
        competitor_phases,
    )
}

#[test]
fn falsify_crux_perf_001_missing_competitor_version_does_not_hold() {
    let r = receipt("v0.70.0-rc.2", 1.0);
    assert!(r.holds(TAG_SHA), "control: a complete receipt holds");
    for v in [None, Some(String::new()), Some("  ".to_string())] {
        let mut bad = r.clone();
        bad.competitor.version = v;
        assert!(!bad.holds(TAG_SHA));
        assert_eq!(bad.check(TAG_SHA), Err(Refusal::CompetitorVersionMissing));
    }
}

#[test]
fn falsify_crux_perf_002_planted_ten_percent_decode_is_red() {
    let prev = receipt("v0.70.0-rc.1", 1.0);
    let released = receipt("v0.69.3", 1.0);
    let hist = [
        receipt("v0.69.1", 1.0),
        receipt("v0.69.2", 1.0),
        receipt("v0.69.3", 1.0),
    ];
    let hist: Vec<&CruxPerfReceipt> = hist.iter().collect();

    // Control: the same numbers again are GREEN.
    let same = receipt("v0.70.0-rc.2", 1.0);
    let g = gate(&same, Some(&prev), Some(&released), &hist).expect("one series");
    assert!(!g.red(), "control is green: {g:?}");

    // Planted: decode 10% slower per token, everything else identical.
    let slow = receipt("v0.70.0-rc.2", 1.10);
    let g = gate(&slow, Some(&prev), Some(&released), &hist).expect("one series");
    assert!(g.red(), "a +10% decode must be RED: {g:?}");
    assert!(
        g.reasons.iter().all(|r| r.phase == Phase::Decode),
        "only decode moved: {g:?}"
    );
    assert!(g.reasons.iter().any(|r| r.rule == Rule::Released));
    assert!(g.reasons.iter().any(|r| r.rule == Rule::Rolling3));

    // A faster decode is not a regression.
    let fast = receipt("v0.70.0-rc.2", 0.80);
    let g = gate(&fast, Some(&prev), Some(&released), &hist).expect("one series");
    assert!(!g.red(), "faster is green: {g:?}");

    // D_prev fires only when the whole 95% CI is past +10%.
    let far = receipt("v0.70.0-rc.2", 1.25);
    let g = gate(&far, Some(&prev), None, &[]).expect("one series");
    assert!(g.reasons.iter().any(|r| r.rule == Rule::Prev), "{g:?}");
    let near = receipt("v0.70.0-rc.2", 1.10);
    let g = gate(&near, Some(&prev), None, &[]).expect("one series");
    assert!(!g.red(), "+10% is inside the D_prev CI band: {g:?}");
}

#[test]
fn falsify_crux_perf_003_changed_cell_field_under_same_id_is_refused() {
    let a = receipt("v0.70.0-rc.1", 1.0);
    assert_eq!(a.cell_id, a.cell.id(), "the id is derived from the fields");
    let mut b = receipt("v0.70.0-rc.2", 1.0);
    b.cell.driver_cuda = "590.10/13.1".to_string();
    // The claimed id is kept; the fields moved.
    assert_eq!(a.cell_id, b.cell_id);
    assert!(matches!(
        b.check(TAG_SHA),
        Err(Refusal::CellIdMismatch { .. })
    ));
    assert!(matches!(
        gate(&b, Some(&a), None, &[]),
        Err(Refusal::CellIdMismatch { .. })
    ));
    // Every field is in the id.
    let base = cell().id();
    let mut moved = Vec::new();
    for f in 0..9 {
        let mut c = cell();
        match f {
            0 => c.host.push('x'),
            1 => c.gpu.push('x'),
            2 => c.driver_cuda.push('x'),
            3 => c.model.push('x'),
            4 => c.gguf_sha256.push('x'),
            5 => c.quant.push('x'),
            6 => c.ctx += 1,
            7 => c.batch += 1,
            _ => c.prompt_set_sha.push('x'),
        }
        moved.push(c.id() != base);
    }
    assert!(moved.iter().all(|m| *m), "{moved:?}");
    // A different cell is a different series, never a comparison.
    let other = CruxPerfReceipt::new(
        "v0.70.0-rc.2".to_string(),
        TAG_SHA.to_string(),
        Cell {
            host: "intel".to_string(),
            ..cell()
        },
        a.competitor.clone(),
        a.apr.clone(),
        a.competitor_phases.clone(),
    );
    assert!(matches!(
        gate(&other, Some(&a), None, &[]),
        Err(Refusal::NewSeries { .. })
    ));
}

#[test]
fn falsify_crux_perf_004_t2_without_trace_overhead_is_refused() {
    let mut r = receipt("v0.70.0-rc.2", 1.0);
    assert_eq!(r.t2_status(), T2Status::Absent);
    r.t2_blob_sha = Some("cc33".to_string());
    assert_eq!(r.t2_status(), T2Status::Refused);
    r.trace_overhead_pct = Some(f64::NAN);
    assert_eq!(r.t2_status(), T2Status::Refused);
    r.trace_overhead_pct = Some(7.5);
    assert_eq!(r.t2_status(), T2Status::AttributionOnly);
    r.trace_overhead_pct = Some(5.0);
    assert_eq!(r.t2_status(), T2Status::Diffable);
    // A refused T2 does not void T0: the receipt still holds.
    r.trace_overhead_pct = None;
    assert!(r.holds(TAG_SHA));
}

#[test]
fn falsify_crux_perf_005_binary_sha_not_the_tag_asset_is_refused() {
    let r = receipt("v0.70.0-rc.2", 1.0);
    assert!(r.holds(TAG_SHA));
    let other = "ff".repeat(32);
    assert!(matches!(
        r.check(&other),
        Err(Refusal::BinaryShaMismatch { .. })
    ));
    // A tag with no published asset sha cannot certify anything.
    assert!(matches!(
        r.check(""),
        Err(Refusal::BinaryShaMismatch { .. })
    ));
    // Case is not a difference.
    assert!(r.holds(&TAG_SHA.to_uppercase()));
}

#[test]
fn receipt_is_self_consistent_and_round_trips() {
    let r = receipt("v0.70.0-rc.2", 1.0);
    assert_eq!(r.schema_version, SCHEMA_VERSION);
    assert_eq!(r.prompt_set_sha, "a4910a65");
    assert_eq!(r.n_runs, 7);
    // apr/competitor, per phase, of the medians.
    let want = r.apr.prefill.median / r.competitor_phases.prefill.median;
    assert!((r.ratio_vs_competitor.prefill - want).abs() < 1e-12);
    let s = serde_json::to_string(&r).expect("serialize");
    let back: CruxPerfReceipt = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(back, r);
    // A tampered ratio is refused.
    let mut bad = r.clone();
    bad.ratio_vs_competitor.decode = 0.5;
    assert_eq!(
        bad.check(TAG_SHA),
        Err(Refusal::RatioInconsistent(Phase::Decode))
    );
    // A claimed run count that is not the samples' is refused.
    let mut bad = r.clone();
    bad.n_runs = 200;
    assert_eq!(bad.check(TAG_SHA), Err(Refusal::TooFewRuns(7)));
    // Fewer than MIN_RUNS is never a gate number.
    assert!(PhaseStats::from_samples(&[1.0, 2.0, 3.0, 4.0], 1).is_none());
    assert!(PhaseStats::from_samples(&[1.0, 2.0, 3.0, 4.0, f64::NAN], 1).is_none());
    let p = PhaseStats::from_samples(&samples(10.0, 1.0), 1).expect("seven samples");
    assert!(p.ci_lo <= p.median && p.median <= p.ci_hi && p.median <= p.p95);
}

fn row(engine: &str, i: u32, prompt_ms: Option<f64>) -> crate::replay::Row {
    let prompt_ms = prompt_ms.map_or_else(|| "null".to_string(), |m| m.to_string());
    serde_json::from_str(&format!(
        r#"{{"schema":"review-replay-receipt-v1","replay_version":"review-replay-v2",
        "set_sha":"a4910a65","diff_sha256":"d{i}","stratum":"2k","engine":"{engine}",
        "engine_version":"x","apr_tag":"t","cell":"gx10-cuda","gguf_sha256":"00fe7986",
        "wall_ms":1000.0,"prompt_ms":{prompt_ms},"prompt_tps":4000.0,"decode_tps":64.0,
        "input_tokens":1000,"output_tokens":100,"peak_rss_mb":8000.0,
        "verdict":{{"state":"Fail"}}}}"#
    ))
    .expect("a replay row")
}

#[test]
fn replay_rows_give_phases_only_when_every_item_was_timed() {
    use crate::replay::Engine;
    let mut rows: Vec<_> = (0..6).map(|i| row("apr", i, Some(250.0))).collect();
    rows.extend((0..6).map(|i| row("llama_cpp", i, Some(240.0))));
    let p = Phases::from_replay_rows(&rows, Engine::Apr, 1).expect("six timed rows");
    assert_eq!(p.ttft.median, 250.0);
    assert!(
        (p.prefill.median - 0.25).abs() < 1e-12,
        "ms/token = 1000/tps"
    );
    assert!((p.decode.median - 15.625).abs() < 1e-12);
    assert_eq!(p.decode.n, 6);
    // One untimed apr item (apr serve before SRV-TIM-001) voids the engine's figure.
    rows[3].prompt_ms = None;
    assert_eq!(
        Phases::from_replay_rows(&rows, Engine::Apr, 1),
        Err(RowsError::UntimedRow("d3".to_string()))
    );
    assert!(Phases::from_replay_rows(&rows, Engine::LlamaCpp, 1).is_ok());
    rows[7].set_sha = "other".to_string();
    assert_eq!(
        Phases::from_replay_rows(&rows, Engine::LlamaCpp, 1),
        Err(RowsError::MixedSets)
    );
    assert_eq!(
        Phases::from_replay_rows(&rows[..0], Engine::Apr, 1),
        Err(RowsError::NoRows)
    );
}

#[test]
fn falsify_crux_perf_006_rc_missing_t0_or_t1_for_an_admitted_cell_is_red() {
    let tag = "v0.70.0-rc.2";
    let gx10 = cell().id();
    let intel = Cell {
        host: "intel".to_string(),
        ..cell()
    }
    .id();
    let admitted = vec![gx10.clone(), intel.clone()];
    let mut a = receipt(tag, 1.0);
    a.t1_topk_sha = Some("dd44".to_string());
    // Only gx10 measured: intel has no T0.
    assert_eq!(
        coverage_gaps(tag, TAG_SHA, &admitted, &[a.clone()]),
        vec![Gap::T0 {
            cell_id: intel.clone()
        }]
    );
    let mut b = a.clone();
    b.cell = Cell {
        host: "intel".to_string(),
        ..cell()
    };
    b.cell_id = b.cell.id();
    b.t1_topk_sha = None;
    // Both measured, intel without its T1 profile.
    assert_eq!(
        coverage_gaps(tag, TAG_SHA, &admitted, &[a.clone(), b.clone()]),
        vec![Gap::T1 {
            cell_id: intel.clone()
        }]
    );
    b.t1_topk_sha = Some("ee55".to_string());
    assert!(coverage_gaps(tag, TAG_SHA, &admitted, &[a.clone(), b.clone()]).is_empty());
    // Another tag's receipt, or a receipt of another binary, fills nothing.
    assert_eq!(
        coverage_gaps("v0.70.0-rc.3", TAG_SHA, &admitted, &[a.clone(), b.clone()]).len(),
        2
    );
    assert_eq!(
        coverage_gaps(tag, &"ff".repeat(32), &admitted, &[a, b]).len(),
        2
    );
}
