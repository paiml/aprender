use super::*;

/// One agent-trace-v1 row, as much of it as the probe reads.
fn row(lane: &str, key: u32, verdict: &str, outcome: &str) -> String {
    row_with(lane, key, verdict, outcome, "gold", "val", true)
}

fn row_with(
    lane: &str,
    key: u32,
    verdict: &str,
    outcome: &str,
    tier: &str,
    split: &str,
    matured: bool,
) -> String {
    let counted = lane != "qwen-shadow";
    let matured = if matured {
        r#""2026-09-01T00:00:00Z""#
    } else {
        "null"
    };
    format!(
        r#"{{"schema":"agent-trace-v1","quorum_id":"q{key}","round":1,"lane":"{lane}","counted":{counted},
        "lane_state":"Verdict","verdict":"{verdict}","parse_status":"ok","label_tier":"{tier}",
        "outcome":"{outcome}","outcome_matured_at":{matured},"split_guard":{{"split":"{split}"}},"trace_status":"ok"}}"#
    )
    .replace('\n', "")
}

/// `n` items; `defects` of them regressed. Each lane's verdict per item comes from `flag(i)`.
fn lane_rows(
    lane: &str,
    n: u32,
    defect: impl Fn(u32) -> bool,
    flag: impl Fn(u32) -> bool,
) -> Vec<String> {
    (0..n)
        .map(|i| {
            let v = if flag(i) {
                "request_changes"
            } else {
                "approve"
            };
            let o = if defect(i) {
                "reverted_le14d"
            } else {
                "merged"
            };
            row(lane, i, v, o)
        })
        .collect()
}

fn board(lines: &[String], min_n: usize) -> Scoreboard {
    let rows = parse_rows(&lines.join("\n")).expect("fixture parses");
    probe(&rows, "val", "qwen-shadow", min_n)
}

fn pair<'a>(b: &'a Scoreboard, lane: &str) -> &'a PairKappa {
    b.pairs
        .iter()
        .find(|p| p.counted_lane == lane)
        .expect("pair present")
}

/// Qwen and sonnet share half their errors: a known, non-degenerate κ.
fn base_fixture() -> Vec<String> {
    let defect = |i: u32| i % 2 == 0;
    let mut v = lane_rows("sonnet", 40, defect, |i| (i % 2 == 0) != (i % 5 == 0));
    v.extend(lane_rows("qwen-shadow", 40, defect, |i| {
        (i % 2 == 0) != (i % 5 == 0 || i % 7 == 0)
    }));
    v
}

#[test]
fn falsify_lind_001_only_matured_gold_val_rows_are_scored() {
    let clean = board(&base_fixture(), 10);
    let k = pair(&clean, "sonnet").kappa_err.expect("defined");
    // Planted noise: every lane gets rows that would be pure agreement-errors if counted.
    let mut noisy = base_fixture();
    for (i, (tier, split, matured, outcome)) in [
        ("silver", "val", true, "reverted_le14d"),
        ("pending", "val", true, "reverted_le14d"),
        ("quarantined", "val", true, "reverted_le14d"),
        ("gold", "train", true, "reverted_le14d"),
        ("gold", "test", true, "reverted_le14d"),
        ("gold", "val", false, "reverted_le14d"),
        ("gold", "val", true, "pending"),
        ("gold", "val", true, "closed_unmerged"),
    ]
    .into_iter()
    .enumerate()
    {
        let key = 1000 + i as u32;
        for lane in ["sonnet", "qwen-shadow"] {
            noisy.push(row_with(
                lane, key, "approve", outcome, tier, split, matured,
            ));
        }
    }
    // Unparsed / non-verdict rows are excluded, never counted as errors.
    noisy.push(row("sonnet", 2000, "abstain", "reverted_le14d"));
    noisy.push(row("qwen-shadow", 2000, "approve", "reverted_le14d"));
    let n = board(&noisy, 10);
    assert_eq!(
        pair(&n, "sonnet").n,
        pair(&clean, "sonnet").n,
        "noise rows were scored"
    );
    assert_eq!(pair(&n, "sonnet").kappa_err, Some(k));
    // And the κ is the frozen stats::error_kappa on the error vectors, not a second formula.
    let (a, b): (Vec<bool>, Vec<bool>) = (0..40u32)
        .map(|i| (i % 5 == 0 || i % 7 == 0, i % 5 == 0))
        .unzip();
    assert_eq!(Some(k), crate::stats::error_kappa(&a, &b));
}

#[test]
fn falsify_lind_002_thin_or_undefined_pairs_are_insufficient() {
    let thin = board(&base_fixture(), 41);
    let p = pair(&thin, "sonnet");
    assert_eq!(p.status, PairStatus::Insufficient);
    assert_eq!(p.kappa_err, None, "a thin pair must not print a number");
    // Both lanes always right → constant error vector → κ undefined.
    let defect = |i: u32| i % 2 == 0;
    let mut v = lane_rows("sonnet", 40, defect, defect);
    v.extend(lane_rows("qwen-shadow", 40, defect, defect));
    let u = board(&v, 10);
    assert_eq!(pair(&u, "sonnet").status, PairStatus::Insufficient);
    assert_eq!(pair(&u, "sonnet").kappa_err, None);
    // The gate refuses on insufficient data (S-14), it never passes by default.
    let g = gate(&thin, &thin, &manifest(0.05)).expect("manifest ok");
    assert!(!g.pass, "insufficient data passed the gate");
    assert!(
        g.refusals.iter().any(|r| r.contains("S-14")),
        "{:?}",
        g.refusals
    );
    // The manifest's min_n binds even when the board was scored with a lower floor.
    let loose = board(&base_fixture(), 10);
    let mut m = manifest(1.0);
    m.min_n = 41;
    let g = gate(&loose, &loose, &m).expect("manifest ok");
    assert!(
        !g.pass,
        "the gate took a board scored below the manifest's min_n"
    );
}

fn manifest(delta: f64) -> Manifest {
    Manifest {
        delta,
        min_n: 10,
        counted_lanes: vec!["sonnet".into()],
        registered_at: "2026-09-20T00:00:00Z".into(),
        training_started_at: "2026-09-21T00:00:00Z".into(),
    }
}

#[test]
fn falsify_lind_003_a_kappa_rise_past_delta_refuses() {
    let base = board(&base_fixture(), 10);
    let k0 = pair(&base, "sonnet").kappa_err.expect("defined");
    // Candidate copies sonnet's errors exactly: κ_err = 1.
    let defect = |i: u32| i % 2 == 0;
    let mut v = lane_rows("sonnet", 40, defect, |i| (i % 2 == 0) != (i % 5 == 0));
    v.extend(lane_rows("qwen-shadow", 40, defect, |i| {
        (i % 2 == 0) != (i % 5 == 0)
    }));
    let copy = board(&v, 10);
    assert_eq!(pair(&copy, "sonnet").kappa_err, Some(1.0));
    let g = gate(&base, &copy, &manifest(0.05)).expect("manifest ok");
    assert!(!g.pass, "a voter-copying candidate passed");
    assert!(
        g.refusals.iter().any(|r| r.contains("sonnet")),
        "{:?}",
        g.refusals
    );
    // Exactly baseline + δ passes (the gate is strict >).
    let edge = gate(&base, &copy, &manifest(1.0 - k0)).expect("manifest ok");
    assert!(edge.pass, "{:?}", edge.refusals);
    // The unchanged candidate passes with δ = 0.
    assert!(gate(&base, &base, &manifest(0.0)).expect("ok").pass);
    // A counted lane the manifest names but the boards lack is a refusal, not a skip.
    let mut m = manifest(0.5);
    m.counted_lanes.push("haiku".into());
    let g = gate(&base, &base, &m).expect("ok");
    assert!(
        !g.pass && g.refusals.iter().any(|r| r.contains("haiku")),
        "{:?}",
        g.refusals
    );
}

#[test]
fn falsify_lind_004_delta_must_be_pre_registered() {
    let ok = r#"{"lane_independence":{"delta":0.05,"min_n":30,"counted_lanes":["sonnet","haiku","agy"],
        "registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#;
    let m = parse_manifest(ok).expect("valid manifest");
    assert_eq!((m.delta, m.min_n, m.counted_lanes.len()), (0.05, 30, 3));
    for bad in [
        "{}",
        r#"{"lane_independence":{}}"#,
        r#"{"lane_independence":{"min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        r#"{"lane_independence":{"delta":-0.01,"min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        r#"{"lane_independence":{"delta":"NaN","min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        r#"{"lane_independence":{"delta":0.05,"min_n":0,"counted_lanes":["sonnet"],"registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        r#"{"lane_independence":{"delta":0.05,"min_n":30,"counted_lanes":[],"registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        // registered AFTER training started: δ could have been picked after seeing κ
        r#"{"lane_independence":{"delta":0.05,"min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-22T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        // same instant is not "before"
        r#"{"lane_independence":{"delta":0.05,"min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-21T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        // a timestamp that does not compare lexicographically (offset, not Z)
        r#"{"lane_independence":{"delta":0.05,"min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-20T00:00:00+02:00","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        "not json",
    ] {
        assert!(parse_manifest(bad).is_err(), "accepted: {bad}");
    }
}
