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
    // The gate refuses an undefined κ (S-14), it never passes by default.
    v.extend(lane_rows("haiku", 40, defect, defect));
    let g = run_gate(&[v], &manifest(&["sonnet", "haiku"], 10));
    assert!(
        !g.pass && g.refusals.iter().any(|r| r.contains("S-14")),
        "{:?}",
        g.refusals
    );
}

fn manifest(voters: &[&str], min_n: usize) -> Manifest {
    Manifest {
        min_n,
        counted_lanes: voters.iter().map(|v| (*v).to_string()).collect(),
        registered_at: "2026-09-20T00:00:00Z".into(),
        training_started_at: "2026-09-21T00:00:00Z".into(),
    }
}

/// `n` items, half regressed; `lane` errs exactly on the items where `err(i)`.
fn errs(lane: &str, n: u32, err: impl Fn(u32) -> bool) -> Vec<String> {
    let defect = |i: u32| i % 2 == 0;
    lane_rows(lane, n, defect, move |i| defect(i) != err(i))
}

fn run_gate(lines: &[Vec<String>], m: &Manifest) -> GateVerdict {
    let rows = parse_rows(&lines.concat().join("\n")).expect("fixture parses");
    gate(&rows, "val", "qwen-shadow", m)
}

fn five(i: u32) -> bool {
    i % 5 == 0
}

/// Sonnet and haiku share most errors; qwen errs elsewhere.
fn voters() -> Vec<Vec<String>> {
    vec![
        errs("sonnet", 40, five),
        errs("haiku", 40, |i| five(i) || i % 11 == 0),
    ]
}

#[test]
fn falsify_lind_003_h7_is_qwen_kappa_at_most_max_voter_kappa() {
    let m = manifest(&["sonnet", "haiku"], 10);
    let mut v = voters();
    v.push(errs("qwen-shadow", 40, |i| i % 3 == 0));
    let g = run_gate(&v, &m);
    assert!(g.pass, "{:?}", g.refusals);
    assert_eq!(g.n, 40);
    let c = g.ceiling.expect("defined");
    assert!(g
        .qwen_vs_voter
        .iter()
        .all(|p| p.kappa_err.expect("defined") <= c));
    // A qwen that copies sonnet's blind spots: κ = 1 > the voter–voter max.
    let mut v = voters();
    v.push(errs("qwen-shadow", 40, five));
    let g = run_gate(&v, &m);
    assert!(c < 1.0 && !g.pass, "a voter-copying qwen passed");
    assert!(
        g.refusals.iter().any(|r| r.contains("sonnet")),
        "{:?}",
        g.refusals
    );
    // Inclusive: identical voters put the ceiling at 1, so the same copy passes.
    let v = vec![
        errs("sonnet", 40, five),
        errs("haiku", 40, five),
        errs("qwen-shadow", 40, five),
    ];
    let g = run_gate(&v, &m);
    assert_eq!(g.ceiling, Some(1.0));
    assert!(g.pass, "κ = ceiling must pass (≤): {:?}", g.refusals);
    // The ceiling is the MAX over every voter pair, not the first or the least.
    let mut v = voters();
    v.push(errs("agy", 40, |i| i % 3 == 0));
    v.push(errs("qwen-shadow", 40, |i| i % 4 == 0));
    let g = run_gate(&v, &manifest(&["sonnet", "haiku", "agy"], 10));
    let vv: Vec<f64> = g
        .voter_vs_voter
        .iter()
        .map(|p| p.kappa_err.expect("defined"))
        .collect();
    assert_eq!(vv.len(), 3);
    assert_eq!(g.ceiling, vv.iter().copied().reduce(f64::max));
    assert!(
        vv.iter().any(|&k| Some(k) != g.ceiling),
        "the fixture must separate max from the rest"
    );
}

#[test]
fn falsify_lind_002_the_gate_refuses_what_it_cannot_decide() {
    let mut v = voters();
    v.push(errs("qwen-shadow", 40, |i| i % 3 == 0));
    // One voter: no voter–voter κ, so H7 is undecided — a refusal, not a pass.
    let g = run_gate(&v, &manifest(&["sonnet"], 10));
    assert!(
        !g.pass && g.refusals.iter().any(|r| r.contains("undecided")),
        "{:?}",
        g.refusals
    );
    // A voter the manifest names but the rows lack is a refusal, not a skip.
    let g = run_gate(&v, &manifest(&["sonnet", "haiku", "agy"], 10));
    assert!(
        !g.pass && g.refusals.iter().any(|r| r.contains("agy")),
        "{:?}",
        g.refusals
    );
    // Below min_n common items.
    let g = run_gate(&v, &manifest(&["sonnet", "haiku"], 41));
    assert!(
        !g.pass && g.refusals.iter().any(|r| r.contains("insufficient")),
        "{:?}",
        g.refusals
    );
    assert!(run_gate(&v, &manifest(&["sonnet", "haiku"], 40)).pass);
}

/// Every κ, the ceiling included, is taken on I, the items qwen AND every
/// voter scored — not on each pair's own overlap.
#[test]
fn falsify_lind_002_kappas_are_on_the_common_items() {
    let haiku = |i: u32| if i < 20 { i % 3 == 0 } else { five(i) };
    let v = vec![
        errs("sonnet", 40, five),
        errs("haiku", 40, haiku),
        errs("qwen-shadow", 20, |i| i % 4 == 0),
    ];
    // An item whose gold the lanes disagree on is not gold: it is not in I.
    let mut v = v;
    v.push(vec![
        row("sonnet", 99, "approve", "merged"),
        row("haiku", 99, "approve", "reverted_le14d"),
        row("qwen-shadow", 99, "approve", "merged"),
    ]);
    let g = run_gate(&v, &manifest(&["sonnet", "haiku"], 10));
    assert_eq!(g.n, 20);
    let e = |f: &dyn Fn(u32) -> bool, n: u32| (0..n).map(f).collect::<Vec<_>>();
    let on_i = crate::stats::error_kappa(&e(&five, 20), &e(&haiku, 20));
    let on_all = crate::stats::error_kappa(&e(&five, 40), &e(&haiku, 40));
    assert_ne!(
        on_i, on_all,
        "the fixture must separate the two, or this proves nothing"
    );
    assert_eq!(g.ceiling, on_i);
}

#[test]
fn falsify_lind_004_voters_and_min_n_are_pre_registered_and_delta_is_gone() {
    let ok = r#"{"lane_independence":{"min_n":30,"counted_lanes":["sonnet","haiku","agy"],
        "registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#;
    let m = parse_manifest(ok).expect("valid manifest");
    assert_eq!((m.min_n, m.counted_lanes.len()), (30, 3));
    for bad in [
        "{}",
        r#"{"lane_independence":{}}"#,
        // v2's δ: H7 is δ-free in v3, and a δ nothing reads is a false assurance
        r#"{"lane_independence":{"delta":0.05,"min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        r#"{"lane_independence":{"min_n":0,"counted_lanes":["sonnet"],"registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        r#"{"lane_independence":{"min_n":30,"counted_lanes":[],"registered_at":"2026-09-20T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        // registered AFTER training started: the voters could be picked after seeing κ
        r#"{"lane_independence":{"min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-22T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        // same instant is not "before"
        r#"{"lane_independence":{"min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-21T00:00:00Z","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        // a timestamp that does not compare lexicographically (offset, not Z)
        r#"{"lane_independence":{"min_n":30,"counted_lanes":["sonnet"],"registered_at":"2026-09-20T00:00:00+02:00","training_started_at":"2026-09-21T00:00:00Z"}}"#,
        "not json",
    ] {
        assert!(parse_manifest(bad).is_err(), "accepted: {bad}");
    }
}
