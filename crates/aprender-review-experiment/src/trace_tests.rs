use super::*;

fn sha(c: char) -> String {
    c.to_string().repeat(64)
}

fn row(lane: &str, pr: u64, at: &str, verdict: Verdict) -> TraceRow {
    TraceRow {
        schema: SCHEMA.into(),
        at: at.into(),
        quorum_id: format!("q-{pr}"),
        repo: "paiml/aprender".into(),
        pr,
        head: "abc".into(),
        diff_sha256: sha('d'),
        lane: lane.into(),
        model_id: "claude-sonnet-5".into(),
        provider: Provider::Anthropic,
        served_by: "api.anthropic.com".into(),
        input_sha256: sha('1'),
        params: BTreeMap::from([("temperature".into(), serde_json::Value::from(0))]),
        run: Run::Ran {
            output_sha256: sha('2'),
            verdict: Some(verdict),
            findings: vec!["ledger.rs:205 drops the weights sha".into()],
            tokens_in: 9000,
            tokens_out: 400,
            latency_ms: 31_000,
        },
        logits_sha256: None,
        label_tier: LabelTier::Silver,
        split_guard: split_key("paiml/aprender", pr),
        agreed: Some(true),
        lane_disagreement: false,
        outcome: "pending".into(),
        unique_diff: false,
        bytes: Some(Bytes {
            raw: 40_000,
            zstd: 9_000,
        }),
    }
}

fn good() -> TraceRow {
    row("sonnet", 4354, "2026-09-25T12:00:00Z", Verdict::Pass)
}

#[test]
fn a_complete_row_lints_clean_and_round_trips() {
    let r = good();
    assert!(lint(&r).is_empty(), "{:?}", lint(&r));
    let line = serde_json::to_string(&r).expect("ser");
    assert_eq!(parse_index(&line).expect("parses"), [r]);
}

#[test]
fn falsify_atr_001_an_unknown_identity_field_is_red() {
    let cases: Vec<(&str, Box<dyn Fn(&mut TraceRow)>)> = vec![
        ("model_id", Box::new(|r| r.model_id = "unknown".into())),
        ("served_by", Box::new(|r| r.served_by = String::new())),
        ("lane", Box::new(|r| r.lane = " ".into())),
        ("quorum_id", Box::new(|r| r.quorum_id = "null".into())),
        ("head", Box::new(|r| r.head = String::new())),
        ("pr", Box::new(|r| r.pr = 0)),
        ("input_sha256", Box::new(|r| r.input_sha256 = "abc".into())),
        ("diff_sha256", Box::new(|r| r.diff_sha256 = sha('D'))),
        ("params", Box::new(|r| r.params.clear())),
        ("at", Box::new(|r| r.at = "2026-09-25 12:00".into())),
        (
            "output_sha256",
            Box::new(|r| {
                if let Run::Ran { output_sha256, .. } = &mut r.run {
                    output_sha256.clear();
                }
            }),
        ),
    ];
    for (field, mutate) in cases {
        let mut r = good();
        mutate(&mut r);
        let red = lint(&r);
        assert!(red.iter().any(|m| m.contains(field)), "{field}: {red:?}");
    }
    let bad_provider = serde_json::to_string(&good())
        .expect("ser")
        .replace(r#""provider":"anthropic""#, r#""provider":"unknown""#);
    assert!(
        parse_index(&bad_provider).is_err(),
        "provider is a closed set"
    );
    let extra = serde_json::to_string(&good())
        .expect("ser")
        .replacen('{', r#"{"note":"x","#, 1);
    assert!(
        parse_index(&extra).is_err(),
        "an unknown key fails to parse"
    );
}

#[test]
fn falsify_atr_002_the_local_lane_carries_top_k_logits_and_hosted_lanes_do_not() {
    let mut local = good();
    local.provider = Provider::Local;
    local.model_id = "qwen3.5-4b-q4_k_m".into();
    assert!(lint(&local).iter().any(|m| m.contains("logits")), "missing");
    local.logits_sha256 = Some(sha('3'));
    assert!(lint(&local).is_empty(), "{:?}", lint(&local));
    let mut hosted = good();
    hosted.logits_sha256 = Some(sha('3'));
    assert!(!lint(&hosted).is_empty(), "hosted lanes have no logits");
    // qwen writes NotRun rows until its executor exists: no output, no logits.
    local.run = Run::NotRun(NotRun::NoExecutor);
    local.logits_sha256 = None;
    assert!(lint(&local).is_empty(), "{:?}", lint(&local));
}

#[test]
fn falsify_atr_003_tiers_stay_separable_and_gold_needs_an_outcome() {
    let mut r = good();
    r.label_tier = LabelTier::Gold;
    assert!(lint(&r).iter().any(|m| m.contains("gold")));
    r.outcome = "merged".into();
    assert!(lint(&r).is_empty());
    let mut guard = good();
    guard.split_guard = "paiml/aprender#1".into();
    assert!(lint(&guard).iter().any(|m| m.contains("split_guard")));
}

#[test]
fn falsify_atr_004_a_pr_never_straddles_splits() {
    // PR 1 first seen before dev_from; its second round lands after test_from.
    let rows = [
        row("sonnet", 1, "2026-09-01T00:00:00Z", Verdict::Pass),
        row("sonnet", 2, "2026-09-15T00:00:00Z", Verdict::Pass),
        row("haiku", 1, "2026-09-29T00:00:00Z", Verdict::Fail),
        row("sonnet", 3, "2026-09-29T00:00:00Z", Verdict::Pass),
    ];
    let s = splits(&rows, "2026-09-10", "2026-09-20");
    assert_eq!(s.len(), 3, "one entry per repo#PR, never per row");
    assert_eq!(s["paiml/aprender#1"], Split::Train);
    assert_eq!(s["paiml/aprender#2"], Split::Dev);
    assert_eq!(s["paiml/aprender#3"], Split::Test);
    // Order-independent: the PR's earliest row decides.
    let mut rev = rows.clone();
    rev.reverse();
    assert_eq!(splits(&rev, "2026-09-10", "2026-09-20"), s);
}

#[test]
fn derived_columns_mark_first_diffs_and_split_rounds() {
    let mut rows = vec![
        row("sonnet", 7, "2026-09-25T00:00:00Z", Verdict::Pass),
        row("haiku", 7, "2026-09-25T00:00:01Z", Verdict::Fail),
        row("sonnet", 8, "2026-09-25T00:00:02Z", Verdict::Pass),
    ];
    rows[2].diff_sha256 = sha('e');
    derive_columns(&mut rows);
    let u: Vec<bool> = rows.iter().map(|r| r.unique_diff).collect();
    assert_eq!(u, [true, false, true]);
    let d: Vec<bool> = rows.iter().map(|r| r.lane_disagreement).collect();
    assert_eq!(d, [true, true, false]);
}

#[test]
fn falsify_atr_005_the_weekly_receipt_never_reports_an_unmeasured_size_as_zero() {
    let mut rows = vec![
        row("sonnet", 1, "2026-09-21T00:00:00Z", Verdict::Pass),
        row("haiku", 1, "2026-09-22T00:00:00Z", Verdict::Fail),
        row("sonnet", 2, "2026-09-28T00:00:00Z", Verdict::Pass),
    ];
    rows[1].label_tier = LabelTier::Gold;
    rows[1].outcome = "merged".into();
    let w = weekly(&rows, "2026-09-21", "2026-09-28");
    assert_eq!((w.rows, w.unique_diffs), (2, 1));
    assert_eq!((w.bytes_raw, w.bytes_zstd), (Some(80_000), Some(18_000)));
    assert_eq!(w.by_lane_tier_outcome["sonnet|silver|pending"], 1);
    assert_eq!(w.by_lane_tier_outcome["haiku|gold|merged"], 1);
    rows[0].bytes = None;
    let w = weekly(&rows, "2026-09-21", "2026-09-28");
    assert_eq!((w.bytes_raw, w.bytes_zstd), (None, None));
}
