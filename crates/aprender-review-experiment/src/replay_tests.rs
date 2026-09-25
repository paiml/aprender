use super::*;
use crate::contamination::Index;
use crate::corpus::{hunk_fingerprints, sha256_hex, Sealed};
use crate::receipt::NotRun;

fn diff(tag: &str) -> String {
    format!("diff --git a/{tag}.rs b/{tag}.rs\n--- a/{tag}.rs\n+++ b/{tag}.rs\n@@ -1,3 +1,4 @@\n fn {tag}() {{\n-    old_{tag}();\n+    new_{tag}();\n+    more_{tag}();\n }}\n")
}

fn cand(group: &str, tag: &str, tokens: u64) -> Candidate {
    let d = diff(tag);
    Candidate {
        group: group.to_string(),
        diff_sha256: sha256_hex(d.as_bytes()),
        input_tokens: tokens,
        diff: d,
    }
}

fn sealed_of(c: &Candidate) -> Sealed {
    Sealed {
        id: format!("G-{}", c.group),
        diff_sha256: c.diff_sha256.clone(),
        hunks: hunk_fingerprints(&c.diff),
    }
}

/// `n` fresh candidates per stratum, each its own group.
fn pool(n: usize) -> Vec<Candidate> {
    let mut v = Vec::new();
    for s in Stratum::ALL {
        for i in 0..n {
            let tag = format!("{}_{i}", s.as_str());
            v.push(cand(
                &format!("paiml/x#{tag}"),
                &format!("f{tag}"),
                s.upper() - i as u64,
            ));
        }
    }
    v
}

fn unrelated_seal() -> Index {
    Index::new(&[sealed_of(&cand("paiml/sealed#1", "sealedonly", 10))])
}

#[test]
fn stratum_bounds_are_inclusive_and_refuse_zero_and_overflow() {
    assert_eq!(Stratum::of(0), None);
    assert_eq!(Stratum::of(1), Some(Stratum::K2));
    assert_eq!(Stratum::of(2048), Some(Stratum::K2));
    assert_eq!(Stratum::of(2049), Some(Stratum::K8));
    assert_eq!(Stratum::of(8192), Some(Stratum::K8));
    assert_eq!(Stratum::of(8193), Some(Stratum::K16));
    assert_eq!(Stratum::of(16384), Some(Stratum::K16));
    assert_eq!(Stratum::of(16385), Some(Stratum::K32));
    assert_eq!(Stratum::of(32768), Some(Stratum::K32));
    assert_eq!(Stratum::of(32769), None);
}

/// FALSIFY-REPLAY-001: a sealed item never enters the set, by diff sha,
/// by hunk or by a copy under another group.
#[test]
fn falsify_replay_001_sealed_items_are_excluded() {
    let mut cands = pool(3);
    let leak = cands[0].clone();
    // Same diff under another PR: the diff-sha / hunk hit must still drop it.
    let mut copy = leak.clone();
    copy.group = "paiml/other#9".into();
    cands.push(copy);
    let ix = Index::new(&[sealed_of(&leak)]);
    let err = build("v1", 4354, 3, &cands, &ix).unwrap_err();
    assert_eq!(
        err,
        BuildError::Short(vec![Short {
            stratum: Stratum::K2,
            have: 2,
            need: 3
        }]),
        "only the sealed diff's stratum may come up short"
    );
    let set = build("v1", 4354, 2, &cands, &ix).expect("2 per stratum fits");
    assert!(set.items.iter().all(|i| i.diff_sha256 != leak.diff_sha256));
}

#[test]
fn an_empty_sealed_index_is_refused() {
    assert_eq!(
        build("v1", 1, 1, &pool(1), &Index::default()),
        Err(BuildError::EmptySealedIndex)
    );
}

/// FALSIFY-REPLAY-002: a short stratum is an error, never padded.
#[test]
fn falsify_replay_002_short_strata_are_refused_not_padded() {
    let mut cands = pool(2);
    cands.retain(|c| Stratum::of(c.input_tokens) != Some(Stratum::K32));
    cands.push(cand("paiml/x#big", "big", 40_000));
    let err = build("v1", 1, 2, &cands, &unrelated_seal()).unwrap_err();
    assert_eq!(
        err,
        BuildError::Short(vec![Short {
            stratum: Stratum::K32,
            have: 0,
            need: 2
        }])
    );
}

/// FALSIFY-REPLAY-003: one item per repo#PR and per diff sha, and the set
/// does not depend on candidate order.
#[test]
fn falsify_replay_003_one_per_group_and_order_free() {
    let mut cands = pool(4);
    // Two more diffs on an existing PR and a re-post of an existing diff.
    let g = cands[0].group.clone();
    cands.push(cand(&g, "extra_a", 100));
    cands.push(cand(&g, "extra_b", 200));
    let mut repost = cands[1].clone();
    repost.group = "paiml/y#1".into();
    cands.push(repost);
    let a = build("v1", 7, 4, &cands, &unrelated_seal()).expect("fits");
    cands.reverse();
    let b = build("v1", 7, 4, &cands, &unrelated_seal()).expect("fits");
    assert_eq!(a, b);
    assert_eq!(a.sha(), b.sha());
    let groups: BTreeSet<_> = a.items.iter().map(|i| &i.group).collect();
    let shas: BTreeSet<_> = a.items.iter().map(|i| &i.diff_sha256).collect();
    assert_eq!(groups.len(), a.items.len());
    assert_eq!(shas.len(), a.items.len());
    assert_eq!(a.items.len(), 16);
    for s in Stratum::ALL {
        assert_eq!(a.items.iter().filter(|i| i.stratum == s).count(), 4);
    }
    let c = build("v1", 8, 4, &cands, &unrelated_seal()).expect("fits");
    assert_ne!(a.sha(), c.sha(), "the seed is part of the frozen identity");
}

#[test]
fn render_parse_round_trips_and_rejects_a_foreign_header() {
    let set = build("v1", 3, 2, &pool(3), &unrelated_seal()).expect("fits");
    let text = set.render();
    assert_eq!(Set::parse(&text), Ok(set.clone()));
    assert!(Set::parse(&text.replace(SCHEMA, "other-v1")).is_err());
    assert!(Set::parse("").is_err());
    assert_eq!(set.sha(), sha256_hex(text.as_bytes()));
}

fn row(engine: Engine, i: usize, wall_ms: f64) -> Row {
    let stratum = Stratum::ALL[i % 4];
    let (pt, dt) = match engine {
        Engine::Apr => (500.0, 20.0),
        Engine::LlamaCpp => (1000.0, 40.0),
    };
    Row {
        schema: ROW_SCHEMA.into(),
        replay_version: "v1".into(),
        set_sha: "s".repeat(64),
        diff_sha256: format!("{i:064x}"),
        stratum,
        engine,
        engine_version: "x".into(),
        apr_tag: "v0.70.1".into(),
        cell: "lambda-4090".into(),
        gguf_sha256: "g".repeat(64),
        wall_ms,
        prompt_ms: Some(if engine == Engine::Apr { 300.0 } else { 100.0 }),
        prompt_tps: Some(pt * (1.0 + (i % 4) as f64)),
        decode_tps: Some(dt),
        input_tokens: Some(1000),
        output_tokens: Some(64),
        peak_rss_mb: Some(if engine == Engine::Apr {
            3000.0 + i as f64
        } else {
            2000.0
        }),
        verdict: if i % 5 == 0 {
            Verdict::Unparsed
        } else {
            Verdict::Pass
        },
    }
}

fn run(n: usize) -> Vec<Row> {
    let mut v: Vec<Row> = (0..n)
        .map(|i| row(Engine::Apr, i, 1000.0 * (i + 1) as f64))
        .collect();
    v.extend((0..n).map(|i| row(Engine::LlamaCpp, i, 500.0)));
    v
}

fn voters() -> Vec<(String, Vec<f64>)> {
    vec![
        ("haiku".into(), (1..=20).map(f64::from).collect()),
        ("agy".into(), (1..=20).map(|x| f64::from(x) * 3.0).collect()),
    ]
}

/// FALSIFY-REPLAY-004: ratios are apr / llama_cpp and the queue budget is
/// the slowest counted voter's p95.
#[test]
fn falsify_replay_004_ratios_and_queue_budget() {
    let s = summarize(&run(20), &voters(), None, 1).expect("valid run");
    assert_eq!(s.n_items, 20);
    assert!((s.p50_s - 10.0).abs() < 1e-9, "p50 {}", s.p50_s);
    assert!((s.p95_s - 19.0).abs() < 1e-9, "p95 {}", s.p95_s);
    assert!(
        s.p95_ci[0] <= s.p95_s && s.p95_s <= s.p95_ci[1],
        "{:?}",
        s.p95_ci
    );
    assert_eq!(s.queue_budget_lane, "agy");
    assert!((s.queue_budget_p95_s - 57.0).abs() < 1e-9);
    assert!(s.within_budget);
    assert_eq!(s.ttft_ms.ratio, Some(3.0));
    assert_eq!(s.decode_tps.ratio, Some(0.5));
    assert_eq!(s.prefill_tps.len(), 4);
    for (st, r) in &s.prefill_tps {
        let k = 1.0 + Stratum::ALL.iter().position(|x| x == st).unwrap_or(9) as f64;
        assert_eq!(r.apr, Some(500.0 * k), "{st:?}");
        assert_eq!(r.ratio, Some(0.5));
    }
    assert_eq!(s.peak_rss_mb.apr, Some(3019.0));
    assert!((s.parse_rate - 0.8).abs() < 1e-9);
    assert_eq!(s.verdict_identity_vs_prev_tag, None);

    let slow = vec![("haiku".to_string(), vec![5.0])];
    assert!(
        !summarize(&run(20), &slow, None, 1)
            .expect("valid")
            .within_budget
    );
}

#[test]
fn falsify_replay_004_a_null_server_timing_is_null_not_a_subset_median() {
    let mut rows = run(8);
    rows[9].prompt_ms = None; // one llama row
    let s = summarize(&rows, &voters(), None, 1).expect("valid");
    assert_eq!(s.ttft_ms.llama_cpp, None);
    assert_eq!(s.ttft_ms.ratio, None);
    assert_eq!(s.ttft_ms.apr, Some(300.0));
}

/// FALSIFY-REPLAY-005: an inconsistent run never prints a number.
#[test]
fn falsify_replay_005_inconsistent_runs_are_refused() {
    let v = voters();
    let base = run(8);
    assert_eq!(summarize(&[], &v, None, 1), Err(SummaryError::NoRows));

    let mut r = base.clone();
    r[3].set_sha = "t".repeat(64);
    assert_eq!(summarize(&r, &v, None, 1), Err(SummaryError::MixedSet));
    let mut r = base.clone();
    r[3].replay_version = "v2".into();
    assert_eq!(summarize(&r, &v, None, 1), Err(SummaryError::MixedSet));
    let mut r = base.clone();
    r[9].gguf_sha256 = "h".repeat(64);
    assert_eq!(
        summarize(&r, &v, None, 1),
        Err(SummaryError::Mismatch("gguf_sha256"))
    );
    let mut r = base.clone();
    r[9].cell = "gx10".into();
    assert_eq!(
        summarize(&r, &v, None, 1),
        Err(SummaryError::Mismatch("cell"))
    );
    let mut r = base.clone();
    r[9].apr_tag = "v0.70.0".into();
    assert_eq!(
        summarize(&r, &v, None, 1),
        Err(SummaryError::Mismatch("apr_tag"))
    );

    let mut r = base.clone();
    r.pop();
    assert_eq!(
        summarize(&r, &v, None, 1),
        Err(SummaryError::CoverageDiffers)
    );
    let mut r = base.clone();
    r.push(r[0].clone());
    assert!(matches!(
        summarize(&r, &v, None, 1),
        Err(SummaryError::DuplicateRow { .. })
    ));
    let mut r = base.clone();
    r[2].verdict = Verdict::NotRun(NotRun::ContextOverflow);
    assert!(matches!(
        summarize(&r, &v, None, 1),
        Err(SummaryError::NotRun {
            engine: Engine::Apr,
            ..
        })
    ));
    let llama_only: Vec<Row> = base
        .iter()
        .filter(|r| r.engine == Engine::LlamaCpp)
        .cloned()
        .collect();
    assert_eq!(
        summarize(&llama_only, &v, None, 1),
        Err(SummaryError::CoverageDiffers)
    );

    assert_eq!(
        summarize(&base, &[], None, 1),
        Err(SummaryError::NoVoterLatency)
    );
    let empty_lane = vec![("haiku".to_string(), vec![])];
    assert_eq!(
        summarize(&base, &empty_lane, None, 1),
        Err(SummaryError::NoVoterLatency)
    );
}

/// FALSIFY-REPLAY-006: verdict identity compares the same set, item by item.
#[test]
fn falsify_replay_006_verdict_identity_vs_previous_tag() {
    let now = run(10);
    let mut prev: Vec<Row> = now.clone();
    for r in prev.iter_mut().filter(|r| r.engine == Engine::Apr).take(3) {
        r.verdict = Verdict::Fail;
        r.apr_tag = "v0.70.0".into();
    }
    let s = summarize(&now, &voters(), Some(&prev), 1).expect("valid");
    assert_eq!(s.verdict_identity_vs_prev_tag, Some(0.7));
    // llama rows in `prev` are ignored.
    let apr_prev: Vec<Row> = prev
        .iter()
        .filter(|r| r.engine == Engine::Apr)
        .cloned()
        .collect();
    assert_eq!(
        summarize(&now, &voters(), Some(&apr_prev), 1)
            .expect("valid")
            .verdict_identity_vs_prev_tag,
        Some(0.7)
    );
    let mut other = apr_prev.clone();
    other[0].set_sha = "t".repeat(64);
    assert_eq!(
        summarize(&now, &voters(), Some(&other), 1),
        Err(SummaryError::PrevSetDiffers)
    );
    let mut fewer = apr_prev;
    fewer.pop();
    assert_eq!(
        summarize(&now, &voters(), Some(&fewer), 1),
        Err(SummaryError::PrevCoverageDiffers)
    );
}

#[test]
fn row_serializes_with_stable_engine_and_stratum_names() {
    let v = serde_json::to_value(row(Engine::LlamaCpp, 1, 1.0)).expect("serializes");
    assert_eq!(v["engine"], "llama_cpp");
    assert_eq!(v["stratum"], "8k");
    assert_eq!(v["schema"], ROW_SCHEMA);
}
