use super::*;

const TERMS: &str = r#"{"url":"https://www.anthropic.com/legal/commercial-terms","effective_date":"2025-06-17","fetched_at":"2026-09-25T17:00:00Z"}"#;
const LOCAL_SHA: &str = "3f1a9c0b5e2d4f6a8b7c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c";

fn row(id: &str, provider: &str, model: &str, channel: &str, terms: &str) -> String {
    format!(
        r#"{{"schema":"agent-trace-v1","trace_id":"{id}","provider":"{provider}","model_id":"{model}","access_channel":"{channel}","terms_ref":{terms}}}"#
    )
}

fn terms(url: &str, effective: &str, fetched: &str) -> String {
    format!(r#"{{"url":"{url}","effective_date":"{effective}","fetched_at":"{fetched}"}}"#)
}

/// One row per channel, all correctly tagged.
fn mixed_batch() -> String {
    [
        row("t1", "anthropic", "claude-sonnet-5", "claude-code-cli", TERMS),
        row("t2", "anthropic", "claude-haiku-4-5-20251001", "anthropic-api", TERMS),
        row("t3", "google", "gemini-3.1-pro-high", "antigravity", TERMS),
        row("t4", "google", "gemini-3.1-pro", "gemini-api", TERMS),
        row("t5", "google", "gemini-3.1-pro", "vertex", TERMS),
        row("t6", "local", LOCAL_SHA, "apr-serve", TERMS),
    ]
    .join("\n")
}

fn ids(rows: &[&str]) -> Vec<String> {
    rows.iter()
        .map(|r| {
            let v: Value = serde_json::from_str(r).expect("row is JSON");
            v["trace_id"].as_str().expect("trace_id").to_string()
        })
        .collect()
}

#[test]
fn falsify_terms_001_pool_rebuild_excluding_a_provider() {
    let batch = mixed_batch();
    let ex = Exclude::from_args(&["--exclude-provider", "google"]).expect("valid args");
    let r = rebuild(&batch, &ex);
    assert!(r.refused.is_empty(), "every row is tagged: {:?}", r.refused);
    assert_eq!(ids(&r.kept), ["t1", "t2", "t6"]);
    assert_eq!(ids(&r.excluded), ["t3", "t4", "t5"]);
    for kept in &r.kept {
        let t = tags(&serde_json::from_str(kept).expect("json")).expect("tagged");
        assert_ne!(t.provider, Provider::Google, "a google row survived: {kept}");
        assert!(batch.lines().any(|l| l == *kept), "kept row rewritten: {kept}");
    }
}

#[test]
fn falsify_terms_002_channel_exclusion_is_narrower_than_provider() {
    let batch = mixed_batch();
    let ex = Exclude::from_args(&["--exclude-channel", "antigravity"]).expect("valid args");
    let r = rebuild(&batch, &ex);
    assert_eq!(ids(&r.excluded), ["t3"]);
    assert_eq!(ids(&r.kept), ["t1", "t2", "t4", "t5", "t6"]);
}

#[test]
fn falsify_terms_003_identity_case_table() {
    let https = "https://x.example/terms";
    let at = "2026-09-25T17:00:00Z";
    let prefixed = format!("sha256:{LOCAL_SHA}");
    let upper = LOCAL_SHA.to_uppercase();
    // (trace_id, provider, model_id, channel, terms_ref, must be tagged?)
    let cases: Vec<(&str, &str, &str, &str, String, bool)> = vec![
        ("ok-a", "anthropic", "claude-opus-5-5", "claude-code-cli", TERMS.into(), true),
        ("ok-g", "google", "gemini-3.1-pro-high", "antigravity", TERMS.into(), true),
        ("ok-l", "local", LOCAL_SHA, "apr-serve", TERMS.into(), true),
        ("ok-l2", "local", &prefixed, "apr-serve", TERMS.into(), true),
        ("alias", "anthropic", "claude-sonnet", "claude-code-cli", TERMS.into(), false),
        ("bare", "anthropic", "sonnet", "claude-code-cli", TERMS.into(), false),
        ("gem-alias", "google", "gemini-pro", "gemini-api", TERMS.into(), false),
        ("unk-model", "anthropic", "unknown", "claude-code-cli", TERMS.into(), false),
        ("empty-model", "anthropic", "", "claude-code-cli", TERMS.into(), false),
        ("unk-prov", "openai", "gpt-5", "anthropic-api", TERMS.into(), false),
        ("unk-chan", "anthropic", "claude-opus-5-5", "web", TERMS.into(), false),
        ("mismatch", "google", "gemini-3.1-pro", "claude-code-cli", TERMS.into(), false),
        ("local-alias", "local", "qwen3.5-4b", "apr-serve", TERMS.into(), false),
        ("local-short", "local", "sha256:3f1a9c", "apr-serve", TERMS.into(), false),
        ("local-upper", "local", &upper, "apr-serve", TERMS.into(), false),
        ("no-terms", "anthropic", "claude-opus-5-5", "claude-code-cli", "null".into(), false),
        ("http", "anthropic", "claude-opus-5-5", "claude-code-cli",
            terms("http://x.example/terms", "2025-06-17", at), false),
        ("bad-date", "anthropic", "claude-opus-5-5", "claude-code-cli",
            terms(https, "June 17", at), false),
        ("bad-month", "anthropic", "claude-opus-5-5", "claude-code-cli",
            terms(https, "2025-13-17", at), false),
        ("bad-fetch", "anthropic", "claude-opus-5-5", "claude-code-cli",
            terms(https, "2025-06-17", "yesterday"), false),
        ("fetch-no-zone", "anthropic", "claude-opus-5-5", "claude-code-cli",
            terms(https, "2025-06-17", "2026-09-25T17:00:00"), false),
        // Terms announced ahead of their effective date are still a valid snapshot.
        ("fetch-before-effective", "anthropic", "claude-opus-5-5", "claude-code-cli",
            terms(https, "2026-10-01", at), true),
    ];
    for (id, p, m, c, t, want) in &cases {
        let v: Value = serde_json::from_str(&row(id, p, m, c, t)).expect("case is JSON");
        let got = tags(&v);
        assert_eq!(got.is_ok(), *want, "case {id} -> {got:?}");
        if let Err(gaps) = got {
            assert!(!gaps.is_empty(), "case {id}: a refusal names its gaps");
        }
    }
    let missing: Value = serde_json::from_str(
        r#"{"schema":"agent-trace-v1","trace_id":"missing","model_id":"claude-opus-5-5","access_channel":"claude-code-cli"}"#,
    )
    .expect("json");
    let gaps = tags(&missing).expect_err("no provider, no terms_ref");
    assert!(gaps.contains(&Gap::Missing("provider")), "{gaps:?}");
    assert!(gaps.contains(&Gap::Missing("terms_ref")), "{gaps:?}");
}

#[test]
fn falsify_terms_004_untagged_row_is_refused_never_kept() {
    let batch = format!(
        "{}\n{}\n\n",
        row("good", "anthropic", "claude-opus-5-5", "claude-code-cli", TERMS),
        row("alias", "google", "gemini-pro", "gemini-api", TERMS),
    );
    // Excluding anthropic must not let an untagged row into the pool.
    let ex = Exclude::from_args(&["--exclude-provider", "anthropic"]).expect("valid");
    let r = rebuild(&batch, &ex);
    assert!(r.kept.is_empty(), "kept: {:?}", r.kept);
    assert_eq!(ids(&r.excluded), ["good"]);
    assert_eq!(r.refused.len(), 1);
    assert!(!r.refused[0].1.is_empty(), "a refusal names its gaps");
    // With nothing excluded the untagged row is still refused.
    let r = rebuild(&batch, &Exclude::default());
    assert_eq!(ids(&r.kept), ["good"]);
    assert_eq!(r.refused.len(), 1);
    // A line that is not JSON is refused too, never dropped silently.
    let r = rebuild("not json\n", &Exclude::default());
    assert!(r.kept.is_empty());
    assert_eq!(r.refused.len(), 1);
}

#[test]
fn exclude_args_reject_unknown_values_and_flags() {
    assert!(Exclude::from_args(&["--exclude-provider", "openai"]).is_err());
    assert!(Exclude::from_args(&["--exclude-channel", "web"]).is_err());
    assert!(Exclude::from_args(&["--exclude-provider"]).is_err());
    assert!(Exclude::from_args(&["--include-provider", "google"]).is_err());
    let ex = Exclude::from_args(&[
        "--exclude-provider",
        "anthropic",
        "--exclude-channel",
        "vertex",
        "--exclude-provider",
        "google",
    ])
    .expect("valid");
    assert_eq!(ex.providers, [Provider::Anthropic, Provider::Google]);
    assert_eq!(ex.channels, [Channel::Vertex]);
    assert_eq!(Exclude::from_args(&[]).expect("empty"), Exclude::default());
}

#[test]
fn public_release_keeps_only_local_rows() {
    let batch = mixed_batch();
    let r = rebuild(&batch, &Exclude::public_release());
    assert_eq!(ids(&r.kept), ["t6"]);
    assert_eq!(r.excluded.len(), 5);
}

#[test]
fn every_channel_maps_to_its_provider() {
    let table = [
        ("anthropic-api", Provider::Anthropic),
        ("claude-code-cli", Provider::Anthropic),
        ("antigravity", Provider::Google),
        ("gemini-api", Provider::Google),
        ("vertex", Provider::Google),
        ("apr-serve", Provider::Local),
    ];
    for (name, p) in table {
        let c = Channel::parse(name).expect("known channel");
        assert_eq!(c.provider(), p, "{name}");
        assert_eq!(c.as_str(), name);
    }
    assert_eq!(table.len(), Channel::ALL.len());
    for (name, p) in [
        ("anthropic", Provider::Anthropic),
        ("google", Provider::Google),
        ("local", Provider::Local),
    ] {
        assert_eq!(Provider::parse(name), Some(p));
    }
    assert_eq!(Provider::parse("Anthropic"), None);
}
