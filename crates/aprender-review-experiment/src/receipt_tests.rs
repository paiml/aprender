// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;

pub(crate) const PREREG: &str = "ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0";
pub(crate) const CORPUS: &str = "review-corpus-v1@787d2026256cc08b";
pub(crate) const EXPECT: Expect<'static> = Expect {
    prereg_sha: PREREG,
    corpus_version: CORPUS,
};

fn h(c: char) -> String {
    std::iter::repeat_n(c, 64).collect()
}

/// A complete, admissible executed row.
pub(crate) fn row(item: &str, class: Class, verdict: Verdict) -> Receipt {
    Receipt {
        schema: SCHEME.into(),
        item_id: item.into(),
        item_sha256: h('1'),
        class,
        stratum: Stratum::S,
        split: Split::Test,
        cell: "C2".into(),
        host: "intel".into(),
        backend: "cpu".into(),
        arm: Arm::Apr4b,
        apr_tag: "v0.69.3".into(),
        apr_sha256: h('2'),
        model_id: "Qwen3.5-4B-Q4_K_M".into(),
        weights_sha256: h('3'),
        prompt_sha256: h('4'),
        request_sha256: h('5'),
        decoding: Decoding {
            temperature: 0.0,
            seed: 4354,
            max_tokens: 512,
        },
        corpus_version: CORPUS.into(),
        prereg_sha: PREREG.into(),
        started_at: "2026-09-25T10:00:00Z".into(),
        cold: false,
        rerun: false,
        verdict,
        tokens: Some(Tokens {
            prompt: 900,
            completion: 80,
        }),
        timings: Some(Timings {
            wall_ms: 4200.0,
            server: None,
        }),
        output: Some(Output {
            path: format!("raw/C2/apr-4b/{item}.txt"),
            sha256: h('6'),
        }),
        load: Some(Load {
            loadavg1: 1.5,
            cpus: 32,
            co_running: None,
            peak_anon_bytes: None,
        }),
    }
}

fn line(r: &Receipt) -> String {
    serde_json::to_string(r).expect("serialize")
}

#[test]
fn complete_row_is_admissible_and_round_trips() {
    let r = row("R-pr1", Class::R, Verdict::Fail);
    assert_eq!(admissible(&line(&r), EXPECT), Ok(r));
}

#[test]
fn falsify_rxr_001_missing_model_sha_is_inadmissible() {
    let mut v: serde_json::Value =
        serde_json::to_value(row("R-pr1", Class::R, Verdict::Fail)).expect("v");
    v.as_object_mut().expect("obj").remove("weights_sha256");
    let e =
        admissible(&v.to_string(), EXPECT).expect_err("missing weights sha must be inadmissible");
    assert!(e[0].contains("weights_sha256"), "{e:?}");

    let mut r = row("R-pr1", Class::R, Verdict::Fail);
    r.weights_sha256 = "unknown".into();
    assert!(
        admissible(&line(&r), EXPECT).is_err(),
        "`unknown` weights sha"
    );
    r.weights_sha256 = h('3')[..12].into();
    assert!(
        admissible(&line(&r), EXPECT).is_err(),
        "a sha prefix is not a sha"
    );
}

#[test]
fn identity_prereg_corpus_and_decoding_are_enforced() {
    let ok = row("G-pr1", Class::G, Verdict::Pass);
    let cases: Vec<(&str, Box<dyn Fn(&mut Receipt)>)> = vec![
        ("host", Box::new(|r| r.host = "unknown".into())),
        ("cell", Box::new(|r| r.cell = String::new())),
        ("prereg", Box::new(|r| r.prereg_sha = h('9'))),
        (
            "corpus",
            Box::new(|r| r.corpus_version = "review-corpus-v0@x".into()),
        ),
        ("greedy", Box::new(|r| r.decoding.temperature = 0.7)),
        ("tokens", Box::new(|r| r.tokens = None)),
        ("output", Box::new(|r| r.output = None)),
        ("load", Box::new(|r| r.load = None)),
        ("hosted", Box::new(|r| r.arm = Arm::Haiku)),
        ("schema", Box::new(|r| r.schema = "v0".into())),
    ];
    for (name, mutate) in cases {
        let mut r = ok.clone();
        mutate(&mut r);
        assert!(
            admissible(&line(&r), EXPECT).is_err(),
            "{name} must be inadmissible"
        );
    }
    let mut v: serde_json::Value = serde_json::to_value(&ok).expect("v");
    v["extra"] = serde_json::json!(1);
    assert!(admissible(&v.to_string(), EXPECT).is_err(), "unknown field");
}

#[test]
fn not_run_rows_need_identity_but_not_timings_and_hosted_arms_say_hosted() {
    let mut r = row("P-1", Class::P, Verdict::NotRun(NotRun::NoDeclaredExecutor));
    (r.tokens, r.timings, r.output, r.load) = (None, None, None, None);
    assert!(admissible(&line(&r), EXPECT).is_ok());
    let mut hosted = row("P-1", Class::P, Verdict::Fail);
    hosted.arm = Arm::Haiku;
    hosted.apr_sha256 = "hosted".into();
    hosted.weights_sha256 = "hosted".into();
    hosted.model_id = "claude-haiku-4-5-20251001".into();
    assert!(admissible(&line(&hosted), EXPECT).is_ok());
}

#[test]
fn verdict_parser_table() {
    let cases = [
        ("VERDICT: FAIL\n- off by one", Verdict::Fail),
        ("VERDICT: PASS", Verdict::Pass),
        ("**VERDICT: FAIL** — index", Verdict::Fail),
        ("## VERDICT: **PASS**\n", Verdict::Pass),
        ("Sure.\nVERDICT: FAIL\nVERDICT: PASS", Verdict::Fail),
        (
            "<think>VERDICT: PASS maybe</think>\nVERDICT: FAIL",
            Verdict::Fail,
        ),
        ("VERDICT: PASS or FAIL", Verdict::Unparsed),
        ("VERDICT: LGTM", Verdict::Unparsed),
        ("VERDICT: pass", Verdict::Unparsed),
        ("VERDICT: PASSED", Verdict::Unparsed),
        ("The verdict is FAIL", Verdict::Unparsed),
        ("", Verdict::Unparsed),
    ];
    for (out, want) in cases {
        assert_eq!(parse_verdict(out), want, "{out:?}");
    }
}

#[test]
fn unparsed_and_not_run_are_never_correct() {
    for v in [Verdict::Unparsed, Verdict::NotRun(NotRun::ContextOverflow)] {
        assert!(!v.correct(true) && !v.correct(false), "{v:?}");
    }
    assert!(Verdict::Fail.correct(true) && !Verdict::Fail.correct(false));
    assert!(Verdict::Pass.correct(false) && !Verdict::Pass.correct(true));
}

#[test]
fn localization_matches_path_or_last_two_components_after_the_verdict() {
    let loc = [Loc {
        file: "crates/aprender-serve/src/api/router.rs".into(),
        line: 10,
    }];
    assert!(localized(
        "VERDICT: FAIL\n- api/router.rs: wrong status",
        &loc
    ));
    assert!(localized(
        "VERDICT: FAIL\n- crates/aprender-serve/src/api/router.rs:10",
        &loc
    ));
    assert!(
        !localized("VERDICT: FAIL\n- router.rs looks off", &loc),
        "bare file name"
    );
    assert!(
        !localized("api/router.rs\nVERDICT: FAIL\n- fine", &loc),
        "before the verdict"
    );
    assert!(!localized("no verdict api/router.rs", &loc));
}
