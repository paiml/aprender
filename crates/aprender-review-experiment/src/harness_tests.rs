// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;
use crate::corpus::Class;
use crate::receipt::{admissible, Expect};
use crate::score::{collect, score};
use std::io::{Read, Write};

const PREREG: &str = "ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0";
const CORPUS: &str = "review-corpus-v1@787d2026256cc08b";

fn run() -> Run {
    Run {
        cell: "C2".into(),
        host: "intel".into(),
        backend: "cpu".into(),
        arm: Arm::Apr4b,
        apr_tag: "v0.69.3".into(),
        apr_sha256: "a".repeat(64),
        model_id: "Qwen3.5-4B-Q4_K_M".into(),
        weights_sha256: "b".repeat(64),
        prompt: "Review. VERDICT: PASS or FAIL.\n".into(),
        corpus_version: CORPUS.into(),
        prereg_sha: PREREG.into(),
    }
}

fn ok_body(text: &str) -> String {
    json!({
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 9, "total_tokens": 129},
        "timings": {"prompt_n": 120, "prompt_ms": 80.0, "prompt_per_second": 1500.0,
                    "predicted_n": 9, "predicted_ms": 90.0, "predicted_per_second": 100.0, "clock": "x"}
    })
    .to_string()
}

fn reply(status: u16, body: &str) -> Reply {
    Reply {
        status,
        body: body.into(),
        wall_ms: 12.0,
    }
}

#[test]
fn compose_is_fixed_and_never_truncates() {
    assert_eq!(compose("P\n", "d\n"), "P\n\n```diff\nd\n```\n");
    assert_eq!(compose("P", "d"), "P\n\n```diff\nd\n```\n");
    let big = "+x\n".repeat(50_000);
    assert!(compose("P", &big).contains(&big));
}

#[test]
fn run_order_is_a_seeded_permutation_and_rerun_is_ten_percent() {
    let ids: Vec<String> = (0..105).map(|i| format!("i{i:03}")).collect();
    let o = run_order(&ids, 4354);
    assert_eq!(o, run_order(&ids, 4354));
    assert_ne!(o, run_order(&ids, 4355));
    let mut s = o.clone();
    s.sort();
    assert_eq!(s, ids);
    let mut rev = ids.clone();
    rev.reverse();
    assert_eq!(run_order(&rev, 4354), o, "input order does not matter");
    assert_eq!(rerun_subset(&o).len(), 11);
    assert_eq!(rerun_subset(&o)[..], o[..11]);
}

#[test]
fn classify_table() {
    let p = classify(&reply(200, &ok_body("VERDICT: FAIL\n- x")));
    assert_eq!(p.verdict, Verdict::Fail);
    assert_eq!(
        p.tokens,
        Some(Tokens {
            prompt: 120,
            completion: 9
        })
    );
    assert_eq!(p.server.map(|s| s.prompt_ms), Some(80.0));
    let cases = [
        (
            400,
            r#"{"error":"prompt exceeds context window"}"#,
            Verdict::NotRun(NotRun::ContextOverflow),
        ),
        (413, "Too long", Verdict::NotRun(NotRun::ContextOverflow)),
        (500, "context poisoned", Verdict::NotRun(NotRun::ServeError)),
        (400, "bad request", Verdict::NotRun(NotRun::ServeError)),
        (200, "not json", Verdict::NotRun(NotRun::ServeError)),
        (
            200,
            r#"{"choices":[]}"#,
            Verdict::NotRun(NotRun::ServeError),
        ),
        (200, &ok_body("looks fine"), Verdict::Unparsed),
    ];
    for (status, body, want) in cases {
        assert_eq!(
            classify(&reply(status, body)).verdict,
            want,
            "{status} {body}"
        );
    }
}

#[test]
fn receipts_from_the_harness_are_admissible_and_score() {
    let r = run();
    let d = "--- a/src/x.rs\n+++ b/src/x.rs\n@@ -1 +1 @@\n-a\n+b\n";
    let item = crate::corpus::Item::new("R-pr1".into(), Class::R, "t".into(), d);
    let body = request_body(&r.model_id, &r.prompt, d);
    let p = classify(&reply(200, &ok_body("VERDICT: FAIL\n- src/x.rs: b")));
    let rec = r.receipt(
        &item,
        &body,
        &p,
        12.0,
        "raw/R-pr1.txt",
        "2026-09-25T10:00:00Z",
    );
    let expect = Expect {
        prereg_sha: PREREG,
        corpus_version: CORPUS,
    };
    let line = serde_json::to_string(&rec).expect("json");
    assert!(
        admissible(&line, expect).is_ok(),
        "{:?}",
        admissible(&line, expect)
    );
    let (rows, rej) = collect(&line, expect, &[&item], |_| true, |_| Some(p.text.clone()));
    assert!(rej.is_empty(), "{rej:?}");
    assert_eq!((score(&rows).recall.k, rows[0].localized), (1, true));

    // No usage block: the row cannot be admitted as a run, so it is ServeError.
    let mut bare: Value = serde_json::from_str(&ok_body("VERDICT: PASS")).expect("v");
    bare.as_object_mut().expect("o").remove("usage");
    let p = classify(&reply(200, &bare.to_string()));
    let rec = r.receipt(&item, &body, &p, 1.0, "raw/x", "2026-09-25T10:00:00Z");
    assert_eq!(rec.verdict, Verdict::NotRun(NotRun::ServeError));

    let nr = r.not_run(&item, NotRun::NoDeclaredExecutor, "2026-09-25T10:00:00Z");
    assert!(admissible(&serde_json::to_string(&nr).expect("j"), expect).is_ok());
}

#[test]
fn post_talks_http_to_a_loopback_serve() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let canned = ok_body("VERDICT: PASS");
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().expect("accept");
        let mut buf = vec![0u8; 65_536];
        let mut got = Vec::new();
        while !String::from_utf8_lossy(&got).contains("\"stream\"") {
            let n = s.read(&mut buf).expect("read");
            if n == 0 {
                break;
            }
            got.extend_from_slice(&buf[..n]);
        }
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{canned}",
            canned.len()
        );
        s.write_all(resp.as_bytes()).expect("write");
        String::from_utf8_lossy(&got).into_owned()
    });
    let body = request_body("m", "P", "d\n");
    let rep = post(&format!("http://{addr}/v1/chat/completions"), &body).expect("post");
    let seen = server.join().expect("join");
    assert!(seen.starts_with("POST /v1/chat/completions"), "{seen}");
    assert!(
        seen.contains("\"temperature\":0.0") || seen.contains("\"temperature\":0"),
        "{seen}"
    );
    assert_eq!(rep.status, 200);
    assert_eq!(classify(&rep).verdict, Verdict::Pass);
}

#[test]
fn rfc3339_by_hand() {
    assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
    assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
    assert_eq!(rfc3339(1_790_337_600), "2026-09-25T12:00:00Z");
    assert_eq!(rfc3339(4_107_542_399), "2100-02-28T23:59:59Z");
}
