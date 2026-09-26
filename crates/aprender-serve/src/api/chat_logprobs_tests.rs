use super::*;
use crate::gguf::TopLogprob;

fn req(logprobs: Option<bool>, top: Option<usize>, stream: bool) -> ChatCompletionRequest {
    ChatCompletionRequest {
        logprobs,
        top_logprobs: top,
        stream,
        ..Default::default()
    }
}

#[test]
fn requested_top_logprobs_case_table() {
    // (logprobs, top_logprobs, stream) -> Ok(want) | Err(substring)
    let table: &[(
        Option<bool>,
        Option<usize>,
        bool,
        Result<Option<usize>, &str>,
    )] = &[
        (None, None, false, Ok(None)),
        (Some(false), None, false, Ok(None)),
        (None, None, true, Ok(None)),
        (Some(true), None, false, Ok(Some(0))),
        (Some(true), Some(0), false, Ok(Some(0))),
        (Some(true), Some(20), false, Ok(Some(20))),
        (Some(true), Some(21), false, Err("at most 20")),
        (None, Some(5), false, Err("requires `logprobs: true`")),
        (
            Some(false),
            Some(5),
            false,
            Err("requires `logprobs: true`"),
        ),
        (Some(true), Some(5), true, Err("stream")),
    ];
    for (lp, top, stream, want) in table {
        let got = requested_top_logprobs(&req(*lp, *top, *stream));
        match (want, &got) {
            (Ok(w), Ok(g)) => assert_eq!(w, g, "{lp:?} {top:?} {stream}"),
            (Err(s), Err(e)) => assert!(e.contains(s), "{lp:?} {top:?} {stream}: {e}"),
            _ => panic!("{lp:?} {top:?} {stream}: want {want:?}, got {got:?}"),
        }
    }
}

#[test]
fn refusal_names_the_architecture() {
    assert!(unsupported_backend_reason(Some("llama")).contains("(llama)"));
    assert!(unsupported_backend_reason(None).contains("unknown architecture"));
}

fn step(chosen: u32, top: &[(u32, f32)], chosen_logprob: f32, lse: f32) -> StepLogprobs {
    StepLogprobs {
        step: 0,
        chosen,
        top: top
            .iter()
            .map(|&(token_id, logprob)| TopLogprob {
                token_id,
                logit: logprob + lse,
                logprob,
            })
            .collect(),
        chosen_logprob,
        logsumexp_full: lse,
    }
}

/// Every field a C11 row needs is present, `n_top` truncates the
/// alternatives only, and a sampled token outside the top-K keeps its own
/// logprob instead of borrowing the top entry's.
#[test]
fn chat_logprobs_json_carries_ids_logprobs_and_lse() {
    let steps = [
        step(7, &[(7, -0.1), (3, -2.5), (9, -3.0)], -0.1, 12.5),
        step(42, &[(1, -0.2), (2, -1.9), (4, -2.2)], -6.0, 11.0),
    ];
    let v = chat_logprobs_json(&steps, 2, &|id| format!("t{id}"));
    let c = v["content"].as_array().expect("content");
    assert_eq!(c.len(), 2);
    assert_eq!(c[0]["token"], "t7");
    assert_eq!(c[0]["token_id"], 7);
    assert_eq!(c[0]["logsumexp_full"], 12.5);
    assert_eq!(c[0]["top_logprobs"].as_array().expect("top").len(), 2);
    assert_eq!(c[0]["top_logprobs"][1]["token_id"], 3);
    assert_eq!(c[0]["top_logprobs"][1]["token"], "t3");
    assert_eq!(c[1]["token_id"], 42);
    assert_eq!(c[1]["logprob"], -6.0);
    assert!(c[1]["top_logprobs"]
        .as_array()
        .expect("top")
        .iter()
        .all(|t| t["token_id"] != 42));
    let none = chat_logprobs_json(&steps, 0, &|id| id.to_string());
    assert!(none["content"][0]["top_logprobs"]
        .as_array()
        .expect("top")
        .is_empty());
    assert_eq!(
        none["content"][0]["logprob"].as_f64(),
        Some(f64::from(-0.1_f32))
    );
}
