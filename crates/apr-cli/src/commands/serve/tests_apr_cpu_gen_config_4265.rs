//! #4265: the APR CPU serve path stops at the model's EOS / chat-turn end and
//! honours the request's `top_p`, the same as every other serve backend.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use std::collections::HashMap;

fn state_with(
    embedded: Option<realizar::apr::BpeTokenizer>,
    eos_from_tokenizer_json: Option<u32>,
) -> AprServerState {
    AprServerState {
        transformer: None,
        model_type: "test".to_string(),
        architecture: "test".to_string(),
        is_transformer: false,
        tokenizer: eos_from_tokenizer_json.map(|eos| SafeTensorsTokenizerInfo {
            tokenizer: std::sync::Arc::new(
                realizar::tokenizer::BPETokenizer::new(vec!["<unk>".to_string()], vec![], "<unk>")
                    .expect("tiny tokenizer"),
            ),
            vocab: vec![],
            bos_token_id: None,
            eos_token_id: Some(eos),
        }),
        embedded_tokenizer: embedded,
        model_name: "apr".to_string(),
        demo_scripted_tokens: None,
    }
}

fn embedded(eos: Option<u32>, specials: &[(&str, u32)]) -> realizar::apr::BpeTokenizer {
    realizar::apr::BpeTokenizer {
        token_to_id: HashMap::new(),
        id_to_token: vec![],
        merge_rules: vec![],
        bos_id: None,
        eos_id: eos,
        special_tokens: specials
            .iter()
            .map(|(s, i)| ((*s).to_string(), *i))
            .collect(),
    }
}

#[test]
fn stop_set_holds_embedded_eos_and_chat_turn_end() {
    // A Qwen-shaped GGUF import: EOS <|endoftext|> = 151643, turn end <|im_end|> = 151645.
    let s = state_with(
        Some(embedded(
            Some(151_643),
            &[("<|im_end|>", 151_645), ("<|im_start|>", 151_644)],
        )),
        None,
    );
    let mut stop = apr_cpu_stop_tokens(&s);
    stop.sort_unstable();
    assert_eq!(
        stop,
        vec![151_643, 151_645],
        "<|im_start|> is not a stop token"
    );
}

#[test]
fn stop_set_holds_sibling_tokenizer_json_eos() {
    let s = state_with(None, Some(2));
    assert_eq!(apr_cpu_stop_tokens(&s), vec![2]);
}

#[test]
fn no_tokenizer_means_no_invented_stop_ids() {
    assert!(apr_cpu_stop_tokens(&state_with(None, None)).is_empty());
}

#[test]
fn config_carries_the_stop_set_and_the_requests_top_p() {
    let c = apr_cpu_generate_config(16, 0.7, Some(0.5), vec![151_645]);
    assert_eq!(c.stop_tokens, vec![151_645]);
    assert!(
        (c.top_p - 0.5).abs() < f32::EPSILON,
        "request top_p ignored: {}",
        c.top_p
    );
}

#[test]
fn absent_top_p_takes_the_other_backends_default() {
    let want = realizar::gguf::QuantizedGenerateConfig::default().top_p;
    let c = apr_cpu_generate_config(16, 0.7, None, vec![]);
    assert!(
        (c.top_p - want).abs() < f32::EPSILON,
        "top_p {} != default {want}",
        c.top_p
    );
}

#[test]
fn reply_drops_the_stop_id_the_loop_ended_on() {
    let stop = [151_645, 0];
    assert_eq!(apr_cpu_reply_tokens(&[9, 8, 151_645], &stop), &[9, 8]);
    // Token 0 ends the loop even when no tokenizer named it (is_eos_token).
    assert_eq!(apr_cpu_reply_tokens(&[9, 8, 0], &stop), &[9, 8]);
}

#[test]
fn reply_keeps_a_budget_cut_and_an_inner_stop_id() {
    let stop = [151_645, 0];
    assert_eq!(apr_cpu_reply_tokens(&[9, 8, 7], &stop), &[9, 8, 7]);
    assert_eq!(apr_cpu_reply_tokens(&[151_645, 8], &stop), &[151_645, 8]);
    assert!(apr_cpu_reply_tokens(&[], &stop).is_empty());
}

#[test]
fn stop_set_holds_sibling_tokenizer_json_turn_end_by_vocab_index() {
    let mut s = state_with(None, Some(2));
    if let Some(tok) = s.tokenizer.as_mut() {
        tok.vocab = ["a", "<|im_start|>", "b", "<|im_end|>"]
            .iter()
            .map(|t| (*t).to_string())
            .collect();
    }
    // <|im_end|> sits at index 3; <|im_start|> (index 1) is not a stop.
    assert_eq!(apr_cpu_stop_tokens(&s), vec![2, 3]);
}
