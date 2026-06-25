
// PMAT-125 B2: unit tests for inference_result.rs trace-parsing helpers
// (rosetta-scoped). Included into rosetta_08.rs's `mod tests`.

#[test]
fn parse_selected_token_extracts_id_and_logit() {
    let line = "  Selected token: 42 (logit: 7.5)";
    assert_eq!(parse_selected_token(line), Some((42, Some(7.5))));
}

#[test]
fn parse_selected_token_id_without_logit() {
    let line = "Selected token: 100 (foo)";
    assert_eq!(parse_selected_token(line), Some((100, None)));
}

#[test]
fn parse_selected_token_none_on_unrelated_line() {
    assert_eq!(parse_selected_token("Loading model..."), None);
}

#[test]
fn parse_selected_token_none_on_non_numeric_id() {
    assert_eq!(parse_selected_token("Selected token: abc (logit: 1.0)"), None);
}

#[test]
fn parse_top5_line_extracts_ids() {
    let line = "Top 5 tokens: (10, 0.5), (20, 0.3), (30, 0.1)";
    assert_eq!(parse_top5_line(line), Some(vec![10, 20, 30]));
}

#[test]
fn parse_top5_line_none_on_unrelated() {
    assert_eq!(parse_top5_line("nothing here"), None);
}

#[test]
fn parse_top5_line_none_when_empty() {
    // "Top 5 tokens:" marker but no parseable pairs.
    assert_eq!(parse_top5_line("Top 5 tokens: garbage"), None);
}

#[test]
fn parse_trace_lines_collects_tokens_logits_top5() {
    let combined = "\
Loading\n\
Selected token: 5 (logit: 1.5)\n\
Top 5 tokens: (5, 0.9), (6, 0.05)\n\
Selected token: 6 (logit: 2.0)\n\
Top 5 tokens: (6, 0.8), (7, 0.1)\n";
    let (tokens, logits, top5) = parse_trace_lines(combined);
    assert_eq!(tokens, vec![5, 6]);
    assert_eq!(logits, vec![1.5, 2.0]);
    assert_eq!(top5, vec![vec![5, 6], vec![6, 7]]);
}

#[test]
fn parse_trace_lines_token_without_logit_skips_logit_vec() {
    let combined = "Selected token: 9 (no logit field)\n";
    let (tokens, logits, top5) = parse_trace_lines(combined);
    assert_eq!(tokens, vec![9]);
    assert!(logits.is_empty());
    assert!(top5.is_empty());
}

#[test]
fn strip_ansi_removes_color_codes() {
    let colored = "\x1b[31mred\x1b[0m text";
    assert_eq!(strip_ansi(colored), "red text");
}

#[test]
fn strip_ansi_passes_through_plain_text() {
    assert_eq!(strip_ansi("plain"), "plain");
}

#[test]
fn extract_clean_output_filters_noise_lines() {
    let stdout = "\
Loading model\n\
Model loaded in 1s\n\
Prompt tokens: 3\n\
Temperature: 0.0\n\
Hello world\n\
Generated (5 tokens)\n\
12.0 tok/s\n";
    // Only the real content line survives.
    assert_eq!(extract_clean_output(stdout), "Hello world");
}

#[test]
fn extract_clean_output_strips_spinner_and_ansi() {
    let stdout = "\x1b[32m⠋⠙Answer\x1b[0m\n";
    assert_eq!(extract_clean_output(stdout), "Answer");
}

#[test]
fn extract_clean_output_empty_when_all_noise() {
    let stdout = "Loading\n[debug]\nERROR boom\n";
    assert_eq!(extract_clean_output(stdout), "");
}

#[test]
fn is_transposed_dims_detects_swap() {
    assert!(is_transposed_dims(&[10, 20], &[20, 10]));
}

#[test]
fn is_transposed_dims_false_on_identical_square() {
    assert!(!is_transposed_dims(&[896, 896], &[896, 896]));
}

#[test]
fn is_transposed_dims_false_on_non_2d() {
    assert!(!is_transposed_dims(&[10, 20, 30], &[30, 20, 10]));
    assert!(!is_transposed_dims(&[10], &[10]));
}

#[test]
fn normalize_tensor_name_gguf_to_canonical() {
    assert_eq!(
        normalize_tensor_name("blk.0.attn_q.weight"),
        "0.q_proj.weight"
    );
    assert_eq!(
        normalize_tensor_name("blk.3.ffn_gate.weight"),
        "3.gate_proj.weight"
    );
}

#[test]
fn normalize_tensor_name_hf_to_canonical() {
    assert_eq!(
        normalize_tensor_name("model.layers.0.self_attn.q_proj.weight"),
        "0.q_proj.weight"
    );
}

#[test]
fn normalize_tensor_name_output_to_lm_head() {
    assert_eq!(normalize_tensor_name("output.weight"), "lm_head.weight");
}

#[test]
fn normalize_tensor_name_token_embd() {
    assert_eq!(
        normalize_tensor_name("token_embd.weight"),
        "embed_tokens.weight"
    );
}
