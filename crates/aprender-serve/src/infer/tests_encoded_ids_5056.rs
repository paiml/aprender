//! #5056: the `-v` encoded-ids line names every prompt id, never a prefix.

use super::encoded_tokens_debug_line;

#[test]
fn encoded_ids_line_names_every_id_past_thirty_5056() {
    let tokens: Vec<u32> = (100..156).collect();
    let line = encoded_tokens_debug_line(true, &tokens);
    let list = line
        .split_once("tokens: ")
        .map(|(_, l)| l)
        .expect("line has an id list");
    let ids: Vec<u32> = list
        .trim_matches(|c| c == '[' || c == ']')
        .split(", ")
        .map(|s| s.parse().expect("id"))
        .collect();
    assert!(
        line.starts_with("[DEBUG] add_bos=true, encoded 56 tokens: "),
        "{line}"
    );
    assert_eq!(ids, tokens, "the printed ids are the encoded ids, all 56");
}

#[test]
fn encoded_ids_line_keeps_the_harness_format_5056() {
    // scripts/check_crux_greedy_rows.sh fixtures parse exactly this shape.
    assert_eq!(
        encoded_tokens_debug_line(false, &[7, 9, 9, 1]),
        "[DEBUG] add_bos=false, encoded 4 tokens: [7, 9, 9, 1]"
    );
}
