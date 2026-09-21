/// #3787: the greedy encoder (every non-byte-level GGUF, and a byte-level one whose
/// pre-tokenizer is not implemented) is linear in the prompt and gives the ids it gave
/// before. The reference functions below are the pre-#3787 code, verbatim.
#[cfg(test)]
mod greedy_linear_tests {
    use super::*;
    use crate::gguf::test_factory::GGUFBuilder;

    /// The pre-#3787 prefix offsets: every character's offset, then the first 32 used.
    fn reference_prefix_ends(remaining: &str) -> Vec<usize> {
        let char_indices: Vec<usize> = remaining
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(remaining.len()))
            .collect();
        (1..=char_indices.len().saturating_sub(1).min(32))
            .map(|c| char_indices[c])
            .collect()
    }

    /// The pre-#3787 special split: every special searched for in the rest, every split.
    fn reference_split<'t>(
        text: &'t str,
        special_tokens: &[(&'t str, u32)],
    ) -> Vec<(bool, &'t str)> {
        let mut segments = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let earliest = special_tokens
                .iter()
                .filter_map(|&(tok, _)| rest.find(tok).map(|pos| (pos, tok)))
                .min_by_key(|&(pos, _)| pos);
            let Some((pos, tok)) = earliest else {
                segments.push((false, rest));
                break;
            };
            if pos > 0 {
                segments.push((false, &rest[..pos]));
            }
            segments.push((true, tok));
            rest = &rest[pos + tok.len()..];
        }
        segments
    }

    /// The pre-#3787 greedy encode loop over one non-special segment (already prefixed and
    /// with its spaces replaced).
    fn reference_greedy(processed: &str, token_to_id: &HashMap<&str, u32>, out: &mut Vec<u32>) {
        let mut remaining = processed;
        while !remaining.is_empty() {
            let mut best_byte_len = 0;
            let mut best_id = None;
            for byte_end in reference_prefix_ends(remaining) {
                if let Some(&id) = token_to_id.get(&remaining[..byte_end]) {
                    best_byte_len = byte_end;
                    best_id = Some(id);
                }
            }
            if let Some(id) = best_id {
                out.push(id);
                remaining = &remaining[best_byte_len..];
            } else {
                let ch = remaining.chars().next().expect("non-empty");
                for byte in remaining[..ch.len_utf8()].bytes() {
                    let byte_token = format!("<0x{byte:02X}>");
                    out.push(token_to_id.get(byte_token.as_str()).copied().unwrap_or(0));
                }
                remaining = &remaining[ch.len_utf8()..];
            }
        }
    }

    /// The pre-#3787 `encode` for a vocabulary that takes the greedy path.
    fn reference_encode(vocab: &[&str], gpt2: bool, text: &str) -> Vec<u32> {
        let token_to_id: HashMap<&str, u32> = vocab
            .iter()
            .enumerate()
            .map(|(id, t)| (*t, id as u32))
            .collect();
        let specials: Vec<(&str, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(_, t)| t.starts_with("<|") && t.ends_with("|>"))
            .map(|(id, t)| (*t, id as u32))
            .collect();
        let mut out = Vec::new();
        for (is_special, segment) in reference_split(text, &specials) {
            if is_special {
                out.extend(token_to_id.get(segment));
                continue;
            }
            let processed = if gpt2 {
                segment.replace(' ', "\u{0120}").replace('\n', "\u{010A}")
            } else if segment.starts_with(' ') {
                segment.replace(' ', "▁")
            } else {
                format!(" {segment}").replace(' ', "▁")
            };
            reference_greedy(&processed, &token_to_id, &mut out);
        }
        out
    }

    /// A SentencePiece-style vocabulary: words, pieces, multi-byte characters, byte tokens,
    /// a 40-character token (longer than the scan's 32), and three specials, one of which
    /// (`<|endoftext|>`) the texts below never contain.
    const SPM_VOCAB: &[&str] = &[
        "<unk>",
        "▁the",
        "▁qu",
        "ick",
        "▁",
        "é",
        "日本",
        "<0xE2>",
        "<0x80>",
        "<0x94>",
        "▁fn",
        "(",
        ")",
        "▁x",
        "x",
        "e",
        "t",
        "h",
        "▁th",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "user",
        "\n",
        "▁aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "a",
        "▁a",
    ];

    fn spm_model() -> GGUFModel {
        let data = GGUFBuilder::new()
            .architecture("llama")
            .add_string("tokenizer.ggml.model", "llama")
            .add_string_array("tokenizer.ggml.tokens", SPM_VOCAB)
            .build();
        GGUFModel::from_bytes(&data).expect("parse")
    }

    /// One chat turn with multi-byte text, an em dash only byte tokens can spell, a word
    /// longer than the scan, and specials at both ends.
    const TURN: &str = "<|im_start|>user\nthe quick fn(x) — é日本 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa the<|im_end|>\n";

    #[test]
    fn prefix_ends_are_the_offsets_they_replaced() {
        let long_ascii = "a".repeat(100);
        let long_multi = "é日—x".repeat(30);
        for s in [
            "a",
            "é",
            "日本",
            TURN,
            &long_ascii,
            &long_multi,
            &"b".repeat(32),
            &"b".repeat(33),
        ] {
            assert_eq!(
                greedy_prefix_ends(s).collect::<Vec<_>>(),
                reference_prefix_ends(s),
                "{s:?}"
            );
        }
    }

    #[test]
    fn special_split_is_the_split_it_replaced() {
        let specials = [("<|a|>", 0), ("<|a|>b|>", 1), ("<|none|>", 2), ("<|c|>", 3)];
        let reversed: Vec<_> = specials.iter().rev().copied().collect();
        for text in [
            "",
            "plain text",
            "<|a|>",
            "<|a|>b|> tie: both match here; the first listed wins",
            "x<|c|><|c|>y<|a|>z<|a|>",
            "<|c|>tail",
            "lead<|c|>",
            TURN,
        ] {
            assert_eq!(
                split_on_special_tokens(text, &specials),
                reference_split(text, &specials),
                "{text:?}"
            );
            assert_eq!(
                split_on_special_tokens(text, &reversed),
                reference_split(text, &reversed),
                "{text:?}"
            );
        }
    }

    #[test]
    fn greedy_encode_gives_the_ids_it_gave_before() {
        let model = spm_model();
        for text in [
            TURN,
            &TURN.repeat(5),
            " leading space",
            "no specials at all — é",
            "日本日本x",
        ] {
            assert_eq!(
                model.encode(text).expect("vocab"),
                reference_encode(SPM_VOCAB, false, text),
                "{text:?}"
            );
        }
        // A byte-level vocabulary with no merges takes the greedy fallback too.
        let gpt2_vocab = [
            "t",
            "h",
            "e",
            "\u{0120}the",
            "\u{010A}",
            "<|im_start|>",
            "<|none|>",
            "user",
        ];
        let data = GGUFBuilder::new()
            .architecture("gpt2")
            .add_string("tokenizer.ggml.model", "gpt2")
            .add_string_array("tokenizer.ggml.tokens", &gpt2_vocab)
            .build();
        let gpt2 = GGUFModel::from_bytes(&data).expect("parse");
        let text = "<|im_start|>user\nthe the\nthe";
        assert_eq!(
            gpt2.encode(text).expect("vocab"),
            reference_encode(&gpt2_vocab, true, text)
        );
    }

    /// The work `encode` does on `text`, in characters read and bytes searched.
    fn greedy_work(model: &GGUFModel, text: &str) -> usize {
        GREEDY_WORK.with(|w| w.set(0));
        model.encode(text).expect("vocab");
        GREEDY_WORK.with(std::cell::Cell::get)
    }

    /// `k` chat turns, then one special-free body `k` lines long. Both must grow: the greedy
    /// scan runs inside one special-free segment, so turns alone never show a scan that
    /// reads to the end of its segment, and the body alone never shows the special split.
    fn prompt(k: usize) -> String {
        let body_line = TURN
            .trim_start_matches("<|im_start|>")
            .trim_end_matches("<|im_end|>\n");
        format!("{}{}", TURN.repeat(k), format!("{body_line}\n").repeat(k))
    }

    /// #3787 done_when 4, the complexity class: 4× the prompt costs at most 5× the work.
    /// It counts work, not time, so it cannot flake on a loaded runner. A scan that reads
    /// the rest of its segment per token, or searches the rest of the text for every
    /// special at every split, grows ~9-16× here.
    #[test]
    fn greedy_encode_work_is_linear_in_the_prompt() {
        let model = spm_model();
        let n = prompt(16);
        let n4 = prompt(64);
        assert!(
            !n4.contains("<|endoftext|>"),
            "one special must be absent from the text"
        );
        let (w, w4) = (greedy_work(&model, &n), greedy_work(&model, &n4));
        assert!(w > 0, "the counter must see the greedy scan");
        assert!(
            w4 <= 5 * w,
            "4x the prompt ({} -> {} bytes) cost {:.1}x the work ({w} -> {w4}): superlinear",
            n.len(),
            n4.len(),
            w4 as f64 / w as f64
        );
    }
}
