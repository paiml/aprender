//! #4979: serve tokenized GGUF prompts by greedy longest-match, so `TANGERINE` became
//! `ĠTA|NG|ER|INE` where the ranked merges give `ĠT|ANGER|INE`. A `BPETokenizer` built from a
//! GGUF now carries the model's `ByteLevelBpe` (the encoder `apr run` uses) and `encode`
//! defers to it. These rows use a vocabulary where greedy and ranked-BPE ids differ.

use crate::gguf::byte_level_bpe::{ByteLevelBpe, PreTokenizer};
use crate::tokenizer::BPETokenizer;

fn glyph_vocab(extra: &[&str]) -> Vec<String> {
    // GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100+n.
    let mut glyphs = Vec::with_capacity(256);
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        if printable {
            glyphs.push(char::from_u32(b).expect("latin1"));
        } else {
            glyphs.push(char::from_u32(0x100 + n).expect("glyph"));
            n += 1;
        }
    }
    let mut v: Vec<String> = glyphs.iter().map(char::to_string).collect();
    v.extend(extra.iter().map(|s| (*s).to_string()));
    v
}

fn id_of(vocab: &[String], t: &str) -> u32 {
    u32::try_from(vocab.iter().position(|v| v == t).expect("in vocab")).expect("id")
}

#[test]
fn encode_uses_ranked_merges_not_greedy_longest_match() {
    // Greedy longest-match would take "ĠTA" (longest prefix); ranked merges build "ĠT" first.
    let extra = ["ĠT", "ĠTA", "AN", "ANG", "ANGE", "ANGER", "ER", "INE", "NG", "GER", "IN"];
    let vocab = glyph_vocab(&extra);
    let merges = [
        "Ġ T", "A N", "AN G", "ANG E", "ANGE R", "I N", "IN E",
    ];
    let bpe = std::sync::Arc::new(ByteLevelBpe::build(PreTokenizer::Qwen2, &vocab, &merges, &[]));
    let want = bpe.encode(" TANGERINE");
    assert_eq!(
        want,
        vec![id_of(&vocab, "ĠT"), id_of(&vocab, "ANGER"), id_of(&vocab, "INE")]
    );

    let plain = BPETokenizer::new(vocab.clone(), vec![], None).expect("tokenizer");
    assert_ne!(
        plain.encode(" TANGERINE"),
        want,
        "the vocabulary-only tokenizer is the greedy one (the #4979 defect)"
    );

    let served = plain.with_gguf_encoder(std::sync::Arc::clone(&bpe));
    assert_eq!(served.encode(" TANGERINE"), want, "serve ids == run ids");
    assert!(served.encode("").is_empty());
}
