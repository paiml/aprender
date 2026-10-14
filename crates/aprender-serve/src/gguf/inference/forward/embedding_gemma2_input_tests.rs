//! Tests for `embedding_gemma2_input` (APR-EMBED-001 EG-2, M5).
//!
//! Tests cover:
//! - Every task prompt and the document form equal the corpus prompt table, row for row
//! - The framing: BOS and EOS where the file's flags ask for them, and nowhere else
//! - Special tokens in the text are matched, as llama.cpp does with `parse_special`
//! - Refusals: a vocabulary that is not SPM-style BPE, no merges, a missing or non-bool flag,
//!   a missing or out-of-range special id

use super::{EmbedTokenizer, Prompt, Task};
use crate::gguf::test_factory::GGUFBuilder;
use crate::gguf::GGUFModel;

const NORMAL: i32 = 1;
const CONTROL: i32 = 3;

/// The vocabulary of the fixture: the four specials at the ids the real files use, then pieces.
const TOKENS: [&str; 10] = [
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "a",
    "b",
    "ab",
    "\u{2581}",
    "\u{2581}a",
    "\u{2581}ab",
];

/// Ranked merges: `▁ a` first, so "ab ab" spells `ab` then `▁ab`.
const MERGES: [&str; 3] = ["\u{2581} a", "a b", "\u{2581}a b"];

/// The fixture's tokenizer metadata; a field set to `None` (or `false`) leaves its key out.
struct Vocab {
    model: &'static str,
    merges: bool,
    add_bos: Option<bool>,
    add_eos: Option<bool>,
    add_eos_as_u32: bool,
    bos: Option<u32>,
    eos: Option<u32>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self {
            model: "gemma4",
            merges: true,
            add_bos: Some(true),
            add_eos: Some(true),
            add_eos_as_u32: false,
            bos: Some(2),
            eos: Some(1),
        }
    }
}

impl Vocab {
    fn model(&self) -> GGUFModel {
        let types: Vec<i32> = TOKENS
            .iter()
            .map(|t| if t.starts_with('<') { CONTROL } else { NORMAL })
            .collect();
        let mut b = GGUFBuilder::new()
            .architecture("gemma-embedding2")
            .add_string("tokenizer.ggml.model", self.model)
            .add_string_array("tokenizer.ggml.tokens", &TOKENS)
            .add_i32_array("tokenizer.ggml.token_type", &types);
        if self.merges {
            b = b.add_string_array("tokenizer.ggml.merges", &MERGES);
        }
        if let Some(flag) = self.add_bos {
            b = b.add_bool("tokenizer.ggml.add_bos_token", flag);
        }
        if self.add_eos_as_u32 {
            b = b.add_u32("tokenizer.ggml.add_eos_token", 1);
        } else if let Some(flag) = self.add_eos {
            b = b.add_bool("tokenizer.ggml.add_eos_token", flag);
        }
        if let Some(id) = self.bos {
            b = b.add_u32("tokenizer.ggml.bos_token_id", id);
        }
        if let Some(id) = self.eos {
            b = b.add_u32("tokenizer.ggml.eos_token_id", id);
        }
        GGUFModel::from_bytes(&b.build()).expect("fixture parses")
    }

    fn tokenizer(&self) -> EmbedTokenizer {
        EmbedTokenizer::from_gguf(&self.model()).expect("fixture tokenizer builds")
    }

    fn refused(&self, needle: &str) {
        let err = EmbedTokenizer::from_gguf(&self.model())
            .expect_err("the fixture must be refused")
            .to_string();
        assert!(err.contains(needle), "wanted {needle:?} in: {err}");
    }
}

/// `docs/audits/APR-EMBED-001/corpus/prompts.tsv`: `name \t query_template \t document_template`.
fn prompt_table() -> Vec<(String, String, String)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/audits/APR-EMBED-001/corpus/prompts.tsv"
    );
    let text = std::fs::read_to_string(path).expect("the corpus prompt table is in the workspace");
    text.lines()
        .skip(1)
        .map(|line| {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(cols.len(), 3, "row {line:?}");
            (
                cols[0].to_string(),
                cols[1].to_string(),
                cols[2].to_string(),
            )
        })
        .collect()
}

fn task_named(name: &str) -> Task {
    let task = match name {
        "SearchQuery" => Task::SearchQuery,
        "QuestionAnswering" => Task::QuestionAnswering,
        "FactChecking" => Task::FactChecking,
        "CodeRetrieval" => Task::CodeRetrieval,
        "Classification" => Task::Classification,
        "Clustering" => Task::Clustering,
        "SentenceSimilarity" => Task::SentenceSimilarity,
        other => panic!("no task named {other}"),
    };
    assert_eq!(format!("{task:?}"), name, "the table names the variant");
    task
}

#[test]
fn every_prompt_equals_the_corpus_prompt_table_eg2() {
    let rows = prompt_table();
    let mut tasks = Vec::new();
    for (name, query, document) in &rows {
        if name == "Document" {
            assert_eq!(query, "-");
            assert_eq!(
                Prompt::Document { title: None }.apply("{content}"),
                *document
            );
            continue;
        }
        let task = task_named(name);
        tasks.push(task);
        assert_eq!(Prompt::Query(task).apply("{content}"), *query, "{name}");
        if document != "-" {
            assert_eq!(
                Prompt::Document {
                    title: Some("{title}")
                }
                .apply("{content}"),
                *document,
                "{name}"
            );
        }
    }
    assert_eq!(tasks, Task::ALL, "the table lists every task, in order");
    assert_eq!(
        rows.len(),
        Task::ALL.len() + 1,
        "seven tasks plus the document form"
    );
}

#[test]
fn the_raw_prompt_is_the_text_alone_eg2() {
    assert_eq!(Prompt::Raw.apply("a b"), "a b");
    assert_eq!(Prompt::Raw.name(), "raw");
    assert_eq!(Prompt::Query(Task::Clustering).name(), "clustering");
    assert_eq!(Prompt::Document { title: Some("t") }.name(), "document");
}

#[test]
fn bos_and_eos_frame_the_body_when_the_file_asks_eg2() {
    let tok = Vocab::default().tokenizer();
    // "ab ab" -> "ab▁ab": `▁ a`, then `a b`, then `▁a b`, so two pieces.
    assert_eq!(tok.tokens(&Prompt::Raw, "ab ab"), [2, 6, 9, 1]);
    assert_eq!(
        tok.tokens(&Prompt::Raw, ""),
        [2, 1],
        "an empty text is the frame alone"
    );
}

#[test]
fn a_flag_that_is_false_adds_nothing_eg2() {
    let bos_only = Vocab {
        add_eos: Some(false),
        ..Vocab::default()
    };
    assert_eq!(bos_only.tokenizer().tokens(&Prompt::Raw, "ab"), [2, 6]);
    let eos_only = Vocab {
        add_bos: Some(false),
        ..Vocab::default()
    };
    assert_eq!(eos_only.tokenizer().tokens(&Prompt::Raw, "ab"), [6, 1]);
    // An id the file never uses need not exist when its flag is false.
    let bare = Vocab {
        add_bos: Some(false),
        add_eos: Some(false),
        bos: None,
        eos: Some(99),
        ..Vocab::default()
    };
    assert_eq!(bare.tokenizer().tokens(&Prompt::Raw, "ab"), [6]);
}

#[test]
fn special_tokens_in_the_text_are_matched_eg2() {
    let tok = Vocab::default().tokenizer();
    assert_eq!(tok.tokens(&Prompt::Raw, "ab<eos>"), [2, 6, 1, 1]);
}

#[test]
fn the_prompt_reaches_the_tokens_eg2() {
    let tok = Vocab::default().tokenizer();
    let prompts = [
        Prompt::Query(Task::SearchQuery),
        Prompt::Document { title: None },
        Prompt::Document { title: Some("ab") },
    ];
    for prompt in prompts {
        assert_eq!(
            tok.tokens(&prompt, "ab"),
            tok.tokens(&Prompt::Raw, &prompt.apply("ab")),
            "{prompt:?}"
        );
    }
    assert_ne!(
        tok.tokens(&Prompt::Raw, "ab"),
        tok.tokens(&prompts[0], "ab")
    );
}

#[test]
fn a_vocabulary_that_is_not_spm_bpe_is_refused_eg2() {
    Vocab {
        model: "llama",
        ..Vocab::default()
    }
    .refused("tokenizer.ggml.model 'llama' is not SPM-style BPE");
}

#[test]
fn a_vocabulary_without_merges_is_refused_eg2() {
    Vocab {
        merges: false,
        ..Vocab::default()
    }
    .refused("no tokenizer.ggml.merges");
}

#[test]
fn a_missing_framing_flag_is_refused_eg2() {
    Vocab {
        add_bos: None,
        ..Vocab::default()
    }
    .refused("tokenizer.ggml.add_bos_token missing");
    Vocab {
        add_eos: None,
        ..Vocab::default()
    }
    .refused("tokenizer.ggml.add_eos_token missing");
}

#[test]
fn a_framing_flag_that_is_not_a_bool_is_refused_eg2() {
    Vocab {
        add_eos_as_u32: true,
        ..Vocab::default()
    }
    .refused("tokenizer.ggml.add_eos_token is not a bool");
}

#[test]
fn a_special_id_the_flag_asks_for_must_exist_eg2() {
    Vocab {
        bos: None,
        ..Vocab::default()
    }
    .refused("tokenizer.ggml.bos_token_id is missing");
    Vocab {
        eos: Some(10),
        ..Vocab::default()
    }
    .refused("tokenizer.ggml.eos_token_id 10 is outside the vocabulary of 10");
}
