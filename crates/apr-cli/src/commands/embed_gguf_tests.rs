//! Tests for `embed_gguf` (APR-EMBED-001 EG-2, #4901).
//!
//! Tests cover:
//! - `--task` names the model card's prompts, in the card's order, and `--task document` carries
//!   `--title`; the GGUF flags parse with no `--vocab`
//! - A GGUF file refuses each flag it answers itself, and each combination that means nothing,
//!   before the model is read; a generative or contradicting file is refused by kind
//! - An APR file refuses the GGUF-only flags and still requires `--vocab`

use super::*;
use crate::commands::model_kind_gate::tests::eg1_gguf;
use clap::ValueEnum;
use std::io::Write;

fn texts() -> Vec<String> {
    vec!["a".to_string()]
}

fn flags(texts: &[String]) -> EmbedFlags<'_> {
    EmbedFlags {
        texts,
        text_file: None,
        vocab: None,
        pool: "mean",
        normalize: true,
        task: None,
        title: None,
        dim: None,
        sizes: BertSizes::default(),
        json: true,
    }
}

fn file(bytes: &[u8]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(bytes).expect("write fixture");
    f
}

fn embedding_file() -> tempfile::NamedTempFile {
    file(&eg1_gguf("gemma-embedding2", None))
}

fn apr_file() -> tempfile::NamedTempFile {
    file(b"APR\0not a GGUF file")
}

fn assert_refused(model: &Path, flags: &EmbedFlags<'_>, code: u8, needle: &str) {
    let err = run(model, flags).expect_err("the call must be refused");
    let msg = err.to_string();
    assert_eq!(err.exit_code_value(), code, "{msg}");
    assert!(msg.contains(needle), "wanted {needle:?} in: {msg}");
}

#[test]
fn every_task_names_the_model_card_prompt_eg2() {
    let queries: Vec<Task> = EmbedTask::value_variants()
        .iter()
        .filter_map(|t| match prompt_of(Some(*t), None) {
            Prompt::Query(task) => Some(task),
            _ => None,
        })
        .collect();
    assert_eq!(queries, Task::ALL, "the seven query prompts, in order");
    assert_eq!(prompt_of(None, Some("t")), Prompt::Raw);
    assert_eq!(
        prompt_of(Some(EmbedTask::Document), Some("t")),
        Prompt::Document { title: Some("t") }
    );
    let names: Vec<String> = EmbedTask::value_variants()
        .iter()
        .filter_map(|t| t.to_possible_value().map(|v| v.get_name().to_string()))
        .collect();
    assert_eq!(
        names,
        [
            "search-query",
            "question-answering",
            "fact-checking",
            "code-retrieval",
            "classification",
            "clustering",
            "sentence-similarity",
            "document",
        ]
    );
}

/// The GGUF flags of a parsed `apr embed` argv: `(vocab given, task, title, dim)`.
type Parsed = (bool, Option<EmbedTask>, Option<String>, Option<usize>);

/// Parsed on a 16 MiB thread, as the `--normalize` argv tests in `embed.rs` are.
fn parse(argv: &[&str]) -> std::result::Result<Parsed, String> {
    let argv: Vec<String> = argv.iter().map(|s| (*s).to_string()).collect();
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            use clap::Parser;
            match crate::Cli::try_parse_from(&argv) {
                Ok(cli) => match *cli.command {
                    crate::Commands::Extended(crate::ExtendedCommands::Embed {
                        vocab,
                        task,
                        title,
                        dim,
                        ..
                    }) => Ok((
                        vocab.is_some(),
                        task,
                        title.map(|t| t.as_str().to_string()),
                        dim,
                    )),
                    other => panic!("expected ExtendedCommands::Embed, got {other:?}"),
                },
                Err(e) => Err(e.to_string()),
            }
        })
        .expect("spawn parse thread")
        .join()
        .expect("parse thread must not panic")
}

#[test]
fn the_gguf_flags_parse_without_a_vocab_eg2() {
    let parsed = parse(&[
        "apr", "embed", "m.gguf", "--text", "a", "--task", "document", "--title", "T", "--dim",
        "256",
    ]);
    assert_eq!(
        parsed,
        Ok((
            false,
            Some(EmbedTask::Document),
            Some("T".to_string()),
            Some(256)
        ))
    );
    let parsed = parse(&["apr", "embed", "m.gguf", "--text", "a"]);
    assert_eq!(parsed, Ok((false, None, None, None)));
    assert!(parse(&["apr", "embed", "m.gguf", "--task", "summarize"]).is_err());
}

#[test]
fn a_gguf_file_refuses_the_flags_it_answers_itself_eg2() {
    let model = embedding_file();
    let texts = texts();
    let vocab = Path::new("tokenizer.json");
    let with_vocab = EmbedFlags {
        vocab: Some(vocab),
        ..flags(&texts)
    };
    assert_refused(model.path(), &with_vocab, 5, "--vocab is for an APR file");
    let with_size = EmbedFlags {
        sizes: BertSizes {
            num_layers: Some(6),
            ..BertSizes::default()
        },
        ..flags(&texts)
    };
    assert_refused(
        model.path(),
        &with_size,
        5,
        "--num-layers is for an APR file",
    );
    let cls = EmbedFlags {
        pool: "cls",
        ..flags(&texts)
    };
    assert_refused(model.path(), &cls, 5, "--pool cls");
    let raw = EmbedFlags {
        normalize: false,
        ..flags(&texts)
    };
    assert_refused(model.path(), &raw, 5, "--normalize false");
}

#[test]
fn a_gguf_file_refuses_a_title_without_a_document_and_an_untrained_dim_eg2() {
    let model = embedding_file();
    let texts = texts();
    for task in [None, Some(EmbedTask::SearchQuery)] {
        let titled = EmbedFlags {
            task,
            title: Some("T"),
            ..flags(&texts)
        };
        assert_refused(model.path(), &titled, 5, "--title names a document");
    }
    for dim in [0, 300, 1024] {
        let wide = EmbedFlags {
            dim: Some(dim),
            ..flags(&texts)
        };
        assert_refused(model.path(), &wide, 5, &format!("--dim {dim}"));
    }
    assert_refused(model.path(), &flags(&[]), 5, "at least one --text");
}

#[test]
fn a_gguf_file_that_is_not_an_embedding_model_is_refused_by_kind_eg2() {
    let texts = texts();
    let generative = file(&eg1_gguf("llama", None));
    assert_refused(generative.path(), &flags(&texts), 6, "Use `apr run`");
    let causal = file(&eg1_gguf("gemma-embedding2", Some(true)));
    assert_refused(causal.path(), &flags(&texts), 6, "attention.causal = true");
}

#[test]
fn an_embedding_file_without_the_graph_fails_to_load_eg2() {
    // The fixture passes every flag check and the kind gate, then has one tensor of the graph.
    let model = embedding_file();
    let texts = texts();
    let document = EmbedFlags {
        task: Some(EmbedTask::Document),
        title: Some("T"),
        dim: Some(256),
        ..flags(&texts)
    };
    let err = run(model.path(), &document).expect_err("an incomplete graph is refused");
    assert_eq!(err.exit_code_value(), 6, "{err}");
}

#[test]
fn an_apr_file_refuses_the_gguf_flags_eg2() {
    let model = apr_file();
    let texts = texts();
    let vocab = Path::new("vocab.txt");
    let base = EmbedFlags {
        vocab: Some(vocab),
        ..flags(&texts)
    };
    let cases = [
        (
            EmbedFlags {
                task: Some(EmbedTask::SearchQuery),
                ..base
            },
            "--task",
        ),
        (
            EmbedFlags {
                title: Some("T"),
                ..base
            },
            "--title",
        ),
        (
            EmbedFlags {
                dim: Some(256),
                ..base
            },
            "--dim",
        ),
    ];
    for (case, flag) in &cases {
        assert_refused(
            model.path(),
            case,
            5,
            &format!("{flag} applies to an EmbeddingGemma 2 GGUF file"),
        );
    }
}

#[test]
fn an_apr_file_still_requires_a_vocab_eg2() {
    let model = apr_file();
    let texts = texts();
    assert_refused(
        model.path(),
        &flags(&texts),
        5,
        "--vocab is required for an APR file",
    );
}

#[test]
fn an_unreadable_file_is_refused_before_either_path_eg2() {
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = dir.path().join("absent.gguf");
    let texts = texts();
    assert_refused(&missing, &flags(&texts), 5, "Failed to read");
}
