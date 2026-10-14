//! `apr embed` on either file kind (APR-EMBED-001 EG-2, #4901).
//!
//! The file's magic bytes pick the path. An APR file takes the BERT path of GH-326 unchanged
//! (`super::embed`). A GGUF file is an EmbeddingGemma 2 model: its tokenizer, sizes and pooling
//! come from the file, so the flags that would restate them are refused rather than ignored,
//! and the vectors come from realizar's forward over the input that `apr serve` builds too
//! (`realizar::gguf::embedding_gemma2_input`). Contract:
//! `contracts/embeddinggemma2-forward-v1.yaml`.

use super::embed;
use super::model_header::read_prefix;
use crate::error::{CliError, Result};
use crate::EmbedTask;
use realizar::gguf::embedding_gemma2_input::{EmbedTokenizer, Prompt, Task};
use realizar::gguf::forward_embedding_gemma2::{truncate_to, EmbeddingGemma2, MATRYOSHKA_DIMS};
use realizar::gguf::{GGUFModel, MappedGGUFModel};
use realizar::model_kind::{gguf_model_kind, ModelKind};
use std::path::Path;

/// The BERT size overrides; each is `None` unless its flag was given.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BertSizes {
    pub(crate) hidden_dim: Option<usize>,
    pub(crate) num_layers: Option<usize>,
    pub(crate) num_heads: Option<usize>,
    pub(crate) intermediate_dim: Option<usize>,
    pub(crate) vocab_size: Option<usize>,
    pub(crate) max_position_embeddings: Option<usize>,
    pub(crate) type_vocab_size: Option<usize>,
}

impl BertSizes {
    /// The long flag of the first size that was given.
    fn first_given(&self) -> Option<&'static str> {
        [
            ("--hidden-dim", self.hidden_dim),
            ("--num-layers", self.num_layers),
            ("--num-heads", self.num_heads),
            ("--intermediate-dim", self.intermediate_dim),
            ("--vocab-size", self.vocab_size),
            ("--max-position-embeddings", self.max_position_embeddings),
            ("--type-vocab-size", self.type_vocab_size),
        ]
        .into_iter()
        .find_map(|(flag, value)| value.map(|_| flag))
    }
}

/// The parsed `apr embed` flags.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EmbedFlags<'a> {
    pub(crate) texts: &'a [String],
    pub(crate) text_file: Option<&'a Path>,
    pub(crate) vocab: Option<&'a Path>,
    pub(crate) pool: &'a str,
    pub(crate) normalize: bool,
    pub(crate) task: Option<EmbedTask>,
    pub(crate) title: Option<&'a str>,
    pub(crate) dim: Option<usize>,
    pub(crate) sizes: BertSizes,
    pub(crate) json: bool,
}

/// Entry point: route `model` by its magic bytes.
///
/// # Errors
/// An unreadable file, a flag the file's kind does not take, and every error of the path taken.
pub(crate) fn run(model: &Path, flags: &EmbedFlags<'_>) -> Result<()> {
    let head = read_prefix(model, 4).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to read {}: {e}", model.display()))
    })?;
    if head.as_slice() == b"GGUF" {
        run_gguf(model, flags)
    } else {
        run_bert(model, flags)
    }
}

fn refused(reason: String) -> CliError {
    CliError::ValidationFailed(reason)
}

/// An APR file: the GH-326 BERT path, with its defaults (MiniLM-L6) for the sizes not given.
fn run_bert(model: &Path, flags: &EmbedFlags<'_>) -> Result<()> {
    let gguf_only = [
        ("--task", flags.task.is_some()),
        ("--title", flags.title.is_some()),
        ("--dim", flags.dim.is_some()),
    ];
    if let Some((flag, _)) = gguf_only.iter().find(|(_, given)| *given) {
        return Err(refused(format!(
            "{flag} applies to an EmbeddingGemma 2 GGUF file; {} is not a GGUF file",
            model.display()
        )));
    }
    let vocab = flags.vocab.ok_or_else(|| {
        refused(format!(
            "--vocab is required for an APR file: {} carries no tokenizer",
            model.display()
        ))
    })?;
    let s = flags.sizes;
    embed::run(
        model,
        flags.texts,
        flags.text_file,
        vocab,
        flags.pool,
        flags.normalize,
        s.hidden_dim.unwrap_or(384),
        s.num_layers.unwrap_or(6),
        s.num_heads.unwrap_or(12),
        s.intermediate_dim.unwrap_or(1536),
        s.vocab_size.unwrap_or(30522),
        s.max_position_embeddings.unwrap_or(512),
        s.type_vocab_size.unwrap_or(2),
        flags.json,
    )
}

/// The flags a GGUF file answers itself, and the combinations that mean nothing, refused before
/// the model is read.
fn refuse_gguf_flags(flags: &EmbedFlags<'_>) -> Result<()> {
    if flags.vocab.is_some() {
        return Err(refused(
            "--vocab is for an APR file; a GGUF file carries its own tokenizer".to_string(),
        ));
    }
    if let Some(flag) = flags.sizes.first_given() {
        return Err(refused(format!(
            "{flag} is for an APR file; a GGUF file's sizes come from the file"
        )));
    }
    if flags.pool != "mean" {
        return Err(refused(format!(
            "--pool {}: an EmbeddingGemma 2 file declares mean pooling, and that is the one it takes",
            flags.pool
        )));
    }
    if !flags.normalize {
        return Err(refused(
            "--normalize false: an EmbeddingGemma 2 vector is unit length by construction"
                .to_string(),
        ));
    }
    if flags.title.is_some() && flags.task != Some(EmbedTask::Document) {
        return Err(refused(
            "--title names a document; pass it with --task document".to_string(),
        ));
    }
    match flags.dim {
        Some(dim) if !MATRYOSHKA_DIMS.contains(&dim) => Err(refused(format!(
            "--dim {dim}: the model is trained at {MATRYOSHKA_DIMS:?}"
        ))),
        _ => Ok(()),
    }
}

/// The prompt `--task` names; no `--task` embeds the text as given.
pub(crate) fn prompt_of(task: Option<EmbedTask>, title: Option<&str>) -> Prompt<'_> {
    let query = match task {
        None => return Prompt::Raw,
        Some(EmbedTask::Document) => return Prompt::Document { title },
        Some(EmbedTask::SearchQuery) => Task::SearchQuery,
        Some(EmbedTask::QuestionAnswering) => Task::QuestionAnswering,
        Some(EmbedTask::FactChecking) => Task::FactChecking,
        Some(EmbedTask::CodeRetrieval) => Task::CodeRetrieval,
        Some(EmbedTask::Classification) => Task::Classification,
        Some(EmbedTask::Clustering) => Task::Clustering,
        Some(EmbedTask::SentenceSimilarity) => Task::SentenceSimilarity,
    };
    Prompt::Query(query)
}

/// `apr embed` reads an embedding model; a generative GGUF file is refused by kind (I-3).
fn refuse_generative(model: &GGUFModel) -> Result<()> {
    match gguf_model_kind(model) {
        Ok(ModelKind::Embedding) => Ok(()),
        Ok(ModelKind::Generative) => Err(CliError::ModelLoadFailed(format!(
            "'{}' is a generative model (ModelKind::Generative): `apr embed` reads an embedding \
             model. Use `apr run`.",
            model.architecture().unwrap_or_default()
        ))),
        Err(reason) => Err(CliError::ModelLoadFailed(reason)),
    }
}

/// One embedded text.
struct Embedded<'a> {
    text: &'a str,
    tokens: usize,
    vector: Vec<f32>,
}

/// A GGUF file: the EmbeddingGemma 2 forward on the file's own tokenizer.
fn run_gguf(model: &Path, flags: &EmbedFlags<'_>) -> Result<()> {
    refuse_gguf_flags(flags)?;
    let texts = embed::collect_texts(flags.texts, flags.text_file)?;
    let load_failed =
        |e: realizar::RealizarError| CliError::ModelLoadFailed(format!("{}: {e}", model.display()));
    let mapped = MappedGGUFModel::from_path(model).map_err(load_failed)?;
    refuse_generative(&mapped.model)?;
    let encoder = EmbeddingGemma2::from_mapped(&mapped).map_err(load_failed)?;
    let tokenizer = EmbedTokenizer::from_gguf(&mapped.model).map_err(load_failed)?;
    let prompt = prompt_of(flags.task, flags.title);
    if prompt == Prompt::Raw {
        eprintln!(
            "note: no --task, so each text is embedded as given; the model card says a task \
             prompt gives better vectors"
        );
    }
    let mut results = Vec::with_capacity(texts.len());
    for text in &texts {
        let tokens = tokenizer.tokens(&prompt, text);
        let full = encoder
            .embed(&tokens)
            .map_err(|e| CliError::InferenceFailed(format!("{text:?}: {e}")))?;
        let vector = match flags.dim {
            None => full,
            Some(dim) => truncate_to(&full, dim).map_err(|e| refused(e.to_string()))?,
        };
        results.push(Embedded {
            text,
            tokens: tokens.len(),
            vector,
        });
    }
    let width = flags.dim.unwrap_or_else(|| encoder.output_dim());
    print_gguf(model, &prompt, width, &results, flags.json);
    Ok(())
}

fn print_gguf(
    model: &Path,
    prompt: &Prompt<'_>,
    width: usize,
    results: &[Embedded<'_>],
    json: bool,
) {
    if json {
        #[allow(clippy::disallowed_methods)]
        {
            let array: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "text": r.text,
                        "embedding": r.vector,
                        "dim": r.vector.len(),
                        "tokens": r.tokens,
                    })
                })
                .collect();
            let payload = serde_json::json!({
                "model": model.display().to_string(),
                "kind": "embedding",
                "pool": "mean",
                "normalize": true,
                "prompt": prompt.name(),
                "dim": width,
                "results": array,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&payload).unwrap_or_default()
            );
        }
        return;
    }
    for r in results {
        let preview: Vec<String> = r
            .vector
            .iter()
            .take(4)
            .map(|x| format!("{x:+.4}"))
            .collect();
        println!(
            "text={:?} dim={} preview=[{}, …]",
            r.text,
            r.vector.len(),
            preview.join(", ")
        );
    }
}

#[cfg(test)]
#[path = "embed_gguf_tests.rs"]
mod tests;
