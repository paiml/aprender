//! `apr tokenize encode` (#3726): the token ids a GGUF model's own tokenizer gives a text.
//!
//! The ids come from `GGUFModel::encode`, the function `apr run`, `apr serve` and
//! `apr parity` tokenize prompts with, so a regression in how that function routes a
//! vocabulary shows up here too. The output names the path that produced the ids: the
//! canonical byte-level BPE (pre-tokenizer plus ranked merges, identical to llama.cpp), or the
//! greedy longest-match fallback and why it was taken. `scripts/tokenizer_parity.sh` compares
//! these ids with the pinned llama.cpp's and refuses a fallback.
//!
//! Only the file's metadata is read (the GGUF is memory-mapped); no weights are loaded.

use std::path::Path;

use crate::error::CliError;

type Result<T> = std::result::Result<T, CliError>;

/// Run `apr tokenize encode MODEL (-p TEXT | -f FILE)`.
pub(crate) fn run_encode(
    model: &Path,
    prompt: Option<&str>,
    file: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    let text = match (prompt, file) {
        (Some(p), None) => p.to_string(),
        (None, Some(f)) => std::fs::read_to_string(f)
            .map_err(|e| CliError::ValidationFailed(format!("cannot read {}: {e}", f.display())))?,
        _ => {
            return Err(CliError::ValidationFailed(
                "give exactly one of -p/--prompt TEXT or -f/--file FILE".to_string(),
            ))
        }
    };
    if !model.exists() {
        return Err(CliError::FileNotFound(model.to_path_buf()));
    }
    let mapped = realizar::gguf::MappedGGUFModel::from_path(model).map_err(|e| {
        CliError::ValidationFailed(format!("{} is not a readable GGUF: {e}", model.display()))
    })?;
    let meta_str = |key: &str| match mapped.model.metadata.get(key) {
        Some(realizar::gguf::GGUFValue::String(s)) => Some(s.clone()),
        _ => None,
    };
    let tokenizer_model = meta_str("tokenizer.ggml.model");
    let tokenizer_pre = meta_str("tokenizer.ggml.pre");
    let vocab = mapped.model.vocabulary().ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "{} has no tokenizer.ggml.tokens: there is no tokenizer to run",
            model.display()
        ))
    })?;

    // Which path GGUFModel::encode takes for this file. It is asked of the same function
    // encode consults, so the label cannot disagree with the ids.
    let byte_level = matches!(tokenizer_model.as_deref(), Some("gpt2" | "bpe"));
    let path = if byte_level {
        match realizar::gguf::byte_level_bpe::ByteLevelBpe::from_gguf(
            &mapped.model.metadata,
            &vocab,
        ) {
            Ok(_) => "canonical".to_string(),
            Err(refusal) => format!("greedy-fallback: {refusal}"),
        }
    } else {
        format!(
            "greedy-fallback: tokenizer.ggml.model is {:?}, not a byte-level vocabulary",
            tokenizer_model.as_deref().unwrap_or("absent")
        )
    };

    let ids = mapped.model.encode(&text).ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "{}: the tokenizer returned nothing",
            model.display()
        ))
    })?;

    // decode(encode(x)) == x: the byte-level map loses nothing (#3726 done_when 1).
    let roundtrip = mapped.model.decode(&ids) == text;

    if json_output {
        let out = serde_json::json!({
            "model": model.display().to_string(),
            "tokenizer_model": tokenizer_model,
            "tokenizer_pre": tokenizer_pre,
            "path": path,
            "count": ids.len(),
            "roundtrip": roundtrip,
            "ids": ids,
        });
        println!("{out}");
    } else {
        let joined: Vec<String> = ids.iter().map(u32::to_string).collect();
        println!("{}", joined.join(" "));
        eprintln!("{} ids; path: {path}; roundtrip: {roundtrip}", ids.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The first quorum round on #3726 found `Encode` spliced into the middle of
    /// `EncodeCorpus`'s doc comment: `apr tokenize --help` described `encode` as "Encode a
    /// JSONL corpus into .bin shards" and `encode-corpus` lost its summary. Each subcommand
    /// must carry its own. (The clap tree is deep; it is built on a 16 MiB stack, as the
    /// other `Cli::command()` tests do.)
    #[test]
    fn tokenize_help_gives_encode_and_encode_corpus_their_own_summaries() {
        let (encode, corpus) = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                use clap::CommandFactory;
                let cli = crate::Cli::command();
                let tokenize = cli
                    .find_subcommand("tokenize")
                    .expect("apr tokenize exists");
                let about = |name: &str| {
                    tokenize
                        .find_subcommand(name)
                        .and_then(|c| c.get_about())
                        .map(ToString::to_string)
                };
                (about("encode"), about("encode-corpus"))
            })
            .expect("spawn")
            .join()
            .expect("clap tree builds");
        let encode = encode.expect("apr tokenize encode exists");
        assert!(encode.contains("token ids"), "encode's summary: {encode:?}");
        assert!(
            !encode.contains("JSONL"),
            "encode took encode-corpus's summary: {encode:?}"
        );
        // encode-corpus exists only with the `training` feature.
        if let Some(corpus) = corpus {
            assert!(
                corpus.starts_with("Encode a JSONL corpus"),
                "encode-corpus's summary: {corpus:?}"
            );
        }
    }
}
