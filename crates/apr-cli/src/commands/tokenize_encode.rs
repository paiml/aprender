//! `apr tokenize encode` (#3726, #3742): the token ids a model file's own tokenizer gives a
//! text, for a GGUF or an `.apr`.
//!
//! A GGUF's ids come from `GGUFModel::encode` and an `.apr`'s from its embedded
//! `BpeTokenizer`, the functions `apr run`, `apr serve` and `apr parity` tokenize prompts
//! with, so a regression in how either routes a vocabulary shows up here too. The output
//! names the path that produced the ids: `canonical` (the model's pre-tokenizer plus its
//! ranked merges, identical to llama.cpp) or the fallback and why it was taken. It also
//! prints where the pre-tokenizer came from and a fingerprint of the (vocabulary, merges)
//! tables: `scripts/tokenizer_parity.sh` pairs an `.apr` with a GGUF of the same fingerprint
//! and compares both with the pinned llama.cpp.
//!
//! Only the tokenizer tables are read (both formats are memory-mapped); no weights are
//! materialised.

use std::path::Path;

use crate::error::CliError;

type Result<T> = std::result::Result<T, CliError>;

/// What one file's tokenizer made of the text.
struct Encoded {
    ids: Vec<u32>,
    path: String,
    pre_source: Option<String>,
    roundtrip: bool,
    fingerprint: Option<u64>,
    tokenizer_model: Option<String>,
}

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
    let enc = if model.extension().is_some_and(|e| e == "apr") {
        encode_apr(model, &text)?
    } else {
        encode_gguf(model, &text)?
    };
    let fingerprint = enc.fingerprint.map(|f| format!("{f:016x}"));

    if json_output {
        let out = serde_json::json!({
            "model": model.display().to_string(),
            "tokenizer_model": enc.tokenizer_model,
            "path": enc.path,
            "pre_source": enc.pre_source,
            "tokenizer_fingerprint": fingerprint,
            "count": enc.ids.len(),
            "roundtrip": enc.roundtrip,
            "ids": enc.ids,
        });
        println!("{out}");
    } else {
        let joined: Vec<String> = enc.ids.iter().map(u32::to_string).collect();
        println!("{}", joined.join(" "));
        if let Some(src) = &enc.pre_source {
            eprintln!("pre-tokenizer: {src}");
        }
        if let Some(f) = &fingerprint {
            eprintln!("tokenizer-fingerprint: {f}");
        }
        eprintln!(
            "{} ids; path: {}; roundtrip: {}",
            enc.ids.len(),
            enc.path,
            enc.roundtrip
        );
    }
    Ok(())
}

fn fingerprint(vocab: &[String], merges: &[(String, String)]) -> u64 {
    let joined: Vec<String> = merges.iter().map(|(a, b)| format!("{a} {b}")).collect();
    let refs: Vec<&str> = joined.iter().map(String::as_str).collect();
    realizar::gguf::byte_level_bpe::ByteLevelBpe::tables_fingerprint(vocab, &refs)
}

fn encode_gguf(model: &Path, text: &str) -> Result<Encoded> {
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

    let ids = mapped.model.encode(text).ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "{}: the tokenizer returned nothing",
            model.display()
        ))
    })?;
    // decode(encode(x)) == x: the byte-level map loses nothing (#3726 done_when 1).
    let roundtrip = mapped.model.decode(&ids) == text;
    let merges = mapped.model.merge_rules().unwrap_or_default();
    Ok(Encoded {
        fingerprint: Some(fingerprint(&vocab, &merges)),
        pre_source: tokenizer_pre.map(|p| format!("tokenizer.ggml.pre = {p}")),
        ids,
        path,
        roundtrip,
        tokenizer_model,
    })
}

fn encode_apr(model: &Path, text: &str) -> Result<Encoded> {
    use realizar::apr::canonical_tokenizer::{apr_pre_tokenizer, is_byte_level};
    let apr = realizar::apr::AprV2Model::load(model).map_err(|e| {
        CliError::ValidationFailed(format!("{} is not a readable APR: {e}", model.display()))
    })?;
    let meta = apr.metadata();
    let tokenizer_model = meta.get_embedded_model_type();
    let tok = apr.load_embedded_bpe_tokenizer().ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "{} carries no embedded vocabulary and merges: there is no tokenizer to run",
            model.display()
        ))
    })?;
    let pre = apr_pre_tokenizer(meta);
    let path = match (&tok.canonical, &pre) {
        (Some(_), _) => "canonical".to_string(),
        (None, _) if !is_byte_level(&tok.id_to_token, tokenizer_model.as_deref()) => format!(
            "legacy-fallback: tokenizer.model_type is {:?}, not a byte-level vocabulary",
            tokenizer_model.as_deref().unwrap_or("absent")
        ),
        (None, _) => format!(
            "legacy-fallback: no implemented pre-tokenizer (tokenizer.pre_type {:?}, architecture {:?})",
            meta.extra.get("tokenizer.pre_type").and_then(|v| v.as_str()),
            meta.architecture
        ),
    };
    let ids = tok.encode(text);
    let roundtrip = tok.decode(&ids) == text;
    Ok(Encoded {
        fingerprint: Some(fingerprint(&tok.id_to_token, &tok.merge_rules)),
        pre_source: pre.map(|(p, src)| format!("{p:?} ({src})")),
        ids,
        path,
        roundtrip,
        tokenizer_model,
    })
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
