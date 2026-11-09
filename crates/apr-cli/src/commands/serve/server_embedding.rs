/// APR-EMBED-001 EG-3 (#4902): serve an EmbeddingGemma 2 file.
///
/// The encoder is the one `apr embed` builds, so `/v1/embeddings`, `/api/embeddings` and
/// `/realize/embed` return the CLI's vectors; every generative route refuses by kind. It
/// runs on the CPU only, so an explicit accelerator request is refused, not dropped (#4089).
#[cfg(feature = "inference")]
fn start_embedding_server(
    mapped_model: &realizar::gguf::MappedGGUFModel,
    config: &ServerConfig,
) -> Result<()> {
    let state = build_embedding_state(mapped_model, config)?;
    serve_router(state, config)
}

/// The serving state for an EmbeddingGemma 2 file: the resident encoder and its measured
/// source — everything [`start_embedding_server`] serves, short of the socket.
#[cfg(feature = "inference")]
fn build_embedding_state(
    mapped_model: &realizar::gguf::MappedGGUFModel,
    config: &ServerConfig,
) -> Result<realizar::api::AppState> {
    use realizar::api::AppState;

    if config.wants_accelerator() {
        config.refuse_unengaged_accelerator("the EmbeddingGemma 2 encoder runs on the CPU only")?;
    }
    let state = AppState::with_embedding_gemma2(mapped_model).map_err(|e| {
        CliError::ModelLoadFailed(format!("Failed to load the embedding model: {e}"))
    })?;
    let architecture = mapped_model
        .model
        .architecture()
        .unwrap_or_default()
        .to_string();
    let width = state
        .embedding_model()
        .map_or(0, |served| served.encoder.output_dim());
    println!(
        "{}",
        format!(
            "Model ready: '{architecture}' embedding model (ModelKind::Embedding), {width}-wide \
             vectors on the CPU; the generative routes refuse by kind"
        )
        .green()
    );
    let model_source = config
        .model_path
        .as_deref()
        .map(realizar::api::ModelSourceInfo::from_path)
        .unwrap_or_default()
        .with_architecture(&architecture)
        .with_parameter_count(gguf_parameter_count(&mapped_model.model));
    let model_source = match gguf_dominant_quantization(&mapped_model.model) {
        Some(q) => model_source.with_quantization(q),
        None => model_source,
    };
    Ok(state
        .with_model_source(model_source)
        .with_verbose(config.verbose))
}

/// EG-3: whether `apr serve` routes this file to [`start_embedding_server`].
///
/// # Errors
/// A file whose `attention.causal` flag contradicts its architecture (I-3), refused at load.
#[cfg(feature = "inference")]
fn serve_route_is_embedding(model: &realizar::gguf::GGUFModel) -> Result<bool> {
    use realizar::model_kind::{gguf_model_kind, ModelKind};
    match gguf_model_kind(model) {
        Ok(kind) => Ok(kind == ModelKind::Embedding),
        Err(reason) => Err(CliError::ModelLoadFailed(reason)),
    }
}

#[cfg(all(test, feature = "inference"))]
mod embedding_route_tests {
    use super::*;
    use crate::commands::model_kind_gate::tests::eg1_gguf;

    fn mapped(bytes: &[u8]) -> (tempfile::NamedTempFile, realizar::gguf::MappedGGUFModel) {
        let file = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp file");
        std::fs::write(file.path(), bytes).expect("write");
        let mapped = realizar::gguf::MappedGGUFModel::from_path(file.path()).expect("map");
        (file, mapped)
    }

    /// EG-3: `apr serve` routes an embedding file to its encoder and a generative one to
    /// the generative loader; a causal flag that contradicts the architecture is refused.
    #[test]
    fn serve_routes_by_model_kind_eg3() {
        let (_f, embedding) = mapped(&eg1_gguf("gemma-embedding2", Some(false)));
        assert!(serve_route_is_embedding(&embedding.model).expect("a kind"));
        let (_f, llama) = mapped(&eg1_gguf("llama", None));
        assert!(!serve_route_is_embedding(&llama.model).expect("a kind"));
        let (_f, contradiction) = mapped(&eg1_gguf("gemma-embedding2", Some(true)));
        match serve_route_is_embedding(&contradiction.model) {
            Err(CliError::ModelLoadFailed(msg)) => {
                assert!(msg.contains("attention.causal = true"), "{msg}");
            }
            other => panic!("a contradicting causal flag must be refused: {other:?}"),
        }
    }

    /// An embedding file without the encoder's tensors fails to load, by name.
    #[test]
    fn an_embedding_file_without_the_graph_fails_to_load_eg3() {
        let (_f, stub) = mapped(&eg1_gguf("gemma-embedding2", Some(false)));
        match build_embedding_state(&stub, &ServerConfig::default()) {
            Err(CliError::ModelLoadFailed(msg)) => {
                assert!(msg.contains("Failed to load the embedding model"), "{msg}");
            }
            Err(other) => panic!("refused for the wrong reason: {other}"),
            Ok(_) => panic!("a file with only token_embd must not serve"),
        }
    }
}
