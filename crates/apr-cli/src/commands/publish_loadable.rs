//! A trained model publishes as a loadable repo (hf-rc-publish-v1 HRP-002).
//!
//! A training run writes one `.apr`. Publishing that directory used to plan
//! the `.apr` alone: no `model.safetensors`, no `config.json`, so nothing on
//! the Hub could load it. When the upload set has an `.apr` and no SafeTensors
//! weights, the `.apr` is now exported to `DIRECTORY/.apr-publish/` as
//! `model.safetensors` + `config.json` (+ `tokenizer.json` when the `.apr`
//! carries one), and those files join the plan. The config is the source
//! config stored at import (qwen35-format-roundtrip-v1 QFR-003), so the repo
//! loads with the source architecture.

use crate::error::CliError;
use std::fs;
use std::path::{Path, PathBuf};

/// Where the export lands, inside the published directory.
pub(crate) const STAGE_DIR: &str = ".apr-publish";

/// What the loadable repo needs beside the weights.
const NEEDED: &[&str] = &["config.json"];

fn named<'a>(paths: &'a [PathBuf], name: &str) -> Option<&'a PathBuf> {
    paths
        .iter()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(name))
}

fn has_ext(p: &Path, ext: &str) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Export the `.apr` in `files` when the upload set has no SafeTensors
/// weights, adding the export to `files` and `companion_files`. A file the
/// directory already carries (a hand-written `config.json`, a
/// `tokenizer.json`) is kept over the exported one. More than one `.apr` and
/// no SafeTensors is refused: which one is the model is not a guess.
pub(crate) fn stage_loadable(
    directory: &Path,
    files: &mut Vec<PathBuf>,
    companion_files: &mut Vec<PathBuf>,
) -> Result<(), CliError> {
    if files.iter().any(|f| has_ext(f, "safetensors")) {
        return Ok(());
    }
    let aprs: Vec<&PathBuf> = files.iter().filter(|f| has_ext(f, "apr")).collect();
    let apr = match aprs.as_slice() {
        [] => return Ok(()),
        [one] => (*one).clone(),
        many => {
            return Err(CliError::ValidationFailed(format!(
                "apr publish: {} .apr files and no model.safetensors in {}; a loadable repo \
                 needs one set of weights. Keep one .apr, or export it yourself \
                 (hf-rc-publish-v1 HRP-002)",
                many.len(),
                directory.display()
            )))
        }
    };
    let stage = directory.join(STAGE_DIR);
    fs::create_dir_all(&stage).map_err(|e| {
        CliError::ValidationFailed(format!(
            "apr publish: cannot create {}: {e}",
            stage.display()
        ))
    })?;
    let weights = stage.join("model.safetensors");
    let options = aprender::format::ExportOptions {
        format: aprender::format::ExportFormat::SafeTensors,
        include_config: true,
        include_tokenizer: true,
        ..Default::default()
    };
    aprender::format::apr_export(apr.as_path(), weights.as_path(), options).map_err(|e| {
        CliError::ValidationFailed(format!(
            "apr publish: exporting {} to {} failed: {e}; the repo would not load without \
             SafeTensors weights and a config.json (hf-rc-publish-v1 HRP-002)",
            apr.display(),
            weights.display()
        ))
    })?;
    files.push(weights.clone());
    for name in ["config.json", "tokenizer.json"] {
        let exported = stage.join(name);
        if exported.is_file() && named(companion_files, name).is_none() {
            companion_files.push(exported);
        }
    }
    companion_files.sort();
    if let Some(missing) = NEEDED.iter().find(|n| named(companion_files, n).is_none()) {
        return Err(CliError::ValidationFailed(format!(
            "apr publish: export of {} wrote no {missing}; the repo would not load \
             (hf-rc-publish-v1 HRP-002)",
            apr.display()
        )));
    }
    let tokenizer = ["tokenizer.json", "vocab.json"]
        .iter()
        .any(|n| named(companion_files, n).is_some());
    eprintln!(
        "[HRP-002] exported {} -> {}/ (model.safetensors, config.json{})",
        apr.display(),
        stage.display(),
        if tokenizer { ", tokenizer" } else { "" }
    );
    if !tokenizer {
        eprintln!(
            "apr publish: warning: no tokenizer.json in DIRECTORY or the .apr; the repo loads \
             weights but cannot tokenize (HRP-002)."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::{apr_import, ImportOptions};

    /// A Qwen3.5 source config the tensors alone cannot reveal.
    fn source_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "model_type": "qwen3_5",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 8,
            "intermediate_size": 8,
            "linear_num_value_heads": 16,
            "layer_types": ["linear_attention"],
            "tie_word_embeddings": false
        })
    }

    /// A one-layer SafeTensors file with the tensor names of an HF decoder.
    fn safetensors_bytes() -> Vec<u8> {
        let (v, h, i) = (8usize, 4usize, 8usize);
        let mut tensors: Vec<(String, Vec<usize>)> =
            vec![("model.embed_tokens.weight".into(), vec![v, h])];
        let l = "model.layers.0";
        for (n, s) in [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![h, h]),
            ("self_attn.k_proj.weight", vec![h, h]),
            ("self_attn.v_proj.weight", vec![h, h]),
            ("self_attn.o_proj.weight", vec![h, h]),
            ("mlp.gate_proj.weight", vec![i, h]),
            ("mlp.up_proj.weight", vec![i, h]),
            ("mlp.down_proj.weight", vec![h, i]),
        ] {
            tensors.push((format!("{l}.{n}"), s));
        }
        tensors.push(("model.norm.weight".into(), vec![h]));
        tensors.push(("lm_head.weight".into(), vec![v, h]));
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (k, (name, shape)) in tensors.iter().enumerate() {
            let n: usize = shape.iter().product();
            let start = data.len();
            for j in 0..n {
                let x = if name.ends_with("norm.weight") {
                    1.0f32
                } else {
                    ((k * 31 + j) as f32 * 0.37).sin() * 0.1
                };
                data.extend_from_slice(&x.to_le_bytes());
            }
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let head = serde_json::Value::Object(header).to_string().into_bytes();
        let mut out = (head.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&head);
        out.extend_from_slice(&data);
        out
    }

    /// A trained-run output directory: one `.apr` (imported from an HF dir,
    /// so it carries the source config) and a tokenizer, nothing else.
    fn trained_dir() -> (tempfile::TempDir, tempfile::TempDir) {
        let src = tempfile::tempdir().expect("tempdir");
        let st = src.path().join("model.safetensors");
        fs::write(&st, safetensors_bytes()).expect("write safetensors");
        fs::write(src.path().join("config.json"), source_config().to_string()).expect("config");
        let out = tempfile::tempdir().expect("tempdir");
        apr_import(
            st.to_str().expect("utf8"),
            out.path().join("model.apr"),
            ImportOptions::default(),
        )
        .expect("import");
        (src, out)
    }

    fn plan_names(dir: &Path) -> Vec<String> {
        let files = super::super::find_model_files(dir).expect("model files");
        let (files, comp) = super::super::upload_set(dir, files, None).expect("upload set");
        let targets = super::super::upload_targets(&files, &comp, &[], None).expect("targets");
        targets.iter().map(|t| t.path_in_repo.clone()).collect()
    }

    /// FALSIFY-HRP-002: publishing the output directory of a trained `.apr`
    /// plans model.safetensors + config.json + tokenizer, and importing
    /// exactly the planned files keeps the source architecture. Removing the
    /// staging step (plan = the `.apr` alone) turns this RED.
    #[test]
    fn falsify_hrp_002_trained_apr_publishes_loadable_repo() {
        let (_src, out) = trained_dir();
        fs::write(
            out.path().join("tokenizer.json"),
            r#"{"model":{"type":"BPE"}}"#,
        )
        .expect("tok");
        let names = plan_names(out.path());
        for want in [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "model.apr",
        ] {
            assert!(
                names.iter().any(|n| n == want),
                "{want} not planned: {names:?}"
            );
        }

        // A consumer that downloads the planned files gets a loadable model.
        let repo = tempfile::tempdir().expect("tempdir");
        for name in ["model.safetensors", "config.json", "tokenizer.json"] {
            let from = [out.path().join(STAGE_DIR).join(name), out.path().join(name)]
                .into_iter()
                .find(|p| p.is_file())
                .expect("planned file on disk");
            fs::copy(from, repo.path().join(name)).expect("copy");
        }
        let back = repo.path().join("back.apr");
        apr_import(
            repo.path()
                .join("model.safetensors")
                .to_str()
                .expect("utf8"),
            &back,
            ImportOptions::default(),
        )
        .expect("the planned files import");
        let meta = super::super::publish_license::apr_metadata(&back)
            .expect("read")
            .expect("apr v2");
        assert_eq!(meta.hf_architecture.as_deref(), Some("Qwen3_5ForCausalLM"));
        assert_eq!(meta.architecture.as_deref(), Some("qwen3_5"));
    }

    /// A directory that already ships SafeTensors is not re-exported, and a
    /// hand-written config.json beats the exported one.
    #[test]
    fn existing_weights_are_untouched_and_user_config_wins() {
        let (src, out) = trained_dir();
        let before = plan_names(src.path());
        assert!(!src.path().join(STAGE_DIR).exists(), "{before:?}");

        fs::write(out.path().join("config.json"), r#"{"hand":"written"}"#).expect("config");
        let mut files = super::super::find_model_files(out.path()).expect("files");
        let mut comp = super::super::find_companion_files(out.path()).expect("comp");
        stage_loadable(out.path(), &mut files, &mut comp).expect("stage");
        assert_eq!(
            named(&comp, "config.json"),
            Some(&out.path().join("config.json"))
        );
    }

    #[test]
    fn two_aprs_and_no_weights_is_refused() {
        let (_src, out) = trained_dir();
        fs::copy(out.path().join("model.apr"), out.path().join("other.apr")).expect("copy");
        let mut files = super::super::find_model_files(out.path()).expect("files");
        let err = stage_loadable(out.path(), &mut files, &mut Vec::new())
            .expect_err("ambiguous")
            .to_string();
        assert!(err.contains("HRP-002") && err.contains("2 .apr"), "{err}");
    }

    /// An `.apr` that cannot be exported is refused, not published unloadable.
    #[test]
    fn unexportable_apr_is_refused() {
        let d = tempfile::tempdir().expect("tempdir");
        fs::write(d.path().join("model.apr"), "APR2test").expect("write");
        let mut files = super::super::find_model_files(d.path()).expect("files");
        let err = stage_loadable(d.path(), &mut files, &mut Vec::new())
            .expect_err("garbage .apr")
            .to_string();
        assert!(
            err.contains("HRP-002") && err.contains("would not load"),
            "{err}"
        );
    }
}
