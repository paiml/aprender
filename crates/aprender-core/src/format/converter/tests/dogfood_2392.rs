//! Falsifiers for #2392 — convert / quantize / export defects found by
//! dogfooding the crates.io 0.63.0 `apr` binary.
//!
//! Every test here asserts BEHAVIOUR that was observably wrong in 0.63.0, not
//! the shape of a struct. Each one turns RED when its fix is reverted.

#[allow(unused_imports)]
use super::super::*;

// =============================================================================
// Finding 1 — convert/quantize discarded the tokenizer for every scheme but Q4K
// =============================================================================
#[cfg(test)]
mod finding_1_tokenizer_survives_every_scheme {
    use super::*;
    use crate::format::gguf::GgufTokenizer;
    use std::collections::BTreeMap;

    fn tiny_tokenizer() -> GgufTokenizer {
        GgufTokenizer {
            vocabulary: vec![
                "<|endoftext|>".to_string(),
                "hello".to_string(),
                "world".to_string(),
            ],
            merges: vec!["h e".to_string(), "l l".to_string()],
            model_type: Some("gpt2".to_string()),
            bos_token_id: Some(0),
            eos_token_id: Some(0),
            architecture: Some("qwen2".to_string()),
            model_name: Some("dogfood-2392".to_string()),
            ..Default::default()
        }
    }

    fn tiny_tensors() -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.25; 8 * 256], vec![8, 256]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.125; 256 * 256], vec![256, 256]),
        );
        tensors
    }

    /// Read back the APR container and report whether it carries a usable
    /// embedded tokenizer (a non-empty `tokenizer.vocabulary` array).
    fn embedded_vocab_len(path: &std::path::Path) -> usize {
        let data = std::fs::read(path).expect("read APR back");
        let reader = crate::format::v2::AprV2Reader::from_bytes(&data).expect("parse APR v2");
        reader
            .metadata()
            .custom
            .get("tokenizer.vocabulary")
            .and_then(|v| v.as_array())
            .map_or(0, Vec::len)
    }

    /// #2392 finding 1: the convert/quantize fallback save path located the
    /// tokenizer, parsed it, printed how many tokens it had read — and then
    /// wrote an APR with no tokenizer at all, exiting 0 with "Conversion
    /// successful". `apr run` on the result died with rc=8 and
    /// "APR format requires self-contained tokenizer", so the command produced
    /// an artifact violating the format contract the same binary enforces.
    ///
    /// The behaviour was systematic in the scheme: Q4K embedded the tokenizer in
    /// 4/4 outputs, int8/int4/fp16/none in 0/7. So this asserts across schemes.
    #[test]
    fn every_quantization_scheme_embeds_the_tokenizer_it_was_given() {
        let tensors = tiny_tensors();
        let tok = tiny_tokenizer();
        let dir = std::env::temp_dir().join(format!("apr-2392-f1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");

        // The four schemes reachable from `apr convert --quantize` / `apr
        // quantize -s`, plus "no quantization at all" (`--compress zstd`).
        let schemes: [(&str, Option<QuantizationType>); 5] = [
            ("none", None),
            ("int8", Some(QuantizationType::Int8)),
            ("int4", Some(QuantizationType::Int4)),
            ("fp16", Some(QuantizationType::Fp16)),
            ("q4k", Some(QuantizationType::Q4K)),
        ];

        for (name, quant) in schemes {
            let path = dir.join(format!("{name}.apr"));
            save_model_tensors_with_config(&tensors, &path, None, quant, Some(&tok))
                .unwrap_or_else(|e| panic!("save failed for {name}: {e:?}"));

            let vocab_len = embedded_vocab_len(&path);
            assert_eq!(
                vocab_len,
                tok.vocabulary.len(),
                "#2392 finding 1: scheme '{name}' produced an APR with {vocab_len} embedded \
                 vocabulary entries, expected {}. An APR without a tokenizer cannot run \
                 inference — the loader rejects it with 'APR format requires self-contained \
                 tokenizer' — so writing one and reporting success is silent data loss.",
                tok.vocabulary.len()
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A tokenizer with BPE merges must keep them too: an APR that has the
    /// vocabulary but not the merge rules still cannot encode a prompt.
    #[test]
    fn int8_apr_keeps_bpe_merges_not_just_the_vocabulary() {
        let tensors = tiny_tensors();
        let tok = tiny_tokenizer();
        let dir = std::env::temp_dir().join(format!("apr-2392-f1m-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("int8.apr");

        save_model_tensors_with_config(
            &tensors,
            &path,
            None,
            Some(QuantizationType::Int8),
            Some(&tok),
        )
        .expect("save int8 APR");

        let data = std::fs::read(&path).expect("read APR back");
        let reader = crate::format::v2::AprV2Reader::from_bytes(&data).expect("parse APR v2");
        let merges = reader
            .metadata()
            .custom
            .get("tokenizer.merges")
            .and_then(|v| v.as_array())
            .map_or(0, Vec::len);

        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            merges,
            tok.merges.len(),
            "#2392 finding 1: int8 APR embedded {merges} BPE merge rules, expected {}",
            tok.merges.len()
        );
    }

    /// #2392 finding 1, second half: a tokenizer is necessary but not
    /// sufficient. With the tokenizer embedded, `apr run` got one step further
    /// and then died with "C-01: APR model missing 'architecture' metadata —
    /// cannot infer model type", because this save path never stamped it. The
    /// artifact still could not run, so the command still reported a success
    /// that was not one.
    #[test]
    fn the_saved_apr_declares_its_architecture() {
        let tensors = tiny_tensors();
        let tok = tiny_tokenizer();
        let dir = std::env::temp_dir().join(format!("apr-2392-f1a-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("int8.apr");

        save_model_tensors_with_config(
            &tensors,
            &path,
            None,
            Some(QuantizationType::Int8),
            Some(&tok),
        )
        .expect("save int8 APR");

        let data = std::fs::read(&path).expect("read APR back");
        let reader = crate::format::v2::AprV2Reader::from_bytes(&data).expect("parse APR v2");
        let arch = reader.metadata().architecture.clone();
        let _ = std::fs::remove_dir_all(&dir);

        let arch = arch.expect(
            "#2392 finding 1: the APR must declare an architecture — without it the loader \
             refuses with 'C-01: APR model missing architecture metadata'",
        );
        assert_ne!(arch, "unknown", "an 'unknown' architecture is not a declaration");
        // These tensors are `model.layers.N.self_attn.*` with no attention bias
        // and no QK norm — the inference helper calls that llama.
        assert_eq!(arch, "llama", "architecture must be inferred from the tensor names");
    }

    /// Guard the honest negative: when there genuinely is no tokenizer to
    /// embed, the save must still succeed and simply not invent one. (A fix
    /// that unconditionally wrote a key would pass the tests above.)
    #[test]
    fn no_tokenizer_available_means_no_tokenizer_key() {
        let tensors = tiny_tensors();
        let dir = std::env::temp_dir().join(format!("apr-2392-f1n-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("no-tok.apr");

        save_model_tensors_with_config(&tensors, &path, None, Some(QuantizationType::Int8), None)
            .expect("save APR without tokenizer");

        let vocab_len = embedded_vocab_len(&path);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            vocab_len, 0,
            "no tokenizer was supplied, so none may appear in the output"
        );
    }
}

// =============================================================================
// Finding 3 — `quantize --plan -s q4k` returned a hardcoded 7.111x ratio
// =============================================================================
#[cfg(test)]
mod finding_3_q4k_plan_is_shape_aware {
    use super::*;
    use std::collections::BTreeMap;

    /// Build an APR whose tensors are deliberately a mix of Q4K-eligible and
    /// Q4K-ineligible shapes, mirroring a real model: a big embedding table
    /// (skipped by name), a norm (skipped by name), a row width that is a clean
    /// multiple of 256, and a row width that is not (384 — the case that made
    /// the shipped estimator 4.34x optimistic on a real 87 MB model).
    fn mixed_model() -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        let mut t: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        t.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.5; 64 * 384], vec![64, 384]),
        );
        t.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            (vec![1.0; 384], vec![384]),
        );
        t.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.25; 384 * 384], vec![384, 384]),
        );
        t.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            (vec![0.25; 256 * 512], vec![256, 512]),
        );
        t
    }

    fn write_apr(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>, path: &std::path::Path) {
        save_model_tensors_with_config(tensors, path, None, None, None).expect("write f32 APR");
    }

    /// #2392 finding 3: the estimate must track the real Q4K writer, which
    /// blocks per ROW with 256-element super-blocks (144 bytes each) and leaves
    /// embeddings/norms/biases/scales as F32. The shipped estimator applied a
    /// flat 4.5 bits/weight, so it returned the identical 7.111x reduction ratio
    /// for a 4.8 MB model, an 87 MB model and a 992 MB model alike.
    #[test]
    fn q4k_estimate_matches_the_bytes_quantization_actually_writes() {
        let tensors = mixed_model();
        let dir = std::env::temp_dir().join(format!("apr-2392-f3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let src = dir.join("src.apr");
        let dst = dir.join("dst-q4k.apr");
        write_apr(&tensors, &src);

        let estimate = crate::format::q4k_output_size_estimate(&src)
            .expect("estimator must read an APR tensor index");

        // Now actually quantize and measure the tensor payload that resulted.
        save_model_tensors_q4k(&tensors, &dst, None).expect("q4k quantize");
        let actual_payload: u64 = {
            let data = std::fs::read(&dst).expect("read q4k APR");
            let reader = crate::format::v2::AprV2Reader::from_bytes(&data).expect("parse q4k APR");
            reader
                .tensor_names()
                .iter()
                .filter_map(|n| reader.get_tensor(n))
                .map(|e| e.size as u64)
                .sum()
        };
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            estimate, actual_payload,
            "#2392 finding 3: `--plan -s q4k` estimated {estimate} tensor bytes but Q4K \
             quantization wrote {actual_payload}. --plan exists to size disk and RAM before a \
             long run; an estimate that ignores which tensors are eligible and how rows are \
             padded is worse than none."
        );
    }

    /// The old estimator's signature failure was that its answer did not depend
    /// on the model at all. Two models with the same on-disk size but different
    /// tensor shapes must get different Q4K estimates.
    #[test]
    fn q4k_estimate_is_not_a_constant_ratio() {
        let dir = std::env::temp_dir().join(format!("apr-2392-f3c-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");

        // Same element count (and therefore near-identical F32 file size), but
        // one is all-embedding (Q4K-ineligible) and one is all-projection.
        let mut skipped: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        skipped.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.5; 256 * 512], vec![256, 512]),
        );
        let mut quantized: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        quantized.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            (vec![0.5; 256 * 512], vec![256, 512]),
        );

        let a = dir.join("skipped.apr");
        let b = dir.join("quantized.apr");
        write_apr(&skipped, &a);
        write_apr(&quantized, &b);

        let est_a = crate::format::q4k_output_size_estimate(&a).expect("estimate a");
        let est_b = crate::format::q4k_output_size_estimate(&b).expect("estimate b");
        let size_a = std::fs::metadata(&a).expect("stat a").len();
        let size_b = std::fs::metadata(&b).expect("stat b").len();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            est_a > est_b,
            "#2392 finding 3: an all-embedding model stays F32 under Q4K and an all-projection \
             model shrinks, so their estimates must differ (got {est_a} vs {est_b} for inputs of \
             {size_a} and {size_b} bytes). The shipped estimator returned the same 7.111x ratio \
             for every model ever passed to it."
        );
        assert_eq!(
            est_a,
            256 * 512 * 4,
            "an ineligible tensor is written verbatim as F32"
        );
        assert_eq!(
            est_b,
            256 * 2 * 144,
            "a [256, 512] projection is 2 super-blocks per row at 144 bytes"
        );
    }

    /// A path with no readable tensor index must yield `None` so the caller can
    /// fall back, rather than a confidently wrong zero.
    #[test]
    fn unreadable_input_yields_none_not_zero() {
        let dir = std::env::temp_dir().join(format!("apr-2392-f3n-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let junk = dir.join("junk.apr");
        std::fs::write(&junk, b"not a model at all").expect("write junk");
        let got = crate::format::q4k_output_size_estimate(&junk);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(got.is_none(), "unreadable input must not produce a number");
    }
}

// =============================================================================
// Finding 6 — a missing LOCAL path was answered with a Hugging Face Hub hint
// =============================================================================
#[cfg(test)]
mod finding_6_local_path_gets_a_local_remedy {
    use super::*;

    /// #2392 finding 6: `apr import /nonexistent.safetensors` failed correctly
    /// (rc=5) but told the user to "verify the model name exists on
    /// huggingface.co/models" — a remedy that cannot apply to an absolute local
    /// path. `resolve_local_source` already encodes the distinction by setting
    /// `status: 0` ("local file, not HTTP"); the message just never read it.
    #[test]
    fn local_not_found_does_not_advise_a_hub_lookup() {
        let err = ImportError::NotFound {
            resource: "/nonexistent.safetensors".to_string(),
            status: 0,
        };
        let msg = err.to_string();
        assert!(
            !msg.contains("huggingface.co"),
            "#2392 finding 6: a local filesystem path must not be answered with a Hub \
             lookup hint. Got: {msg}"
        );
        assert!(
            msg.contains("/nonexistent.safetensors"),
            "the message must name the path that was not found. Got: {msg}"
        );
        assert!(
            msg.contains("File not found"),
            "the message must say what actually went wrong. Got: {msg}"
        );
    }

    /// #2392 finding 6, second half: the Pacha cache stores one model's files
    /// under a shared hash stem (`<hash>.safetensors`, `<hash>.tokenizer.json`,
    /// `<hash>.config.json`). The tokenizer loader understands that layout and
    /// says so on stderr; the CONFIG loader did not, so `apr convert` on a Pacha
    /// model silently fell back to shape inference and emitted an APR with no
    /// `num_heads` — which `apr run` rejects with "C-03: APR model missing
    /// 'num_heads' metadata", with the real value sitting in a file beside the
    /// weights the whole time.
    #[test]
    fn pacha_cache_config_json_is_found() {
        let dir = std::env::temp_dir().join(format!("apr-2392-f6p-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let model = dir.join("064a3693fa1ea02c.safetensors");
        std::fs::write(&model, b"placeholder").expect("write model");
        std::fs::write(
            dir.join("064a3693fa1ea02c.config.json"),
            br#"{"model_type":"qwen2","hidden_size":8,"num_hidden_layers":2,
                 "num_attention_heads":4,"num_key_value_heads":2,
                 "intermediate_size":32,"vocab_size":151665,"rope_theta":10000.0}"#,
        )
        .expect("write pacha config");

        let cfg = crate::format::converter::import::load_model_config_from_json(&model);
        let _ = std::fs::remove_dir_all(&dir);

        let cfg = cfg.expect(
            "#2392 finding 6: the Pacha `<hash>.config.json` layout must be recognised — the \
             tokenizer loader in the same binary already does",
        );
        assert_eq!(cfg.num_heads, Some(4), "num_heads must come from config.json");
        assert_eq!(cfg.num_kv_heads, Some(2), "num_kv_heads must come from config.json");
        assert_eq!(cfg.hidden_size, Some(8));
        assert_eq!(cfg.architecture.as_deref(), Some("qwen2"));
    }

    /// A bare sibling `config.json` must keep working, and a model with neither
    /// layout must still yield `None` rather than a fabricated config.
    #[test]
    fn standard_layout_still_works_and_absence_is_still_none() {
        let dir = std::env::temp_dir().join(format!("apr-2392-f6s-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let model = dir.join("model.safetensors");
        std::fs::write(&model, b"placeholder").expect("write model");

        assert!(
            crate::format::converter::import::load_model_config_from_json(&model).is_none(),
            "no config.json anywhere means no config"
        );

        std::fs::write(
            dir.join("config.json"),
            br#"{"model_type":"llama","hidden_size":16,"num_attention_heads":8}"#,
        )
        .expect("write standard config");
        let cfg = crate::format::converter::import::load_model_config_from_json(&model);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(
            cfg.and_then(|c| c.num_heads),
            Some(8),
            "the standard HuggingFace layout must keep working"
        );
    }

    /// The Hub hint is still right for a real Hub 404 — the fix must not have
    /// deleted it, only stopped applying it to local paths.
    #[test]
    fn hub_404_still_advises_a_hub_lookup() {
        let err = ImportError::NotFound {
            resource: "openai/no-such-model".to_string(),
            status: 404,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("huggingface.co/models"),
            "a genuine Hub 404 must keep its Hub remedy. Got: {msg}"
        );
        assert!(msg.contains("404"), "the status belongs in the message: {msg}");
    }
}

// =============================================================================
// Finding 2 — `apr export` reported the dequantized tensor total as
//             `original_size`, a number independent of the input file
// =============================================================================
#[cfg(test)]
mod finding_2_export_original_size_is_a_file_size {
    use super::*;
    use std::collections::BTreeMap;

    /// #2392 finding 2: `original_size` is printed directly beside
    /// `exported_size`, which is a real on-disk size, so the two must be
    /// comparable. As shipped, `original_size` was the sum of dequantized F32
    /// tensor bytes: it read an identical 9714528 for five different input APRs
    /// spanning 1.37 MB to 30.5 MB on disk, turning a true 3x shrink into a
    /// reported slight growth.
    #[test]
    fn export_reports_the_input_files_on_disk_size() {
        let dir = std::env::temp_dir().join(format!("apr-2392-f2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            (vec![0.5; 256 * 512], vec![256, 512]),
        );

        // Two inputs holding the SAME tensors, so the F32 tensor total is
        // identical, but stored at different precisions so the FILES differ.
        let f32_src = dir.join("f32.apr");
        let fp16_src = dir.join("fp16.apr");
        save_model_tensors_with_config(&tensors, &f32_src, None, None, None).expect("write f32");
        save_model_tensors_with_config(
            &tensors,
            &fp16_src,
            None,
            Some(QuantizationType::Fp16),
            None,
        )
        .expect("write fp16");

        let f32_bytes = std::fs::metadata(&f32_src).expect("stat f32").len() as usize;
        let fp16_bytes = std::fs::metadata(&fp16_src).expect("stat fp16").len() as usize;
        assert!(
            f32_bytes > fp16_bytes,
            "precondition: the fp16 APR must actually be the smaller file \
             ({f32_bytes} vs {fp16_bytes})"
        );

        let opts = || ExportOptions {
            format: ExportFormat::SafeTensors,
            skip_completeness_check: true,
            ..Default::default()
        };
        let r_f32 = apr_export(f32_src.as_path(), dir.join("a.safetensors").as_path(), opts())
            .expect("export f32 source");
        let r_fp16 = apr_export(
            fp16_src.as_path(),
            dir.join("b.safetensors").as_path(),
            opts(),
        )
        .expect("export fp16 source");
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            r_f32.original_size, f32_bytes,
            "#2392 finding 2: original_size must be the input file's on-disk size"
        );
        assert_eq!(
            r_fp16.original_size, fp16_bytes,
            "#2392 finding 2: original_size must be the input file's on-disk size"
        );
        assert_ne!(
            r_f32.original_size, r_fp16.original_size,
            "#2392 finding 2: two inputs of different on-disk size reported the SAME \
             original_size — that is the defect: it was the dequantized tensor total, a \
             property of the parameter count, not of the file."
        );
    }
}
