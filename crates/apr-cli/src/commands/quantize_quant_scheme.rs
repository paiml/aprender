
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_quant_scheme_parse_int8() {
        let scheme: QuantScheme = "int8".parse().expect("parse int8");
        assert!(matches!(scheme, QuantScheme::Int8));
    }

    #[test]
    fn test_quant_scheme_parse_int4() {
        let scheme: QuantScheme = "int4".parse().expect("parse int4");
        assert!(matches!(scheme, QuantScheme::Int4));
    }

    #[test]
    fn test_quant_scheme_parse_fp16() {
        let scheme: QuantScheme = "fp16".parse().expect("parse fp16");
        assert!(matches!(scheme, QuantScheme::Fp16));
    }

    #[test]
    fn test_quant_scheme_parse_q4k() {
        let scheme: QuantScheme = "q4k".parse().expect("parse q4k");
        assert!(matches!(scheme, QuantScheme::Q4K));
    }

    #[test]
    fn test_quant_scheme_parse_aliases() {
        assert!("i8".parse::<QuantScheme>().is_ok());
        assert!("i4".parse::<QuantScheme>().is_ok());
        assert!("q8_0".parse::<QuantScheme>().is_ok());
        assert!("q4_0".parse::<QuantScheme>().is_ok());
        assert!("f16".parse::<QuantScheme>().is_ok());
        assert!("half".parse::<QuantScheme>().is_ok());
        assert!("q4_k".parse::<QuantScheme>().is_ok());
        assert!("q4_k_m".parse::<QuantScheme>().is_ok());
    }

    #[test]
    fn test_quant_scheme_parse_unknown() {
        assert!("unknown".parse::<QuantScheme>().is_err());
    }

    #[test]
    fn test_quant_scheme_to_quant_type() {
        assert!(matches!(
            QuantizationType::from(QuantScheme::Int8),
            QuantizationType::Int8
        ));
        assert!(matches!(
            QuantizationType::from(QuantScheme::Q4K),
            QuantizationType::Q4K
        ));
    }

    #[test]
    fn test_estimate_memory_int4() {
        let (input, output, ratio) = estimate_memory(1_000_000, QuantScheme::Int4);
        assert_eq!(input, 1_000_000);
        assert_eq!(output, 125_000); // 4/32 = 0.125
        assert!((ratio - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_memory_fp16() {
        let (_, output, ratio) = estimate_memory(1_000_000, QuantScheme::Fp16);
        assert_eq!(output, 500_000); // 16/32 = 0.5
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            "int4",
            Some(Path::new("/tmp/output.apr")),
            None,
            None,
            false,
            false,
            false,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_run_unknown_scheme() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            input.path(),
            "bad_scheme",
            Some(Path::new("/tmp/output.apr")),
            None,
            None,
            false,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Unknown quantization scheme"));
            }
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_overwrite_protection() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let output = NamedTempFile::with_suffix(".apr").expect("create output");
        let result = run(
            input.path(),
            "int4",
            Some(output.path()),
            None,
            None,
            false,
            false, // force=false
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("already exists"));
            }
            _ => panic!("Expected overwrite protection error"),
        }
    }

    #[test]
    fn test_run_plan_mode() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 1024]).expect("write");
        let result = run(
            input.path(),
            "int4",
            None, // plan mode doesn't need output
            None,
            None,
            true, // plan only
            false,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_plan_json() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 1024]).expect("write");
        let result = run(
            input.path(),
            "int4",
            None, // plan mode doesn't need output
            None,
            None,
            true,
            false,
            true, // json
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_invalid_apr_content() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(b"not valid APR data").expect("write");
        let result = run(
            input.path(),
            "int4",
            Some(Path::new("/tmp/output.apr")),
            None,
            None,
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_empty_schemes() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run_batch(input.path(), "", Path::new("/tmp/batch/"), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_invalid_scheme() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run_batch(
            input.path(),
            "int4,unknown",
            Path::new("/tmp/batch/"),
            false,
            false,
        );
        assert!(result.is_err());
    }

    // ── #2392 finding 3: `--plan -s q4k` returned a hardcoded 7.111x ratio ──

    /// Write a SafeTensors file whose tensors are deliberately a mix of
    /// Q4K-eligible and Q4K-ineligible: a big embedding table (skipped by name,
    /// stays F32) plus a projection whose row width is a clean multiple of 256.
    fn mixed_safetensors(path: &Path) {
        use aprender::serialization::safetensors::save_safetensors;
        use std::collections::BTreeMap;
        let mut t: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        t.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.5; 128 * 512], vec![128, 512]),
        );
        t.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            (vec![0.25; 128 * 512], vec![128, 512]),
        );
        save_safetensors(path, &t).expect("write safetensors fixture");
    }

    /// The CLI must actually ASK the shape-aware estimator. The flat model
    /// returns `file_size * 4.5 / 32` for Q4K no matter what is in the file —
    /// on a model that is half ineligible embedding, that is wildly optimistic.
    #[test]
    fn q4k_plan_uses_the_shape_aware_estimate_not_the_flat_ratio() {
        let dir = std::env::temp_dir().join(format!("apr-2392-f3cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let src = dir.join("mixed.safetensors");
        mixed_safetensors(&src);
        let file_size = std::fs::metadata(&src).expect("stat").len();

        let (_, planned, planned_ratio) = estimate_sizes(&src, file_size, QuantScheme::Q4K);
        let (_, flat, flat_ratio) = estimate_memory(file_size, QuantScheme::Q4K);
        let direct = aprender::format::q4k_output_size_estimate(&src);
        let _ = std::fs::remove_dir_all(&dir);

        assert_ne!(
            planned, flat,
            "#2392 finding 3: `--plan -s q4k` still returned the flat bits-per-weight \
             estimate ({flat} bytes, {flat_ratio:.3}x). That number is a constant function of \
             the file size — it was identical (7.111x) for a 4.8 MB, an 87 MB and a 992 MB \
             model, and 4.34x optimistic on a real one."
        );
        assert!(
            planned_ratio < flat_ratio,
            "#2392 finding 3: half of this model is a Q4K-ineligible embedding table that \
             stays F32, so the honest ratio ({planned_ratio:.3}x) must be below the flat \
             assumption ({flat_ratio:.3}x)"
        );
        assert_eq!(
            Some(planned),
            direct,
            "the CLI must report exactly what the estimator computed"
        );
    }

    /// The other three schemes were measured accurate against real conversions,
    /// so they must keep the flat model — this fix is Q4K-only by design.
    #[test]
    fn non_q4k_schemes_keep_the_flat_estimate() {
        let dir = std::env::temp_dir().join(format!("apr-2392-f3cli2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let src = dir.join("mixed.safetensors");
        mixed_safetensors(&src);
        let file_size = std::fs::metadata(&src).expect("stat").len();

        for scheme in [QuantScheme::Int8, QuantScheme::Int4, QuantScheme::Fp16] {
            assert_eq!(
                estimate_sizes(&src, file_size, scheme),
                estimate_memory(file_size, scheme),
                "{scheme:?} must be unaffected by the Q4K estimator change"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
