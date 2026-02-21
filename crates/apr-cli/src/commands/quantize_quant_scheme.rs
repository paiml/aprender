
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
}
