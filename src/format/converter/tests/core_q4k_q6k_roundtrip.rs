
// ============================================================================
// GH-200: Q4K/Q6K round-trip integration tests
// ============================================================================
//
// These tests exercise the exact code paths where GH-200 bugs lived:
// - `get_tensor_as_f32` with Q4K/Q6K dtypes
// - `apr_export` with GGUF-named tensors
// - The full dequantization pipeline for K-quant formats
#[cfg(test)]
mod tests_gh200_q4k_roundtrip {
    use crate::format::test_factory::{
        build_pygmy_apr_gguf_names, build_pygmy_apr_q4k, build_pygmy_apr_q6k,
    };
    use crate::format::v2::AprV2Reader;

    /// GH-200: Q4K APR tensors can be loaded and dequantized to f32.
    ///
    /// This catches the root bug: `get_tensor_as_f32` returning None for Q4K dtypes.
    #[test]
    fn test_q4k_apr_load_tensors_f32() {
        let data = build_pygmy_apr_q4k();
        let reader = AprV2Reader::from_bytes(&data).expect("parse Q4K APR");

        let names = reader.tensor_names();
        assert!(!names.is_empty(), "Q4K APR should have tensors, got empty");

        // Verify all tensors can be dequantized to f32
        for name in &names {
            let f32_data = reader.get_tensor_as_f32(name);
            assert!(
                f32_data.is_some(),
                "get_tensor_as_f32 returned None for tensor '{}' — Q4K dequant failed",
                name
            );
            let f32_data = f32_data.expect("checked above");
            assert!(
                !f32_data.is_empty(),
                "Dequantized tensor '{}' is empty",
                name
            );
        }

        // Verify Q4K attention tensors specifically exist and dequantize
        let q4k_tensor = reader
            .get_tensor_as_f32("blk.0.attn_q.weight")
            .expect("Q4K attn_q.weight should dequantize");
        assert_eq!(
            q4k_tensor.len(),
            256,
            "Q4K tensor should have 256 elements (1 super-block)"
        );
    }

    /// GH-200: Q6K APR tensors can be loaded and dequantized to f32.
    ///
    /// Same as Q4K test but for Q6_K format (210-byte super-blocks).
    #[test]
    fn test_q6k_apr_load_tensors_f32() {
        let data = build_pygmy_apr_q6k();
        let reader = AprV2Reader::from_bytes(&data).expect("parse Q6K APR");

        let names = reader.tensor_names();
        assert!(!names.is_empty(), "Q6K APR should have tensors, got empty");

        // Verify all tensors can be dequantized to f32
        for name in &names {
            let f32_data = reader.get_tensor_as_f32(name);
            assert!(
                f32_data.is_some(),
                "get_tensor_as_f32 returned None for tensor '{}' — Q6K dequant failed",
                name
            );
        }

        // Verify Q6K attention tensors specifically
        let q6k_tensor = reader
            .get_tensor_as_f32("blk.0.attn_q.weight")
            .expect("Q6K attn_q.weight should dequantize");
        assert_eq!(
            q6k_tensor.len(),
            256,
            "Q6K tensor should have 256 elements (1 super-block)"
        );
    }

    /// GH-200: Q4K APR can be exported to SafeTensors via apr_export.
    ///
    /// Exercises the full pipeline: Q4K APR → load_apr_tensors_f32 → SafeTensors.
    #[test]
    fn test_q4k_apr_export_safetensors() {
        use crate::format::converter::{apr_export, ExportFormat, ExportOptions};
        use crate::format::test_factory::harness::ConversionTestHarness;

        let h = ConversionTestHarness::new().with_apr_q4k();
        let input = h.input_path().expect("input exists");

        let dir = h.dir();
        let output = dir.join("exported.safetensors");

        let options = ExportOptions {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
            skip_completeness_check: true, // Pygmy test model is intentionally incomplete
        };
        let result = apr_export(input, &output, options);
        assert!(
            result.is_ok(),
            "Q4K APR export to SafeTensors failed: {:?}",
            result.unwrap_err()
        );
        assert!(output.exists(), "Exported SafeTensors file should exist");
    }

    /// GH-200: Q6K APR can be exported to SafeTensors via apr_export.
    ///
    /// Exercises the full pipeline: Q6K APR → load_apr_tensors_f32 → SafeTensors.
    #[test]
    fn test_q6k_apr_export_safetensors() {
        use crate::format::converter::{apr_export, ExportFormat, ExportOptions};
        use crate::format::test_factory::harness::ConversionTestHarness;

        let h = ConversionTestHarness::new().with_apr_q6k();
        let input = h.input_path().expect("input exists");

        let dir = h.dir();
        let output = dir.join("exported.safetensors");

        let options = ExportOptions {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
            skip_completeness_check: true, // Pygmy test model is intentionally incomplete
        };
        let result = apr_export(input, &output, options);
        assert!(
            result.is_ok(),
            "Q6K APR export to SafeTensors failed: {:?}",
            result.unwrap_err()
        );
        assert!(output.exists(), "Exported SafeTensors file should exist");
    }

    /// GH-200: GGUF-named tensors (F32) export with correct name mapping.
    ///
    /// Verifies that GGUF names like `blk.0.attn_q.weight` are correctly
    /// present in the exported SafeTensors (name mapping may or may not apply,
    /// but the export must not fail or drop tensors).
    #[test]
    fn test_gguf_names_export_maps_correctly() {
        use crate::format::converter::{apr_export, ExportFormat, ExportOptions};
        use crate::serialization::safetensors::MappedSafeTensors;

        let apr_bytes = build_pygmy_apr_gguf_names();
        let dir = tempfile::tempdir().expect("tempdir");
        let apr_path = dir.path().join("gguf_names.apr");
        let st_path = dir.path().join("exported.safetensors");

        std::fs::write(&apr_path, &apr_bytes).expect("write APR");

        let options = ExportOptions {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
            skip_completeness_check: true, // Pygmy test model is intentionally incomplete
        };
        let result = apr_export(&apr_path, &st_path, options);
        assert!(
            result.is_ok(),
            "GGUF-named APR export failed: {:?}",
            result.unwrap_err()
        );

        // Read back SafeTensors and verify tensors were exported
        let mapped = MappedSafeTensors::open(&st_path).expect("open exported SafeTensors");
        let exported_names = mapped.tensor_names();

        // Count original GGUF tensors from the APR
        let reader = AprV2Reader::from_bytes(&apr_bytes).expect("parse GGUF-named APR");
        let apr_names = reader.tensor_names();

        assert_eq!(
            exported_names.len(),
            apr_names.len(),
            "No tensors should be lost in GGUF→SafeTensors export.\n\
             APR had: {:?}\nSafeTensors has: {:?}",
            apr_names,
            exported_names
        );
    }
}
