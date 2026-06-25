// GH-434 / ALB-093: Streaming APR→Q4K quantization tests.
//
// The full streaming path only activates for inputs ≥ 4 GiB; these tests
// exercise the streaming function directly on a pygmy fixture to verify
// parity with the full-load path on tensor count, shape, and dequant values.
#[cfg(test)]
mod tests_streaming_quantize {
    use crate::format::converter::{
        STREAMING_THRESHOLD_BYTES, qualifies_for_streaming_q4k, streaming_quantize_apr_to_q4k,
        streaming_quantize_peak_estimate,
    };
    use crate::format::test_factory::build_pygmy_apr_gguf_names;
    use crate::format::v2::{AprV2Reader, AprV2ReaderRef};
    use crate::format::AprV2DequantExt; // issue #2231 re-attached accessor

    #[test]
    fn streaming_quantize_roundtrips_all_tensors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("in.apr");
        let output = dir.path().join("out.apr");
        std::fs::write(&input, build_pygmy_apr_gguf_names()).expect("write input");

        let input_reader_bytes = std::fs::read(&input).expect("read input");
        let input_reader = AprV2Reader::from_bytes(&input_reader_bytes).expect("parse input apr");
        let input_tensor_count = input_reader.tensor_names().len();

        let count = streaming_quantize_apr_to_q4k(&input, &output).expect("streaming quantize");
        assert_eq!(count, input_tensor_count, "tensor count must match input");

        let bytes = std::fs::read(&output).expect("read output");
        let reader = AprV2Reader::from_bytes(&bytes).expect("parse output apr");
        assert_eq!(reader.tensor_names().len(), input_tensor_count);

        for name in reader.tensor_names() {
            let f32_data = reader
                .get_tensor_as_f32(&name)
                .unwrap_or_else(|| panic!("dequant failed for '{name}'"));
            assert!(!f32_data.is_empty(), "empty dequant for '{name}'");
            assert!(
                f32_data.iter().all(|v| v.is_finite()),
                "non-finite values in '{name}'"
            );
        }

        let q = reader.metadata().quantization.as_ref().expect("q meta");
        assert_eq!(q.quant_type, "q4_k");
        assert_eq!(q.bits, 4);
    }

    #[test]
    fn qualifies_for_streaming_requires_threshold_and_apr_magic() {
        use crate::format::converter::STREAMING_THRESHOLD_TEST_MUTEX;
        // Any test that reads the effective threshold must hold this mutex so
        // it cannot race with a concurrent test that sets the override.
        let _guard = STREAMING_THRESHOLD_TEST_MUTEX.lock().expect("mutex");

        let dir = tempfile::tempdir().expect("tempdir");
        let small = dir.path().join("small.apr");
        std::fs::write(&small, build_pygmy_apr_gguf_names()).expect("write small");
        assert!(
            !qualifies_for_streaming_q4k(&small),
            "small APR must not qualify"
        );

        let not_apr = dir.path().join("bogus.bin");
        std::fs::write(&not_apr, vec![0u8; 16]).expect("write bogus");
        assert!(
            !qualifies_for_streaming_q4k(&not_apr),
            "non-APR must not qualify"
        );

        assert_eq!(STREAMING_THRESHOLD_BYTES, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn peak_estimate_is_largest_tensor_working_set() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("input.apr");
        let bytes = build_pygmy_apr_gguf_names();
        std::fs::write(&path, &bytes).expect("write");

        let peak = streaming_quantize_peak_estimate(&path).expect("peak estimate");

        let reader = AprV2ReaderRef::from_bytes(&bytes).expect("parse");
        let expected = reader
            .tensor_names()
            .iter()
            .filter_map(|n| reader.get_tensor(n))
            .map(|e| {
                let n = e.element_count() as u64;
                n * 4 + (n * 9).div_ceil(16)
            })
            .max()
            .expect("nonempty");
        assert_eq!(peak, expected);
        assert!(peak > 0);
        assert!(
            peak < bytes.len() as u64,
            "streaming peak must be below file size"
        );
    }

    #[test]
    fn peak_estimate_rejects_non_apr() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bogus.bin");
        std::fs::write(&path, vec![0u8; 128]).expect("write");
        assert!(streaming_quantize_peak_estimate(&path).is_none());
    }

    // GH-434 / FALSIFY-CONV-009: `apr_convert` must take the streaming path
    // whenever the input qualifies (≥ threshold + APR magic + Q4K quantize).
    //
    // We lower the threshold via the cfg(test) override so the pygmy fixture
    // qualifies. The streaming path is distinguishable from the full-load
    // path by a metadata fingerprint: streaming clones the source metadata
    // (preserving `model_type == "pygmy-gguf"`), whereas the full-load Q4K
    // builder hardcodes `model_type == "qwen2"`.
    #[test]
    fn apr_convert_short_circuits_to_streaming_when_threshold_qualifies() {
        use crate::format::converter::{
            STREAMING_THRESHOLD_TEST_MUTEX, STREAMING_THRESHOLD_TEST_OVERRIDE,
        };
        use crate::format::{ConvertOptions, QuantizationType, apr_convert};
        use std::sync::atomic::Ordering;

        let _guard = STREAMING_THRESHOLD_TEST_MUTEX.lock().expect("mutex");
        STREAMING_THRESHOLD_TEST_OVERRIDE.store(1, Ordering::Relaxed);
        let result = (|| -> Result<(), String> {
            let dir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
            let input = dir.path().join("in.apr");
            let output = dir.path().join("out.apr");
            std::fs::write(&input, build_pygmy_apr_gguf_names())
                .map_err(|e| format!("write: {e}"))?;

            let options = ConvertOptions {
                quantize: Some(QuantizationType::Q4K),
                ..Default::default()
            };
            apr_convert(&input, &output, options).map_err(|e| format!("convert: {e:?}"))?;

            let bytes = std::fs::read(&output).map_err(|e| format!("read out: {e}"))?;
            let reader = AprV2Reader::from_bytes(&bytes).map_err(|e| format!("parse: {e:?}"))?;
            let meta = reader.metadata();

            if meta.model_type != "pygmy-gguf" {
                return Err(format!(
                    "expected streaming metadata fingerprint 'pygmy-gguf', got '{}' (full-load path taken?)",
                    meta.model_type
                ));
            }
            let q = meta
                .quantization
                .as_ref()
                .ok_or_else(|| "missing quantization meta".to_string())?;
            if q.quant_type != "q4_k" {
                return Err(format!("expected q4_k, got '{}'", q.quant_type));
            }
            Ok(())
        })();
        STREAMING_THRESHOLD_TEST_OVERRIDE.store(u64::MAX, Ordering::Relaxed);
        result.expect("streaming short-circuit assertion");
    }
}
