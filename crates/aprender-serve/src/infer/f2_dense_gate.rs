// #3602: the dense GGUF F2 guard behind its receipt. The decision table and its
// falsifiers are CUDA-free in `gguf/inference/forward/f2_dense_receipt.rs`; this is
// the GPU-facing half, the dense twin of `f2_validate_qwen35_receipted`.

/// What the receipted dense guard decided.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DenseF2 {
    /// A matching receipt admitted the GPU; the CPU reference forward did not run.
    Receipt,
    /// The guard ran; this is its outcome.
    Fresh(F2Outcome),
}

/// [`validate_gpu_first_token`] once per (model sha256, apr build, device,
/// prefill precision). Every path that is not a match validates; a receipt is
/// written after a [`F2Outcome::Validated`] only. Nothing here fails the run
/// except the guard's own rejection: no cache dir, an unhashable model file or
/// an unwritable receipt are printed, and that run simply validates.
#[cfg(feature = "cuda")]
pub(crate) fn validate_dense_f2_receipted(
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    input_tokens: &[u32],
    model_path: &std::path::Path,
) -> DenseF2 {
    use crate::gguf::f2_dense_receipt::{
        decide_dense, dense_device_key, DenseDecision, DensePrecision,
    };
    use crate::gguf::f2_receipt::{
        apr_version, model_sha256_file, read_receipt, receipt_dir, receipt_path,
        revalidate_requested, unix_now, write_receipt, F2Receipt, F2ReceiptKey, F2_RECEIPT_SCHEMA,
    };

    let sha_start = std::time::Instant::now();
    let sha256 = match model_sha256_file(model_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "F2 guard: cannot hash {} ({e}), so no receipt applies — validating [source=fresh]",
                model_path.display()
            );
            return DenseF2::Fresh(validate_gpu_first_token(
                cuda_model,
                gen_config,
                input_tokens,
            ));
        },
    };
    let sha256_ms = sha_start.elapsed().as_secs_f64() * 1000.0;
    let apr = apr_version();
    let device = cuda_model.device_name().to_string();
    let fp8_before = cuda_model.executor.gpu_profile.fp8_prefill;

    let path = receipt_dir().map(|d| receipt_path(&d, &sha256));
    let found = match path.as_deref() {
        Some(p) => read_receipt(p),
        None => Ok(None),
    };

    match decide_dense(
        found,
        &sha256,
        &apr,
        &device,
        fp8_before,
        revalidate_requested(),
    ) {
        DenseDecision::Skip {
            receipt,
            force_fp16,
        } => {
            if force_fp16 {
                f2_force_fp16_prefill(cuda_model);
            }
            eprintln!(
                "F2 guard: receipt matches (model sha256 {}…, apr {}, {}) — validated {}s ago on {} positions; CPU reference forward skipped [source=receipt, sha256 {sha256_ms:.0} ms]{}. `apr run --revalidate` forces a fresh run.",
                &sha256[..12],
                apr,
                receipt.key.device,
                unix_now().saturating_sub(receipt.validated_at),
                receipt.positions_judged,
                if force_fp16 {
                    "; FP8 prefill OFF, as the validating run ended"
                } else {
                    ""
                },
            );
            return DenseF2::Receipt;
        },
        DenseDecision::Validate(reason) => {
            eprintln!("F2 guard: validating on this run ({reason}) [source=fresh]");
        },
    }

    let start = std::time::Instant::now();
    let outcome = validate_gpu_first_token(cuda_model, gen_config, input_tokens);
    let validate_ms = start.elapsed().as_secs_f64() * 1000.0;

    if matches!(outcome, F2Outcome::Validated { .. }) {
        let precision = DensePrecision::after_validation(
            fp8_before,
            cuda_model.executor.gpu_profile.fp8_prefill,
        );
        // The count the guard itself passed to `f2_accept_or_reject`: its CPU
        // reference holds probe.len() + 1 logits (the probe plus one decode step)
        // and judges all but pos0, i.e. probe.len().
        let positions_judged =
            gpu_probe(cuda_model.model(), input_tokens).map_or(0, |(_, _, probe)| probe.len());
        match path.as_deref() {
            Some(p) => {
                let receipt = F2Receipt {
                    schema: F2_RECEIPT_SCHEMA,
                    key: F2ReceiptKey {
                        model_sha256: sha256,
                        apr_version: apr,
                        device: dense_device_key(&device, precision),
                    },
                    validated_at: unix_now(),
                    positions_judged,
                };
                match write_receipt(p, &receipt) {
                    Ok(()) => eprintln!(
                        "F2 guard: passed in {validate_ms:.0} ms ({}); receipt written to {} — the next run of this (model, apr, device, precision) skips it [sha256 {sha256_ms:.0} ms].",
                        receipt.key.device,
                        p.display()
                    ),
                    Err(e) => eprintln!(
                        "F2 guard: passed in {validate_ms:.0} ms, but the receipt could not be written ({e}); the next run validates again. Set APR_F2_RECEIPT_DIR to a writable directory."
                    ),
                }
            },
            None => eprintln!(
                "F2 guard: passed in {validate_ms:.0} ms; no cache directory (no HOME, XDG_CACHE_HOME or APR_F2_RECEIPT_DIR), so no receipt — every run validates."
            ),
        }
    }
    DenseF2::Fresh(outcome)
}
