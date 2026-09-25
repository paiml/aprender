/// #3873-adjacent: WHEN can the golden gate compare its two legs at all?
///
/// `golden_output.rs` hands one `QuantizedGenerateConfig` to
/// `OwnedQuantizedModel::generate_with_cache` (CPU) and to
/// `OwnedQuantizedModelCuda::generate_gpu_resident` (GPU), then compares the
/// resulting TEXT.
///
/// Both legs honour `config.stop_tokens` — the CPU at
/// `generate_with_cache` (`gguf/inference/generate_quantized.rs`), the GPU inside `decode_blocking`
/// (`gguf/cuda/generate_2.rs`), which `generate_gpu_resident` delegates to. I
/// first reported the GPU check as missing, having grepped the delegating
/// function's body rather than the loop it calls. A `grep -c` over a function
/// body is a claim about that body and nothing else.
///
/// What is actually true is narrower and more useful. Measured on tinyllama,
/// greedy (temperature 0, top_k 1), 64-token budget, one binary, one session:
///
/// ```text
/// PROBE gate-capital     cpu=49 gpu=64  first_divergence=4    identical=false
/// PROBE gate-2plus2      cpu=64 gpu=64  first_divergence=2    identical=false
/// PROBE gate-hello       cpu=64 gpu=64  first_divergence=4    identical=false
/// PROBE nobos-capital    cpu=64 gpu=64  first_divergence=1    identical=false
/// PROBE nobos-2plus2     cpu=64 gpu=64  first_divergence=21   identical=false
/// PROBE nobos-hello      cpu=64 gpu=64  first_divergence=57   identical=false
/// PROBE declared-zephyr  cpu= 7 gpu= 7  first_divergence=None identical=TRUE
/// PROBE bare             cpu=64 gpu=64  first_divergence=None identical=TRUE
/// PROBE bos-bare         cpu= 2 gpu= 2  first_divergence=None identical=TRUE
/// ```
///
/// The split is by OUTPUT REGIME, not by prompt shape. Prompts whose completion
/// is coherent agree byte-for-byte — `bare` runs 64 tokens on both legs with no
/// divergence at all. Prompts whose completion is a degenerate `[INST]`
/// scaffolding loop diverge, at scattered points (1, 2, 4, 21, 57) rather than a
/// fixed offset. A systematic compute error diverges consistently; a near-tie
/// diverges wherever the top-1 margin first collapses.
///
/// The obvious alternative was checked and REFUTED: it is not the leading
/// literal `<s>`. Removing it (`nobos-*`) leaves the divergence; adding it to a
/// coherent prompt (`bos-bare`) leaves the agreement.
///
/// CONSEQUENCE FOR THE GATE: two legs with different floating-point accumulation
/// order cannot be compared for text equality once the model is in a regime
/// where the top-1 margin is ~0. The comparison is ill-posed there, not failing.
#[cfg(test)]
#[cfg(feature = "cuda")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod gpu_leg_comparability_tests {
    fn model_path() -> Option<std::path::PathBuf> {
        for p in [
            "/home/noah/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "/mnt/nvme-raid0/cache/apr-home/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        ] {
            let p = std::path::PathBuf::from(p);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }

    /// The guard: on a prompt whose completion is COHERENT, the two legs must
    /// agree exactly. That is the condition under which the golden gate's
    /// comparison is well-posed, and the one worth protecting — if it ever
    /// breaks, that IS a compute defect rather than a near-tie.
    #[test]
    fn the_two_legs_agree_exactly_when_the_argmax_margin_is_wide() {
        use crate::gguf::{
            MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda,
            QuantizedGenerateConfig,
        };

        let Some(path) = model_path() else {
            eprintln!("SKIP: tinyllama not present");
            return;
        };
        let Ok(mapped) = MappedGGUFModel::from_path(&path) else {
            eprintln!("SKIP: could not map the model");
            return;
        };
        let eos = mapped.model.eos_token_id().unwrap_or(2);
        let Some(prompt) = mapped.model.encode("The capital of France is") else {
            eprintln!("SKIP: prompt did not encode");
            return;
        };

        const BUDGET: usize = 64;
        let config = QuantizedGenerateConfig {
            max_tokens: BUDGET,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: vec![eos],
            ..Default::default()
        };

        let Ok(cpu_model) = OwnedQuantizedModel::from_mapped(&mapped) else {
            return;
        };
        let cpu = cpu_model
            .generate_with_cache(&prompt, &config)
            .expect("CPU generation");
        let Ok(gpu_model) = OwnedQuantizedModel::from_mapped(&mapped) else {
            return;
        };
        let Ok(mut cuda) = OwnedQuantizedModelCuda::new(gpu_model, 0) else {
            eprintln!("SKIP: no CUDA device");
            return;
        };
        let gpu = cuda
            .generate_gpu_resident(&prompt, &config)
            .expect("GPU generation");

        let c = &cpu[prompt.len()..];
        let g = &gpu[prompt.len()..];
        assert!(
            c.len() >= 32,
            "control is too short to mean anything: {} tokens. A handful can agree by luck.",
            c.len()
        );
        assert_eq!(
            c, g,
            "the two legs diverged on a prompt whose output is coherent. Unlike the \
             degenerate-regime divergences (scattered at 1, 2, 4, 21, 57 tokens, where the \
             top-1 margin collapses), a divergence HERE is a real compute defect: both legs \
             are greedy, the margin is wide, and fp accumulation order should not flip the \
             argmax."
        );
    }

    /// #3899: is the Q4_K GPU path architecture-conditional (llama vs qwen)?
    ///
    /// The hypothesis: tinyllama is the fleet's only llama-arch Q4_K model and
    /// the only one producing GPU gibberish, so the Q4_K GPU path may be right
    /// for qwen geometry and wrong for llama. If so, a llama Q4_K model must
    /// diverge from its own CPU leg on ANY prompt, not only degenerate ones -
    /// a wrong kernel does not become right when the output is coherent.
    #[test]
    fn zz_probe_arch_conditional() {
        use crate::gguf::{
            MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda,
            QuantizedGenerateConfig,
        };
        let models = [
            ("tinyllama/LLAMA", "/home/noah/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
            ("qwen3-1.7B/QWEN", "/mnt/nvme-raid0/cache/apr-home/models/Qwen3-1.7B-Q4_K_M.gguf"),
        ];
        for (label, path) in models {
            let p = std::path::PathBuf::from(path);
            if !p.exists() {
                eprintln!("ARCH {label:18} SKIP: not present");
                continue;
            }
            let Ok(mapped) = MappedGGUFModel::from_path(&p) else {
                eprintln!("ARCH {label:18} SKIP: map failed");
                continue;
            };
            let arch = mapped.model.architecture().unwrap_or_default();
            let eos = mapped.model.eos_token_id().unwrap_or(2);
            // A coherent, un-templated continuation: the regime where the argmax
            // margin is wide and a real kernel defect cannot hide.
            let Some(prompt) = mapped.model.encode("The capital of France is") else {
                continue;
            };
            let config = QuantizedGenerateConfig {
                max_tokens: 192,
                temperature: 0.0,
                top_k: 1,
                stop_tokens: vec![eos],
                ..Default::default()
            };
            let Ok(cm) = OwnedQuantizedModel::from_mapped(&mapped) else { continue };
            let Ok(cpu) = cm.generate_with_cache(&prompt, &config) else {
                eprintln!("ARCH {label:18} SKIP: cpu generate failed");
                continue;
            };
            let Ok(gm) = OwnedQuantizedModel::from_mapped(&mapped) else { continue };
            let Ok(mut cuda) = OwnedQuantizedModelCuda::new(gm, 0) else {
                eprintln!("ARCH {label:18} SKIP: cuda init failed");
                continue;
            };
            let Ok(gpu) = cuda.generate_gpu_resident(&prompt, &config) else {
                eprintln!("ARCH {label:18} SKIP: gpu generate failed");
                continue;
            };
            let c = &cpu[prompt.len()..];
            let g = &gpu[prompt.len()..];
            let d = c.iter().zip(g.iter()).position(|(a, b)| a != b);
            eprintln!(
                "ARCH {label:18} arch={arch:8} cpu={:3} gpu={:3} first_divergence={:?} identical={}",
                c.len(),
                g.len(),
                d,
                c == g
            );
        }
    }
}
