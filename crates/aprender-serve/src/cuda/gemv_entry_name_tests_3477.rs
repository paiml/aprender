// #3477: the launcher looks a kernel up BY NAME inside the module it just
// compiled. `kernel_name()` and the emitted PTX's `.visible .entry` are two
// strings in two files that must agree, and nothing checked that they did.
//
// A mismatch does not fail to compile and does not fail to load — the module
// builds, and the failure is a function lookup at launch time, on a GPU box
// only, reported as a CUDA error rather than as "these two strings disagree".
// The existing PTX tests in this crate assert `ptx.contains(".version")`, which
// no rename can violate.
//
// This is GPU-FREE: it generates PTX and reads it. It does need `--features
// cuda` to compile, so it runs in the cuda-unit CI job and NOT in the coverage
// target (which does not pass the feature).
#[cfg(test)]
mod gemv_entry_name_tests_3477 {
    use crate::cuda::{CudaKernels, KernelType};

    /// Every GEMV kernel the dispatch can reach, with the shape it is built at.
    fn every_gemv_kernel() -> Vec<KernelType> {
        let (k, n) = (4096u32, 4096u32);
        vec![
            KernelType::Gemv { k, n },
            KernelType::CoalescedGemv { k, n },
            KernelType::Q4KGemv { k, n },
            KernelType::Q5KGemv { k, n },
            KernelType::Q6KGemv { k, n },
            KernelType::Q8_0Gemv { k, n },
            KernelType::Q5_0Gemv { k, n },
            KernelType::Q4_0Gemv { k, n },
            KernelType::Q4_1Gemv { k, n },
            KernelType::F16Gemv { k, n },
            KernelType::Iq4XsGemv { k, n },
            KernelType::Iq4NlGemv { k, n },
            KernelType::Iq3SGemv { k, n },
            KernelType::Iq2XxsGemv { k, n },
            KernelType::Iq3XxsGemv { k, n },
            KernelType::Bf16Gemv { k, n },
            KernelType::Iq2SGemv { k, n },
            KernelType::Q5_1Gemv { k, n },
        ]
    }

    /// #3970: the Q4_K / Q6_K STRATEGY kernels (tiled, DP4A, multi-warp, batched,
    /// fused, ...) picked by the selection path rather than by `BoundWeight`.
    /// They were excluded from the completeness check by a named rule, so none of
    /// them — hot-path `MwvDp4aQ4KGemv` included — was ever ptxas-assembled.
    /// Parameters are the values the executor dispatches with (4 warps / 4 rows /
    /// 4 outputs per block, eps 1e-6).
    fn every_strategy_gemv_kernel() -> Vec<KernelType> {
        let (k, n, w) = (4096u32, 4096u32, 4u32);
        vec![
            KernelType::TiledQ4KGemv { k, n, outputs_per_block: w },
            KernelType::ChunkedTiledQ4KGemv { k, n, outputs_per_block: w },
            KernelType::CoalescedQ4KGemv { k, n },
            KernelType::WideQ4KGemv { k, n },
            KernelType::VectorizedQ4KGemv { k, n },
            KernelType::MwvQ4KGemv { k, n, num_warps: w },
            KernelType::MwvDp4aQ4KGemv { k, n, num_warps: w },
            KernelType::HwDp4aQ4KGemv { k, n, num_warps: w },
            KernelType::Dp4aQ4KGemv { k, n },
            KernelType::Dp4aSIMDQ4KGemv { k, n },
            KernelType::TrueDp4aQ4KGemv { k, n },
            KernelType::BatchedQ4KGemv { m: w, k, n },
            KernelType::MultiWarpBatchedQ4KGemv { k, n, warps: w },
            KernelType::BatchedHwDp4aQ4KGemv { k, n, m: w, num_warps: w },
            KernelType::FusedFp32Q4KGemv { k, n, m: w, num_warps: w },
            KernelType::InlineQ8Dp4aQ4KGemv { k, n, m: w, num_warps: w },
            KernelType::CoalescedQ6KGemv { k, n },
            KernelType::BatchedQ6KGemv { k, n, m: w },
            KernelType::MwvQ6KGemv { k, n, num_warps: w },
            KernelType::Dp4aQ6KGemv { k, n, num_warps: w },
            KernelType::HwDp4aQ6KGemv { k, n, num_warps: w },
            KernelType::Fp16Q4KGemv { k, n },
            KernelType::FusedRmsNormQ4KGemv { k, n, epsilon: 1e-6 },
            KernelType::FusedGateUpQ4KGemv { k, n },
            KernelType::FusedGateUpSwigluHwDp4aQ4KGemv { k, n },
        ]
    }

    /// Every GEMV kernel the crate can emit: the per-weight-type set plus the
    /// strategy set. The assembly and entry-name guards iterate THIS.
    fn every_emitted_gemv_kernel() -> Vec<KernelType> {
        let mut all = every_gemv_kernel();
        all.extend(every_strategy_gemv_kernel());
        all
    }

    /// #3931: `every_gemv_kernel` is a HAND-WRITTEN list, and every assembly and
    /// entry-name guard in this file iterates it. A new kernel that is not added
    /// here is never assembled by `every_emitted_kernel_assembles`, and that test
    /// stays green — which is what happened to `Iq2XxsGemv` on its first build.
    ///
    /// So the list's completeness is derived from the enum's own source: every
    /// `KernelType` variant whose name ends in `Gemv` and takes `{ k, n }` must
    /// appear in `every_gemv_kernel`. A kernel the dispatch can reach cannot be
    /// skipped by the guards meant to catch it.
    #[test]
    fn every_gemv_variant_is_in_the_list_the_guards_iterate() {
        let enum_src = include_str!("kernel_type.rs");
        let this_src = include_str!("gemv_entry_name_tests_3477.rs");
        let listed: std::collections::BTreeSet<&str> =
            ["fn every_gemv_kernel()", "fn every_strategy_gemv_kernel()"]
                .iter()
                .flat_map(|marker| {
                    this_src
                        .split(marker)
                        .nth(1)
                        .and_then(|t| t.split("]").next())
                        .expect("a kernel list's vec! literal")
                        .split("KernelType::")
                        .skip(1)
                        .filter_map(|t| t.split_whitespace().next())
                })
                .collect();
        let declared: Vec<&str> = enum_src
            .lines()
            .map(str::trim)
            .filter_map(|l| l.strip_suffix(" {"))
            .filter(|name| {
                name.ends_with("Gemv")
                    && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            })
            .collect();
        assert!(
            declared.len() >= 10,
            "found only {} `*Gemv {{` variants in kernel_type.rs; the parse broke and \
             this test would pass vacuously",
            declared.len()
        );
        // No exclusions (#3970): the Q4_K/Q6_K strategy variants used to be
        // skipped here by a named rule, and so were never assembled.
        let missing: Vec<&str> =
            declared.iter().copied().filter(|v| !listed.contains(v)).collect();
        assert!(
            missing.is_empty(),
            "GEMV kernel variant(s) declared in kernel_type.rs but absent from \
             every_gemv_kernel() / every_strategy_gemv_kernel(), so never assembled \
             or name-checked: {missing:?}"
        );
    }

    /// #3953: a generator's OUTPUT must not contain generator SOURCE, and each module
    /// must declare exactly one kernel -- its own.
    ///
    /// WHAT THE COMPILER ALREADY CATCHES, stated so this guard does not claim it. Every PTX
    /// literal in these generators is a plain `r"..."`, where any `"` ENDS the string. So
    /// Rust source that leaks into a literal -- a function body almost always carries
    /// quotes -- is a `mismatched closing delimiter` compile error, not a silent pass.
    /// (The IQ2_S generator was once inserted inside IQ3_S's literal by a bad anchor; that
    /// would NOT have compiled. The first draft of this comment said it would; it did not
    /// check, and a planted copy failed to compile.)
    ///
    /// WHAT IT DOES NOT CATCH, and this guard does:
    ///   (a) a QUOTE-FREE fragment -- `fn generate_x_ptx(k: u32) -> String {` or
    ///       `let mut acc = 0;` -- which a raw string accepts silently;
    ///   (b) duplicated PTX inside ONE literal. PTX has no quotes, so a kernel body pasted
    ///       twice compiles cleanly, and only ptxas objects, with an opaque duplicate-symbol
    ///       message. Here it is named: "declares 2 `.visible .entry`";
    ///   (c) any literal later switched to `r#"..."#`, where a bare `"` no longer ends it
    ///       and every leak becomes compile-invisible.
    #[test]
    fn no_generator_leaks_into_another_kernels_ptx_3953() {
        // None of these is valid PTX. `///` is deliberately absent: a PTX comment can
        // legitimately contain it, and a guard with false positives gets disabled.
        const RUST_ONLY: &[&str] = &["fn generate_", "String::from(", ".push_str(", "let mut "];
        let kernels = CudaKernels::new();
        let all: Vec<(KernelType, String)> = every_gemv_kernel()
            .into_iter()
            .map(|kt| {
                let name = kernels.kernel_name(&kt).to_string();
                (kt, name)
            })
            .collect();
        let mut broken = Vec::new();
        for (kt, own) in &all {
            let ptx = kernels.generate_ptx(kt);
            for tok in RUST_ONLY {
                if let Some(line) = ptx.lines().find(|l| l.contains(tok)) {
                    broken.push(format!(
                        "\n  - {kt:?}: emitted PTX contains the Rust token {tok:?} -- generator \
                         SOURCE leaked into this kernel's literal: {:?}",
                        line.trim()
                    ));
                }
            }
            let entries = ptx.matches(".visible .entry ").count();
            if entries != 1 {
                broken.push(format!(
                    "\n  - {kt:?}: declares {entries} `.visible .entry`, expected exactly 1 (`{own}`)"
                ));
            }
            for (okt, other) in &all {
                if other != own && ptx.contains(&format!(".visible .entry {other}(")) {
                    broken.push(format!(
                        "\n  - {kt:?}: contains ANOTHER kernel's entry `{other}` ({okt:?}) -- one \
                         generator's body is inside another's literal"
                    ));
                }
            }
        }
        assert!(broken.is_empty(), "PTX generators leaked into each other:{}", broken.concat());
    }

    #[test]
    fn the_name_the_launcher_looks_up_exists_in_the_ptx_it_compiles() {
        let kernels = CudaKernels::new();
        let mut broken = Vec::new();
        for kt in every_emitted_gemv_kernel() {
            let name = kernels.kernel_name(&kt);
            let ptx = kernels.generate_ptx(&kt);
            // ptxas rejects a non-ASCII byte ANYWHERE, comments included, and
            // fails the whole module with `ptxas fatal : Unexpected non-ASCII
            // character`. The entry name can be perfectly correct and the module
            // still never assemble — which is exactly what an em dash in one of
            // MY comments did to the F16 and IQ4_XS kernels (#3477, caught by
            // aprender-45's A/B on its first real run, not by this guard).
            if let Some(bad) = ptx.chars().find(|c| !c.is_ascii()) {
                let line = ptx
                    .lines()
                    .find(|l| l.chars().any(|c| !c.is_ascii()))
                    .unwrap_or("")
                    .trim();
                broken.push(format!(
                    "\n  - {kt:?}: emitted PTX contains U+{:04X}, which ptxas refuses \
                     even inside a comment, so the module never assembles: {line:?}",
                    bad as u32
                ));
                continue;
            }
            let entry = format!(".visible .entry {name}(");
            if !ptx.contains(&entry) {
                let found: Vec<&str> = ptx
                    .lines()
                    .filter(|l| l.contains(".visible .entry"))
                    .map(str::trim)
                    .collect();
                broken.push(format!(
                    "\n  - {kt:?}: kernel_name() says {name:?}, but the PTX declares {found:?}"
                ));
            }
        }
        assert!(
            broken.is_empty(),
            "{} GEMV kernel(s) would fail the function lookup at launch — the name \
             the launcher asks for is not an entry in the PTX it just compiled:{}",
            broken.len(),
            broken.join("")
        );
    }

    /// The four places a new quant type has to be registered, asserted
    /// together. They were consistent before F16 and the compiler enforces the
    /// two `match`es; these are the ones it cannot check — a wrong BYTE COUNT
    /// compiles fine and reads the wrong bytes.
    #[test]
    fn f16_is_registered_consistently_in_every_table() {
        use crate::cuda::types::{GemvKernel, WeightQuantType};

        assert_eq!(WeightQuantType::from_ggml_type(1), Some(WeightQuantType::F16));
        assert_eq!(WeightQuantType::F16.bytes_per_superblock(), 512, "256 elems x 2 bytes");
        assert_eq!(WeightQuantType::F16.bytes_per_block(), 64, "32 elems x 2 bytes");

        // A [2560, 32] ssm_alpha — the actual shape in Qwen3.5-4B-UD-Q4_K_XL.
        assert!(WeightQuantType::F16.matches_size(2560 * 32 * 2, 2560, 32));
        assert!(!WeightQuantType::F16.matches_size(2560 * 32 * 4, 2560, 32), "that is F32's size");

        // from_size must not let F32's check swallow F16, nor the reverse.
        assert_eq!(WeightQuantType::from_size(2560 * 32 * 2, 2560, 32), Some(WeightQuantType::F16));
        assert_eq!(WeightQuantType::from_size(2560 * 32 * 4, 2560, 32), Some(WeightQuantType::F32));

        assert_eq!(
            crate::cuda::types::BoundWeight::bind(0x1000, 2560 * 32 * 2, WeightQuantType::F16, 2560, 32)
                .kernel(),
            GemvKernel::F16,
            "binding F16 to any other kernel decodes 2-byte weights as something else"
        );
    }

    /// THE ORDERING INVARIANT, as a mechanism rather than a habit.
    ///
    /// A type may be admitted by `gpu_unsupported_quant_qtype` ONLY if
    /// `from_ggml_type` maps it to a real kernel. The converse is allowed and
    /// is the deliberate state of IQ4_XS right now: a kernel can exist while
    /// the gate stays shut, pending its own evidence.
    ///
    /// Whitelisting without a kernel is the 0.68.1 defect exactly —
    /// `resolve_qtype` would fall through to Q4_K and decode the weights as a
    /// different scheme, silently. That ordering has had to be re-established
    /// by someone noticing three times on this row; this makes it fail a test
    /// instead.
    #[test]
    fn nothing_is_whitelisted_without_a_kernel_behind_it() {
        use crate::cuda::types::WeightQuantType;
        let wrong: Vec<String> = (0u32..=40)
            .filter(|&t| {
                // admitted by the whitelist ...
                !crate::gguf::gpu_unsupported_quant_qtype(t)
                    // ... but no kernel to decode it with
                    && WeightQuantType::from_ggml_type(t).is_none()
            })
            .map(|t| format!("\n  - GGML type {t} is admitted but has no GEMV kernel"))
            .collect();
        assert!(
            wrong.is_empty(),
            "{} type(s) would be silently decoded as Q4_K by resolve_qtype's fallback, \
             which is the 0.68.1 garbage-logits defect:{}",
            wrong.len(),
            wrong.join("")
        );
    }

    /// F16 IS admitted, and this records WHY — with the sha, because a bare
    /// "217/217" in a comment outlives its provenance and the numbers are not
    /// portable across trees.
    ///
    /// Opened on the evidence, not on the kernel's existence: **217 of 217 F16
    /// tensors exact** against the CPU decoder, worst cosine 1.00000000 —
    /// 169 in Qwen2.5-0.5B-Instruct-f16 and 48 in Qwen3.5-4B-UD-Q4_K_XL,
    /// enumerated from the files — with a planted-fault control going RED on an
    /// F16 tensor specifically, so the greens are licensed rather than merely
    /// reported. Measured by aprender-45 against **`88d25d265`**.
    #[test]
    fn f16_is_admitted_because_its_kernel_was_measured() {
        assert!(
            !crate::gguf::gpu_unsupported_quant_qtype(1),
            "F16 has a measured kernel (217/217 exact) and must be GPU-eligible"
        );
    }

    /// IQ4_XS IS admitted, on its own evidence rather than on F16's.
    ///
    /// **10/10 IQ4_XS tensors exact, cosine 1.00000000, measured at
    /// `b782b4257`** — every IQ4_XS tensor in Qwen3.5-4B-UD-Q4_K_XL,
    /// `[2560, 9216]` each, with a planted fault going RED on an IQ4_XS tensor
    /// specifically (`blk.12.ffn_gate --perturb` -> cosine 0.99120069). An
    /// IQ4_XS green needs an IQ4_XS red; F16's control licenses nothing here.
    ///
    /// The exactness is the part worth keeping: the split 6-bit scale
    /// reassembly and the `ib == m` / `jj == tid` collapse would both have
    /// produced plausible near-matches if wrong, not garbage. 1.00000000 on a
    /// 23.6M-element tensor is not a shape a bad index collapse hides in.
    #[test]
    fn iq4_xs_is_admitted_because_its_kernel_was_measured() {
        assert!(
            !crate::gguf::gpu_unsupported_quant_qtype(23),
            "IQ4_XS has a measured kernel (10/10 exact) and must be GPU-eligible"
        );
    }

    /// #3850: every type the census found in the wild that has no kernel must
    /// map to `None`, which is what `resolve_qtype` now refuses on. Before this
    /// it fell through to Q4_K and the bytes were read as a different scheme.
    ///
    /// These are not hypothetical: all were measured in lambda's inventory.
    /// IQ2_XXS/IQ3_XXS/Q2_K in `Qwen3.5-0.8B-UD-IQ2_XXS`, IQ3_S in two more.
    ///
    /// #3869/#3884/#3885/#3908: IQ4_NL, IQ3_S, Q5_1 and BF16 were on this list
    /// and have been REMOVED because they now have kernels. BF16 sat in
    /// `Qwen3-0.6B-BF16` and in `qwen2.5-coder-0.5b-instruct.apr` (290 of its
    /// 291 tensors), the model whose rc=14 #3908 was filed for. It is not deleted from the guard - it moved to
    /// `iq4_nl_has_a_kernel_but_is_not_admitted_until_it_is_measured` below,
    /// which asserts the other half. A row that outlives its premise is
    /// converted, never dropped.
    #[test]
    fn the_types_found_in_the_wild_without_kernels_resolve_to_none() {
        use crate::cuda::types::WeightQuantType;
        let census: &[(u32, &str)] = &[
            (10, "Q2_K"),
            (11, "Q3_K"),
            (18, "IQ3_XXS"),
            (22, "IQ2_S"),
        ];
        let admitted: Vec<String> = census
            .iter()
            .filter(|(t, _)| WeightQuantType::from_ggml_type(*t).is_some())
            .map(|(t, n)| format!("\n  - {n} (type {t}) claims a kernel it does not have"))
            .collect();
        assert!(admitted.is_empty(), "census drift:{}", admitted.join(""));
    }

    /// #3869: IQ4_NL is admitted BECAUSE its kernel was measured.
    ///
    /// This row was `iq4_nl_has_a_kernel_but_is_not_admitted_until_it_is_measured`
    /// and it said, in its own body, to flip this assertion and open the
    /// whitelist in the same commit once the device A/B passed. It has:
    ///
    /// ```text
    /// #3869 A/B: 64 rows, worst relative disagreement 0.000e0
    /// ```
    ///
    /// EXACT against `iq_parallel_matvec` on the same bytes, RTX 4090 sm_89, plus
    /// a k=100 shape whose last block is padding. Proved able to fail rather than
    /// trusted for passing first time:
    ///   FAULT (nibble select disabled): worst row 10, GPU -6614 vs CPU -571
    ///   FAULT (row stride 18 -> 17):    worst row 47, GPU -166775700 vs CPU -225
    ///
    /// Both halves stay asserted together. "Has a kernel" and "is admitted" are
    /// the two claims whose conflation produced #3850 — an open whitelist in
    /// front of a `from_ggml_type` returning `None` is how every F16 tensor
    /// nearly got decoded as Q4_K — so they remain separate, simultaneously
    /// checked facts rather than one implying the other.
    #[test]
    fn iq4_nl_is_admitted_because_its_kernel_was_measured() {
        use crate::cuda::types::{GemvKernel, WeightQuantType};

        assert_eq!(
            WeightQuantType::from_ggml_type(20),
            Some(WeightQuantType::IQ4NL),
            "#3869: the kernel exists, so the type must resolve"
        );
        assert_eq!(
            WeightQuantType::IQ4NL.bytes_per_block(),
            18,
            "IQ4_NL is natively 18 bytes per 32 elements"
        );
        assert_eq!(
            crate::cuda::types::BoundWeight::bind(
                0x1000,
                2560 * (9216 / 32) * 18,
                WeightQuantType::IQ4NL,
                2560,
                9216
            )
            .kernel(),
            GemvKernel::IQ4NL
        );
        assert!(
            !crate::gguf::gpu_unsupported_quant_qtype(20),
            "#3869: the kernel was measured EXACT against the CPU decoder on device \
             (iq4_nl_device_ab_tests), so IQ4_NL is GPU-eligible"
        );
    }

    /// #3884: IQ3_S is admitted BECAUSE its kernel was measured.
    ///
    /// This row said, in its own body, to flip the admission assertion and open
    /// the whitelist in the same commit once the device A/B passed. It has:
    ///
    /// ```text
    /// #3884 A/B: 48 rows, worst relative disagreement 0.000e0
    /// ```
    ///
    /// EXACT against `iq_parallel_matvec` on the same bytes, RTX 4090 sm_89, at
    /// `in_dim = 512` so the row stride spans two super-blocks. Proved able to
    /// fail first, on the three mechanisms this format actually has:
    ///   scale nibble inverted    -> row 13  GPU 9311   vs CPU 1209
    ///   9th grid bit from 2l+1   -> row 13  GPU 11085  vs CPU 1209
    ///   sign bits ignored        -> row 1   GPU -17453 vs CPU 1523
    #[test]
    fn iq3_s_is_admitted_because_its_kernel_was_measured() {
        use crate::cuda::types::{GemvKernel, WeightQuantType};

        assert_eq!(
            WeightQuantType::from_ggml_type(21),
            Some(WeightQuantType::IQ3S)
        );
        assert_eq!(WeightQuantType::IQ3S.bytes_per_superblock(), 110);
        let nsb = 2560 * (9216 / 256);
        assert_eq!(
            crate::cuda::types::BoundWeight::bind(
                0x1000,
                nsb * 110,
                WeightQuantType::IQ3S,
                2560,
                9216
            )
            .kernel(),
            GemvKernel::IQ3S
        );
        // 110 per 256 elements is unique, so unlike IQ4_NL this type IS safe in
        // the size ladder. Asserted rather than assumed.
        assert_eq!(
            WeightQuantType::from_size(nsb * 110, 2560, 9216),
            Some(WeightQuantType::IQ3S),
            "110 bytes per 256 elements collides with nothing (144 is Q4_K/Q4_0/IQ4_NL, \
             176 is Q5_K/Q5_0), so size inference may name IQ3_S"
        );
        assert!(
            !crate::gguf::gpu_unsupported_quant_qtype(21),
            "#3884: the kernel was measured EXACT against the CPU decoder on device \
             (iq4_nl_device_ab_tests), so IQ3_S is GPU-eligible"
        );
    }

    /// #3885: Q5_1 is admitted BECAUSE its kernel was measured.
    ///
    /// ```text
    /// #3885 A/B: 64 rows, worst relative disagreement 0.000e0
    /// ```
    ///
    /// EXACT against `dequantize_q5_1` + an explicit row-major dot, RTX 4090
    /// sm_89, `in_dim = 256` so each row spans 8 blocks. Proved able to fail
    /// first, on the two mechanisms that define this format plus one index:
    ///   5th bit dropped (i.e. Q4_1)  -> row 41  GPU 288   vs CPU -24
    ///   affine min m dropped         -> row 34  GPU -13.5 vs CPU -9
    ///   qh bit index missing +16     -> row 34  GPU -585  vs CPU -9
    #[test]
    fn q5_1_is_admitted_because_its_kernel_was_measured() {
        use crate::cuda::types::{GemvKernel, WeightQuantType};

        assert_eq!(
            WeightQuantType::from_ggml_type(7),
            Some(WeightQuantType::Q5_1)
        );
        assert_eq!(WeightQuantType::Q5_1.bytes_per_block(), 24);
        let nb = 2560 * (9216 / 32);
        assert_eq!(
            crate::cuda::types::BoundWeight::bind(
                0x1000,
                nb * 24,
                WeightQuantType::Q5_1,
                2560,
                9216
            )
            .kernel(),
            GemvKernel::Q5_1
        );
        // 192 bytes per 256 elements is unique (144 is Q4_K/Q4_0/IQ4_NL, 176 is
        // Q5_K/Q5_0), so size inference MAY name Q5_1 - the opposite of IQ4_NL.
        assert_eq!(
            WeightQuantType::from_size(nb * 24, 2560, 9216),
            Some(WeightQuantType::Q5_1)
        );
        assert!(
            !crate::gguf::gpu_unsupported_quant_qtype(7),
            "#3885: the kernel was measured EXACT against the CPU decoder on device \
             (iq4_nl_device_ab_tests), so Q5_1 is GPU-eligible"
        );
    }

    /// #3908: BF16 is admitted BECAUSE its kernel was measured, and it is the
    /// only type here measured BIT-EXACTLY rather than within a tolerance.
    ///
    /// ```text
    /// #3908 A/B: 64 rows, 0 ULP (bit-exact) on integer-exact data
    /// #3908 A/B: 64 rows, worst relative disagreement 1.468e-5
    /// ```
    ///
    /// RTX 4090 sm_89. bf16 decoding is `bits << 16` reinterpreted as f32, which
    /// rounds nothing, so the decode can be held to 0 ULP. The GEMV still SUMS
    /// and the GPU sums in a different order, so the exact comparison is made
    /// well-posed with integer-exact data (every partial sum exact in f32) rather
    /// than demanded of an arbitrary dot product -- which would fail a CORRECT
    /// kernel. Ordinary bf16 values are measured separately at a tolerance.
    ///
    /// Proved able to fail first, on the three mechanisms this format has:
    ///   shift 8 instead of 16       -> RED (all 3 A/B tests)
    ///   byte-swapped halfword       -> RED (all 3)
    ///   row stride k not k*2        -> RED (all 3)  the LAYOUT-001 fault
    #[test]
    fn bf16_is_admitted_because_its_kernel_was_measured() {
        use crate::cuda::types::{GemvKernel, WeightQuantType};

        assert_eq!(
            WeightQuantType::from_ggml_type(30),
            Some(WeightQuantType::BF16)
        );
        // 2 bytes per element, exactly F16's rule -- which is why size can never
        // tell them apart and BF16 stays out of `from_size`'s ladder.
        assert!(WeightQuantType::BF16.matches_size(2560 * 9216 * 2, 2560, 9216));
        assert_eq!(
            crate::cuda::types::BoundWeight::bind(
                0x1000,
                2560 * 9216 * 2,
                WeightQuantType::BF16,
                2560,
                9216
            )
            .kernel(),
            GemvKernel::BF16
        );
        assert!(
            !crate::gguf::gpu_unsupported_quant_qtype(30),
            "#3908: the kernel was measured BIT-EXACT against the CPU decoder on \
             device (iq4_nl_device_ab_tests), so BF16 is GPU-eligible"
        );
    }

    /// The Q5_1 kernel's INDEX MATH in Rust, against the verified CPU decoder.
    ///
    /// Two things this pins that a port gets wrong. The 5TH BIT: byte `j`'s low
    /// nibble takes bit `j` of `qh` and its high nibble bit `j + 16`, each worth
    /// 16 quantization levels - that bit is the entire difference between Q5_1
    /// and Q4_1. And the OFFSET: `w = q * d + m`, the only affine
    /// dequantization here, so dropping `m` yields the right spread around the
    /// wrong centre and every row is off by a constant.
    #[test]
    fn the_q5_1_thread_mapping_reproduces_the_cpu_decoder() {
        let mut block = [0u8; 24];
        let mut x: u32 = 0x0BAD_C0DE;
        for b in block.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        block[0] = 0x00;
        block[1] = 0x3e; // d = 1.5
        block[2] = 0x00;
        block[3] = 0xb4; // m = -0.25

        let expected = crate::quantize::dequantize_q5_1(&block).expect("one whole block");

        let d = f32::from(half_from_le(block[0], block[1]));
        let m = f32::from(half_from_le(block[2], block[3]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let mut got = [0f32; 32];
        for tid in 0..32usize {
            let jlow = tid & 15;
            let jhalf = tid >> 4;
            let byte = u32::from(block[8 + jlow]);
            let nib = (byte >> (4 * jhalf)) & 0xf;
            let hb = (qh >> (jlow + 16 * jhalf)) & 1;
            #[allow(clippy::cast_precision_loss)]
            let q = (nib | (hb << 4)) as f32;
            got[tid] = q * d + m;
        }

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-5,
                "element {i}: kernel mapping {g}, CPU decoder {e}"
            );
        }
        assert!(
            (0..32).any(|i| (qh >> i) & 1 != 0),
            "this fixture never sets a 5th bit, so it cannot see a 5th-bit bug"
        );
    }

    /// #3950: IQ2_XXS is admitted BECAUSE its kernel was measured.
    ///
    /// This row was `iq2_xxs_has_a_kernel_but_is_not_admitted_until_it_is_measured`,
    /// which said in its body to flip this assertion and open the whitelist in the
    /// same commit once the device A/B passed. Before that it was `(16, "IQ2_XXS")`
    /// in `the_types_found_in_the_wild_without_kernels_resolve_to_none`. Converted
    /// twice, never dropped.
    ///
    /// ```text
    /// synthetic k=512 n=64, k=300 n=16          EXACT, 0.000e0
    /// Qwen3.5-0.8B-UD-IQ2_XXS (a369165c...)       95/95 tensors, all 6 shapes:
    ///   (k,n) (512,1024) (1024,2048) (1024,3584) (3584,1024) (4096,1024) (6144,1024)
    ///   worst |err| / sum|w||x| = 2.713e-7   (bar 1e-5, NaN-prefilled, 0 unwritten)
    ///   positive control, corrupted block scales on real bytes: 4.156e3 RED
    /// ```
    ///
    /// Synthetic blocks have f16 scale 1.0 and agree exactly; real scales make
    /// the GPU warp-tree sum and the CPU sequential sum round differently, which
    /// is what the 2.7e-7 is. RTX 4090 sm_89, oracle `iq_parallel_matvec`. Proved able to fail rather than trusted for passing first time:
    ///   FAULT sign-code width 7 -> 8:   row 49 GPU 378.875 vs CPU -2.625
    ///   FAULT aux1 u16 halves swapped:  row 49 GPU 2753.375 vs CPU -2.625
    ///   FAULT row stride 66 -> 64:      row 62 GPU -10820048 vs CPU 4815.125
    ///   FAULT scale nibble dropped:     row 49 GPU 351.375 vs CPU -2.625
    ///
    /// Re-runnable: `every_iq2_xxs_tensor_in_a_real_model_agrees_on_device`
    /// (`APR_IQ_AB_MODEL=<gguf> ... -- --ignored`).
    #[test]
    fn iq2_xxs_is_admitted_because_its_kernel_was_measured() {
        use crate::cuda::types::{GemvKernel, WeightQuantType};
        assert_eq!(WeightQuantType::from_ggml_type(16), Some(WeightQuantType::IQ2XXS));
        assert_eq!(
            crate::cuda::types::BoundWeight::bind(0x1000, 66 * 4, WeightQuantType::IQ2XXS, 4, 256)
                .kernel(),
            GemvKernel::IQ2XXS,
            "binding IQ2_XXS to any other kernel decodes 66-byte blocks as another scheme"
        );
        assert!(
            !crate::gguf::gpu_unsupported_quant_qtype(16),
            "IQ2_XXS has a measured kernel (95/95 real tensors) and must be GPU-eligible"
        );
        // 66 bytes per 256 elements collides with no other format, so unlike
        // IQ4_NL (byte-identical to Q4_0) it is safely inferable from size — and
        // must actually be inferred there, at a real shape from the model.
        let (k, n) = (3584usize, 1024usize);
        assert_eq!(
            WeightQuantType::from_size(n * k.div_ceil(256) * 66, n, k),
            Some(WeightQuantType::IQ2XXS),
            "a [1024 x 3584] IQ2_XXS tensor's byte count must resolve to IQ2XXS"
        );
    }

    /// The IQ2_XXS kernel's INDEX MATH in Rust — the PTX's own fetches, lane by
    /// lane — against the verified CPU decoder on the same bytes.
    ///
    /// What it pins that a port gets wrong:
    ///   * `aux1` sits at block offset 6 + 8*ib, which is 2 mod 4. The PTX reads it
    ///     as two u16 halves; reassembling them in the wrong order swaps the sign
    ///     codes and the scale nibble and still yields plausible magnitudes.
    ///   * the grid entry is a u64 fed to two lanes of four: LOW word -> columns
    ///     j, HIGH word -> columns j+4, with sign bits j and j+4.
    ///   * the sign code for index `l` is `(aux1 >> 7l) & 127` — a 7-bit field, not
    ///     8. An 8-bit shift walks off the field after l = 0.
    ///
    /// Same limit as the rows below: this proves the mapping, not that the PTX
    /// TEXT implements it. That closes only with the device A/B.
    #[test]
    fn the_iq2_xxs_thread_mapping_reproduces_the_cpu_decoder() {
        use crate::quantize::iq2_xxs::{
            dequantize_iq2_xxs_block, IQ2_XXS_BLOCK_BYTES, IQ2_XXS_BLOCK_ELEMS,
        };
        use crate::quantize::iq_grids::{IQ2XXS_GRID, KSIGNS_IQ2XS};

        let mut block = [0u8; IQ2_XXS_BLOCK_BYTES];
        let mut x: u32 = 0x2468_ace1;
        for b in block.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0

        let mut expected = [0f32; IQ2_XXS_BLOCK_ELEMS];
        dequantize_iq2_xxs_block(&block, &mut expected);

        // ---- exactly what the PTX does, one lane at a time ----
        let d = f32::from(half_from_le(block[0], block[1]));
        let grid_u32: Vec<u32> = IQ2XXS_GRID
            .iter()
            .flat_map(|&v| [(v & 0xffff_ffff) as u32, (v >> 32) as u32])
            .collect();
        let u16_at = |o: usize| u32::from(u16::from_le_bytes([block[o], block[o + 1]]));
        let mut scales_seen = std::collections::BTreeSet::new();
        let mut got = [0f32; IQ2_XXS_BLOCK_ELEMS];
        for tid in 0..32usize {
            let ib = tid >> 2;
            let l = tid & 3;
            let sub = 2 + 8 * ib;
            let idx = usize::from(block[sub + l]);
            let aux1 = u16_at(sub + 4) | (u16_at(sub + 6) << 16);
            let scale = aux1 >> 28;
            scales_seen.insert(scale);
            #[allow(clippy::cast_precision_loss)]
            let db = d * ((0.5 + scale as f32) * 0.25);
            let signs = u32::from(KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize]);
            let (lo, hi) = (grid_u32[2 * idx], grid_u32[2 * idx + 1]);
            let col0 = 32 * ib + 8 * l;
            for j in 0..4usize {
                #[allow(clippy::cast_precision_loss)]
                let m1 = ((lo >> (8 * j)) & 0xff) as f32;
                #[allow(clippy::cast_precision_loss)]
                let m2 = ((hi >> (8 * j)) & 0xff) as f32;
                let s1 = if (signs >> j) & 1 != 0 { -1.0 } else { 1.0 };
                let s2 = if (signs >> (j + 4)) & 1 != 0 { -1.0 } else { 1.0 };
                got[col0 + j] = m1 * db * s1;
                got[col0 + j + 4] = m2 * db * s2;
            }
        }

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6,
                "element {i}: kernel mapping {g}, CPU decoder {e}"
            );
        }
        // The fixture must be able to see the bugs above, or agreement is empty.
        assert!(scales_seen.len() >= 3, "fixture exercises only scales {scales_seen:?}");
        assert!(expected.iter().any(|v| *v < 0.0), "fixture never sets a sign bit");
        assert!(expected.iter().any(|v| *v > 0.0), "fixture is all negative");
    }

    /// #3963: the IQ3_XXS kernel's INDEX MATH in Rust — the PTX's own fetches,
    /// lane by lane — against the CPU decoder (bit-exact vs gguf-py) on one block.
    ///
    /// Pins: the scale/sign word at 66 + 4*ib assembled from two u16 halves in
    /// the right order; grid indices at 2 + 8*ib + 2*l and +1 (a PAIR per lane,
    /// not IQ2_XXS's single index); the 0.5 scale factor (IQ2_XXS uses 0.25);
    /// and the 7-bit sign-code field.
    #[test]
    fn the_iq3_xxs_thread_mapping_reproduces_the_cpu_decoder() {
        use crate::quantize::iq3_xxs::{dequantize_iq3_xxs_block, IQ3_XXS_BLOCK_BYTES, IQ3_XXS_BLOCK_ELEMS};
        use crate::quantize::iq_grids::{IQ3XXS_GRID, KSIGNS_IQ2XS};

        let mut block = [0u8; IQ3_XXS_BLOCK_BYTES];
        let mut x: u32 = 0x1357_9bdf;
        for b in block.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0
        let mut expected = [0f32; IQ3_XXS_BLOCK_ELEMS];
        dequantize_iq3_xxs_block(&block, &mut expected);

        let d = f32::from(half_from_le(block[0], block[1]));
        let u16_at = |o: usize| u32::from(u16::from_le_bytes([block[o], block[o + 1]]));
        let mut scales = std::collections::BTreeSet::new();
        let mut got = [0f32; IQ3_XXS_BLOCK_ELEMS];
        for tid in 0..32usize {
            let (ib, l) = (tid >> 2, tid & 3);
            let aux = u16_at(66 + 4 * ib) | (u16_at(68 + 4 * ib) << 16);
            scales.insert(aux >> 28);
            #[allow(clippy::cast_precision_loss)]
            let db = d * ((0.5 + (aux >> 28) as f32) * 0.5);
            let signs = u32::from(KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize]);
            let q = 2 + 8 * ib + 2 * l;
            let (g1, g2) = (IQ3XXS_GRID[usize::from(block[q])], IQ3XXS_GRID[usize::from(block[q + 1])]);
            let col0 = 32 * ib + 8 * l;
            for j in 0..4usize {
                #[allow(clippy::cast_precision_loss)]
                let m1 = ((g1 >> (8 * j)) & 0xff) as f32;
                #[allow(clippy::cast_precision_loss)]
                let m2 = ((g2 >> (8 * j)) & 0xff) as f32;
                let s1 = if (signs >> j) & 1 != 0 { -1.0 } else { 1.0 };
                let s2 = if (signs >> (j + 4)) & 1 != 0 { -1.0 } else { 1.0 };
                got[col0 + j] = m1 * db * s1;
                got[col0 + j + 4] = m2 * db * s2;
            }
        }
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() <= 1e-6, "element {i}: kernel mapping {g}, CPU decoder {e}");
        }
        assert!(scales.len() >= 3, "fixture exercises only scales {scales:?}");
        assert!(expected.iter().any(|v| *v < 0.0) && expected.iter().any(|v| *v > 0.0));
    }

    /// The IQ3_S kernel's INDEX MATH in Rust, against the verified CPU decoder.
    ///
    /// The mapping a port gets wrong here is the SCALE PAIRING: ggml walks
    /// `ib32` in steps of 2 with an inner `half`, indexing `scales[ib32/2]` and
    /// taking the low nibble when `half == 0`. Flattened to one `ib` per warp
    /// lane that is `scales[ib >> 1]`, low nibble when `ib & 1 == 0` — one byte
    /// serving two consecutive sub-blocks. Getting it wrong swaps the two scales
    /// within every pair and still produces plausible magnitudes.
    #[test]
    fn the_iq3_s_thread_mapping_reproduces_the_cpu_decoder() {
        use crate::quantize::iq3_s::{
            dequantize_iq3_s_block, IQ3_S_BLOCK_BYTES, IQ3_S_BLOCK_ELEMS,
        };
        use crate::quantize::iq_grids::IQ3S_GRID;

        let mut block = [0u8; IQ3_S_BLOCK_BYTES];
        let mut x: u32 = 0x1234_5678;
        for b in block.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0

        let mut expected = [0f32; IQ3_S_BLOCK_ELEMS];
        dequantize_iq3_s_block(&block, &mut expected);

        // ---- exactly what the PTX does, one lane at a time ----
        let d = f32::from(half_from_le(block[0], block[1]));
        let mut got = [0f32; IQ3_S_BLOCK_ELEMS];
        for tid in 0..32usize {
            let ib = tid >> 2;
            let l = tid & 3;
            let sc = u32::from(block[106 + (ib >> 1)]);
            let v = (sc >> (4 * (ib & 1))) & 0xf;
            #[allow(clippy::cast_precision_loss)]
            let db = d * (1.0 + 2.0 * (v as f32));
            let qh = u32::from(block[66 + ib]);
            let i1 = usize::from(block[2 + 8 * ib + 2 * l]) | (((qh >> (2 * l)) & 1) << 8) as usize;
            let i2 =
                usize::from(block[2 + 8 * ib + 2 * l + 1]) | (((qh >> (2 * l + 1)) & 1) << 8) as usize;
            let (g1, g2) = (IQ3S_GRID[i1], IQ3S_GRID[i2]);
            let sb = u32::from(block[74 + 4 * ib + l]);
            let col0 = 32 * ib + 8 * l;
            for j in 0..4usize {
                #[allow(clippy::cast_precision_loss)]
                let m1 = ((g1 >> (8 * j)) & 0xff) as f32;
                #[allow(clippy::cast_precision_loss)]
                let m2 = ((g2 >> (8 * j)) & 0xff) as f32;
                let s1 = if (sb >> j) & 1 != 0 { -1.0 } else { 1.0 };
                let s2 = if (sb >> (j + 4)) & 1 != 0 { -1.0 } else { 1.0 };
                got[col0 + j] = db * m1 * s1;
                got[col0 + j + 4] = db * m2 * s2;
            }
        }

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6,
                "element {i}: kernel mapping {g}, CPU decoder {e}"
            );
        }
    }

    /// The IQ4_NL kernel's INDEX MATH, executed in Rust exactly as the PTX
    /// executes it, checked against the verified CPU decoder on the same bytes.
    ///
    /// Same contract as the IQ4_XS row below, and the same limit: this proves
    /// the thread mapping, not that the PTX TEXT implements it. That gap closes
    /// only with a device A/B.
    ///
    /// The mapping it pins is the one most likely to be wrong, because it is the
    /// one the format makes counter-intuitive: byte `j`'s two nibbles land at
    /// elements `j` and `j + 16`, NOT at `2j` and `2j + 1`. Reading them
    /// adjacently transposes every block's halves and still yields plausible
    /// magnitudes, so nothing downstream would look obviously broken.
    #[test]
    fn the_iq4_nl_thread_mapping_reproduces_the_cpu_decoder() {
        use crate::quantize::iq4_nl::{
            dequantize_iq4_nl_block, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS,
        };

        // KVALUES_IQ4NL, the 16 non-linear levels, as the PTX embeds them.
        const KV: [i32; 16] =
            [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

        let mut block = [0u8; IQ4_NL_BLOCK_BYTES];
        let mut x: u32 = 0x1234_5678;
        for b in block.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0

        let mut expected = [0f32; IQ4_NL_BLOCK_ELEMS];
        dequantize_iq4_nl_block(&block, &mut expected);

        // ---- exactly what the PTX does, one lane at a time ----
        // jlow = tid & 15 (byte index), jhalf = tid >> 4 (nibble select),
        // element index = tid. There is no sub-block loop and no scale
        // reassembly: one f16 d covers all 32 elements.
        let d = f32::from(half_from_le(block[0], block[1]));
        let mut got = [0f32; IQ4_NL_BLOCK_ELEMS];
        for tid in 0..IQ4_NL_BLOCK_ELEMS {
            let jhalf = tid >> 4;
            let jlow = tid & 15;
            let byte = u32::from(block[2 + jlow]);
            let nib = (byte >> (jhalf * 4)) & 15;
            #[allow(clippy::cast_precision_loss)]
            let w = d * (KV[nib as usize] as f32);
            got[tid] = w;
        }

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6,
                "element {i}: kernel mapping {g}, CPU decoder {e}"
            );
        }

        // State the counter-intuitive half as its own assertion, so a failure
        // names the mechanism rather than an index.
        assert_eq!(
            got[0], expected[0],
            "byte 0's LOW nibble is element 0"
        );
        assert_eq!(
            got[16], expected[16],
            "byte 0's HIGH nibble is element 16, not element 1"
        );
    }

    /// The IQ4_XS kernel's INDEX MATH, executed in Rust exactly as the PTX
    /// executes it, checked against the verified CPU decoder on the same bytes.
    ///
    /// WHAT THIS PROVES: the thread mapping is right — that `ib == m` and
    /// `jj == tid` (which holds only because `tid < 32`), that `tid >> 4`
    /// selects the nibble and `tid & 15` the byte, that the 6-bit scale is
    /// reassembled from `scales_l`/`scales_h` correctly, and that the codebook
    /// is indexed correctly. Index math is the likely bug class in a port and
    /// it is checkable with no GPU.
    ///
    /// WHAT IT DOES NOT PROVE: that the PTX *text* implements this model. That
    /// gap closes with the tensor-level A/B on real device bytes, and nothing
    /// on a CPU-only box can close it. Recorded rather than glossed.
    #[test]
    fn the_iq4_xs_thread_mapping_reproduces_the_cpu_decoder() {
        use crate::quantize::iq4_xs::{dequantize_iq4_xs_block, IQ4_XS_BLOCK_BYTES};

        // KVALUES_IQ4NL, the 16 non-linear levels, as the PTX embeds them.
        const KV: [i32; 16] =
            [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

        // A deterministic pseudo-random block; byte 0..2 is the f16 scale, kept
        // to a cleanly representable value so the comparison is about indexing.
        let mut block = [0u8; IQ4_XS_BLOCK_BYTES];
        let mut x: u32 = 0x1234_5678;
        for b in block.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0

        let mut expected = [0f32; 256];
        dequantize_iq4_xs_block(&block, &mut expected);

        // ---- exactly what the PTX does, one lane at a time ----
        let d = f32::from(half_from_le(block[0], block[1]));
        let scales_h = u32::from(u16::from_le_bytes([block[2], block[3]]));
        let mut got = [0f32; 256];
        for tid in 0..32usize {
            let jhalf = tid >> 4;
            let jlow = tid & 15;
            for m in 0..8usize {
                let ls_low = (u32::from(block[4 + (m >> 1)]) >> (4 * (m & 1))) & 0xf;
                let ls_high = ((scales_h >> (2 * m)) & 3) << 4;
                #[allow(clippy::cast_precision_loss)]
                let dl = d * ((ls_low | ls_high) as f32 - 32.0);
                let byte = u32::from(block[8 + 16 * m + jlow]);
                let nib = (byte >> (jhalf * 4)) & 15;
                #[allow(clippy::cast_precision_loss)]
                let w = dl * KV[nib as usize] as f32;
                got[32 * m + tid] = w;
            }
        }

        let mut wrong = Vec::new();
        for i in 0..256 {
            if (got[i] - expected[i]).abs() > 1e-6 {
                wrong.push(format!("\n  - element {i}: kernel {} vs decoder {}", got[i], expected[i]));
            }
        }
        assert!(
            wrong.is_empty(),
            "{} of 256 elements disagree with dequantize_iq4_xs_block — the kernel's \
             index math does not match the reference it was ported from:{}",
            wrong.len(),
            wrong.join("")
        );
    }

    /// f16 bits -> f32, for the test's own scale. Kept local so the test does
    /// not depend on a `pub(crate)` helper's visibility.
    fn half_from_le(lo: u8, hi: u8) -> f32 {
        let bits = u16::from_le_bytes([lo, hi]);
        let sign = f32::from_bits(u32::from(bits & 0x8000) << 16);
        let exp = i32::from((bits >> 10) & 0x1f);
        let frac = f32::from(bits & 0x3ff);
        let mag = if exp == 0 {
            frac * 2f32.powi(-24)
        } else {
            (1.0 + frac / 1024.0) * 2f32.powi(exp - 15)
        };
        if sign.is_sign_negative() { -mag } else { mag }
    }

    /// IQ4_XS must be registered with the right block size. 136 bytes per 256
    /// elements: a wrong number here walks the row at the wrong stride and
    /// every block after the first is misread.
    #[test]
    fn iq4_xs_is_registered_with_the_right_block_size() {
        use crate::cuda::types::{GemvKernel, WeightQuantType};
        assert_eq!(WeightQuantType::from_ggml_type(23), Some(WeightQuantType::IQ4XS));
        assert_eq!(WeightQuantType::IQ4XS.bytes_per_superblock(), 136);
        // [2560, 9216] ffn_gate — the real shape in the UD model.
        let nsb = 2560 * (9216 / 256);
        assert!(WeightQuantType::IQ4XS.matches_size(nsb * 136, 2560, 9216));
        assert_eq!(WeightQuantType::from_size(nsb * 136, 2560, 9216), Some(WeightQuantType::IQ4XS));
        assert_eq!(
            crate::cuda::types::BoundWeight::bind(0x1000, nsb * 136, WeightQuantType::IQ4XS, 2560, 9216)
                .kernel(),
            GemvKernel::IQ4XS
        );
    }

    /// #3869: IQ4_NL MUST NOT be inferred from size, and this records why.
    ///
    /// `block_iq4_nl` and `block_q4_0` are the **identical C struct** —
    /// `{ ggml_half d; uint8_t qs[16]; }`, 18 bytes per 32 elements. They are
    /// byte-for-byte indistinguishable. Only the DECODE differs: Q4_0 is linear,
    /// `(q - 8) * d`; IQ4_NL indexes the non-linear codebook,
    /// `d * kvalues_iq4nl[q]`.
    ///
    /// Normalized to 256 elements that is 144 bytes, which is ALSO Q4_K's
    /// super-block size. So three types collide:
    ///
    /// | type | block | bytes | bytes per 256 elems |
    /// |---|---|---|---|
    /// | Q4_0   |  32 |  18 | **144** |
    /// | IQ4_NL |  32 |  18 | **144** |
    /// | Q4_K   | 256 | 144 | **144** |
    ///
    /// `from_size` already documents the Q4_0/Q4_K half of this
    /// (CORRECTNESS-002) and resolves it by trying super-block formats first.
    /// Adding IQ4_NL to that ladder would make a THIRD indistinguishable member
    /// and hand a wrong codebook to a real tensor — the #3850
    /// `resolve_qtype().unwrap_or(Q4K)` failure mode arriving by size inference
    /// instead of by fallback, and just as silent: the wrong decode of a valid
    /// block produces plausible numbers, not an error.
    ///
    /// **When the IQ4_NL GPU path lands, its type must come from the DECLARED
    /// ggml type id and never from `from_size`.** This test exists to say that
    /// where the person adding it will read it.
    #[test]
    fn iq4_nl_is_byte_identical_to_q4_0_so_size_can_never_name_it() {
        use crate::cuda::types::WeightQuantType;
        use crate::gguf::ggml_type_table;

        let q4_0 = ggml_type_table::traits(2).expect("Q4_0 is in the loader table");
        let iq4_nl = ggml_type_table::traits(20).expect("IQ4_NL is in the loader table");
        assert_eq!(
            (q4_0.blck_size, q4_0.type_size),
            (iq4_nl.blck_size, iq4_nl.type_size),
            "if these ever differ, this whole hazard is gone and the test should say so"
        );

        // A tensor that is 18 bytes per 32 elements. Size alone cannot say which
        // of the three it is.
        let (rows, cols) = (2560usize, 9216usize);
        let size = rows * (cols / 32) * 18;
        assert_eq!(size, rows * (cols / 256) * 144, "the 144-per-256 collision");

        let inferred = WeightQuantType::from_size(size, rows, cols);
        assert!(
            matches!(
                inferred,
                Some(WeightQuantType::Q4K | WeightQuantType::Q4_0)
            ),
            "size inference resolves this to Q4_K or Q4_0 today; it got {inferred:?}. It must \
             never resolve it to IQ4_NL, because nothing in the bytes distinguishes them"
        );
    }

    /// ASSEMBLE every emitted kernel with `ptxas`. This is the check the
    /// em-dash defect actually needed: `ptxas` is a compiler, it needs **no
    /// GPU**, and it is present on any box that can build `--features cuda`
    /// (nvcc is required for that). So the authoritative answer to "does this
    /// module load" is available on a CPU-only machine, in milliseconds, and
    /// neither the entry-name guard nor the ASCII guard is a substitute for it.
    ///
    /// The ASCII check above is kept because it names the exact codepoint and
    /// runs without a toolchain; this one is the ground truth.
    #[test]
    fn every_emitted_kernel_assembles() {
        use std::io::Write as _;
        let kernels = CudaKernels::new();
        let mut broken = Vec::new();

        for kt in every_emitted_gemv_kernel() {
            let ptx = kernels.generate_ptx(&kt);
            // Assemble at the target the module itself declares.
            let target = ptx
                .lines()
                .find_map(|l| l.trim().strip_prefix(".target "))
                .unwrap_or("sm_70")
                .trim()
                .to_string();

            let dir = std::env::temp_dir().join(format!("apr_ptx_{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("{}.ptx", kernels.kernel_name(&kt)));
            // A kernel that could not be written was not assembled; say so
            // rather than `continue` past it and report it as clean.
            let written = std::fs::File::create(&path).and_then(|mut f| f.write_all(ptx.as_bytes()));
            if let Err(e) = written {
                broken.push(format!("\n  - {kt:?}: could not write PTX to {} ({e})", path.display()));
                continue;
            }

            let out = std::process::Command::new("ptxas")
                .args(["--gpu-name", &target, "-o"])
                .arg(dir.join("out.cubin"))
                .arg(&path)
                .output();

            match out {
                Ok(o) if !o.status.success() => {
                    let err = String::from_utf8_lossy(&o.stderr);
                    broken.push(format!(
                        "\n  - {kt:?} at {target}: {}",
                        err.trim().lines().take(3).collect::<Vec<_>>().join(" | ")
                    ));
                },
                Ok(_) => {},
                Err(e) => {
                    // A cuda-feature build implies a CUDA toolchain, so a
                    // missing ptxas is a real finding, not a reason to skip.
                    broken.push(format!("\n  - {kt:?}: could not run ptxas ({e})"));
                },
            }
            let _ = std::fs::remove_file(&path);
        }

        assert!(
            broken.is_empty(),
            "{} emitted PTX module(s) do not assemble. ptxas is the ground truth and \
             needs no GPU — a module that fails here never loads on any device, whatever \
             its entry name says:{}",
            broken.len(),
            broken.join("")
        );
    }

    /// #3477 specifically: F16 is the new one, and its PTX must carry the
    /// row-major 2-byte stride LAYOUT-001 requires. A 4-byte stride here reads
    /// every other weight and is the silent-garbage shape this ticket exists to
    /// avoid.
    #[test]
    fn the_f16_kernel_strides_two_bytes_per_weight() {
        let kernels = CudaKernels::new();
        let ptx = kernels.generate_ptx(&KernelType::F16Gemv { k: 4096, n: 4096 });
        assert!(
            ptx.contains("cvt.f32.f16"),
            "an F16 GEMV that never converts f16 to f32 is reading raw bits as floats"
        );
        assert!(
            ptx.contains("ld.global.b16"),
            "the weight load must be 16-bit; a 32-bit load reads two weights at once"
        );
        // row_base = w_ptr + row * (k << 1): the `<< 1` IS the 2-bytes-per-f16.
        assert!(
            ptx.contains("shl.b32 %r4, %r3, 1;"),
            "the row stride must be k*2 bytes — row-major, one f16 per weight"
        );
        assert!(
            !ptx.contains("colmajor") && !ptx.contains("col_major"),
            "LAYOUT-001: GGUF/APR data is row-major on this path"
        );
    }

    /// #3908: a size can AGREE with a declaration or CONTRADICT it; it cannot
    /// out-rank one it agrees with. BF16 and F16 are both 2 bytes/element, so
    /// size-first resolution relabelled a tied BF16 LM head as F16 and the GPU
    /// decoded bf16 bytes as IEEE half (F2 gate: CPU argmax 319, GPU 14880).
    #[test]
    fn a_consistent_declaration_wins_over_the_size_guess() {
        use crate::cuda::types::WeightQuantType as W;
        let (rows, cols) = (151_936usize, 896usize); // qwen2.5-0.5b lm_head
        let two_bytes = rows * cols * 2;

        // The #3908 case: declared BF16, size agrees -> BF16, NOT F16.
        assert_eq!(
            W::resolve_declared_or_sized(Some(W::BF16), two_bytes, rows, cols),
            Some(W::BF16)
        );
        assert_eq!(
            W::resolve_declared_or_sized(Some(W::F16), two_bytes, rows, cols),
            Some(W::F16)
        );
        // PAR-058 is preserved: a declaration the size CONTRADICTS is overridden.
        // Q4_1 bytes (20/32) declared as Q4_0 (18/32) resolve to Q4_1.
        let (r, c) = (896usize, 4864usize);
        let q4_1_bytes = r * (c / 32) * 20;
        assert_eq!(
            W::resolve_declared_or_sized(Some(W::Q4_0), q4_1_bytes, r, c),
            Some(W::Q4_1)
        );
        // No declaration: the size guess is all there is, unchanged.
        assert_eq!(W::resolve_declared_or_sized(None, two_bytes, rows, cols), Some(W::F16));
        // Neither matches: fall back to the declaration rather than inventing one.
        assert_eq!(
            W::resolve_declared_or_sized(Some(W::Q4K), 12_345, rows, cols),
            Some(W::Q4K)
        );
    }
}
