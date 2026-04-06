// ---------------------------------------------------------------------------
// Macro: exact-match golden file tests
// ---------------------------------------------------------------------------

macro_rules! golden_test {
    ($name:ident, $contract:literal, $format:literal, $expected:expr) => {
        #[test]
        fn $name() {
            let actual = run_equations($contract, $format);
            let expected = $expected;
            assert_eq!(
                actual,
                expected,
                "\n=== GOLDEN MISMATCH: {} --format {} ===\n\
                 --- expected ({} bytes) ---\n{}\n\
                 --- actual ({} bytes) ---\n{}",
                $contract,
                $format,
                expected.len(),
                expected,
                actual.len(),
                actual
            );
        }
    };
}

// ---------------------------------------------------------------------------
// ReLU: y = max(0, x) — invariant, bound, idempotency, equivalence
// ---------------------------------------------------------------------------

golden_test!(
    relu_text,
    "relu-kernel-v1.yaml",
    "text",
    include_str!("../fixtures/expected/relu-text.txt")
);
golden_test!(
    relu_latex,
    "relu-kernel-v1.yaml",
    "latex",
    include_str!("../fixtures/expected/relu-latex.txt")
);
golden_test!(
    relu_ptx,
    "relu-kernel-v1.yaml",
    "ptx",
    include_str!("../fixtures/expected/relu-ptx.txt")
);
golden_test!(
    relu_asm,
    "relu-kernel-v1.yaml",
    "asm",
    include_str!("../fixtures/expected/relu-asm.txt")
);

// ---------------------------------------------------------------------------
// Clamp: y = clamp(x, lo, hi) — bound, monotonicity, idempotency, equiv
// ---------------------------------------------------------------------------

golden_test!(
    clamp_text,
    "clamp-kernel-v1.yaml",
    "text",
    include_str!("../fixtures/expected/clamp-text.txt")
);
golden_test!(
    clamp_latex,
    "clamp-kernel-v1.yaml",
    "latex",
    include_str!("../fixtures/expected/clamp-latex.txt")
);
golden_test!(
    clamp_ptx,
    "clamp-kernel-v1.yaml",
    "ptx",
    include_str!("../fixtures/expected/clamp-ptx.txt")
);
golden_test!(
    clamp_asm,
    "clamp-kernel-v1.yaml",
    "asm",
    include_str!("../fixtures/expected/clamp-asm.txt")
);

// ---------------------------------------------------------------------------
// Dot product: y = Σ x_i · w_i — linearity, invariant, equivalence
// ---------------------------------------------------------------------------

golden_test!(
    dot_text,
    "dot-kernel-v1.yaml",
    "text",
    include_str!("../fixtures/expected/dot-text.txt")
);
golden_test!(
    dot_latex,
    "dot-kernel-v1.yaml",
    "latex",
    include_str!("../fixtures/expected/dot-latex.txt")
);
golden_test!(
    dot_ptx,
    "dot-kernel-v1.yaml",
    "ptx",
    include_str!("../fixtures/expected/dot-ptx.txt")
);
golden_test!(
    dot_asm,
    "dot-kernel-v1.yaml",
    "asm",
    include_str!("../fixtures/expected/dot-asm.txt")
);

// ---------------------------------------------------------------------------
// Scale: y = α·x + β — invariant, equivalence (single phase)
// ---------------------------------------------------------------------------

golden_test!(
    scale_text,
    "scale-kernel-v1.yaml",
    "text",
    include_str!("../fixtures/expected/scale-text.txt")
);
golden_test!(
    scale_latex,
    "scale-kernel-v1.yaml",
    "latex",
    include_str!("../fixtures/expected/scale-latex.txt")
);
golden_test!(
    scale_ptx,
    "scale-kernel-v1.yaml",
    "ptx",
    include_str!("../fixtures/expected/scale-ptx.txt")
);
golden_test!(
    scale_asm,
    "scale-kernel-v1.yaml",
    "asm",
    include_str!("../fixtures/expected/scale-asm.txt")
);

// ---------------------------------------------------------------------------
// L2 norm: ||x|| = sqrt(Σ x_i²) — bound, invariant×2, equivalence
// ---------------------------------------------------------------------------

golden_test!(
    l2norm_text,
    "l2norm-kernel-v1.yaml",
    "text",
    include_str!("../fixtures/expected/l2norm-text.txt")
);
golden_test!(
    l2norm_latex,
    "l2norm-kernel-v1.yaml",
    "latex",
    include_str!("../fixtures/expected/l2norm-latex.txt")
);
golden_test!(
    l2norm_ptx,
    "l2norm-kernel-v1.yaml",
    "ptx",
    include_str!("../fixtures/expected/l2norm-ptx.txt")
);
golden_test!(
    l2norm_asm,
    "l2norm-kernel-v1.yaml",
    "asm",
    include_str!("../fixtures/expected/l2norm-asm.txt")
);
