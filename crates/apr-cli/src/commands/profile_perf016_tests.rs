// ========================================================================
// PERF-016 — a verdict may not be decided by a number nothing measured.
//
// PERF-014 deleted a hardcoded CAUSE fired on a threshold. PERF-015 fixed a
// SUBTRACTION between two incomparable measurements. This ticket audited the
// rest of `apr profile` for the same shapes and found the two meeting in one
// place: the roofline block computed an arithmetic intensity from two closed
// forms that differ by a constant factor, then selected MEMORY BOUND from a
// threshold on it and printed a paragraph telling the user where to optimise.
//
// These tests are the gate. `perf016_roofline_ai_is_a_constant_...` states the
// defect as a checkable fact about the code, and
// `perf016_no_bound_verdict_from_an_invariant_arithmetic_intensity` is the rule
// that outlives it: EITHER the arithmetic intensity varies with the workload,
// OR no memory-vs-compute verdict is issued. It never needs deleting — when
// someone gives the byte model real per-kernel counts, `ai` starts varying, the
// implication's antecedent goes false, and the verdict re-arms on its own.
//
// The third test is the DISCRIMINATION case: handed an intensity that does
// vary, `bound_verdict_from` still returns a verdict. The rule refuses the PAIR
// (invariant input + a verdict), not the concept of a verdict — a gate that
// flagged every verdict would be as useless as one that flagged none.
// ========================================================================

/// Five model shapes chosen to be as unlike each other as real models get:
/// qwen2-0.5B, qwen2-7B, llama2-7B, a small llama, and llama3-70B. Hidden dim
/// spans 9x, vocab 4.75x, layers 5x. If the arithmetic intensity depended on
/// the workload at all, it would move across these.
#[cfg(feature = "inference")]
const PERF016_SHAPES: [(usize, usize, usize); 5] = [
    (896, 151_936, 24),
    (3584, 152_064, 28),
    (4096, 32_000, 32),
    (2048, 32_000, 16),
    (8192, 128_256, 80),
];

#[cfg(feature = "inference")]
fn perf016_shape(hidden: usize, vocab: usize, layers: usize) -> RealProfileResults {
    RealProfileResults {
        hidden_dim: hidden,
        vocab_size: vocab,
        num_layers: layers,
        // A nonzero measured time so the achieved-throughput terms are finite;
        // the CPU path divides by this and never touches decode_tok_s.
        total_inference_us: 10_000.0,
        ..Default::default()
    }
}

/// THE DEFECT, as a fact rather than an opinion.
///
/// `roofline_flops_bytes` returns `32h²L + 2hV` FLOPs against `8h²L + 0.5hV`
/// bytes. Both terms of the second are the first divided by four, so the ratio
/// is 4.0 for every model that has ever existed or will. `apr profile` printed
/// it as `Arithmetic int: 4.00` next to this run's measured numbers.
#[test]
#[cfg(feature = "inference")]
fn perf016_roofline_ai_is_a_constant_of_the_closed_form_models() {
    let ais: Vec<f64> = PERF016_SHAPES
        .iter()
        .map(|&(h, v, l)| {
            let (flops, bytes) = roofline_flops_bytes(&perf016_shape(h, v, l));
            safe_ratio(flops, bytes)
        })
        .collect();
    for (ai, (h, v, l)) in ais.iter().zip(PERF016_SHAPES) {
        assert!(
            (ai - 4.0).abs() < 1e-9,
            "the byte model is the FLOP model / 4, so the arithmetic intensity is \
             4.0 for every shape; hidden={h} vocab={v} layers={l} gave {ai} \
             (all five: {ais:?})"
        );
    }
    assert!(
        !roofline_ai_varies_with_workload(),
        "roofline_ai_varies_with_workload() must agree with the five shapes above; \
         if the byte model became independent of the FLOP model, this test is the \
         signal to re-arm the memory-vs-compute verdict"
    );
}

/// THE GATE. Reverting `bound_verdict_from` to the original inline
/// `ai < ai_threshold` branch turns this RED on all five shapes.
#[test]
#[cfg(feature = "inference")]
fn perf016_no_bound_verdict_from_an_invariant_arithmetic_intensity() {
    let mut ais = Vec::new();
    let mut verdicts = Vec::new();
    for &(h, v, l) in &PERF016_SHAPES {
        let analysis = compute_roofline(&perf016_shape(h, v, l));
        ais.push(analysis.arithmetic_intensity);
        verdicts.push((h, v, l, analysis.bottleneck));
    }

    let varies = ais.iter().any(|a| (a - ais[0]).abs() > 1e-9);
    if varies {
        // The premise no longer holds: the intensity is workload-dependent, so a
        // verdict derived from it is earned. Nothing to enforce.
        return;
    }

    for (h, v, l, verdict) in &verdicts {
        assert_eq!(
            verdict,
            ROOFLINE_AI_UNMEASURED,
            "arithmetic intensity is the constant {} across all five shapes, so the \
             verdict {verdict:?} at hidden={h} vocab={v} layers={l} was decided by a \
             number nothing measured — report the magnitude or report UNMEASURED, \
             but do not name a bound the tool did not determine",
            ais[0]
        );
    }
}

/// DISCRIMINATION — the rule refuses the PAIR, not the concept.
///
/// Given an intensity that genuinely varies with the workload, the verdict is
/// still issued in both directions. A gate that returned UNMEASURED for every
/// input would pass `perf016_no_bound_verdict_...` while destroying the feature,
/// and only this half tells the two apart.
#[test]
#[cfg(feature = "inference")]
fn perf016_a_varying_arithmetic_intensity_still_gets_a_verdict() {
    assert_eq!(bound_verdict_from(true, 4.0, 82.0), "MEMORY BOUND");
    assert_eq!(bound_verdict_from(true, 120.0, 82.0), "COMPUTE BOUND");
    assert_eq!(bound_verdict_from(true, 82.0, 82.0), "COMPUTE BOUND");
    assert_eq!(bound_verdict_from(false, 4.0, 82.0), ROOFLINE_AI_UNMEASURED);
    assert_eq!(
        bound_verdict_from(false, 120.0, 82.0),
        ROOFLINE_AI_UNMEASURED
    );
}

/// Every ridge point the tool can pick is above the constant 4.0, which is why
/// the old branch could only ever take one arm. Stated so that a future ridge
/// point below 4.0 does not quietly make the old code look defensible.
#[test]
fn perf016_every_ridge_point_is_above_the_constant_intensity() {
    for name in [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA GeForce RTX 4070",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 3080",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA H100 PCIe",
        "some unknown accelerator",
    ] {
        let (_, _, ridge) = gpu_specs_by_name(name);
        assert!(
            ridge > 4.0,
            "{name} has ridge point {ridge}; with a constant intensity of 4.0 the \
             old `ai < ai_threshold` branch is not a test, it is a constant"
        );
    }
}

/// PERF-016 companion to the PERF-014 fix: the performance grade's F arm named
/// two mechanisms — a wrong backend and a naive implementation — that nothing on
/// this path inspects. A–D describe the magnitude the grade is a threshold on;
/// F must too.
#[test]
fn perf016_perf_grade_descriptions_name_no_mechanism() {
    for grade in [
        PerfGrade::A,
        PerfGrade::B,
        PerfGrade::C,
        PerfGrade::D,
        PerfGrade::F,
    ] {
        let d = grade.description().to_lowercase();
        for mechanism in [
            "backend",
            "naive",
            "implementation",
            "kernel",
            "sync",
            "cache",
            "driver",
        ] {
            assert!(
                !d.contains(mechanism),
                "PerfGrade::{} says {:?}, which names the mechanism {mechanism:?}. \
                 The grade is a threshold on an efficiency percentage whose \
                 numerator is an analytic op count; it cannot see backends, \
                 kernels or caches. Describe the magnitude.",
                grade.label(),
                grade.description()
            );
        }
    }
    // DISCRIMINATION: the descriptions must still say something. A rule
    // satisfied by the empty string is not a rule.
    for grade in [PerfGrade::A, PerfGrade::F] {
        assert!(grade.description().len() > 8);
    }
}
