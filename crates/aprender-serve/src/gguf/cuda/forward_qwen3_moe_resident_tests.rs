//! #3714: [`Qwen3MoeCudaModel`] against the CPU forward it is specified by,
//! on the real Qwen3-Coder-30B-A3B Q4_K_M file (present on both CUDA hosts).
//!
//! The comparison runs the SAME code the runtime F2 guard runs — the probe is
//! the CPU's `forward_single_qwen3_moe_with_cache` and the GPU's
//! `forward_single`, over 64 prompt positions plus one greedy decode step — and
//! is judged by the SAME rule, `f2_multi_position_report`. A test with its own
//! looser floors would certify a GPU path the runtime then refuses.
//!
//! Skips (never fails) when the file is absent, there is no CUDA device, or the
//! device cannot hold the model right now; the last prints the arithmetic.

use super::{capacity_inputs, Qwen3MoeCudaModel, MIB};
use crate::infer::qwen3_moe_dispatch::gpu::{cpu_reference, gpu_logits, load_moe_layers};
use crate::infer::qwen3_moe_dispatch::qwen3_moe_shape;

/// The file this row is specified against; `APR_QWEN3MOE_GGUF` overrides it.
const DEFAULT_MODEL_PATH: &str = "/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf";

/// Real text long enough to tokenize past 64 positions.
const PROBE_TEXT: &str = "The history of the printing press begins in the fifteenth century, \
    when Johannes Gutenberg combined movable metal type, oil-based ink and a wooden screw press \
    into a system that could reproduce books quickly and cheaply. Within fifty years, presses \
    operated in more than two hundred cities across Europe, and the number of books in \
    circulation grew from thousands to millions, changing how ideas spread.";

/// The ≥ 64 positions #3714 done_when 2 names.
const POSITIONS: usize = 64;

/// Per-position cosine floor against the FP32-activation CPU reference, at
/// EVERY position including 0.
///
/// Derived from the pair measured on lambda (RTX 4090) on the Coder file after
/// a `<|im_start|>`: the GPU against the FP32-activation reference, 1.000000
/// at all 65 positions (known-good); the same GPU against the Q8_K-activation
/// reference, 0.985436 minimum (the known-imprecise comparator). A GPU kernel
/// defect lands far below 0.9999; the reference's own quantization would not
/// reach it either.
const EXACT_REFERENCE_COSINE_FLOOR: f32 = 0.9999;

fn model_path() -> String {
    std::env::var("APR_QWEN3MOE_GGUF").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string())
}

/// `<|im_start|>user\n` — how every chat-templated prompt `apr run` builds
/// begins (token ids measured from `apr run -v` on this file).
const CHAT_TURN_START: [u32; 3] = [151_644, 872, 198];

/// GPU forward vs CPU forward over `prefix` + the first `POSITIONS -
/// prefix.len()` tokens of [`PROBE_TEXT`], plus one greedy decode step, judged
/// by the runtime F2 rule. Every position's cosine and argmax is printed.
fn assert_parity_after(prefix: &[u32], label: &str) {
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: {path} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(&path).expect("map the GGUF");
    let model = crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).expect("CPU model");
    let shape = qwen3_moe_shape(&mapped).expect("MoE shape");
    let layers = load_moe_layers(&mapped, model.config().num_layers).expect("MoE layers");

    let text = mapped.model.encode(PROBE_TEXT).expect("tokenize");
    let mut probe = prefix.to_vec();
    probe.extend(text.iter().take(POSITIONS - prefix.len()));
    assert_eq!(
        probe.len(),
        POSITIONS,
        "the probe text must tokenize to >= {POSITIONS} tokens"
    );

    let mut gpu = match Qwen3MoeCudaModel::with_max_seq_len(
        &model,
        &layers,
        shape,
        mapped.data(),
        executor,
        POSITIONS + 2,
    ) {
        Ok(g) => g,
        Err(e @ crate::error::RealizarError::CapacityRefused(_)) => {
            eprintln!("SKIP: {e}");
            return;
        },
        Err(e) => panic!("build the CUDA model: {e}"),
    };

    let t0 = std::time::Instant::now();
    let cpu = cpu_reference(&model, &layers, shape, mapped.data(), &probe).expect("CPU reference");
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let decode_token = crate::infer::argmax_u32(&cpu[POSITIONS - 1]);
    let t1 = std::time::Instant::now();
    let got = gpu_logits(&mut gpu, &probe, decode_token).expect("GPU forward");
    let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;
    assert_eq!(got.len(), POSITIONS + 1);
    assert_eq!(cpu.len(), POSITIONS + 1);

    let mut argmax_mismatches = 0;
    for (pos, (c, g)) in cpu.iter().zip(&got).enumerate() {
        let cos = crate::infer::logits_cosine_similarity(c, g);
        let (ca, ga) = (crate::infer::argmax_u32(c), crate::infer::argmax_u32(g));
        if ca != ga {
            argmax_mismatches += 1;
        }
        eprintln!("[qwen3moe {label}] pos {pos:2}: cosine {cos:.6} argmax cpu {ca} gpu {ga}");
    }
    let report = crate::infer::f2_multi_position_report(&cpu, &got);
    eprintln!(
        "[qwen3moe {label}] {} positions: min cosine (pos >= 1) {:.6}, argmax mismatches \
         {argmax_mismatches}, CPU {cpu_ms:.0} ms, GPU {gpu_ms:.0} ms",
        cpu.len(),
        report.min_cosine_real
    );
    assert!(
        report.accepted,
        "[{label}] the runtime F2 rule rejects this GPU forward: {}",
        crate::infer::f2_divergence_msg(&report, crate::infer::F2ProbePath::Serial)
    );
    // The F2 rule is the runtime's floor; against an exact reference the GPU
    // is held to far more, at every position (position 0 included).
    assert_eq!(
        argmax_mismatches, 0,
        "[{label}] argmax must agree at every position"
    );
    for (pos, (c, g)) in cpu.iter().zip(&got).enumerate() {
        let cos = crate::infer::logits_cosine_similarity(c, g);
        assert!(
            cos >= EXACT_REFERENCE_COSINE_FLOOR,
            "[{label}] pos {pos}: cosine {cos:.6} < {EXACT_REFERENCE_COSINE_FLOOR} against the \
             FP32-activation CPU reference"
        );
    }
}

/// Plain text from position 0.
#[test]
#[serial_test::serial]
fn qwen3moe_cuda_forward_matches_cpu_at_64_positions() {
    assert_parity_after(&[], "e2e");
}

/// The production shape: a chat turn start at position 0 — the token Qwen3
/// parks its attention-sink outliers on, and the shape `apr run --gpu` hit.
#[test]
#[serial_test::serial]
fn qwen3moe_cuda_forward_matches_cpu_after_a_chat_turn_start() {
    assert_parity_after(&CHAT_TURN_START, "chat");
}

/// The 30B-A3B Q4_K_M weights as uploaded, measured on the real file:
/// experts 16788 MiB + attention 492 MiB + lm_head 243 MiB.
const WEIGHTS_30B_A3B: usize = (16788 + 492 + 243) * MIB;

/// A discrete card with too little free VRAM is refused by the shared capacity
/// plan, with the arithmetic #3714 done_when 1 asks for, and classified as a
/// co-tenant (an empty 4090 holds it); one with room fits.
#[test]
fn a_busy_discrete_card_is_refused_with_the_arithmetic() {
    use crate::capacity::{plan, CapacityVerdict, DeviceMemory, RefusalKind};
    let busy = DeviceMemory::Discrete {
        free: 16_000 * MIB as u64,
        total: 24_564 * MIB as u64,
    };
    let CapacityVerdict::Refused(r) = plan(&capacity_inputs(WEIGHTS_30B_A3B, 48, 512, 66, 0, busy))
    else {
        panic!("16000 MiB free cannot hold 17523 MiB of weights");
    };
    assert_eq!(r.kind, RefusalKind::CoTenant, "{}", r.reason);
    for part in ["weights 17523 MiB", "against 16000 MiB free of 24564 MiB"] {
        assert!(r.reason.contains(part), "missing {part:?} in: {}", r.reason);
    }
    let idle = DeviceMemory::Discrete {
        free: 23_900 * MIB as u64,
        total: 24_564 * MIB as u64,
    };
    assert!(matches!(
        plan(&capacity_inputs(WEIGHTS_30B_A3B, 48, 512, 66, 0, idle)),
        CapacityVerdict::Fits(_)
    ));
}

/// gx10, measured: cuMemGetInfo said 16056 MiB free of 122502 MiB while the
/// host had 92443 MiB available. Budgeting cuMemGetInfo refused a model the
/// GB10 holds; the shared plan budgets MemAvailable less the unified headroom.
#[test]
fn unified_memory_is_budgeted_from_mem_available_less_the_headroom() {
    use crate::capacity::{plan, CapacityVerdict, DeviceMemory};
    let gx10 = DeviceMemory::Unified {
        available: 92_443 * MIB as u64,
        total: 122_502 * MIB as u64,
        cuda_free: 16_056 * MIB as u64,
    };
    assert!(
        matches!(
            plan(&capacity_inputs(WEIGHTS_30B_A3B, 48, 512, 66, 0, gx10)),
            CapacityVerdict::Fits(_)
        ),
        "the gx10 measurement must fit"
    );
    let starved = DeviceMemory::Unified {
        available: 30_000 * MIB as u64,
        total: 122_502 * MIB as u64,
        cuda_free: 16_056 * MIB as u64,
    };
    let CapacityVerdict::Refused(r) =
        plan(&capacity_inputs(WEIGHTS_30B_A3B, 48, 512, 66, 0, starved))
    else {
        panic!("30000 MiB available less a 16384 MiB headroom cannot hold 17523 MiB");
    };
    assert!(r.reason.contains("MemAvailable 30000 MiB"), "{}", r.reason);
}

/// A wrong forward the parity checks must reject — injected into the routing
/// the GPU applies, test builds only (`Qwen3MoeCudaModel::fault`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RoutingFault {
    /// Every routed expert id shifted by one: the right weights, the wrong
    /// expert matrices.
    WrongExpert,
    /// The top-ranked expert's weight zeroed: one expert's output dropped.
    DropTopExpert,
    /// The router's weights ignored: every routed expert weighted `1/k`.
    UniformWeights,
}

/// Apply `fault` to one layer's routing.
pub(super) fn inject(
    fault: Option<RoutingFault>,
    mut routes: Vec<(usize, f32)>,
    num_experts: usize,
) -> Vec<(usize, f32)> {
    let k = routes.len().max(1) as f32;
    match fault {
        None => {},
        Some(RoutingFault::WrongExpert) => {
            for r in &mut routes {
                r.0 = (r.0 + 1) % num_experts;
            }
        },
        Some(RoutingFault::DropTopExpert) => {
            if let Some(top) = routes.first_mut() {
                top.1 = 0.0;
            }
        },
        Some(RoutingFault::UniformWeights) => {
            for r in &mut routes {
                r.1 = 1.0 / k;
            }
        },
    }
    routes
}

/// #3714 (cop condition 1): the stricter FP32 reference must still BITE. The
/// same GPU model, the same CPU reference, first clean (the control: it must
/// pass, or the rejections below prove nothing), then with each routing fault
/// injected — every fault must fail the exact-reference check. The runtime F2
/// verdict (floors 0.95 / 0.98) is printed for each, and asserted to reject the
/// two faults that change which experts or how much of them run.
#[test]
#[serial_test::serial]
fn qwen3moe_parity_rejects_injected_routing_faults() {
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: {path} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(&path).expect("map the GGUF");
    let model = crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).expect("CPU model");
    let shape = qwen3_moe_shape(&mapped).expect("MoE shape");
    let layers = load_moe_layers(&mapped, model.config().num_layers).expect("MoE layers");
    let text = mapped.model.encode(PROBE_TEXT).expect("tokenize");
    let mut probe = CHAT_TURN_START.to_vec();
    probe.extend(text.iter().take(POSITIONS - CHAT_TURN_START.len()));

    let mut gpu = match Qwen3MoeCudaModel::with_max_seq_len(
        &model,
        &layers,
        shape,
        mapped.data(),
        executor,
        POSITIONS + 2,
    ) {
        Ok(g) => g,
        Err(e @ crate::error::RealizarError::CapacityRefused(_)) => {
            eprintln!("SKIP: {e}");
            return;
        },
        Err(e) => panic!("build the CUDA model: {e}"),
    };
    let cpu = cpu_reference(&model, &layers, shape, mapped.data(), &probe).expect("CPU reference");
    let decode_token = crate::infer::argmax_u32(&cpu[POSITIONS - 1]);

    let mut verdicts = Vec::new();
    for fault in [
        None,
        Some(RoutingFault::WrongExpert),
        Some(RoutingFault::DropTopExpert),
        Some(RoutingFault::UniformWeights),
    ] {
        gpu.fault = fault;
        let got = gpu_logits(&mut gpu, &probe, decode_token).expect("GPU forward");
        let min_cos = cpu
            .iter()
            .zip(&got)
            .map(|(c, g)| crate::infer::logits_cosine_similarity(c, g))
            .fold(1.0f32, f32::min);
        let mismatches = cpu
            .iter()
            .zip(&got)
            .filter(|(c, g)| crate::infer::argmax_u32(c) != crate::infer::argmax_u32(g))
            .count();
        let exact_ok = min_cos >= EXACT_REFERENCE_COSINE_FLOOR && mismatches == 0;
        let f2 = crate::infer::f2_multi_position_report(&cpu, &got);
        eprintln!(
            "[qwen3moe fault] {fault:?}: min cosine {min_cos:.6}, argmax mismatches \
             {mismatches}/{}, exact-reference check {}, runtime F2 {}",
            cpu.len(),
            if exact_ok { "PASS" } else { "REJECT" },
            if f2.accepted { "ACCEPT" } else { "REJECT" },
        );
        verdicts.push((fault, exact_ok, f2.accepted));
    }
    gpu.fault = None;

    assert!(
        verdicts[0].1 && verdicts[0].2,
        "the control (no fault) must pass both checks, or the rejections prove nothing"
    );
    for &(fault, exact_ok, f2_ok) in &verdicts[1..] {
        assert!(
            !exact_ok,
            "{fault:?} passed the exact-reference check: it cannot fail"
        );
        if matches!(
            fault,
            Some(RoutingFault::WrongExpert | RoutingFault::UniformWeights)
        ) {
            assert!(!f2_ok, "{fault:?} passed the runtime F2 guard");
        }
    }
}
