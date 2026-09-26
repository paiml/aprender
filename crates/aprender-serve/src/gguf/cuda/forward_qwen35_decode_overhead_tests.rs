//! #4215: the host cost of one steady-state Qwen3.5 greedy decode token.
//!
//! Decode is close to launch-bound (~634 launches/token), so the host work
//! around each launch is on the critical path. Before #4215 every token
//! allocated a device buffer for its embedding row, downloaded the whole
//! 248,320-entry logits vector (~1 MB) to run the argmax on the CPU, and
//! allocated on the host around every GDN launch (a `format!` per pointer, the
//! argument `Vec`s, a `format!` cache key per wrapper).
//!
//! These tests read three per-thread counters across steady-state tokens (after
//! warm-up has compiled every module and sized every lazy buffer): host heap
//! allocations (a counting global allocator), device allocations and
//! device→host bytes (`trueno_gpu::driver`'s `GpuBuffer` counters).

use super::Qwen35CudaModel;
use crate::gguf::forward_qwen35::Qwen35Model;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use trueno_gpu::driver::{device_allocs_total, device_to_host_bytes_total};

const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Prompt ids (all well inside the 248,320-row vocabulary).
const PROMPT: [u32; 6] = [9707, 11, 1879, 0, 3555, 374];

/// Steady-state tokens measured, after the prompt and `WARMUP` decode tokens.
const MEASURED: u64 = 4;
const WARMUP: usize = 2;

/// Counts every heap allocation the current thread makes; forwards to `System`.
struct CountingAlloc;

thread_local! {
    static HOST_ALLOCS: Cell<u64> = const { Cell::new(0) };
}

fn count_host_alloc() {
    // `try_with`: an allocation during thread-local teardown is simply not counted.
    let _ = HOST_ALLOCS.try_with(|c| c.set(c.get() + 1));
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged;
// the only addition is a thread-local counter increment, which never allocates.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_host_alloc();
        // SAFETY: forwarded verbatim; the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_host_alloc();
        // SAFETY: forwarded verbatim.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_host_alloc();
        // SAFETY: forwarded verbatim.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded verbatim.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING_ALLOC: CountingAlloc = CountingAlloc;

fn host_allocs_total() -> u64 {
    HOST_ALLOCS.with(Cell::get)
}

/// One greedy decode step: the token temperature 0 picks after `token` at `position`.
fn greedy_step(
    gpu: &mut Qwen35CudaModel<'_>,
    token: u32,
    state: &mut super::Qwen35CudaState,
    position: usize,
) -> u32 {
    gpu.forward_single_greedy(token, state, position)
        .expect("greedy forward")
}

macro_rules! model_or_skip {
    () => {{
        if !std::path::Path::new(MODEL_PATH).exists() {
            eprintln!("SKIP: {MODEL_PATH} is absent");
            return;
        }
        crate::cuda_executor_or_skip!(0)
    }};
}

/// A steady-state greedy token makes no host or device allocation and downloads
/// exactly the 4-byte token id.
#[test]
#[serial_test::serial]
fn a_steady_state_greedy_decode_token_allocates_nothing_and_downloads_only_the_token_id() {
    let executor = model_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    let mut state = gpu.new_state().expect("device state");

    let mut pos = 0;
    let mut next = 0;
    for &t in &PROMPT {
        next = greedy_step(&mut gpu, t, &mut state, pos);
        pos += 1;
    }
    for _ in 0..WARMUP {
        next = greedy_step(&mut gpu, next, &mut state, pos);
        pos += 1;
    }

    // The #3759 key proof `format!`s every cache hit in test builds only; the warmup above
    // ran it over every key this token uses. Measure the lookup a release build does.
    gpu.executor.suspend_module_key_proof(true);
    let (h0, g0, d0) = (
        host_allocs_total(),
        device_allocs_total(),
        device_to_host_bytes_total(),
    );
    for _ in 0..MEASURED {
        next = greedy_step(&mut gpu, next, &mut state, pos);
        pos += 1;
    }
    let (h1, g1, d1) = (
        host_allocs_total(),
        device_allocs_total(),
        device_to_host_bytes_total(),
    );
    gpu.executor.suspend_module_key_proof(false);
    let (host, dev, d2h) = (h1 - h0, g1 - g0, d1 - d0);
    eprintln!(
        "[4215] per steady-state token: host allocs {:.1}, device allocs {:.1}, D2H bytes {:.1}",
        host as f64 / MEASURED as f64,
        dev as f64 / MEASURED as f64,
        d2h as f64 / MEASURED as f64,
    );
    assert_eq!(
        dev, 0,
        "device allocations across {MEASURED} steady-state tokens"
    );
    assert_eq!(
        d2h,
        4 * MEASURED,
        "device→host bytes across {MEASURED} tokens: one u32 token id each"
    );
    assert_eq!(
        host, 0,
        "host heap allocations across {MEASURED} steady-state tokens"
    );
}

/// Temperature 0 on the device picks exactly the token the CPU argmax of the full
/// logits picks, token after token.
#[test]
#[serial_test::serial]
fn greedy_on_the_device_is_token_identical_to_the_cpu_argmax_of_the_logits() {
    let executor = model_or_skip!();
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_PATH).expect("map the GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::new(&qwen, executor).expect("build the CUDA model");
    let mut logits_state = gpu.new_state().expect("device state");
    let mut greedy_state = gpu.new_state().expect("device state");

    let steps = 32;
    let (mut via_logits, mut via_greedy) = (Vec::new(), Vec::new());
    let (mut a, mut b) = (PROMPT[0], PROMPT[0]);
    for pos in 0..PROMPT.len() + steps {
        let (ta, tb) = match PROMPT.get(pos) {
            Some(&t) => (t, t),
            None => (a, b),
        };
        let logits = gpu
            .forward_single(ta, &mut logits_state, pos)
            .expect("logits forward");
        a = crate::gguf::ops::argmax(&logits);
        b = greedy_step(&mut gpu, tb, &mut greedy_state, pos);
        via_logits.push(a);
        via_greedy.push(b);
    }
    eprintln!("[4215] greedy tokens: {via_greedy:?}");
    assert_eq!(
        via_greedy, via_logits,
        "device greedy diverged from the CPU argmax of the downloaded logits"
    );
}

/// Decode tokens/s after an 850-token prompt (#4215's reported number). Timing,
/// not a gate: run in release with `--ignored`; `APR_4215_MODEL` picks the GGUF.
#[test]
#[ignore = "timing: run in release with --ignored"]
#[serial_test::serial]
fn bench_greedy_decode_tok_s_at_850() {
    let path = std::env::var("APR_4215_MODEL").unwrap_or_else(|_| MODEL_PATH.to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: {path} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(&path).expect("map the GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu =
        Qwen35CudaModel::with_max_seq_len(&qwen, executor, 1024).expect("build the CUDA model");
    let mut state = gpu.new_state().expect("device state");

    const PROMPT_LEN: usize = 850;
    const DECODE: usize = 128;
    let mut next = 0;
    for pos in 0..PROMPT_LEN {
        let t = PROMPT[pos % PROMPT.len()] + (pos as u32 % 97) * 13;
        next = greedy_step(&mut gpu, t, &mut state, pos);
    }
    let t0 = std::time::Instant::now();
    for i in 0..DECODE {
        next = greedy_step(&mut gpu, next, &mut state, PROMPT_LEN + i);
    }
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[4215] {path}: decode {:.1} tok/s at {PROMPT_LEN} ({DECODE} tokens, {:.3} ms/token)",
        DECODE as f64 / secs,
        secs * 1e3 / DECODE as f64
    );
}
