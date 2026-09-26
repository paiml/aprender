//! F114 tie-break (#4215): the GPU argmax must return the FIRST maximum, the
//! contract of the CPU `ops::argmax` it replaces on the greedy decode path.
//!
//! The tree reduction compares thread `tid` against `tid + stride`, but after a
//! strided load `tid` can already hold the LARGER index (thread 0 owns
//! 0/256/512/768, thread 1 owns 1/257/…), so a strict `>` keeps index 768 over
//! an exactly equal value at index 1. The same holds across blocks in the final
//! pass once a subtree has absorbed a later block. On a temp-0 decode an exact
//! tie then emits a different token than the CPU path — a silent divergence.
//!
//! Each case below states the index the CPU contract picks; the pre-fix kernel
//! returns the other one.

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{ArgMaxFinalKernel, ArgMaxKernel, Kernel};

/// The CPU contract (`aprender-serve` `gguf::ops::argmax`): strict `>` in index
/// order, so the first maximum wins and NaN never does.
fn cpu_first_argmax(v: &[f32]) -> u32 {
    let (mut best, mut best_val) = (0u32, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i as u32;
        }
    }
    best
}

/// Both passes, exactly as `CudaExecutor::gpu_argmax` launches them. `None`
/// when this host has no CUDA device, so the caller can say it did not measure.
fn gpu_argmax(input: &[f32]) -> Option<u32> {
    let ctx = CudaContext::new(0).ok()?;
    let stream = CudaStream::new(&ctx).expect("stream");
    let len = input.len() as u32;

    let block = ArgMaxKernel::new(len);
    let num_blocks = block.num_blocks();
    let fin = ArgMaxFinalKernel::new(num_blocks);
    let mut block_mod = CudaModule::from_ptx(&ctx, &block.emit_ptx()).expect("block ptx");
    let mut final_mod = CudaModule::from_ptx(&ctx, &fin.emit_ptx()).expect("final ptx");

    let mut in_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, input.len()).expect("alloc");
    let mut vals: GpuBuffer<f32> = GpuBuffer::new(&ctx, num_blocks as usize).expect("alloc");
    let mut idxs: GpuBuffer<u32> = GpuBuffer::new(&ctx, num_blocks as usize).expect("alloc");
    let mut out: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).expect("alloc");
    in_buf.copy_from_host(input).expect("upload");

    let mut args: [*mut c_void; 4] = [
        in_buf.as_kernel_arg(),
        vals.as_kernel_arg(),
        idxs.as_kernel_arg(),
        &len as *const u32 as *mut c_void,
    ];
    let cfg = LaunchConfig {
        grid: (num_blocks, 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };
    // SAFETY: buffers sized to the kernel's immediates; args match `.param` order.
    unsafe { stream.launch_kernel(&mut block_mod, block.name(), &cfg, &mut args) }
        .expect("block launch");

    let mut fargs: [*mut c_void; 4] = [
        vals.as_kernel_arg(),
        idxs.as_kernel_arg(),
        out.as_kernel_arg(),
        &num_blocks as *const u32 as *mut c_void,
    ];
    let fcfg = LaunchConfig {
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_mem: 0,
    };
    // SAFETY: as above.
    unsafe { stream.launch_kernel(&mut final_mod, fin.name(), &fcfg, &mut fargs) }
        .expect("final launch");
    stream.synchronize().expect("sync");

    let mut host = [0u32; 1];
    out.copy_to_host(&mut host).expect("download");
    Some(host[0])
}

/// Qwen3.5's vocabulary: 243 blocks, all of the final pass's tree in play.
const VOCAB: usize = 248_320;

fn case(name: &str, input: &[f32]) {
    let want = cpu_first_argmax(input);
    let Some(got) = gpu_argmax(input) else {
        // Not a pass: nothing was measured. APR_REQUIRE_CUDA=1 (the GPU host
        // that runs this falsifier) turns the absence into a failure.
        assert!(
            std::env::var("APR_REQUIRE_CUDA").as_deref() != Ok("1"),
            "{name}: APR_REQUIRE_CUDA=1 but no CUDA device"
        );
        eprintln!("{name}: NOT MEASURED (no CUDA device)");
        return;
    };
    assert_eq!(got, want, "{name}: GPU argmax {got}, CPU first-max {want}");
}

#[test]
fn f114_tie_within_block_picks_lowest_index() {
    // Thread 0 reaches 768 (its 4th strided load); thread 1 holds 1. Final
    // intra-block step compares tid 0 (idx 768) with tid 1 (idx 1).
    let mut v = vec![-1.0f32; VOCAB];
    v[768] = 5.0;
    v[1] = 5.0;
    case("within-block", &v);
}

#[test]
fn f114_tie_across_blocks_picks_lowest_index() {
    // After the final pass's stride-128 step, tid 0 holds block 128's winner;
    // the last step compares it with tid 1 (block 1). Block 1's element is
    // the lower index.
    let mut v = vec![-1.0f32; VOCAB];
    v[128 * 1024 + 7] = 5.0;
    v[1024 + 3] = 5.0;
    case("across-blocks", &v);
}

#[test]
fn f114_tie_many_equal_maxima_picks_first() {
    // Every element equal: the first index is the only correct answer.
    case("all-equal", &vec![0.25f32; VOCAB]);
}

#[test]
fn f114_nan_never_wins() {
    // A NaN at a lower index than the real maximum must not be chosen, with or
    // without the tie-break (setp.eq and setp.gt are ordered comparisons).
    let mut v = vec![-1.0f32; VOCAB];
    v[3] = f32::NAN;
    v[1] = 2.0;
    v[900] = 2.0;
    case("nan", &v);
}

#[test]
fn f114_no_tie_is_unchanged() {
    // Regression guard: distinct values, maximum late in the vocabulary.
    let v: Vec<f32> = (0..VOCAB)
        .map(|i| ((i as u64 * 2_654_435_761) % 1_000_003) as f32 * 1e-6)
        .collect();
    case("distinct", &v);
}
