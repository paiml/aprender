//! Extreme-trace poison lifecycle test.
//!
//! Five-whys diagnostic: create a poisoned CUDA context, observe every driver
//! call return value, and understand the exact error lifecycle.

#![allow(clippy::wildcard_imports)]

use super::*;

/// Helper: create executor, return Ok or the error string.
fn try_executor() -> Result<CudaExecutor, String> {
    CudaExecutor::new(0).map_err(|e| format!("{e:?}"))
}

/// Helper: check if a context handle is healthy via sync.
fn ctx_sync_status(ctx: &CudaContext) -> &'static str {
    match ctx.make_current() {
        Err(e) => {
            eprintln!("    make_current failed: {e:?}");
            "MAKE_CURRENT_FAILED"
        },
        Ok(()) => match ctx.synchronize() {
            Ok(()) => "HEALTHY",
            Err(e) => {
                eprintln!("    synchronize failed: {e:?}");
                "SYNC_FAILED"
            },
        },
    }
}

/// Inspect and log the current state of sentinel, context pool, and stream pool.
fn log_pool_state(prefix: &str) {
    {
        let sentinel = CUDA_SENTINEL.lock().unwrap();
        let has_sentinel = sentinel.is_some();
        eprintln!("  [{prefix}a] Sentinel exists: {has_sentinel}");
        if let Some(ref ctx) = *sentinel {
            eprintln!("  [{prefix}b] Sentinel status: {}", ctx_sync_status(ctx));
        }
    }
    {
        let pool = CONTEXT_POOL.lock().unwrap();
        eprintln!("  [{prefix}c] Context pool has entry: {}", pool.is_some());
    }
    {
        let pool = STREAM_POOL.lock().unwrap();
        eprintln!("  [{prefix}d] Stream pool has entry: {}", pool.is_some());
    }
    eprintln!();
}

/// Attempt a trivial GPU allocation and log the result.
fn try_gpu_alloc(ctx: &CudaContext, label: &str) {
    let buf = GpuBuffer::from_host(ctx, &[1.0f32, 2.0, 3.0]);
    eprintln!(
        "  [{label}] GpuBuffer::from_host: {}",
        if buf.is_ok() { "OK" } else { "FAILED" }
    );
}

/// Log the result of a try_executor() call.
fn log_executor_result(result: &Result<CudaExecutor, String>, label: &str, num: u32) {
    match result {
        Ok(_) => eprintln!("  [{label}] Executor #{num} created: OK"),
        Err(e) => eprintln!("  [{label}] Executor #{num} created: FAILED — {e}"),
    }
}

#[test]
#[ignore = "Diagnostic test that permanently poisons the GPU device — run manually with --ignored"]
fn test_poison_lifecycle_trace() {
    eprintln!("\n======================================================================");
    eprintln!("  POISON LIFECYCLE TRACE — Five Whys");
    eprintln!("======================================================================\n");

    // ── Phase 1: Establish healthy baseline ──────────────────────────
    eprintln!("── Phase 1: Healthy baseline ──");
    let exec = CudaExecutor::new(0).expect("Phase 1: executor must create");
    eprintln!("  [1a] Executor created OK");

    let status = ctx_sync_status(&exec.context);
    eprintln!("  [1b] Context status: {status}");
    assert_eq!(status, "HEALTHY", "Phase 1: context must be healthy");

    let stream_sync = exec.stream.synchronize();
    eprintln!("  [1c] Stream sync: {stream_sync:?}");
    assert!(stream_sync.is_ok(), "Phase 1: stream must be healthy");

    // Do a trivial GPU operation to prove the context works
    try_gpu_alloc(&exec.context, "1d");
    assert!(
        GpuBuffer::from_host(&exec.context, &[1.0f32]).is_ok(),
        "Phase 1: must be able to allocate GPU memory"
    );

    drop(exec);
    eprintln!("  [1e] Executor dropped\n");

    // ── Phase 2: Check sentinel + pool state ────────────────────────
    eprintln!("── Phase 2: Sentinel + pool state after Phase 1 ──");
    log_pool_state("2");

    // ── Phase 3: Poison the context via flash_attention kernel ──────
    eprintln!("── Phase 3: Poison via flash_attention ──");
    let mut exec2 = CudaExecutor::new(0).expect("Phase 3: executor must create");
    eprintln!("  [3a] Executor #2 created OK");
    eprintln!(
        "  [3b] Context status BEFORE kernel: {}",
        ctx_sync_status(&exec2.context)
    );

    // Launch the known-crashing flash_attention kernel
    let seq_len = 4usize;
    let head_dim = 32usize;
    let total = seq_len * head_dim;
    let q = vec![1.0f32; total];
    let k = vec![1.0f32; total];
    let v = vec![1.0f32; total];
    let mut output = vec![0.0f32; total];
    let scale = 1.0 / (head_dim as f32).sqrt();

    let result = exec2.flash_attention(
        &q,
        &k,
        &v,
        &mut output,
        seq_len as u32,
        head_dim as u32,
        scale,
        true,
    );
    eprintln!(
        "  [3c] flash_attention returned: {}",
        if result.is_ok() { "OK" } else { "ERR" }
    );
    if let Err(ref e) = result {
        eprintln!("  [3c]   error: {e:?}");
    }

    // Now check context health AFTER the kernel call
    let status_after = ctx_sync_status(&exec2.context);
    eprintln!("  [3d] Context status AFTER flash_attention: {status_after}");

    // Check stream health
    let stream_after = exec2.stream.synchronize();
    eprintln!("  [3e] Stream sync AFTER flash_attention: {stream_after:?}");

    // Check if we can still allocate GPU memory
    try_gpu_alloc(&exec2.context, "3f");

    // Sync AGAIN — does it return the same error or OK?
    let sync2 = exec2.context.synchronize();
    eprintln!("  [3g] Context sync #2 (second call): {sync2:?}");
    let sync3 = exec2.context.synchronize();
    eprintln!("  [3h] Context sync #3 (third call): {sync3:?}");

    // Drop executor — this triggers the sync-on-drop
    eprintln!("  [3i] Dropping poisoned executor...");
    drop(exec2);
    eprintln!("  [3j] Executor #2 dropped\n");

    // ── Phase 4: Post-poison sentinel state ─────────────────────────
    eprintln!("── Phase 4: Sentinel + pool state after poisoning ──");
    log_pool_state("4");

    // ── Phase 5: Attempt recovery — create new executor ─────────────
    eprintln!("── Phase 5: Recovery attempt ──");
    let exec3_result = try_executor();
    log_executor_result(&exec3_result, "5a", 3);

    if let Ok(mut exec3) = exec3_result {
        let status = ctx_sync_status(&exec3.context);
        eprintln!("  [5b] Executor #3 context status: {status}");

        let stream_ok = exec3.stream.synchronize();
        eprintln!("  [5c] Executor #3 stream sync: {stream_ok:?}");

        try_gpu_alloc(&exec3.context, "5d");

        // Try a trivial kernel (SiLU — small, simple, known to work)
        let silu_input = GpuBuffer::from_host(&exec3.context, &[0.5f32, -0.5, 1.0, -1.0]);
        match silu_input {
            Ok(buf) => {
                let silu_result = exec3.silu_gpu(&buf, 4);
                eprintln!(
                    "  [5e] SiLU kernel on executor #3: {}",
                    if silu_result.is_ok() { "OK" } else { "FAILED" }
                );
                if let Err(ref e) = silu_result {
                    eprintln!("  [5e]   error: {e:?}");
                }
            },
            Err(e) => eprintln!("  [5e] SiLU input alloc FAILED: {e:?}"),
        }

        drop(exec3);
        eprintln!("  [5f] Executor #3 dropped\n");
    }

    // ── Phase 6: Second recovery attempt ────────────────────────────
    eprintln!("── Phase 6: Second recovery attempt ──");
    let exec4_result = try_executor();
    log_executor_result(&exec4_result, "6a", 4);

    if let Ok(exec4) = exec4_result {
        let status = ctx_sync_status(&exec4.context);
        eprintln!("  [6b] Executor #4 context status: {status}");
        drop(exec4);
    }

    eprintln!("\n======================================================================");
    eprintln!("  END POISON LIFECYCLE TRACE");
    eprintln!("======================================================================\n");
}
