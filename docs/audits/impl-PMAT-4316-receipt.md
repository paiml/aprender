# Implementation receipt — PMAT-4316 (#4316, part of 0.69.4 epic #4249)

Author: claude-opus-5-5. Base: origin/main @ aa7c6ef03.

## Claim
`Qwen35CudaModel::forward_single` (crates/aprender-serve/src/gguf/cuda/forward_qwen35_cuda.rs) called
`GpuBuffer::from_host` on every token: a cuMemAlloc, a blocking HtoD copy and a cuMemFree per token, with
a device pointer that changed each time. No captured CUDA graph can replay that (#4215).

## Change
- `CudaExecutor::upload_into_async` (executor_api.rs): an `unsafe fn` that queues `copy_from_host_async`
  on the execution stream (`self.stream`, the stream `sync_stream` waits on and the layer kernels run on).
  Its safety contract, that `src` stays valid until the stream syncs, is passed on to the caller.
- `Qwen35CudaModel.residual: Option<GpuBuffer<f32>>` (`[hidden_dim]`) is allocated once in `new`.
  `forward_single` validates the input, takes the buffer out of `self` (the layer calls take `&mut self`),
  runs `forward_resident`, and puts the buffer back on every return path, error paths included. It is
  re-allocated only if a panic mid-token ever lost it.
- `forward_resident` = the old body after the checks, starting with an in-stream upload of the
  embedding row into the resident buffer. The row borrows `self.model`'s embedding table (`&'a`),
  which outlives `self`, so the async copy's source memory is alive whenever the copy runs.
- `forward_hidden_deltanet_only` (a test/diagnostic helper, not the decode path) is unchanged.

## Test
- NEW `qwen35_cuda_the_residual_is_one_buffer_across_tokens`: the device pointer is identical after
  construction, after each of 6 tokens, after a refused (out-of-vocab) token, and after the token that
  follows the refusal.
- Unchanged parity oracle `qwen35_cuda_forward_single_matches_cpu_logits_end_to_end`. It catches a missed
  or stale upload, because a stale residual is the previous token's hidden state, not this token's
  embedding.

## Gates
- `cargo test --release -p aprender-serve --lib --features cuda --no-run`: rc=0.
- `cargo clippy -p aprender-serve --lib --features cuda -- -D warnings`: rc=0.
  (`--tests` clippy has 1284 pre-existing errors, all in `quantize/iq*` test tables and none in these files.)
- `cargo fmt --all -- --check`: rc=0. Roadmap guards (sorted, ids unique): PASS.
- GPU run of `qwen35_cuda*` on RTX 4090 through gpu-q, with compute-apps empty at start (the test binary built from
  this code, commit 02729ec9a; the later commits are docs only): **17 passed, 0 failed, 0 skipped**, in 24.6 s.
  e2e logits worst cosine 0.998348 / rel L-inf 6.721e-2 over 6 positions. 4B argmax end-to-end passes.
  `qwen35_cuda_the_residual_is_one_buffer_across_tokens` ok. Log: /mnt/nvme-raid0/tmp/embed-gpu/test-bba6b0924.log
