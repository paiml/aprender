
impl CudaExecutor {
    /// Get number of CUDA devices
    ///
    /// Returns 0 if CUDA is not available.
    #[must_use]
    pub fn num_devices() -> usize {
        device_count().unwrap_or(0)
    }

    /// Number of loaded kernel modules.
    #[must_use]
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Set the CUDA context as current for the calling thread
    ///
    /// CUDA contexts are thread-local. When using async/multi-threaded code
    /// (like axum/tokio), you must call `make_current()` before any CUDA
    /// operation if the operation might run on a different thread than where
    /// the executor was created.
    ///
    /// # Errors
    ///
    /// Returns error if cuCtxSetCurrent fails (e.g., invalid context).
    pub fn make_current(&self) -> Result<(), GpuError> {
        self.context.make_current()
    }

    /// realizr#203: Download hidden_buf2 contents to host.
    /// After prefill, this contains S × hidden_dim hidden states.
    pub fn download_hidden_buf2(&self, dst: &mut [f32]) -> Result<(), GpuError> {
        if let Some(ref buf) = self.workspace.hidden_buf2 {
            let copy_len = dst.len().min(buf.len());
            buf.copy_to_host(&mut dst[..copy_len])
        } else {
            Err(GpuError::InvalidLaunchConfig(
                "hidden_buf2 not initialized".to_string(),
            ))
        }
    }

    // ========================================================================
    // PAR-073: BrickProfiler API for per-brick timing
    // ========================================================================

    /// Enable per-brick profiling for real timing measurements.
    ///
    /// When enabled, each brick operation is timed individually using
    /// `std::time::Instant` with CUDA sync for accurate GPU timing.
    ///
    /// # Performance Impact
    ///
    /// Profiling adds ~1 CUDA sync per brick, which adds overhead.
    /// Use only during development/benchmarking, not production.
    pub fn enable_profiling(&mut self) {
        self.profiler.enable();
    }

    /// Disable per-brick profiling (default state).
    pub fn disable_profiling(&mut self) {
        self.profiler.disable();
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_profiling_enabled(&self) -> bool {
        self.profiler.is_enabled()
    }

    /// Get the brick profiler for reading statistics.
    #[must_use]
    pub fn profiler(&self) -> &trueno::BrickProfiler {
        &self.profiler
    }

    /// Get mutable access to the brick profiler.
    pub fn profiler_mut(&mut self) -> &mut trueno::BrickProfiler {
        &mut self.profiler
    }

    /// Reset profiler statistics.
    pub fn reset_profiler(&mut self) {
        self.profiler.reset();
    }

    /// Get profiler summary report.
    #[must_use]
    pub fn profiler_summary(&self) -> String {
        self.profiler.summary()
    }

    /// Start timing a brick (internal use, legacy string API).
    ///
    /// When profiling is enabled, syncs the stream and starts a timer.
    /// Returns the timer handle for use with `stop_brick_timer()`.
    #[must_use]
    pub(crate) fn start_brick_timer(&mut self, name: &str) -> Option<trueno::BrickTimer> {
        if !self.profiler.is_enabled() {
            return None;
        }
        // Sync to ensure previous work is complete (immediate mode)
        if self.profiler.sync_mode() == trueno::SyncMode::Immediate {
            let _ = self.stream.synchronize();
        }
        Some(self.profiler.start(name))
    }

    /// Stop timing a brick and record the sample.
    ///
    /// When profiling is enabled, syncs the stream and records the elapsed time.
    pub(crate) fn stop_brick_timer(&mut self, timer: Option<trueno::BrickTimer>, elements: u64) {
        if let Some(t) = timer {
            // Sync to capture real GPU time (immediate mode)
            if self.profiler.sync_mode() == trueno::SyncMode::Immediate {
                let _ = self.stream.synchronize();
            }
            self.profiler.stop(t, elements);
        }
    }

    // ========================================================================
    // PAR-200: BrickId-based profiling (O(1) hot path)
    // ========================================================================

    /// Start timing a brick using BrickId (PAR-200 fast path).
    ///
    /// This is the preferred API for known brick types.
    #[inline]
    #[must_use]
    pub(crate) fn start_brick_id(
        &mut self,
        brick_id: trueno::BrickId,
    ) -> Option<trueno::BrickIdTimer> {
        contract_pre_report_fidelity!();
        if !self.profiler.is_enabled() {
            return None;
        }
        if self.profiler.sync_mode() == trueno::SyncMode::Immediate {
            let _ = self.stream.synchronize();
        }
        Some(self.profiler.start_brick(brick_id))
    }

    /// Stop timing a brick using BrickId (PAR-200 fast path).
    #[inline]
    pub(crate) fn stop_brick_id(&mut self, timer: Option<trueno::BrickIdTimer>, elements: u64) {
        contract_pre_report_completeness!();
        if let Some(t) = timer {
            if self.profiler.sync_mode() == trueno::SyncMode::Immediate {
                let _ = self.stream.synchronize();
            }
            self.profiler.stop_brick(t, elements);
        }
    }

    /// Record a deferred measurement (PAR-200, ~5% overhead).
    ///
    /// Use this for production profiling. Call `finalize_profiling()` after
    /// GPU sync to apply all pending measurements.
    #[inline]
    pub(crate) fn record_brick_deferred(
        &mut self,
        brick_id: trueno::BrickId,
        start_ns: u64,
        elements: u64,
    ) {
        self.profiler.record_deferred(brick_id, start_ns, elements);
    }

    /// Finalize all pending profiling measurements.
    ///
    /// Must be called after `stream.synchronize()` when using deferred mode.
    #[inline]
    pub(crate) fn finalize_profiling(&mut self) {
        if self.profiler.has_pending() {
            let end_ns = self.profiler.elapsed_ns();
            self.profiler.finalize(end_ns);
        }
    }

    /// Reset profiler epoch for deferred timing.
    #[inline]
    pub(crate) fn reset_profiler_epoch(&mut self) {
        self.profiler.reset_epoch();
    }

    /// Get profiler category breakdown (PAR-200).
    #[must_use]
    pub fn profiler_category_stats(&self) -> [trueno::CategoryStats; trueno::BrickCategory::COUNT] {
        self.profiler.category_stats()
    }

    /// Print profiler category breakdown (PAR-200).
    pub fn print_profiler_categories(&self) {
        self.profiler.print_category_stats();
    }

    /// Set profiler sync mode (PAR-200).
    ///
    /// # Modes
    /// - `Immediate`: Sync after each kernel (accurate, ~200% overhead)
    /// - `PerLayer`: Sync once per layer (~20% overhead)
    /// - `Deferred`: Sync once per forward pass (~5% overhead)
    /// - `None`: No profiling overhead
    pub fn set_profiler_sync_mode(&mut self, mode: trueno::SyncMode) {
        self.profiler.set_sync_mode(mode);
    }

    /// Get current profiler sync mode.
    #[must_use]
    pub fn profiler_sync_mode(&self) -> trueno::SyncMode {
        self.profiler.sync_mode()
    }

    // ========================================================================
    // PAR-201: Execution Graph Tracking (ASCII tree visualization)
    // ========================================================================

    /// Enable execution graph tracking for brick→kernel→PTX relationships.
    ///
    /// When enabled, each brick operation is recorded in a hierarchical graph
    /// that can be visualized as an ASCII tree for CI/CD and debugging.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// cuda_model.enable_graph_tracking();
    /// cuda_model.forward_gpu_resident(token, &mut cache, pos)?;
    /// println!("{}", cuda_model.execution_graph_ascii());
    /// // Output:
    /// // Layer 0
    /// // ├── RmsNorm  50.0µs (4096 elem)
    /// // │   └── rmsnorm_kernel  <<<16,256,1>>> smem=1024B
    /// // └── QkvProjection  200.0µs (4096 elem)
    /// ```
    pub fn enable_graph_tracking(&mut self) {
        self.profiler.enable_graph();
    }

    /// Disable execution graph tracking.
    pub fn disable_graph_tracking(&mut self) {
        self.profiler.disable_graph();
    }

    /// Check if execution graph tracking is enabled.
    #[must_use]
    pub fn is_graph_tracking_enabled(&self) -> bool {
        self.profiler.is_graph_enabled()
    }

    /// Get the execution graph (immutable).
    #[must_use]
    pub fn execution_graph(&self) -> &trueno::ExecutionGraph {
        self.profiler.execution_graph()
    }

    /// Get the execution graph as ASCII tree (headless mode for CI/CD).
    ///
    /// PAR-201: Zero-dependency tree visualization for snapshot tests, logging,
    /// and CI pipelines.
    #[must_use]
    pub fn execution_graph_ascii(&self) -> String {
        self.profiler.execution_graph().to_ascii_tree()
    }

    // ========================================================================
    // TILING-SPEC-001: Tile-Level Profiling (Phase 15)
    // ========================================================================

    /// Enable tile-level profiling for hierarchical cache analysis.
    ///
    /// When enabled, `start_tile_timer()`/`stop_tile_timer()` record per-tile
    /// statistics (GFLOP/s, arithmetic intensity, throughput) at Macro/Midi/Micro
    /// levels for identifying memory-bound vs compute-bound bottlenecks.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// cuda_model.enable_tile_profiling();
    /// cuda_model.forward_tiled_profiled(...)?;
    /// println!("{}", cuda_model.tile_summary());
    /// // Output:
    /// // === Tile Profiling Summary (TILING-SPEC-001) ===
    /// // Level       Samples   Avg µs    GFLOP/s   AI      Elements
    /// // macro            28    5280.0      0.40  0.50    1048576
    /// ```
    pub fn enable_tile_profiling(&mut self) {
        self.profiler.enable_tile_profiling();
    }

    /// Disable tile-level profiling.
    pub fn disable_tile_profiling(&mut self) {
        self.profiler.disable_tile_profiling();
    }

    /// Check if tile profiling is enabled.
    #[must_use]
    pub fn is_tile_profiling_enabled(&self) -> bool {
        self.profiler.is_tile_profiling_enabled()
    }

    /// Start timing a tile operation.
    ///
    /// # Arguments
    /// * `level` - Tile hierarchy level (Macro/Midi/Micro)
    /// * `layer_idx` - Layer index (for Macro tiles) or head index (for Midi tiles)
    /// * `op_idx` - Operation index within the layer (0=QKV, 1=Attn, 2=FFN)
    ///
    /// # Returns
    /// Timer handle to pass to `stop_tile_timer()`.
    #[must_use]
    pub(crate) fn start_tile_timer(
        &mut self,
        level: trueno::TileLevel,
        layer_idx: u32,
        op_idx: u32,
    ) -> Option<trueno::TileTimer> {
        if !self.profiler.is_tile_profiling_enabled() {
            return None;
        }
        // Sync to ensure previous work is complete
        let _ = self.stream.synchronize();
        Some(self.profiler.start_tile(level, layer_idx, op_idx))
    }

    /// Stop timing a tile operation and record statistics.
    ///
    /// # Arguments
    /// * `timer` - Timer handle from `start_tile_timer()`
    /// * `elements` - Number of elements processed (for throughput calculation)
    /// * `flops` - Number of floating-point operations (for GFLOP/s calculation)
    pub(crate) fn stop_tile_timer(
        &mut self,
        timer: Option<trueno::TileTimer>,
        elements: u64,
        flops: u64,
    ) {
        if let Some(t) = timer {
            // Sync to capture real GPU time
            let _ = self.stream.synchronize();
            self.profiler.stop_tile(t, elements, flops);
        }
    }

    /// Get tile statistics for a given level.
    #[must_use]
    pub fn tile_stats(&self, level: trueno::TileLevel) -> &trueno::TileStats {
        self.profiler.tile_stats(level)
    }

    /// Get tile profiling summary report.
    #[must_use]
    pub fn tile_summary(&self) -> String {
        self.profiler.tile_summary()
    }

    /// Get tile statistics as JSON (PMAT integration).
    #[must_use]
    pub fn tile_stats_json(&self) -> String {
        self.profiler.tile_stats_to_json()
    }

    /// Reset tile statistics.
    pub fn reset_tile_stats(&mut self) {
        self.profiler.reset_tile_stats();
    }

    /// Clear the execution graph.
    pub fn clear_execution_graph(&mut self) {
        self.profiler.execution_graph_mut().clear();
    }

    /// Get device name
    pub fn device_name(&self) -> Result<String, GpuError> {
        self.context.device_name()
    }

    /// PMAT-798: Force the high-precision (non-fused, float MWV) Q4K FFN path.
    ///
    /// The fused gate+up+SwiGLU HW DP4A kernel quantizes RMSNorm activations to
    /// Q8_1 before the integer dot product. For LLaMA-NORM-family models with
    /// massive activations (e.g. TinyLlama dim-624 ~138), this Q8 step costs
    /// enough first-token cosine (~0.972 vs the 0.98 parity gate) to force a
    /// CPU fallback. Switching to the separate float MWV path restores parity
    /// (cosine ~0.990) at a small throughput cost. The parity gate calls this
    /// as a retry before failing closed, so correct models stay on the GPU.
    ///
    /// Returns the previous `fused_gate_up` flag so the caller can detect
    /// whether anything actually changed.
    pub fn force_high_precision_ffn(&mut self) -> bool {
        let was_fused = self.gpu_profile.fused_gate_up;
        self.gpu_profile.fused_gate_up = false;
        // Keep HW DP4A for the separate GEMVs (it stays coherent); only the
        // FUSED gate+up+SwiGLU kernel is the parity-costly one. Disabling just
        // the fusion recovers cosine ~0.987 (TinyLlama) while full float MWV
        // decode is unstable under sm_121 graph capture.
        // Drop any captured decode graph so the next forward re-records the
        // kernel sequence with the new (float MWV, non-fused) FFN path. Without
        // this, a graph captured under the fused kernel would replay unchanged
        // and the precision switch would be a no-op.
        self.decode_graph = None;
        self.decode_token_count = 0;
        self.graph_capture_failed = false;
        was_fused
    }

    /// Get free and total GPU memory in bytes
    pub fn memory_info(&self) -> Result<(usize, usize), GpuError> {
        self.context.memory_info()
    }

    /// PP-LLAMA-001 §5.2: the resolved kernel configuration, for reporting.
    ///
    /// A shared accessor existed only as `executor_mut()`, so the one caller
    /// that needs to READ the profile (the effective-config endpoint, which
    /// holds a read lock on purpose) could not reach it.
    #[must_use]
    pub fn gpu_profile(&self) -> &GpuProfile {
        &self.gpu_profile
    }

    /// PP-LLAMA-001 §5.2: which CUDA graphs are enabled and which are captured.
    ///
    /// All three switches are read through the predicates that DECIDE with them
    /// (`decode_graph_enabled`, `use_graph_dispatch`, `prefill_graph_enabled`)
    /// rather than re-derived here, so the report cannot drift from the
    /// dispatch. `decode_graph_enabled` is the predicate `graphed_fast_path`
    /// itself branches on, env override and compute-capability default
    /// included — not a second reading of `CUDA_GRAPH_ENABLE`, which on an
    /// sm_89+ device would report the OPPOSITE of what runs (the env var is a
    /// force-on, and the default for cc>=89 is already on).
    #[must_use]
    pub fn graph_config(&self) -> crate::cuda::gpu_profile::GraphConfig {
        let mut batched_graph_sizes: Vec<usize> =
            self.batched_decode_graphs.keys().copied().collect();
        batched_graph_sizes.sort_unstable();
        let mut prefill_graph_sizes: Vec<usize> = self.prefill_graphs.keys().copied().collect();
        prefill_graph_sizes.sort_unstable();
        crate::cuda::gpu_profile::GraphConfig {
            cuda_graph_enable: self.decode_graph_enabled(),
            graph_dispatch: self.use_graph_dispatch(),
            prefill_graph: Self::prefill_graph_enabled(),
            decode_graph_captured: self.decode_graph.is_some(),
            batched_graph_sizes,
            batched_graph_batch_size: self.batched_graph_batch_size,
            prefill_graph_sizes,
        }
    }

    /// PP-LLAMA-001 §5.2 / §9 #7: sample VRAM now and fold it into the peak.
    ///
    /// Returns `(free, total, used_peak)`. The peak is a MEASURED high-water
    /// mark over the samples this process actually took (after preload, once
    /// per scheduler batch, and on every effective-config GET) — which is why
    /// it is reported as `used_peak_bytes` and the pool's `peak_usage` is
    /// reported separately as `recorded_alloc_peak_bytes`. The pool counts only
    /// `record_allocation` sites and is a lower bound; labelling it `vram_peak`
    /// would under-report the 9.5 GB §9 #7 asks to account for.
    pub fn sample_vram_used(&self) -> Option<(usize, usize, usize)> {
        let (free, total) = self.context.memory_info().ok()?;
        let used = total.saturating_sub(free);
        let peak = self
            .vram_used_peak
            .fetch_max(used, std::sync::atomic::Ordering::AcqRel)
            .max(used);
        Some((free, total, peak))
    }

    /// The highest `used` VRAM this process has sampled, or `None` before the
    /// first sample — never a fabricated zero.
    #[must_use]
    pub fn vram_used_peak(&self) -> Option<usize> {
        let peak = self.vram_used_peak.load(std::sync::atomic::Ordering::Acquire);
        (peak > 0).then_some(peak)
    }

    /// Get reference to CUDA context (CORRECTNESS-002: for testing Q6K kernel directly)
    #[must_use]
    pub fn context(&self) -> &CudaContext {
        &self.context
    }

    /// Get reference to compute stream (WAPR-PERF-014: reuse stream for KV scatter/attention)
    #[must_use]
    pub fn compute_stream(&self) -> &CudaStream {
        &self.compute_stream
    }

    /// Synchronize the execution stream (wait for all pending operations)
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.stream.synchronize()
    }

    /// Get memory pool statistics (IMP-900d)
    #[must_use]
    pub fn pool_stats(&self) -> PoolStats {
        self.memory_pool.stats()
    }

    /// Get staging buffer pool statistics (PARITY-042)
    #[must_use]
    pub fn staging_pool_stats(&self) -> StagingPoolStats {
        self.staging_pool.stats()
    }

    /// Get a staging buffer for pinned memory transfers (PARITY-042)
    pub fn get_staging_buffer(&mut self, size: usize) -> PinnedHostBuffer<f32> {
        self.staging_pool.get(size)
    }

    /// Return a staging buffer to the pool (PARITY-042)
    pub fn return_staging_buffer(&mut self, buf: PinnedHostBuffer<f32>) {
        self.staging_pool.put(buf);
    }

    /// Clear memory pool buffers (releases GPU memory)
    pub fn clear_pool(&mut self) {
        self.memory_pool.clear();
    }
}
