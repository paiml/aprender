//! GPU execution engine for batched operations
//!
//! Contains the `execute()` and `read()` public entry points, plus sub-modules
//! for operation dispatch and shader pipeline infrastructure.
//!
//! - [`dispatch`]: Pipeline-cached shader dispatch (`encode_unary_op`, `encode_binary_op`, etc.)
//! - [`operations`]: Per-operation routing (`encode_operation`)
//!
//! # KAIZEN-022: Pipeline caching + single encoder
//!
//! All operations in a batch share a single command encoder (one GPU submission)
//! and a pipeline cache (shader compiled once, reused for all operations using
//! that shader).  For Qwen3-4B FFN: reduces 5 pipeline compilations + 5 submissions
//! per layer to 3 compilations (first layer only) + 1 submission.

pub(crate) mod dispatch;
mod operations;

use super::{BufferId, GpuCommandBatch};
use std::sync::Arc;

impl GpuCommandBatch {
    /// Execute all queued operations on GPU
    ///
    /// Uses a single command encoder for all operations (one GPU submission)
    /// and caches pipelines per shader source to avoid redundant compilation.
    /// The pipeline cache is local to this call — see `execute_with_cache()`
    /// for persistent caching across multiple batch executions.
    ///
    /// # Contract (C-BATCH-EXEC-001)
    ///
    /// - **Precondition**: Operations queued via `matmul()`, `relu()`, etc.
    /// - **Postcondition**: All operations executed, results in GPU buffers
    /// - **Invariant**: Pipeline compiled at most once per unique shader source
    /// - **Invariant**: Single `queue.submit()` per `execute()` call
    pub async fn execute(&mut self) -> Result<(), String> {
        contract_pre_single_encoder_batch!();
        let mut local_cache = dispatch::PipelineCache::new();
        let result = self.execute_inner(&mut local_cache);
        contract_post_single_encoder_batch!(result);
        result
    }

    /// Execute with a persistent pipeline cache (KAIZEN-023).
    ///
    /// Same as `execute()` but uses a caller-provided pipeline cache that
    /// persists across multiple batch executions.  Shaders compiled in a
    /// previous batch are reused without recompilation.
    ///
    /// For Qwen3-4B FFN (36 layers × 3 unique shaders per batch):
    /// - `execute()`: 3 compilations per layer × 36 = 108 total
    /// - `execute_with_cache()`: 3 compilations (layer 1) + 0 (layers 2-36) = 3 total
    pub async fn execute_with_cache(
        &mut self,
        cache: &mut dispatch::PipelineCache,
    ) -> Result<(), String> {
        self.execute_inner(cache)
    }

    /// Shared implementation for execute() and execute_with_cache().
    fn execute_inner(
        &mut self,
        pipeline_cache: &mut dispatch::PipelineCache,
    ) -> Result<(), String> {
        // Step 1: Create GPU buffers for all BufferIds
        // Skip imported buffers — already GPU-resident (KAIZEN-015)
        for (buffer_id, buffer_info) in &mut self.buffers {
            if buffer_info.gpu_buffer.is_some() {
                continue;
            }

            let size_bytes = (buffer_info.size * std::mem::size_of::<f32>()) as u64;

            let gpu_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Buffer {:?}", buffer_id)),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            buffer_info.gpu_buffer = Some(Arc::new(gpu_buffer));
        }

        // Step 2: Upload initial data to buffers that have it
        for buffer_info in self.buffers.values() {
            if let Some(data) = &buffer_info.data {
                if let Some(gpu_buffer) = &buffer_info.gpu_buffer {
                    self.device.queue.write_buffer(gpu_buffer, 0, bytemuck::cast_slice(data));
                }
            }
        }

        // Step 3: Encode all operations into a single command encoder
        // with cached pipelines (KAIZEN-022)
        let mut encoder =
            self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Encoder"),
            });

        for op in &self.operations {
            self.encode_operation(op, &mut encoder, pipeline_cache)?;
        }

        // Step 4: Single GPU submission for all operations
        self.device.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Read buffer data back from GPU
    ///
    /// Must call `execute()` first.
    pub async fn read(&self, buffer_id: BufferId) -> Result<Vec<f32>, String> {
        contract_pre_read!();
        let buffer_info = self.buffers.get(&buffer_id).ok_or("Invalid buffer ID")?;

        let gpu_buffer = buffer_info
            .gpu_buffer
            .as_ref()
            .ok_or("Buffer not executed yet - call execute() first")?;

        let size_bytes = (buffer_info.size * std::mem::size_of::<f32>()) as u64;

        // Create staging buffer for reading
        let staging_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        let mut encoder =
            self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Encoder"),
            });

        encoder.copy_buffer_to_buffer(gpu_buffer, 0, &staging_buffer, 0, size_bytes);

        self.device.queue.submit(Some(encoder.finish()));

        // Map the staging buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Drive GPU work to completion — wgpu requires explicit polling
        // for map_async callbacks to fire
        self.device
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| format!("GPU poll failed: {:?}", e))?;

        // Wait for mapping to complete
        receiver
            .receive()
            .await
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        // Read data from mapped buffer
        let data = {
            let mapped_range = buffer_slice.get_mapped_range();
            let float_data: &[f32] = bytemuck::cast_slice(&mapped_range);
            float_data.to_vec()
        };

        staging_buffer.unmap();

        Ok(data)
    }
}
