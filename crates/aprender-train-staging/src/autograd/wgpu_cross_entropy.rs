//! WgslCrossEntropy — fused cross-entropy loss on GPU via wgpu (§26 Step 0d.4)
//!
//! Computes causal LM loss without materializing the full softmax tensor.
//! Forward: loss = -logits[label] + logsumexp(logits) per token
//! Backward: grad = softmax(logits) - one_hot(label), written IN-PLACE into logits
//!
//! Memory savings: only [seq_len] logsumexp scalars saved, not [seq_len × vocab] softmax.
//! For vocab=32000, seq=512: saves ~62 MB per forward pass.
//!
//! Contract: fused-cross-entropy-v1
//! Zero unsafe, zero FFI.

#[cfg(feature = "gpu")]
use trueno::backends::gpu::shaders::backward::{
    CROSS_ENTROPY_BACKWARD_SHADER, CROSS_ENTROPY_FORWARD_SHADER,
};
#[cfg(feature = "gpu")]
use trueno::backends::gpu::wgpu;

/// Fused cross-entropy loss computation on GPU.
#[cfg(feature = "gpu")]
pub struct WgslCrossEntropy {
    device: wgpu::Device,
    queue: wgpu::Queue,
    forward_pipeline: wgpu::ComputePipeline,
    backward_pipeline: wgpu::ComputePipeline,
    forward_bgl: wgpu::BindGroupLayout,
    backward_bgl: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CEForwardParams {
    seq_len: u32,
    vocab_size: u32,
    loss_start: u32,
    loss_end: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CEBackwardParams {
    seq_len: u32,
    vocab_size: u32,
    loss_start: u32,
    loss_end: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[cfg(feature = "gpu")]
impl WgslCrossEntropy {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // Forward: logits(ro), labels(ro), losses(rw), logsumexp(rw), params(uniform)
        let forward_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ce_fwd_bgl"),
            entries: &[storage_ro(0), storage_ro(1), storage_rw(2), storage_rw(3), uniform(4)],
        });
        let fwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ce_forward"),
            source: wgpu::ShaderSource::Wgsl(CROSS_ENTROPY_FORWARD_SHADER.into()),
        });
        let fwd_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ce_fwd_pl"),
            bind_group_layouts: &[&forward_bgl],
            push_constant_ranges: &[],
        });
        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ce_fwd_pipe"),
            layout: Some(&fwd_pl),
            module: &fwd_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Backward: logits(rw), labels(ro), logsumexp(ro), params(uniform)
        let backward_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ce_bwd_bgl"),
            entries: &[storage_rw(0), storage_ro(1), storage_ro(2), uniform(3)],
        });
        let bwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ce_backward"),
            source: wgpu::ShaderSource::Wgsl(CROSS_ENTROPY_BACKWARD_SHADER.into()),
        });
        let bwd_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ce_bwd_pl"),
            bind_group_layouts: &[&backward_bgl],
            push_constant_ranges: &[],
        });
        let backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ce_bwd_pipe"),
            layout: Some(&bwd_pl),
            module: &bwd_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { device, queue, forward_pipeline, backward_pipeline, forward_bgl, backward_bgl }
    }

    /// Compute forward cross-entropy loss on GPU.
    ///
    /// Returns average loss over response tokens.
    /// Saves logsumexp for backward pass.
    /// Dispatch CE forward compute — no GPU sync, no loss download.
    /// KAIZEN: the old `forward()` blocked 10s waiting for ALL prior GPU work.
    /// Call `read_loss()` later (after backward) to get the actual loss value.
    pub fn forward_async(
        &self,
        logits: &wgpu::Buffer,
        labels: &wgpu::Buffer,
        losses: &wgpu::Buffer,
        logsumexp: &wgpu::Buffer,
        seq_len: u32,
        vocab_size: u32,
        loss_start: u32,
        loss_end: u32,
    ) {
        let params = CEForwardParams { seq_len, vocab_size, loss_start, loss_end };
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.forward_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: logits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: labels.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: losses.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: logsumexp.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(seq_len, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Read back loss from GPU (blocks until all prior GPU work completes).
    /// Call this AFTER backward + LoRA updates to avoid blocking the pipeline.
    pub fn read_loss(
        &self,
        losses: &wgpu::Buffer,
        seq_len: u32,
        loss_start: u32,
        loss_end: u32,
    ) -> f32 {
        // PMAT-498 fix: (1) add buffer label (wgpu rejects unlabeled buffers on map_async),
        // (2) poll before copy to ensure CE forward kernel completed (loss buffer has data).
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ce_loss_readback"),
            size: u64::from(seq_len) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(losses, 0, &staging, 0, u64::from(seq_len) * 4);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let loss_data: &[f32] = bytemuck::cast_slice(&data);
        let num_tokens = (loss_end - loss_start) as f32;
        let avg = if num_tokens > 0.0 { loss_data.iter().sum::<f32>() / num_tokens } else { 0.0 };
        drop(data);
        staging.unmap();
        avg
    }

    /// Synchronous forward (legacy — blocks on GPU). Prefer forward_async + read_loss.
    pub fn forward(
        &self,
        logits: &wgpu::Buffer,
        labels: &wgpu::Buffer,
        losses: &wgpu::Buffer,
        logsumexp: &wgpu::Buffer,
        seq_len: u32,
        vocab_size: u32,
        loss_start: u32,
        loss_end: u32,
    ) -> f32 {
        self.forward_async(
            logits, labels, losses, logsumexp, seq_len, vocab_size, loss_start, loss_end,
        );
        self.read_loss(losses, seq_len, loss_start, loss_end)
    }

    /// Compute backward cross-entropy gradient IN-PLACE into logits buffer.
    ///
    /// After this call, logits[i] = (softmax(logits)[i] - one_hot(label)[i]) * scale
    pub fn backward(
        &self,
        logits: &wgpu::Buffer, // [seq_len, vocab_size] — overwritten with gradient
        labels: &wgpu::Buffer, // [seq_len] u32
        logsumexp: &wgpu::Buffer, // [seq_len] from forward
        seq_len: u32,
        vocab_size: u32,
        loss_start: u32,
        loss_end: u32,
    ) {
        let num_tokens = (loss_end - loss_start).max(1);
        let scale = 1.0 / num_tokens as f32;

        let params = CEBackwardParams {
            seq_len,
            vocab_size,
            loss_start,
            loss_end,
            scale,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = self.make_uniform(&params);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.backward_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: logits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: labels.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: logsumexp.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.backward_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total = seq_len * vocab_size;
            let workgroups = total.div_ceil(256);
            if workgroups <= 65535 {
                pass.dispatch_workgroups(workgroups, 1, 1);
            } else {
                // 2D dispatch for large tensors (>65535 workgroups)
                let x = 65535u32;
                let y = workgroups.div_ceil(x);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn make_uniform<T: bytemuck::Pod>(&self, data: &T) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(data));
        buf
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    /// FALSIFY-FCE-001: Fused CE matches naive cross-entropy
    #[test]
    fn test_fused_ce_matches_naive() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = match trueno::backends::gpu::runtime::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        ) {
            Ok(a) => a,
            Err(_) => return,
        };
        let (device, queue) = match trueno::backends::gpu::runtime::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default()),
        ) {
            Ok(dq) => dq,
            Err(_) => return,
        };

        let ce = WgslCrossEntropy::new(device.clone(), queue.clone());

        let seq_len = 4u32;
        let vocab = 8u32;

        // Random logits
        let logits_data: Vec<f32> =
            (0..seq_len * vocab).map(|i| ((i as f32) * 0.3).sin()).collect();
        let labels_data: Vec<u32> = vec![2, 5, 1, 7]; // target tokens

        let buf = |data: &[u8], rw: bool| -> wgpu::Buffer {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: data.len() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
                    | if rw { wgpu::BufferUsages::empty() } else { wgpu::BufferUsages::empty() },
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, data);
            buffer
        };

        let logits = buf(bytemuck::cast_slice(&logits_data), true);
        let labels = buf(bytemuck::cast_slice(&labels_data), false);
        let losses = buf(&vec![0u8; seq_len as usize * 4], true);
        let logsumexp_buf = buf(&vec![0u8; seq_len as usize * 4], true);

        // All tokens are response (loss_start=0, loss_end=4)
        let gpu_loss =
            ce.forward(&logits, &labels, &losses, &logsumexp_buf, seq_len, vocab, 0, seq_len);

        // CPU reference
        let mut cpu_loss = 0.0f32;
        for pos in 0..seq_len as usize {
            let offset = pos * vocab as usize;
            let label = labels_data[pos] as usize;
            let max_val: f32 = logits_data[offset..offset + vocab as usize]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits_data[offset..offset + vocab as usize]
                .iter()
                .map(|x| (x - max_val).exp())
                .sum();
            let lse = max_val + sum_exp.ln();
            cpu_loss += -logits_data[offset + label] + lse;
        }
        cpu_loss /= seq_len as f32;

        let err = (gpu_loss - cpu_loss).abs();
        eprintln!("[PARITY] Fused CE: gpu={gpu_loss:.6}, cpu={cpu_loss:.6}, err={err:.6}");
        assert!(err < 1e-4, "Fused CE parity failed: err={err}");
    }
}
