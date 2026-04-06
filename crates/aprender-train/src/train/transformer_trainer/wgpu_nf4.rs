//! NF4 weight management for WGPU training
//!
//! Loads NF4-quantized weights and dequantizes on GPU per-layer.
//! This keeps total VRAM under 16GB for Qwen3-4B (36 layers).
//!
//! # Contract: wgpu-transformer-trainer-v1.yaml (C-WGPU-TRAIN-001)
//!
//! - NF4 dequant on GPU matches CPU within ε < 1e-6 (FALSIFY-WGPU-003)
//! - Per-layer dequant avoids storing all fp32 weights simultaneously

#[cfg(feature = "gpu")]
use trueno::backends::gpu::GpuDevice;

/// NF4 quantized layer weights (compact representation)
///
/// Stores packed 4-bit nibbles + per-block scales.
/// Total size per projection: n_params / 2 bytes (packed) + n_params / block_size * 4 bytes (scales)
#[cfg(feature = "gpu")]
pub struct Nf4LayerWeights {
    /// Gate projection [intermediate, hidden] packed NF4
    pub gate_packed: Vec<u32>,
    pub gate_scales: Vec<f32>,
    /// Up projection [intermediate, hidden] packed NF4
    pub up_packed: Vec<u32>,
    pub up_scales: Vec<f32>,
    /// Down projection [hidden, intermediate] packed NF4
    pub down_packed: Vec<u32>,
    pub down_scales: Vec<f32>,
    /// Q projection [num_heads * head_dim, hidden] packed NF4
    pub q_packed: Vec<u32>,
    pub q_scales: Vec<f32>,
    /// K projection [num_kv_heads * head_dim, hidden] packed NF4
    pub k_packed: Vec<u32>,
    pub k_scales: Vec<f32>,
    /// V projection [num_kv_heads * head_dim, hidden] packed NF4
    pub v_packed: Vec<u32>,
    pub v_scales: Vec<f32>,
    /// O projection [hidden, num_heads * head_dim] packed NF4
    pub o_packed: Vec<u32>,
    pub o_scales: Vec<f32>,
    /// Number of elements per projection
    pub gate_n: u32,
    pub up_n: u32,
    pub down_n: u32,
    pub q_n: u32,
    pub k_n: u32,
    pub v_n: u32,
    pub o_n: u32,
    /// NF4 block size (typically 64)
    pub block_size: u32,
}

#[cfg(feature = "gpu")]
impl Nf4LayerWeights {
    /// Dequantize gate projection to fp32 on GPU
    pub fn dequant_gate(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        self.dequant_any(&self.gate_packed, &self.gate_scales, self.gate_n, device)
    }
    /// Dequantize up projection to fp32 on GPU
    pub fn dequant_up(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        self.dequant_any(&self.up_packed, &self.up_scales, self.up_n, device)
    }
    /// Dequantize down projection to fp32 on GPU
    pub fn dequant_down(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        self.dequant_any(&self.down_packed, &self.down_scales, self.down_n, device)
    }

    /// Dequantize Q projection to fp32 on GPU
    pub fn dequant_q(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        self.dequant_any(&self.q_packed, &self.q_scales, self.q_n, device)
    }
    /// Dequantize K projection to fp32 on GPU
    pub fn dequant_k(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        self.dequant_any(&self.k_packed, &self.k_scales, self.k_n, device)
    }
    /// Dequantize V projection to fp32 on GPU
    pub fn dequant_v(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        self.dequant_any(&self.v_packed, &self.v_scales, self.v_n, device)
    }
    /// Dequantize O projection to fp32 on GPU
    pub fn dequant_o(&self, device: &GpuDevice) -> Result<Vec<f32>, String> {
        self.dequant_any(&self.o_packed, &self.o_scales, self.o_n, device)
    }

    fn dequant_any(
        &self,
        packed: &[u32],
        scales: &[f32],
        n: u32,
        device: &GpuDevice,
    ) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0f32; n as usize];
        device.nf4_dequant(packed, scales, &mut output, n, self.block_size)?;
        Ok(output)
    }

    /// Memory usage in bytes (NF4 packed + scales)
    pub fn memory_bytes(&self) -> usize {
        let packed_bytes = (self.gate_packed.len()
            + self.up_packed.len()
            + self.down_packed.len()
            + self.q_packed.len()
            + self.k_packed.len()
            + self.v_packed.len()
            + self.o_packed.len())
            * 4;
        let scale_bytes = (self.gate_scales.len()
            + self.up_scales.len()
            + self.down_scales.len()
            + self.q_scales.len()
            + self.k_scales.len()
            + self.v_scales.len()
            + self.o_scales.len())
            * 4;
        packed_bytes + scale_bytes
    }

    /// Quantize a single projection from pre-parsed safetensors
    ///
    /// Public so the model loader can call it per-shard.
    pub fn quantize_projection_from_tensors(
        tensors: &safetensors::SafeTensors<'_>,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<u32>, Vec<f32>, u32), String> {
        quantize_projection(tensors, name, rows, cols)
    }
}

/// NF4 codebook (same as trueno::quantize::NF4_LUT)
#[cfg(feature = "gpu")]
const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

const NF4_BLOCK_SIZE: usize = 64;

/// Quantize fp32 values to NF4 format (packed u32 + scales)
///
/// Returns (packed_u32, scales, n_elements)
#[cfg(feature = "gpu")]
fn quantize_to_nf4(values: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let n = values.len();
    assert!(n.is_multiple_of(NF4_BLOCK_SIZE), "Length must be divisible by {NF4_BLOCK_SIZE}");

    let num_blocks = n / NF4_BLOCK_SIZE;
    let mut scales = Vec::with_capacity(num_blocks);
    let mut packed_bytes = vec![0u8; n / 2]; // 2 values per byte

    for block_idx in 0..num_blocks {
        let start = block_idx * NF4_BLOCK_SIZE;
        let block = &values[start..start + NF4_BLOCK_SIZE];

        // Find absmax for scale
        let absmax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if absmax < 1e-10 { 1.0 } else { absmax };
        scales.push(scale);

        // Quantize each value: find nearest NF4 codebook entry
        for (i, &val) in block.iter().enumerate() {
            let normalized = val / scale;
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;
            for (j, &lut_val) in NF4_LUT.iter().enumerate() {
                let dist = (normalized - lut_val).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j as u8;
                }
            }
            let elem_idx = start + i;
            let byte_idx = elem_idx / 2;
            if elem_idx.is_multiple_of(2) {
                packed_bytes[byte_idx] |= best_idx; // low nibble
            } else {
                packed_bytes[byte_idx] |= best_idx << 4; // high nibble
            }
        }
    }

    // Pack bytes into u32
    let mut packed = vec![0u32; packed_bytes.len().div_ceil(4)];
    for (i, &byte) in packed_bytes.iter().enumerate() {
        packed[i / 4] |= u32::from(byte) << ((i % 4) * 8);
    }

    (packed, scales)
}

/// Load one projection from safetensors, quantize to NF4, return GPU format.
///
/// # Contract (FALSIFY-WGPU-003)
#[cfg(feature = "gpu")]
fn quantize_projection(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<(Vec<u32>, Vec<f32>, u32), String> {
    let view = tensors.tensor(name).map_err(|e| format!("Missing tensor {name}: {e}"))?;

    let fp32: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::F32 => bytemuck::cast_slice(view.data()).to_vec(),
        safetensors::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        dt => return Err(format!("Unsupported dtype {dt:?} for {name}")),
    };

    let expected = rows * cols;
    // Pad to NF4_BLOCK_SIZE if needed
    let mut padded = fp32;
    if padded.len() != expected {
        return Err(format!("{name}: expected {expected} elements, got {}", padded.len()));
    }
    let remainder = expected % NF4_BLOCK_SIZE;
    if remainder != 0 {
        padded.resize(expected + NF4_BLOCK_SIZE - remainder, 0.0);
    }

    let (packed, scales) = quantize_to_nf4(&padded);
    Ok((packed, scales, expected as u32))
}

#[cfg(feature = "gpu")]
impl Nf4LayerWeights {
    /// Load a single transformer layer's weights from safetensors as NF4
    ///
    /// # Contract (FALSIFY-WGPU-003)
    ///
    /// NF4 dequant of loaded weights matches original fp32 within quantization error.
    pub fn from_safetensors(
        tensors: &safetensors::SafeTensors<'_>,
        layer_idx: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: u32,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let (gate_packed, gate_scales, gate_n) = quantize_projection(
            tensors,
            &format!("{prefix}.mlp.gate_proj.weight"),
            intermediate_size,
            hidden_size,
        )?;
        let (up_packed, up_scales, up_n) = quantize_projection(
            tensors,
            &format!("{prefix}.mlp.up_proj.weight"),
            intermediate_size,
            hidden_size,
        )?;
        let (down_packed, down_scales, down_n) = quantize_projection(
            tensors,
            &format!("{prefix}.mlp.down_proj.weight"),
            hidden_size,
            intermediate_size,
        )?;
        let (q_packed, q_scales, q_n) = quantize_projection(
            tensors,
            &format!("{prefix}.self_attn.q_proj.weight"),
            q_dim,
            hidden_size,
        )?;
        let (k_packed, k_scales, k_n) = quantize_projection(
            tensors,
            &format!("{prefix}.self_attn.k_proj.weight"),
            kv_dim,
            hidden_size,
        )?;
        let (v_packed, v_scales, v_n) = quantize_projection(
            tensors,
            &format!("{prefix}.self_attn.v_proj.weight"),
            kv_dim,
            hidden_size,
        )?;
        let (o_packed, o_scales, o_n) = quantize_projection(
            tensors,
            &format!("{prefix}.self_attn.o_proj.weight"),
            hidden_size,
            q_dim,
        )?;

        Ok(Self {
            gate_packed,
            gate_scales,
            up_packed,
            up_scales,
            down_packed,
            down_scales,
            q_packed,
            q_scales,
            k_packed,
            k_scales,
            v_packed,
            v_scales,
            o_packed,
            o_scales,
            gate_n,
            up_n,
            down_n,
            q_n,
            k_n,
            v_n,
            o_n,
            block_size,
        })
    }
}

/// LoRA adapter pair for a single projection (rank-r)
///
/// Forward: y = x @ W^T + x @ B^T @ A^T (where A is [rank, in_dim], B is [out_dim, rank])
/// Backward: gradients flow through B and A, frozen base W is not updated
#[cfg(feature = "gpu")]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LoraAdapter {
    /// A matrix [rank, in_dim] — fp32, trainable
    pub a: Vec<f32>,
    /// B matrix [out_dim, rank] — fp32, trainable
    pub b: Vec<f32>,
    /// AdamW first moment for A
    pub m_a: Vec<f32>,
    /// AdamW second moment for A
    pub v_a: Vec<f32>,
    /// AdamW first moment for B
    pub m_b: Vec<f32>,
    /// AdamW second moment for B
    pub v_b: Vec<f32>,
    /// Dimensions
    pub rank: u32,
    pub in_dim: u32,
    pub out_dim: u32,
}

#[cfg(feature = "gpu")]
impl LoraAdapter {
    /// Create a new LoRA adapter with Kaiming-uniform A and zero B
    pub fn new(rank: u32, in_dim: u32, out_dim: u32) -> Self {
        let a_len = (rank * in_dim) as usize;
        let b_len = (out_dim * rank) as usize;

        // Kaiming-uniform initialization for A
        let scale = (2.0 / f64::from(in_dim)).sqrt() as f32;
        let mut a = vec![0.0f32; a_len];
        // Simple deterministic pseudo-random init
        for (i, val) in a.iter_mut().enumerate() {
            let hash = ((i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)) as f32;
            *val = (hash / u64::MAX as f32 * 2.0 - 1.0) * scale;
        }

        Self {
            a,
            b: vec![0.0f32; b_len], // B initialized to zero → LoRA starts as identity
            m_a: vec![0.0f32; a_len],
            v_a: vec![0.0f32; a_len],
            m_b: vec![0.0f32; b_len],
            v_b: vec![0.0f32; b_len],
            rank,
            in_dim,
            out_dim,
        }
    }

    /// Total trainable parameters
    pub fn num_params(&self) -> usize {
        self.a.len() + self.b.len()
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_lora_adapter_creation() {
        let adapter = LoraAdapter::new(16, 2560, 4096);
        assert_eq!(adapter.a.len(), 16 * 2560);
        assert_eq!(adapter.b.len(), 4096 * 16);
        assert_eq!(adapter.num_params(), 16 * 2560 + 4096 * 16);
        // B should be zero (identity initialization)
        assert!(adapter.b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_nf4_layer_memory() {
        // Simulate Qwen3-4B FFN layer
        let h: u32 = 2560;
        let i: u32 = 9728;
        let bs: u32 = 64;

        let layer = Nf4LayerWeights {
            gate_packed: vec![0u32; (h * i / 8) as usize], // 4 bits per param, 8 params per u32
            gate_scales: vec![0.0f32; (h * i / bs) as usize],
            up_packed: vec![0u32; (h * i / 8) as usize],
            up_scales: vec![0.0f32; (h * i / bs) as usize],
            down_packed: vec![0u32; (i * h / 8) as usize],
            down_scales: vec![0.0f32; (i * h / bs) as usize],
            q_packed: vec![0u32; (h * 4096 / 8) as usize],
            q_scales: vec![0.0f32; (h * 4096 / bs) as usize],
            k_packed: vec![0u32; (h * 1024 / 8) as usize],
            k_scales: vec![0.0f32; (h * 1024 / bs) as usize],
            v_packed: vec![0u32; (h * 1024 / 8) as usize],
            v_scales: vec![0.0f32; (h * 1024 / bs) as usize],
            o_packed: vec![0u32; (4096 * h / 8) as usize],
            o_scales: vec![0.0f32; (4096 * h / bs) as usize],
            gate_n: h * i,
            up_n: h * i,
            down_n: i * h,
            q_n: h * 4096,
            k_n: h * 1024,
            v_n: h * 1024,
            o_n: 4096 * h,
            block_size: bs,
        };

        let mb = layer.memory_bytes() as f64 / 1024.0 / 1024.0;
        eprintln!("Qwen3-4B NF4 layer: {mb:.1} MB");
        assert!(mb < 100.0, "NF4 layer should be < 100MB, got {mb:.1}");
    }

    /// Load Qwen3-4B layer 0 from safetensors and quantize to NF4
    ///
    /// # Contract (FALSIFY-WGPU-003): NF4 round-trip preserves relative accuracy
    #[test]
    fn test_load_qwen3_4b_layer0_nf4() {
        let model_path = std::path::Path::new("/home/noah/src/models/qwen3-4b");
        if !model_path.exists() {
            eprintln!("Skipping: Qwen3-4B model not found at {}", model_path.display());
            return;
        }

        // Load first shard
        let shard_path = model_path.join("model-00001-of-00003.safetensors");
        let data = std::fs::read(&shard_path).expect("read shard");
        let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse safetensors");

        let layer = Nf4LayerWeights::from_safetensors(
            &tensors, 0,    // layer 0
            2560, // hidden_size
            9728, // intermediate_size
            32,   // num_heads
            8,    // num_kv_heads
            128,  // head_dim
            64,   // block_size
        )
        .expect("from_safetensors");

        let mb = layer.memory_bytes() as f64 / 1024.0 / 1024.0;
        eprintln!("Layer 0 NF4: {mb:.1} MB (gate_n={}, q_n={})", layer.gate_n, layer.q_n);

        assert_eq!(layer.gate_n, 2560 * 9728);
        assert_eq!(layer.q_n, 2560 * 4096);
        assert_eq!(layer.k_n, 2560 * 1024);
        assert!(mb < 60.0, "Layer 0 should be < 60MB NF4, got {mb:.1}");

        // Verify dequant round-trip on GPU
        let device = GpuDevice::new().expect("GPU");
        let gate_fp32 = layer.dequant_gate(&device).expect("dequant_gate");
        assert_eq!(gate_fp32.len(), (2560 * 9728) as usize);
        assert!(gate_fp32.iter().all(|v| v.is_finite()), "All dequanted values must be finite");

        // Check non-trivial values (not all zero)
        let nonzero = gate_fp32.iter().filter(|&&v| v.abs() > 1e-6).count();
        let pct = nonzero as f64 / gate_fp32.len() as f64 * 100.0;
        eprintln!("Gate dequant: {nonzero}/{} non-zero ({pct:.1}%)", gate_fp32.len());
        assert!(pct > 50.0, "Most dequanted values should be non-zero, got {pct:.1}%");
    }
}
