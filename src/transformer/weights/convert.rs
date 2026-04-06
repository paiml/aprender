//! Tensor format conversion from SafeTensors to f32

/// Convert SafeTensors tensor view to f32 Vec
///
/// Handles bf16, fp16, and fp32 formats.
pub(crate) fn tensor_to_f32_vec(tensor: &safetensors::tensor::TensorView<'_>) -> Option<Vec<f32>> {
    use safetensors::Dtype;

    let shape = tensor.shape();
    let numel: usize = shape.iter().product();

    if numel == 0 {
        return Some(Vec::new());
    }

    let data = tensor.data();

    match tensor.dtype() {
        Dtype::F32 => {
            // Direct f32 conversion
            let values: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Some(values)
        }
        Dtype::F16 => {
            // fp16 conversion
            let values: Vec<f32> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();
            Some(values)
        }
        Dtype::BF16 => {
            // bf16 conversion
            let values: Vec<f32> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect();
            Some(values)
        }
        Dtype::I32 => {
            // Integer to float (rare for transformer weights)
            let values: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32)
                .collect();
            Some(values)
        }
        _ => {
            // Unsupported dtype
            eprintln!("Warning: Unsupported tensor dtype {:?}, skipping", tensor.dtype());
            None
        }
    }
}
