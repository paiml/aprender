//! Test Tensor Core attention with detailed error output
#[cfg(feature = "cuda")]
use realizar::cuda::CudaExecutor;

fn main() {
    #[cfg(feature = "cuda")]
    {
        let mut executor = CudaExecutor::new(0).expect("CUDA init failed");

        let seq_len = 64u32;
        let head_dim = 64u32;
        let n_heads = 1u32;
        let size = (seq_len * head_dim) as usize;

        let q: Vec<f32> = vec![0.1; size];
        let k: Vec<f32> = vec![0.1; size];
        let v: Vec<f32> = vec![0.1; size];
        let mut output = vec![0.0f32; size];

        println!(
            "Testing Tensor Core attention: seq_len={}, head_dim={}",
            seq_len, head_dim
        );

        match executor.tensor_core_attention(
            &q,
            &k,
            &v,
            &mut output,
            seq_len,
            head_dim,
            n_heads,
            false,
        ) {
            Ok(()) => {
                println!("SUCCESS! Output sample: {:?}", &output[..4]);
            },
            Err(e) => {
                eprintln!("FAILED: {:?}", e);
            },
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled, skipping test");
    }
}
