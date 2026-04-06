//! Check which SIMD path is being used
use std::arch::is_x86_feature_detected;

fn main() {
    println!("SIMD Feature Detection:");

    #[cfg(target_arch = "x86_64")]
    {
        println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        println!("  FMA: {}", is_x86_feature_detected!("fma"));
        println!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
        println!("  AVX-512BW: {}", is_x86_feature_detected!("avx512bw"));
        println!("  AVX-512VNNI: {}", is_x86_feature_detected!("avx512vnni"));
    }

    // Test which path fused_q4k_q8k_dot_simd uses
    let q4k_data = vec![0u8; 144]; // 1 super-block
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![1i8; 256];

    let result = realizar::quantize::fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    println!("\nQ4K×Q8K dot result: {:?}", result);
}
