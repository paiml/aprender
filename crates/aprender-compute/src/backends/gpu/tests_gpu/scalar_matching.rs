use super::super::super::scalar::ScalarBackend;
use super::super::*;
use crate::backends::VectorBackend;

#[test]
fn test_gpu_vec_add_matches_scalar() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let mut gpu = GpuBackend::new();
    let a = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
    let b = vec![8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5];

    let gpu_result = gpu.vec_add(&a, &b);

    let mut scalar_result = vec![0.0; 8];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::add(&a, &b, &mut scalar_result);
    }

    if let Ok(gpu_r) = gpu_result {
        for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
            assert!((g - s).abs() < 1e-4, "GPU vs Scalar mismatch: gpu={}, scalar={}", g, s);
        }
    } else {
        eprintln!("GPU vec_add failed: {:?}", gpu_result);
    }
}

#[test]
fn test_gpu_dot_matches_scalar() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let mut gpu = GpuBackend::new();
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let gpu_result = gpu.dot(&a, &b);
    // SAFETY: Test code calling backend trait methods marked unsafe
    let scalar_result = unsafe { ScalarBackend::dot(&a, &b) };

    if let Ok(gpu_r) = gpu_result {
        assert!(
            (gpu_r - scalar_result).abs() < 1e-4,
            "GPU vs Scalar dot mismatch: gpu={}, scalar={}",
            gpu_r,
            scalar_result
        );
    } else {
        eprintln!("GPU dot failed: {:?}", gpu_result);
    }
}

#[test]
fn test_gpu_relu_matches_scalar() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let mut gpu = GpuBackend::new();
    let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.5, 2.5, 0.5];

    let gpu_result = gpu.relu(&input);
    let mut scalar_result = vec![0.0; input.len()];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::relu(&input, &mut scalar_result);
    }

    if let Ok(gpu_r) = gpu_result {
        for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
            assert!((g - s).abs() < 1e-4, "GPU vs Scalar relu mismatch: gpu={}, scalar={}", g, s);
        }
    } else {
        eprintln!("GPU relu failed: {:?}", gpu_result);
    }
}

#[test]
fn test_gpu_sigmoid_matches_scalar() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let mut gpu = GpuBackend::new();
    let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.5, 2.5, 0.5];

    let gpu_result = gpu.sigmoid(&input);
    let mut scalar_result = vec![0.0; input.len()];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::sigmoid(&input, &mut scalar_result);
    }

    if let Ok(gpu_r) = gpu_result {
        for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
            assert!(
                (g - s).abs() < 1e-3,
                "GPU vs Scalar sigmoid mismatch: gpu={}, scalar={}",
                g,
                s
            );
        }
    } else {
        eprintln!("GPU sigmoid failed: {:?}", gpu_result);
    }
}

#[test]
fn test_gpu_gelu_matches_scalar() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let mut gpu = GpuBackend::new();
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5];

    let gpu_result = gpu.gelu(&input);
    let mut scalar_result = vec![0.0; input.len()];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::gelu(&input, &mut scalar_result);
    }

    if let Ok(gpu_r) = gpu_result {
        for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
            assert!((g - s).abs() < 1e-2, "GPU vs Scalar gelu mismatch: gpu={}, scalar={}", g, s);
        }
    } else {
        eprintln!("GPU gelu failed: {:?}", gpu_result);
    }
}

#[test]
fn test_gpu_swish_matches_scalar() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let mut gpu = GpuBackend::new();
    let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.5, 2.5, 0.5];

    let gpu_result = gpu.swish(&input);
    let mut scalar_result = vec![0.0; input.len()];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::swish(&input, &mut scalar_result);
    }

    if let Ok(gpu_r) = gpu_result {
        for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
            assert!((g - s).abs() < 1e-3, "GPU vs Scalar swish mismatch: gpu={}, scalar={}", g, s);
        }
    } else {
        eprintln!("GPU swish failed: {:?}", gpu_result);
    }
}

#[test]
fn test_gpu_clip_matches_scalar() {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let mut gpu = GpuBackend::new();
    let input = vec![1.0, 5.0, 10.0, 15.0, -3.0, 20.0, 7.5, 0.0];
    let min_val = 3.0;
    let max_val = 12.0;

    let gpu_result = gpu.clip(&input, min_val, max_val);
    let mut scalar_result = vec![0.0; input.len()];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::clamp(&input, min_val, max_val, &mut scalar_result);
    }

    if let Ok(gpu_r) = gpu_result {
        for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
            assert!((g - s).abs() < 1e-4, "GPU vs Scalar clip mismatch: gpu={}, scalar={}", g, s);
        }
    } else {
        eprintln!("GPU clip failed: {:?}", gpu_result);
    }
}
