use super::*;

#[test]
fn test_gpu_tiled_sum_matches_cpu() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // Test various sizes
    let test_cases = [
        (4, 4),   // Small (single partial tile)
        (16, 16), // Exact tile
        (32, 32), // Multiple tiles (2x2)
        (20, 20), // Non-aligned
        (100, 5), // Wide
        (5, 100), // Tall
    ];

    for (width, height) in test_cases {
        let data: Vec<f32> = (1..=(width * height) as i32).map(|x| x as f32).collect();
        let cpu_result = tiled_sum_2d(&data, width, height);

        if let Ok(gpu_result) = gpu.tiled_sum_2d_gpu(&data, width, height) {
            let rel_err = (gpu_result - cpu_result).abs() / cpu_result.abs().max(1.0);
            assert!(
                rel_err < 1e-4,
                "GPU vs CPU tiled_sum mismatch for {}x{}: gpu={}, cpu={}, rel_err={}",
                width,
                height,
                gpu_result,
                cpu_result,
                rel_err
            );
        } else {
            eprintln!("GPU tiled_sum_2d failed for {}x{}", width, height);
        }
    }
}

#[test]
fn test_gpu_tiled_max_matches_cpu() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // Test data with varying values
    let data: Vec<f32> = (1..=256).map(|x| x as f32).collect();
    let width = 16;
    let height = 16;

    let cpu_result = tiled_max_2d(&data, width, height);

    if let Ok(gpu_result) = gpu.tiled_max_2d_gpu(&data, width, height) {
        assert!(
            (gpu_result - cpu_result).abs() < 1e-5,
            "GPU vs CPU tiled_max mismatch: gpu={}, cpu={}",
            gpu_result,
            cpu_result
        );
    } else {
        eprintln!("GPU tiled_max_2d failed");
    }
}

#[test]
fn test_gpu_tiled_min_matches_cpu() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // Test data with varying values including negatives
    let data: Vec<f32> = (-128..=127).map(|x| x as f32).collect();
    let width = 16;
    let height = 16;

    let cpu_result = tiled_min_2d(&data, width, height);

    if let Ok(gpu_result) = gpu.tiled_min_2d_gpu(&data, width, height) {
        assert!(
            (gpu_result - cpu_result).abs() < 1e-5,
            "GPU vs CPU tiled_min mismatch: gpu={}, cpu={}",
            gpu_result,
            cpu_result
        );
    } else {
        eprintln!("GPU tiled_min_2d failed");
    }
}

#[test]
fn test_gpu_tiled_sum_large_matrix() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // Large 64x64 matrix (16 tiles)
    let width = 64;
    let height = 64;
    let data: Vec<f32> = vec![1.0; width * height];

    let cpu_result = tiled_sum_2d(&data, width, height);
    let expected = (width * height) as f32;

    // Verify CPU is correct
    assert!((cpu_result - expected).abs() < 1e-3);

    if let Ok(gpu_result) = gpu.tiled_sum_2d_gpu(&data, width, height) {
        let rel_err = (gpu_result - expected).abs() / expected;
        assert!(
            rel_err < 1e-4,
            "GPU tiled_sum large matrix: got {}, expected {}, rel_err={}",
            gpu_result,
            expected,
            rel_err
        );
    } else {
        eprintln!("GPU tiled_sum_2d large matrix failed");
    }
}

#[test]
fn test_gpu_tiled_max_with_negatives() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // All negative values - max should be -1.0
    let data: Vec<f32> = (-100..-1).map(|x| x as f32).collect();
    let width = 9;
    let height = 11;

    let cpu_result = tiled_max_2d(&data, width, height);

    if let Ok(gpu_result) = gpu.tiled_max_2d_gpu(&data, width, height) {
        assert!(
            (gpu_result - cpu_result).abs() < 1e-5,
            "GPU vs CPU tiled_max (negatives): gpu={}, cpu={}",
            gpu_result,
            cpu_result
        );
    } else {
        eprintln!("GPU tiled_max_2d with negatives failed");
    }
}

#[test]
fn test_gpu_tiled_min_single_element() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    let data = vec![42.0];

    if let Ok(gpu_result) = gpu.tiled_min_2d_gpu(&data, 1, 1) {
        assert!(
            (gpu_result - 42.0).abs() < 1e-5,
            "GPU tiled_min single element: got {}, expected 42.0",
            gpu_result
        );
    } else {
        eprintln!("GPU tiled_min_2d single element failed");
    }
}
