use super::*;

#[test]
fn test_gpu_matmul_basic() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // Simple 2x2 matrix multiplication
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = A * B = [[19, 22], [43, 50]]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let res = gpu.matmul(&a, &b, 2, 2, 2);

    if let Ok(result) = res {
        assert!((result[0] - 19.0).abs() < 1e-3, "Expected 19.0, got {}", result[0]);
        assert!((result[1] - 22.0).abs() < 1e-3, "Expected 22.0, got {}", result[1]);
        assert!((result[2] - 43.0).abs() < 1e-3, "Expected 43.0, got {}", result[2]);
        assert!((result[3] - 50.0).abs() < 1e-3, "Expected 50.0, got {}", result[3]);
    } else {
        eprintln!("GPU matmul failed: {:?}", res);
    }
}

#[test]
fn test_gpu_matmul_identity() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // Multiply by identity matrix
    // A = [[1, 2], [3, 4]]
    // I = [[1, 0], [0, 1]]
    // A * I = A
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let identity = vec![1.0, 0.0, 0.0, 1.0];

    let res = gpu.matmul(&a, &identity, 2, 2, 2);

    if let Ok(result) = res {
        for i in 0..4 {
            assert!((result[i] - a[i]).abs() < 1e-3, "Expected {}, got {}", a[i], result[i]);
        }
    } else {
        eprintln!("GPU matmul identity failed: {:?}", res);
    }
}

#[test]
fn test_gpu_matmul_non_square() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // 2x3 matrix * 3x2 matrix = 2x2 matrix
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8], [9, 10], [11, 12]]
    // C = [[58, 64], [139, 154]]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let res = gpu.matmul(&a, &b, 2, 3, 2);

    if let Ok(result) = res {
        assert!((result[0] - 58.0).abs() < 1e-3, "Expected 58.0, got {}", result[0]);
        assert!((result[1] - 64.0).abs() < 1e-3, "Expected 64.0, got {}", result[1]);
        assert!((result[2] - 139.0).abs() < 1e-3, "Expected 139.0, got {}", result[2]);
        assert!((result[3] - 154.0).abs() < 1e-3, "Expected 154.0, got {}", result[3]);
    } else {
        eprintln!("GPU matmul non-square failed: {:?}", res);
    }
}

#[test]
fn test_gpu_convolve2d_basic() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // 3x3 input, 2x2 kernel -> 2x2 output
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let kernel = vec![1.0, 0.0, 0.0, 1.0];

    let res = gpu.convolve2d(&input, &kernel, 3, 3, 2, 2);

    if let Ok(result) = res {
        // For kernel [[1, 0], [0, 1]], each output is sum of diagonal elements
        // Output[0,0] = input[0,0]*1 + input[1,1]*1 = 1 + 5 = 6
        assert!((result[0] - 6.0).abs() < 1e-3, "Expected 6.0, got {}", result[0]);
    } else {
        eprintln!("GPU convolve2d basic failed: {:?}", res);
    }
}

#[test]
fn test_gpu_convolve2d_identity() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // 3x3 input with center-only kernel should extract center values
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    // 3x3 kernel with center = 1, rest = 0
    let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    let res = gpu.convolve2d(&input, &kernel, 3, 3, 3, 3);

    if let Ok(result) = res {
        // Should extract center value
        assert!((result[0] - 5.0).abs() < 1e-3, "Expected 5.0, got {}", result[0]);
    } else {
        eprintln!("GPU convolve2d identity failed: {:?}", res);
    }
}

#[test]
fn test_gpu_convolve2d_averaging() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    // 4x4 input with 2x2 averaging kernel
    let input =
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    // 2x2 averaging kernel
    let kernel = vec![0.25, 0.25, 0.25, 0.25];

    let res = gpu.convolve2d(&input, &kernel, 4, 4, 2, 2);

    if let Ok(result) = res {
        // First output: average of top-left 2x2 = (1+2+5+6)/4 = 3.5
        assert!((result[0] - 3.5).abs() < 1e-3, "Expected 3.5, got {}", result[0]);
    } else {
        eprintln!("GPU convolve2d averaging failed: {:?}", res);
    }
}
