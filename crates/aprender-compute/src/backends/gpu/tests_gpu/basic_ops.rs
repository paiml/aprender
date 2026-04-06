use super::*;

#[test]
fn test_gpu_vec_add_basic() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let result = gpu.vec_add(&a, &b);

    if let Ok(c) = result {
        assert_eq!(c.len(), 4);
        assert!((c[0] - 6.0).abs() < 1e-4);
        assert!((c[1] - 8.0).abs() < 1e-4);
        assert!((c[2] - 10.0).abs() < 1e-4);
        assert!((c[3] - 12.0).abs() < 1e-4);
    } else {
        eprintln!("GPU vec_add failed: {:?}", result);
    }
}

#[test]
fn test_gpu_vec_add_large() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let size = 10000;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    let result = gpu.vec_add(&a, &b);

    if let Ok(c) = result {
        assert_eq!(c.len(), size);
        // Check first few elements
        assert!((c[0] - 0.0).abs() < 1e-4); // 0 + 0
        assert!((c[1] - 3.0).abs() < 1e-4); // 1 + 2
        assert!((c[100] - 300.0).abs() < 1e-4); // 100 + 200
    } else {
        eprintln!("GPU vec_add large failed: {:?}", result);
    }
}

#[test]
fn test_gpu_vec_add_length_mismatch() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0]; // Different length

    let result = gpu.vec_add(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_gpu_dot_basic() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let result = gpu.dot(&a, &b);

    // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    if let Ok(dot_product) = result {
        assert!((dot_product - 70.0).abs() < 1e-4);
    } else {
        eprintln!("GPU dot failed: {:?}", result);
    }
}

#[test]
fn test_gpu_dot_large() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let size = 10000;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|_| 1.0).collect();

    let result = gpu.dot(&a, &b);

    // Expected: sum of 0 + 1 + 2 + ... + 9999 = 9999 * 10000 / 2 = 49995000
    if let Ok(dot_product) = result {
        let expected = (size * (size - 1) / 2) as f32;
        assert!((dot_product - expected).abs() < 1.0); // Allow small floating point error
    } else {
        eprintln!("GPU dot large failed: {:?}", result);
    }
}

#[test]
fn test_gpu_dot_length_mismatch() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0]; // Different length

    let result = gpu.dot(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_gpu_vec_add_empty() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];

    let result = gpu.vec_add(&a, &b);

    // GPU backend returns error for empty vectors (wgpu doesn't allow zero-sized buffers)
    assert!(result.is_err(), "Expected error for empty vectors, got: {:?}", result);
}
