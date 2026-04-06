//! SIMD-PIXEL-FKR: SIMD Validation (SPEC Section 3.5.3)

use super::helpers::*;
use trueno::Vector;

/// simd-pixel-fkr: Vector operations match scalar baseline
#[test]
fn simd_pixel_fkr_vector_ops() {
    let mut rng = SimpleRng::new(11111);
    let a = rng.gen_vec(10000);
    let b = rng.gen_vec(10000);

    // Scalar baseline
    let scalar_add: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let scalar_mul: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    // SIMD implementation
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);

    let simd_add = va.add(&vb).expect("SIMD add failed");
    let simd_mul = va.mul(&vb).expect("SIMD mul failed");

    assert!(vectors_match(&scalar_add, simd_add.as_slice(), SIMD_TOLERANCE, "simd_add"));
    assert!(vectors_match(&scalar_mul, simd_mul.as_slice(), SIMD_TOLERANCE, "simd_mul"));
}

/// simd-pixel-fkr: Softmax matches scalar baseline
#[test]
fn simd_pixel_fkr_softmax() {
    let mut rng = SimpleRng::new(22222);
    let x = rng.gen_vec(2048);

    // Scalar baseline
    let scalar_result = scalar_softmax(&x);

    // SIMD implementation
    let v = Vector::from_slice(&x);
    let simd_result = v.softmax().expect("SIMD softmax failed");

    assert!(vectors_match(&scalar_result, simd_result.as_slice(), SIMD_TOLERANCE, "simd_softmax"));
}

/// simd-pixel-fkr: Unaligned input (17 elements - not divisible by SIMD width)
#[test]
fn simd_pixel_fkr_unaligned_17() {
    let mut rng = SimpleRng::new(33333);
    let a = rng.gen_vec(17);
    let b = rng.gen_vec(17);

    // Scalar baseline
    let scalar_add: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    // SIMD implementation
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);
    let simd_add = va.add(&vb).expect("SIMD unaligned add failed");

    assert!(vectors_match(&scalar_add, simd_add.as_slice(), SIMD_TOLERANCE, "simd_unaligned_17"));
}

/// simd-pixel-fkr: Remainder handling (255 elements)
#[test]
fn simd_pixel_fkr_remainder_255() {
    let mut rng = SimpleRng::new(44444);
    let a = rng.gen_vec(255);
    let b = rng.gen_vec(255);

    // Scalar baseline
    let scalar_mul: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    // SIMD implementation
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);
    let simd_mul = va.mul(&vb).expect("SIMD remainder mul failed");

    assert!(vectors_match(&scalar_mul, simd_mul.as_slice(), SIMD_TOLERANCE, "simd_remainder_255"));
}

/// simd-pixel-fkr: ReLU activation
#[test]
fn simd_pixel_fkr_relu() {
    let mut rng = SimpleRng::new(55555);
    let x = rng.gen_vec(10000);

    // Scalar baseline
    let scalar_relu: Vec<f32> = x.iter().map(|v| v.max(0.0)).collect();

    // SIMD implementation
    let v = Vector::from_slice(&x);
    let simd_relu = v.relu().expect("SIMD relu failed");

    assert!(vectors_match(&scalar_relu, simd_relu.as_slice(), SIMD_TOLERANCE, "simd_relu"));
}
