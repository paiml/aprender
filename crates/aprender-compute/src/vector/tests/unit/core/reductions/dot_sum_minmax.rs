use super::super::super::super::super::*;

// Dot product tests
#[test]
fn test_dot() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.dot(&b).unwrap();
    assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_dot_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.dot(&b).unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_dot_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[3.0]);
    let result = a.dot(&b);
    assert!(result.is_err());
}

// Sum tests
#[test]
fn test_sum() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v.sum().unwrap(), 10.0);
}

#[test]
fn test_sum_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    assert_eq!(v.sum().unwrap(), 0.0);
}

#[test]
fn test_sum_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.sum().unwrap(), 42.0);
}

// Kahan summation tests (numerically stable)
#[test]
fn test_sum_kahan() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v.sum_kahan().unwrap(), 10.0);
}

#[test]
fn test_sum_kahan_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    assert_eq!(v.sum_kahan().unwrap(), 0.0);
}

#[test]
fn test_sum_kahan_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.sum_kahan().unwrap(), 42.0);
}

#[test]
fn test_sum_kahan_numerical_stability() {
    // Test case that demonstrates rounding error accumulation
    // Using many small values that can lose precision
    let mut data = vec![1e-7f32; 10_000];
    data.push(1.0);

    let v = Vector::from_slice(&data);
    let kahan_result = v.sum_kahan().unwrap();
    let naive_result = v.sum().unwrap();

    // Expected: 1.0 + 10000 * 1e-7 = 1.001
    let expected = 1.001f32;

    // Kahan should be more accurate than naive sum
    let kahan_error = (kahan_result - expected).abs();
    let naive_error = (naive_result - expected).abs();

    // Kahan error should be smaller (or at most equal)
    assert!(
        kahan_error <= naive_error,
        "Kahan sum error ({}) should be <= naive sum error ({})",
        kahan_error,
        naive_error
    );
}

// Max tests
#[test]
fn test_max() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert_eq!(v.max().unwrap(), 5.0);
}

#[test]
fn test_max_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.max().unwrap(), 42.0);
}

#[test]
fn test_max_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.max();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), TruenoError::InvalidInput("Empty vector".to_string()));
}

#[test]
fn test_max_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
    assert_eq!(v.max().unwrap(), -1.0);
}

#[test]
fn test_min() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert_eq!(v.min().unwrap(), 1.0);
}

#[test]
fn test_min_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.min().unwrap(), 42.0);
}

#[test]
fn test_min_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.min();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), TruenoError::InvalidInput("Empty vector".to_string()));
}

#[test]
fn test_min_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
    assert_eq!(v.min().unwrap(), -10.0);
}

#[test]
fn test_argmax() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert_eq!(v.argmax().unwrap(), 1); // max value 5.0 is at index 1
}

#[test]
fn test_argmax_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.argmax().unwrap(), 0);
}

#[test]
fn test_argmax_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.argmax();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), TruenoError::InvalidInput("Empty vector".to_string()));
}

#[test]
fn test_argmax_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
    assert_eq!(v.argmax().unwrap(), 1); // max value -1.0 is at index 1
}

#[test]
fn test_argmax_first_occurrence() {
    // When there are duplicates, should return first occurrence
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 5.0, 2.0]);
    assert_eq!(v.argmax().unwrap(), 1); // first 5.0 is at index 1
}

#[test]
fn test_argmin() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert_eq!(v.argmin().unwrap(), 0); // min value 1.0 is at index 0
}

#[test]
fn test_argmin_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.argmin().unwrap(), 0);
}

#[test]
fn test_argmin_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.argmin();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), TruenoError::InvalidInput("Empty vector".to_string()));
}

#[test]
fn test_argmin_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
    assert_eq!(v.argmin().unwrap(), 2); // min value -10.0 is at index 2
}

#[test]
fn test_argmin_first_occurrence() {
    // When there are duplicates, should return first occurrence
    let v = Vector::from_slice(&[5.0, 1.0, 3.0, 1.0, 2.0]);
    assert_eq!(v.argmin().unwrap(), 1); // first 1.0 is at index 1
}
