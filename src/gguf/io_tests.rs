//! Tests for GGUF primitive I/O readers (io.rs)
//!
//! Coverage target: all 11 reader functions in io.rs (36 missed lines)
//! Each function reads N bytes from a Cursor<&[u8]> in little-endian format.
//! Tests cover: happy path, boundary values, error on truncated input.

use std::io::Cursor;

use crate::gguf::io::*;

// ============================================================================
// read_u8
// ============================================================================

#[test]
fn test_read_u8_zero() {
    let data = [0u8];
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u8(&mut cursor).unwrap(), 0);
}

#[test]
fn test_read_u8_max() {
    let data = [0xFF];
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u8(&mut cursor).unwrap(), 255);
}

#[test]
fn test_read_u8_value() {
    let data = [42u8];
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u8(&mut cursor).unwrap(), 42);
}

#[test]
fn test_read_u8_empty_input() {
    let data: [u8; 0] = [];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_u8(&mut cursor).is_err());
}

#[test]
fn test_read_u8_sequential() {
    let data = [10u8, 20, 30];
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u8(&mut cursor).unwrap(), 10);
    assert_eq!(read_u8(&mut cursor).unwrap(), 20);
    assert_eq!(read_u8(&mut cursor).unwrap(), 30);
    assert!(read_u8(&mut cursor).is_err());
}

// ============================================================================
// read_i8
// ============================================================================

#[test]
fn test_read_i8_zero() {
    let data = [0u8];
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i8(&mut cursor).unwrap(), 0);
}

#[test]
fn test_read_i8_positive() {
    let data = [127u8]; // max positive i8
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i8(&mut cursor).unwrap(), 127);
}

#[test]
fn test_read_i8_negative() {
    let data = [0xFF]; // -1 in two's complement
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i8(&mut cursor).unwrap(), -1);
}

#[test]
fn test_read_i8_min() {
    let data = [0x80]; // -128 in two's complement
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i8(&mut cursor).unwrap(), -128);
}

#[test]
fn test_read_i8_empty_input() {
    let data: [u8; 0] = [];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_i8(&mut cursor).is_err());
}

// ============================================================================
// read_u16
// ============================================================================

#[test]
fn test_read_u16_zero() {
    let data = 0u16.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u16(&mut cursor).unwrap(), 0);
}

#[test]
fn test_read_u16_max() {
    let data = u16::MAX.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u16(&mut cursor).unwrap(), u16::MAX);
}

#[test]
fn test_read_u16_value() {
    let data = 12345u16.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u16(&mut cursor).unwrap(), 12345);
}

#[test]
fn test_read_u16_truncated() {
    let data = [0x01]; // Only 1 byte, needs 2
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_u16(&mut cursor).is_err());
}

#[test]
fn test_read_u16_empty() {
    let data: [u8; 0] = [];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_u16(&mut cursor).is_err());
}

// ============================================================================
// read_i16
// ============================================================================

#[test]
fn test_read_i16_zero() {
    let data = 0i16.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i16(&mut cursor).unwrap(), 0);
}

#[test]
fn test_read_i16_positive() {
    let data = 32767i16.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i16(&mut cursor).unwrap(), 32767);
}

#[test]
fn test_read_i16_negative() {
    let data = (-1i16).to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i16(&mut cursor).unwrap(), -1);
}

#[test]
fn test_read_i16_min() {
    let data = i16::MIN.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i16(&mut cursor).unwrap(), i16::MIN);
}

#[test]
fn test_read_i16_truncated() {
    let data = [0x01];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_i16(&mut cursor).is_err());
}

// ============================================================================
// read_u32
// ============================================================================

#[test]
fn test_read_u32_zero() {
    let data = 0u32.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u32(&mut cursor).unwrap(), 0);
}

#[test]
fn test_read_u32_max() {
    let data = u32::MAX.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u32(&mut cursor).unwrap(), u32::MAX);
}

#[test]
fn test_read_u32_value() {
    let data = 0xDEADBEEFu32.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u32(&mut cursor).unwrap(), 0xDEADBEEF);
}

#[test]
fn test_read_u32_truncated() {
    let data = [0x01, 0x02, 0x03]; // Only 3 bytes, needs 4
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_u32(&mut cursor).is_err());
}

#[test]
fn test_read_u32_empty() {
    let data: [u8; 0] = [];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_u32(&mut cursor).is_err());
}

// ============================================================================
// read_i32
// ============================================================================

#[test]
fn test_read_i32_zero() {
    let data = 0i32.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i32(&mut cursor).unwrap(), 0);
}

#[test]
fn test_read_i32_positive() {
    let data = i32::MAX.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i32(&mut cursor).unwrap(), i32::MAX);
}

#[test]
fn test_read_i32_negative() {
    let data = (-42i32).to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i32(&mut cursor).unwrap(), -42);
}

#[test]
fn test_read_i32_min() {
    let data = i32::MIN.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i32(&mut cursor).unwrap(), i32::MIN);
}

#[test]
fn test_read_i32_truncated() {
    let data = [0x01, 0x02];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_i32(&mut cursor).is_err());
}

// ============================================================================
// read_f32
// ============================================================================

#[test]
fn test_read_f32_zero() {
    let data = 0.0f32.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!((read_f32(&mut cursor).unwrap() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_read_f32_one() {
    let data = 1.0f32.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!((read_f32(&mut cursor).unwrap() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_read_f32_negative() {
    let data = (-3.14f32).to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!((read_f32(&mut cursor).unwrap() - (-3.14)).abs() < 1e-6);
}

#[test]
fn test_read_f32_infinity() {
    let data = f32::INFINITY.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_f32(&mut cursor).unwrap().is_infinite());
}

#[test]
fn test_read_f32_nan() {
    let data = f32::NAN.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_f32(&mut cursor).unwrap().is_nan());
}

#[test]
fn test_read_f32_truncated() {
    let data = [0x01, 0x02, 0x03]; // Only 3 bytes
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_f32(&mut cursor).is_err());
}

// ============================================================================
// read_bool
// ============================================================================

#[test]
fn test_read_bool_false() {
    let data = [0u8];
    let mut cursor = Cursor::new(&data[..]);
    assert!(!read_bool(&mut cursor).unwrap());
}

#[test]
fn test_read_bool_true_one() {
    let data = [1u8];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_bool(&mut cursor).unwrap());
}

#[test]
fn test_read_bool_true_nonzero() {
    // Any non-zero byte should be true
    let data = [0xFF];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_bool(&mut cursor).unwrap());
}

#[test]
fn test_read_bool_true_arbitrary() {
    let data = [42u8];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_bool(&mut cursor).unwrap());
}

#[test]
fn test_read_bool_empty() {
    let data: [u8; 0] = [];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_bool(&mut cursor).is_err());
}

// ============================================================================
// read_u64
// ============================================================================

#[test]
fn test_read_u64_zero() {
    let data = 0u64.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u64(&mut cursor).unwrap(), 0);
}

#[test]
fn test_read_u64_max() {
    let data = u64::MAX.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u64(&mut cursor).unwrap(), u64::MAX);
}

#[test]
fn test_read_u64_value() {
    let data = 0x0102030405060708u64.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_u64(&mut cursor).unwrap(), 0x0102030405060708);
}

#[test]
fn test_read_u64_truncated() {
    let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]; // Only 7 bytes
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_u64(&mut cursor).is_err());
}

#[test]
fn test_read_u64_empty() {
    let data: [u8; 0] = [];
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_u64(&mut cursor).is_err());
}

// ============================================================================
// read_i64
// ============================================================================

#[test]
fn test_read_i64_zero() {
    let data = 0i64.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i64(&mut cursor).unwrap(), 0);
}

#[test]
fn test_read_i64_positive() {
    let data = i64::MAX.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i64(&mut cursor).unwrap(), i64::MAX);
}

#[test]
fn test_read_i64_negative() {
    let data = (-999i64).to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i64(&mut cursor).unwrap(), -999);
}

#[test]
fn test_read_i64_min() {
    let data = i64::MIN.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert_eq!(read_i64(&mut cursor).unwrap(), i64::MIN);
}

#[test]
fn test_read_i64_truncated() {
    let data = [0x01, 0x02, 0x03, 0x04]; // Only 4 bytes
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_i64(&mut cursor).is_err());
}

// ============================================================================
// read_f64
// ============================================================================

#[test]
fn test_read_f64_zero() {
    let data = 0.0f64.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!((read_f64(&mut cursor).unwrap() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_read_f64_one() {
    let data = 1.0f64.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!((read_f64(&mut cursor).unwrap() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_read_f64_pi() {
    let data = std::f64::consts::PI.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!((read_f64(&mut cursor).unwrap() - std::f64::consts::PI).abs() < 1e-15);
}

#[test]
fn test_read_f64_negative() {
    let data = (-2.71828f64).to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!((read_f64(&mut cursor).unwrap() - (-2.71828)).abs() < 1e-10);
}

#[test]
fn test_read_f64_infinity() {
    let data = f64::INFINITY.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_f64(&mut cursor).unwrap().is_infinite());
}

include!("io_tests_read_f64.rs");
