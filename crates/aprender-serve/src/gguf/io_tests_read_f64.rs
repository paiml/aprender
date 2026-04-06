
#[test]
fn test_read_f64_nan() {
    let data = f64::NAN.to_le_bytes();
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_f64(&mut cursor).expect("expected value").is_nan());
}

#[test]
fn test_read_f64_truncated() {
    let data = [0x01, 0x02, 0x03, 0x04, 0x05]; // Only 5 bytes
    let mut cursor = Cursor::new(&data[..]);
    assert!(read_f64(&mut cursor).is_err());
}

// ============================================================================
// Mixed sequential reads (integration)
// ============================================================================

#[test]
fn test_sequential_mixed_reads() {
    // Build a buffer with different types packed together
    let mut data = Vec::new();
    data.push(42u8); // u8
    data.push(0xFF); // i8 = -1
    data.extend_from_slice(&1000u16.to_le_bytes()); // u16
    data.extend_from_slice(&(-500i16).to_le_bytes()); // i16
    data.extend_from_slice(&123456u32.to_le_bytes()); // u32
    data.extend_from_slice(&(-789i32).to_le_bytes()); // i32
    data.extend_from_slice(&3.14f32.to_le_bytes()); // f32
    data.push(1); // bool = true
    data.extend_from_slice(&999999u64.to_le_bytes()); // u64
    data.extend_from_slice(&(-12345i64).to_le_bytes()); // i64
    data.extend_from_slice(&2.71828f64.to_le_bytes()); // f64

    let mut cursor = Cursor::new(&data[..]);

    assert_eq!(read_u8(&mut cursor).expect("expected value"), 42);
    assert_eq!(read_i8(&mut cursor).expect("expected value"), -1);
    assert_eq!(read_u16(&mut cursor).expect("expected value"), 1000);
    assert_eq!(read_i16(&mut cursor).expect("expected value"), -500);
    assert_eq!(read_u32(&mut cursor).expect("expected value"), 123456);
    assert_eq!(read_i32(&mut cursor).expect("expected value"), -789);
    assert!((read_f32(&mut cursor).expect("expected value") - 3.14).abs() < 1e-6);
    assert!(read_bool(&mut cursor).expect("expected value"));
    assert_eq!(read_u64(&mut cursor).expect("expected value"), 999999);
    assert_eq!(read_i64(&mut cursor).expect("expected value"), -12345);
    assert!((read_f64(&mut cursor).expect("expected value") - 2.71828).abs() < 1e-10);

    // Should be at end of cursor now
    assert!(read_u8(&mut cursor).is_err());
}

#[test]
fn test_error_messages_contain_operation_name() {
    let data: [u8; 0] = [];

    let err = read_u8(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_u8"));

    let err = read_i8(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_i8"));

    let err = read_u16(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_u16"));

    let err = read_i16(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_i16"));

    let err = read_u32(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_u32"));

    let err = read_i32(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_i32"));

    let err = read_f32(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_f32"));

    let err = read_bool(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_bool"));

    let err = read_u64(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_u64"));

    let err = read_i64(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_i64"));

    let err = read_f64(&mut Cursor::new(&data[..])).unwrap_err();
    assert!(format!("{:?}", err).contains("read_f64"));
}
