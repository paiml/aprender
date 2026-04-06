use super::*;

#[test]
fn test_address_registry() {
    let mut registry = AddressRegistry::new();

    // Register a buffer
    registry.register("input_buf", 0x7f00000000, 4096, "f32", 4);

    // Test lookup at start
    let info = registry.lookup(0x7f00000000).unwrap();
    assert_eq!(info.name, "input_buf");

    // Test lookup in middle
    let info = registry.lookup(0x7f00000100).unwrap();
    assert_eq!(info.name, "input_buf");
    assert_eq!(info.element_index_of(0x7f00000100), Some(64)); // 256 bytes / 4 = 64

    // Test lookup outside
    assert!(registry.lookup(0x8000000000).is_none());
}

#[test]
fn test_format_address() {
    let mut registry = AddressRegistry::new();
    registry.register("weights", 0x7f00000000, 1024 * 4, "f32", 4);

    // Element-aligned access
    let formatted = registry.format_address(0x7f00000010);
    assert!(formatted.contains("weights[4]"));

    // Non-element-aligned access
    let formatted = registry.format_address(0x7f00000011);
    assert!(formatted.contains("weights[4]+1"));

    // Unknown address
    let formatted = registry.format_address(0x1);
    assert!(formatted.contains("unknown"));
}

#[test]
fn test_parse_sanitizer_output() {
    let output = r#"
========= COMPUTE-SANITIZER
========= Invalid __shared__ read of size 4 bytes
=========     at lz4_compress_warp+0x2160
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x1 is misaligned
========= ERROR SUMMARY: 1 error
"#;

    let violations = SanitizerParser::parse(output);
    assert_eq!(violations.len(), 1);

    let v = &violations[0];
    assert_eq!(v.kernel_name, "lz4_compress_warp");
    assert_eq!(v.sass_offset, 0x2160);
    assert_eq!(v.thread, (0, 0, 0));
    assert_eq!(v.block, (0, 0, 0));
    assert_eq!(v.address, 0x1);

    match &v.violation_type {
        MemoryViolationType::InvalidSharedRead { size } => assert_eq!(*size, 4),
        _ => panic!("Expected InvalidSharedRead"),
    }
}

#[test]
fn test_ptx_source_map() {
    let ptx = r#"
.version 8.0
.target sm_89
.entry test_kernel() {
    mov.u32 %r0, 0;
L_loop:
    add.u32 %r0, %r0, 1;
    bra L_loop;
L_end:
    ret;
}
"#;

    let map = PtxSourceMap::new(ptx);

    // Check label parsing
    assert!(map.label_lines.contains_key("L_loop"));
    assert!(map.label_lines.contains_key("L_end"));

    // Check context retrieval
    let context = map.context_around_label("L_loop", 2);
    assert!(context.is_some());
    let ctx = context.unwrap();
    assert!(ctx.contains("L_loop:"));
}
