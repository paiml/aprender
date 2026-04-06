use super::super::*;

// ========================================================================
// TCB Geometry tests
// ========================================================================

#[test]
fn test_tcb_geometry_creation() {
    let geom = TcbGeometry::new(4, 8, 256);
    assert_eq!(geom.m, 4);
    assert_eq!(geom.n, 8);
    assert_eq!(geom.k, 256);
    assert_eq!(geom.alignment, 16);
}

#[test]
fn test_tcb_geometry_alignment() {
    let geom = TcbGeometry::with_alignment(4, 16, 128, 64);
    assert_eq!(geom.alignment, 64);
}

#[test]
#[should_panic(expected = "TCB dimensions must be non-zero")]
fn test_tcb_geometry_zero_dimension() {
    let _ = TcbGeometry::new(0, 8, 256);
}

#[test]
#[should_panic(expected = "Alignment must be power of 2")]
fn test_tcb_geometry_invalid_alignment() {
    let _ = TcbGeometry::with_alignment(4, 8, 256, 17);
}

#[test]
fn test_arithmetic_intensity() {
    // 4×8×256 tile
    let geom = TcbGeometry::new(4, 8, 256);
    let ai = geom.arithmetic_intensity();
    // AI = 2*4*8*256 / ((4*256 + 256*8) * 4) = 16384 / 12288 ≈ 1.33
    assert!((ai - 1.33).abs() < 0.1);
}

#[test]
fn test_q4k_alignment() {
    let aligned = TcbGeometry::new(4, 8, 256);
    assert!(aligned.is_q4k_aligned());

    let unaligned = TcbGeometry::new(4, 8, 128);
    assert!(!unaligned.is_q4k_aligned());
}

#[test]
fn test_cache_fitting() {
    let geom = TcbGeometry::new(64, 64, 64);
    // A: 64*64*4 = 16KB, B: 64*64*4 = 16KB, total = 32KB
    assert!(geom.fits_in_cache(64 * 1024)); // 64KB cache
    assert!(!geom.fits_in_cache(16 * 1024)); // 16KB cache
}

// F334: TcbGeometry helper methods
#[test]
fn test_tcb_geometry_helpers() {
    let geom = TcbGeometry::new(8, 16, 256);

    // total_elements = m * n
    assert_eq!(geom.total_elements(), 8 * 16);

    // total_flops = 2 * m * n * k
    assert_eq!(geom.total_flops(), 2 * 8 * 16 * 256);

    // Tile bytes
    assert_eq!(geom.a_tile_bytes(), 8 * 256 * 4);
    assert_eq!(geom.b_tile_bytes(), 256 * 16 * 4);
    assert_eq!(geom.c_tile_bytes(), 8 * 16 * 4);

    // Q4_0 alignment
    assert!(geom.is_q4_0_aligned()); // 256 % 32 == 0
    let unaligned = TcbGeometry::new(4, 4, 17);
    assert!(!unaligned.is_q4_0_aligned());
}

// F335: TcbGeometry Display
#[test]
fn test_tcb_geometry_display() {
    let geom = TcbGeometry::new(4, 8, 256);
    let display = format!("{}", geom);
    assert!(display.contains("TCB"));
    assert!(display.contains("4×8×256"));
    assert!(display.contains("align=16"));
    assert!(display.contains("AI="));
}

// F336: TcbGeometry Default
#[test]
fn test_tcb_geometry_default() {
    let geom = TcbGeometry::default();
    assert_eq!(geom.m, 4);
    assert_eq!(geom.n, 4);
    assert_eq!(geom.k, 4);
    assert_eq!(geom.alignment, 16);
}

// F337: TcbLevel typical cache bytes
#[test]
fn test_tcb_level_cache_bytes() {
    assert_eq!(TcbLevel::Macro.typical_cache_bytes(), 32 * 1024 * 1024);
    assert_eq!(TcbLevel::Midi.typical_cache_bytes(), 256 * 1024);
    assert_eq!(TcbLevel::Micro.typical_cache_bytes(), 32 * 1024);
}

// F349: Serialization round-trip for TcbGeometry
#[test]
fn test_tcb_geometry_serde() {
    let geom = TcbGeometry::with_alignment(4, 8, 256, 64);
    let json = serde_json::to_string(&geom).unwrap();
    let decoded: TcbGeometry = serde_json::from_str(&json).unwrap();
    assert_eq!(geom, decoded);
}

// F353: Large tile calculations
#[test]
fn test_large_tile_arithmetic() {
    // Very large geometry to test u64 arithmetic
    let geom = TcbGeometry::new(10000, 10000, 1000);
    let total = geom.total_elements();
    assert_eq!(total, 100_000_000);

    let flops = geom.total_flops();
    assert_eq!(flops, 200_000_000_000);
}

// ========================================================================
// TilingConfig tests
// ========================================================================

#[test]
fn test_tiling_config_gpu_q4k_matvec() {
    let config = TilingConfig::gpu_q4k_matvec();
    assert_eq!(config.macro_tile.m, 1);
    assert_eq!(config.macro_tile.k, 256);
    assert!(config.macro_tile.is_q4k_aligned());
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_cpu_avx2() {
    let config = TilingConfig::cpu_avx2_matmul();
    assert_eq!(config.micro_tile.n, 8); // AVX2 = 8 floats
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_validation_failure() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    // Make midi larger than macro (invalid)
    config.midi_tile.m = config.macro_tile.m + 1;
    assert!(config.validate().is_err());
}

// TILE-004: AVX-512 Register Tiling Tests
#[test]
fn test_tiling_config_avx512_matmul() {
    let config = TilingConfig::cpu_avx512_matmul();
    assert_eq!(config.micro_tile.n, 16); // AVX-512 = 16 floats
    assert_eq!(config.micro_tile.alignment, 64); // 64-byte alignment
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_avx512_q4k_matvec() {
    let config = TilingConfig::cpu_avx512_q4k_matvec();
    assert!(config.micro_tile.is_q4k_aligned());
    assert_eq!(config.micro_tile.m, 4); // 4×1 micro-kernel
    assert_eq!(config.micro_tile.n, 1); // Single output column (matvec)
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_avx512_vnni() {
    let config = TilingConfig::cpu_avx512_vnni_q4k_q8k();
    assert!(config.micro_tile.is_q4k_aligned());
    assert_eq!(config.backend, TilingBackend::CpuAvx512);
    assert!(config.validate().is_ok());
}

#[test]
fn test_avx512_vs_avx2_tile_sizes() {
    let avx2 = TilingConfig::cpu_avx2_matmul();
    let avx512 = TilingConfig::cpu_avx512_matmul();

    // AVX-512 should have 2x wider micro-tiles
    assert_eq!(avx512.micro_tile.n, avx2.micro_tile.n * 2);

    // AVX-512 should have stricter alignment
    assert!(avx512.micro_tile.alignment >= avx2.micro_tile.alignment);
}

// F338: TilingConfig factory methods
#[test]
fn test_tiling_config_gpu_softmax() {
    let config = TilingConfig::gpu_softmax();
    assert_eq!(config.name, "Softmax_GPU");
    assert_eq!(config.macro_tile.m, 1);
    assert_eq!(config.macro_tile.n, 32000); // Vocab size
    assert_eq!(config.backend, TilingBackend::Gpu);
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_cpu_rmsnorm() {
    let config = TilingConfig::cpu_rmsnorm();
    assert_eq!(config.name, "RMSNorm_CPU");
    assert_eq!(config.macro_tile.m, 1);
    assert_eq!(config.backend, TilingBackend::CpuAvx512);
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_gpu_q4k_matmul() {
    let config = TilingConfig::gpu_q4k_matmul();
    assert_eq!(config.name, "Q4K_MatMul_GPU");
    assert_eq!(config.macro_tile.m, 128);
    assert!(config.macro_tile.is_q4k_aligned());
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_cpu_avx2_q4k_matvec() {
    let config = TilingConfig::cpu_avx2_q4k_matvec();
    assert_eq!(config.name, "Q4K_MatVec_AVX2");
    assert!(config.micro_tile.is_q4k_aligned());
    assert_eq!(config.backend, TilingBackend::CpuAvx2);
    assert!(config.validate().is_ok());
}

// F350: Serialization round-trip for TilingConfig
#[test]
fn test_tiling_config_serde() {
    let config = TilingConfig::cpu_avx512_matmul();
    let json = serde_json::to_string(&config).unwrap();
    let decoded: TilingConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.name, decoded.name);
    assert_eq!(config.backend, decoded.backend);
}

// F330: Midi/micro tile divisibility
#[test]
fn test_tile_divisibility() {
    let config = TilingConfig::cpu_avx512_matmul();

    // Macro should be divisible by midi
    assert_eq!(config.macro_tile.m % config.midi_tile.m, 0);
    assert_eq!(config.macro_tile.n % config.midi_tile.n, 0);

    // Midi should be divisible by micro
    assert_eq!(config.midi_tile.m % config.micro_tile.m, 0);
    assert_eq!(config.midi_tile.n % config.micro_tile.n, 0);
}

// F343: TilingConfig validation - divisibility errors
#[test]
fn test_tiling_config_divisibility_error() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    // Make midi not divide evenly into macro
    config.midi_tile.m = 17; // 256 % 17 != 0
    let result = config.validate();
    assert!(result.is_err());
    if let Err(TilingError::DivisibilityError { level, .. }) = result {
        assert_eq!(level, "macro/midi");
    } else {
        panic!("Expected DivisibilityError");
    }
}

#[test]
fn test_tiling_config_micro_divisibility_error() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    // Make micro not divide evenly into midi
    config.micro_tile.m = 17; // midi.m % 17 != 0
    let result = config.validate();
    assert!(result.is_err());
}

// F341: TilingError Display implementations
#[test]
fn test_tiling_error_display() {
    let err1 = TilingError::InvalidHierarchy { reason: "test".into() };
    assert!(format!("{}", err1).contains("Invalid tiling hierarchy"));
    assert!(format!("{}", err1).contains("test"));

    let err2 = TilingError::DivisibilityError {
        level: "macro/midi",
        dimension: "M",
        larger: 256,
        smaller: 17,
    };
    assert!(format!("{}", err2).contains("Tiling divisibility error"));
    assert!(format!("{}", err2).contains("256"));

    let err3 = TilingError::CacheOverflow {
        level: TcbLevel::Midi,
        required_bytes: 1000,
        available_bytes: 500,
    };
    assert!(format!("{}", err3).contains("exceeds"));
    assert!(format!("{}", err3).contains("Midi"));

    let err4 = TilingError::AlignmentError { required: 64, actual: 32 };
    assert!(format!("{}", err4).contains("Alignment error"));

    let err5 = TilingError::QuantAlignmentError { format: "Q4_K", required_k: 256, actual_k: 100 };
    assert!(format!("{}", err5).contains("Quantization alignment"));
    assert!(format!("{}", err5).contains("Q4_K"));
}

// F342: TilingError as std::error::Error
#[test]
fn test_tiling_error_trait() {
    let err = TilingError::InvalidHierarchy { reason: "test".into() };
    // Ensure it implements Error trait
    let _: &dyn std::error::Error = &err;
}

// F346: TilingBackend enum
#[test]
fn test_tiling_backend_equality() {
    assert_eq!(TilingBackend::CpuAvx2, TilingBackend::CpuAvx2);
    assert_ne!(TilingBackend::CpuAvx2, TilingBackend::CpuAvx512);
    assert_ne!(TilingBackend::Gpu, TilingBackend::Scalar);
    assert_eq!(TilingBackend::CpuNeon, TilingBackend::CpuNeon);
}

// F347: PackingLayout enum
#[test]
fn test_packing_layout_equality() {
    assert_eq!(PackingLayout::RowMajor, PackingLayout::RowMajor);
    assert_ne!(PackingLayout::RowMajor, PackingLayout::ColumnMajor);
    assert_ne!(PackingLayout::PanelMajorA, PackingLayout::PanelMajorB);
}

// F348: TcbLevel enum
#[test]
fn test_tcb_level_equality() {
    assert_eq!(TcbLevel::Macro, TcbLevel::Macro);
    assert_ne!(TcbLevel::Macro, TcbLevel::Midi);
    assert_ne!(TcbLevel::Midi, TcbLevel::Micro);
}

// ========================================================================
// TilingConfig coverage gap tests (Refs CB-130)
// ========================================================================

// Validate micro > midi hierarchy error (second branch in validate())
#[test]
fn test_validate_micro_larger_than_midi_m() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    // micro.m > midi.m triggers the second hierarchy check
    config.micro_tile.m = config.midi_tile.m + 1;
    let result = config.validate();
    assert!(result.is_err());
    if let Err(TilingError::InvalidHierarchy { reason }) = result {
        assert!(reason.contains("Micro-tile larger than midi-tile"));
    } else {
        panic!("Expected InvalidHierarchy for micro > midi");
    }
}

#[test]
fn test_validate_micro_larger_than_midi_n() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    config.micro_tile.n = config.midi_tile.n + 1;
    let result = config.validate();
    assert!(result.is_err());
    if let Err(TilingError::InvalidHierarchy { reason }) = result {
        assert!(reason.contains("Micro-tile larger than midi-tile"));
    } else {
        panic!("Expected InvalidHierarchy for micro.n > midi.n");
    }
}

#[test]
fn test_validate_micro_larger_than_midi_k() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    config.micro_tile.k = config.midi_tile.k + 1;
    let result = config.validate();
    assert!(result.is_err());
    if let Err(TilingError::InvalidHierarchy { reason }) = result {
        assert!(reason.contains("Micro-tile larger than midi-tile"));
    } else {
        panic!("Expected InvalidHierarchy for micro.k > midi.k");
    }
}

// Validate midi > macro hierarchy error via N dimension
#[test]
fn test_validate_midi_larger_than_macro_n() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    config.midi_tile.n = config.macro_tile.n + 1;
    let result = config.validate();
    assert!(result.is_err());
    if let Err(TilingError::InvalidHierarchy { reason }) = result {
        assert!(reason.contains("Midi-tile larger than macro-tile"));
    } else {
        panic!("Expected InvalidHierarchy for midi.n > macro.n");
    }
}

// Validate midi > macro hierarchy error via K dimension
#[test]
fn test_validate_midi_larger_than_macro_k() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    config.midi_tile.k = config.macro_tile.k + 1;
    let result = config.validate();
    assert!(result.is_err());
    if let Err(TilingError::InvalidHierarchy { reason }) = result {
        assert!(reason.contains("Midi-tile larger than macro-tile"));
    } else {
        panic!("Expected InvalidHierarchy for midi.k > macro.k");
    }
}

// Validate midi/micro M divisibility error with detail check
#[test]
fn test_validate_midi_micro_divisibility_detail() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    // midi.m = 64, set micro.m = 13 so 64 % 13 != 0
    // But we need micro.m <= midi.m to pass hierarchy check
    config.micro_tile.m = 13;
    let result = config.validate();
    assert!(result.is_err());
    if let Err(TilingError::DivisibilityError { level, dimension, .. }) = result {
        assert_eq!(level, "midi/micro");
        assert_eq!(dimension, "M");
    } else {
        panic!("Expected DivisibilityError for midi/micro M");
    }
}

// Test num_macro_tiles with exact divisibility
#[test]
fn test_num_macro_tiles_exact_divisibility() {
    let config = TilingConfig::cpu_avx2_matmul();
    // macro_tile is 256x256, so 512x512 should give 2x2=4 tiles
    let tiles = config.num_macro_tiles(512, 512);
    assert_eq!(tiles, 4);
}

// Test num_macro_tiles with remainder (ceiling division)
#[test]
fn test_num_macro_tiles_with_remainder() {
    let config = TilingConfig::cpu_avx2_matmul();
    // macro_tile is 256x256, so 257x1 should give ceil(257/256)*ceil(1/256) = 2*1 = 2
    let tiles = config.num_macro_tiles(257, 1);
    assert_eq!(tiles, 2);
}

// Test num_macro_tiles with single element
#[test]
fn test_num_macro_tiles_single_element() {
    let config = TilingConfig::cpu_avx2_matmul();
    let tiles = config.num_macro_tiles(1, 1);
    assert_eq!(tiles, 1);
}

// Test midi_tiles_per_macro for various configs
#[test]
fn test_midi_tiles_per_macro_avx2() {
    let config = TilingConfig::cpu_avx2_matmul();
    // macro: 256x256, midi: 64x64 -> (256/64)*(256/64) = 4*4 = 16
    let midi = config.midi_tiles_per_macro();
    assert_eq!(midi, 16);
}

#[test]
fn test_midi_tiles_per_macro_gpu_softmax() {
    let config = TilingConfig::gpu_softmax();
    // macro: 1x32000, midi: 1x1024 -> (1/1)*(32000/1024) = 1*31 (truncated)
    let midi = config.midi_tiles_per_macro();
    // 32000 / 1024 = 31 (integer division, 32000 = 31*1024 + 256)
    assert_eq!(midi, 31);
}

#[test]
fn test_midi_tiles_per_macro_gpu_q4k_matvec() {
    let config = TilingConfig::gpu_q4k_matvec();
    // macro: 1x4096, midi: 1x256 -> (1/1)*(4096/256) = 1*16 = 16
    let midi = config.midi_tiles_per_macro();
    assert_eq!(midi, 16);
}

// Test micro_tiles_per_midi for various configs
#[test]
fn test_micro_tiles_per_midi_avx2() {
    let config = TilingConfig::cpu_avx2_matmul();
    // midi: 64x64, micro: 4x8 -> (64/4)*(64/8) = 16*8 = 128
    let micro = config.micro_tiles_per_midi();
    assert_eq!(micro, 128);
}

#[test]
fn test_micro_tiles_per_midi_avx512() {
    let config = TilingConfig::cpu_avx512_matmul();
    // midi: 128x128, micro: 4x16 -> (128/4)*(128/16) = 32*8 = 256
    let micro = config.micro_tiles_per_midi();
    assert_eq!(micro, 256);
}

#[test]
fn test_micro_tiles_per_midi_gpu_q4k_matmul() {
    let config = TilingConfig::gpu_q4k_matmul();
    // midi: 32x32, micro: 8x8 -> (32/8)*(32/8) = 4*4 = 16
    let micro = config.micro_tiles_per_midi();
    assert_eq!(micro, 16);
}

// TilingBackend serde round-trip for all variants
#[test]
fn test_tiling_backend_serde_all_variants() {
    let variants = [
        TilingBackend::CpuAvx2,
        TilingBackend::CpuAvx512,
        TilingBackend::CpuNeon,
        TilingBackend::Gpu,
        TilingBackend::Scalar,
    ];
    for variant in &variants {
        let json = serde_json::to_string(variant).unwrap();
        let decoded: TilingBackend = serde_json::from_str(&json).unwrap();
        assert_eq!(*variant, decoded);
    }
}

// TilingBackend Debug formatting
#[test]
fn test_tiling_backend_debug() {
    assert!(format!("{:?}", TilingBackend::CpuAvx2).contains("CpuAvx2"));
    assert!(format!("{:?}", TilingBackend::CpuAvx512).contains("CpuAvx512"));
    assert!(format!("{:?}", TilingBackend::CpuNeon).contains("CpuNeon"));
    assert!(format!("{:?}", TilingBackend::Gpu).contains("Gpu"));
    assert!(format!("{:?}", TilingBackend::Scalar).contains("Scalar"));
}

// TilingBackend Clone and Copy
#[test]
fn test_tiling_backend_clone_copy() {
    let backend = TilingBackend::CpuNeon;
    let cloned = backend;
    let copied = backend;
    assert_eq!(cloned, copied);
    assert_eq!(backend, TilingBackend::CpuNeon);
}

// TilingConfig Debug formatting
#[test]
fn test_tiling_config_debug() {
    let config = TilingConfig::gpu_q4k_matvec();
    let debug = format!("{:?}", config);
    assert!(debug.contains("TilingConfig"));
    assert!(debug.contains("Q4K_MatVec_GPU"));
    assert!(debug.contains("Gpu"));
}

// TilingConfig Clone
#[test]
fn test_tiling_config_clone() {
    let config = TilingConfig::cpu_avx2_matmul();
    let cloned = config.clone();
    assert_eq!(cloned.name, "MatMul_AVX2");
    assert_eq!(cloned.backend, TilingBackend::CpuAvx2);
    assert_eq!(cloned.macro_tile, config.macro_tile);
    assert_eq!(cloned.midi_tile, config.midi_tile);
    assert_eq!(cloned.micro_tile, config.micro_tile);
}

// Serde round-trip for configs with Neon and Scalar backends
#[test]
fn test_tiling_config_serde_neon_backend() {
    let config = TilingConfig {
        name: "NEON_Test".into(),
        macro_tile: TcbGeometry::with_alignment(64, 64, 64, 16),
        midi_tile: TcbGeometry::with_alignment(16, 16, 16, 16),
        micro_tile: TcbGeometry::with_alignment(4, 4, 16, 16),
        backend: TilingBackend::CpuNeon,
    };
    let json = serde_json::to_string(&config).unwrap();
    let decoded: TilingConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.name, "NEON_Test");
    assert_eq!(decoded.backend, TilingBackend::CpuNeon);
    assert_eq!(decoded.macro_tile, config.macro_tile);
    assert!(decoded.validate().is_ok());
}

#[test]
fn test_tiling_config_serde_scalar_backend() {
    let config = TilingConfig {
        name: "Scalar_Test".into(),
        macro_tile: TcbGeometry::with_alignment(32, 32, 32, 16),
        midi_tile: TcbGeometry::with_alignment(8, 8, 8, 16),
        micro_tile: TcbGeometry::with_alignment(4, 4, 8, 16),
        backend: TilingBackend::Scalar,
    };
    let json = serde_json::to_string(&config).unwrap();
    let decoded: TilingConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.name, "Scalar_Test");
    assert_eq!(decoded.backend, TilingBackend::Scalar);
    assert!(decoded.validate().is_ok());
}

// Exhaustive factory method field verification
#[test]
fn test_gpu_q4k_matvec_all_fields() {
    let config = TilingConfig::gpu_q4k_matvec();
    assert_eq!(config.name, "Q4K_MatVec_GPU");
    assert_eq!(config.backend, TilingBackend::Gpu);

    // Macro-tile
    assert_eq!(config.macro_tile.m, 1);
    assert_eq!(config.macro_tile.n, 4096);
    assert_eq!(config.macro_tile.k, 256);
    assert_eq!(config.macro_tile.alignment, 64);

    // Midi-tile
    assert_eq!(config.midi_tile.m, 1);
    assert_eq!(config.midi_tile.n, 256);
    assert_eq!(config.midi_tile.k, 256);
    assert_eq!(config.midi_tile.alignment, 64);

    // Micro-tile
    assert_eq!(config.micro_tile.m, 1);
    assert_eq!(config.micro_tile.n, 32);
    assert_eq!(config.micro_tile.k, 256);
    assert_eq!(config.micro_tile.alignment, 64);
}

#[test]
fn test_gpu_q4k_matmul_all_fields() {
    let config = TilingConfig::gpu_q4k_matmul();
    assert_eq!(config.name, "Q4K_MatMul_GPU");
    assert_eq!(config.backend, TilingBackend::Gpu);

    assert_eq!(config.macro_tile.m, 128);
    assert_eq!(config.macro_tile.n, 128);
    assert_eq!(config.macro_tile.k, 256);
    assert_eq!(config.macro_tile.alignment, 64);

    assert_eq!(config.midi_tile.m, 32);
    assert_eq!(config.midi_tile.n, 32);
    assert_eq!(config.midi_tile.k, 256);

    assert_eq!(config.micro_tile.m, 8);
    assert_eq!(config.micro_tile.n, 8);
    assert_eq!(config.micro_tile.k, 256);
}

#[test]
fn test_gpu_softmax_all_fields() {
    let config = TilingConfig::gpu_softmax();
    assert_eq!(config.name, "Softmax_GPU");
    assert_eq!(config.backend, TilingBackend::Gpu);

    assert_eq!(config.macro_tile.m, 1);
    assert_eq!(config.macro_tile.n, 32000);
    assert_eq!(config.macro_tile.k, 1);
    assert_eq!(config.macro_tile.alignment, 64);

    assert_eq!(config.midi_tile.m, 1);
    assert_eq!(config.midi_tile.n, 1024);
    assert_eq!(config.midi_tile.k, 1);

    assert_eq!(config.micro_tile.m, 1);
    assert_eq!(config.micro_tile.n, 32);
    assert_eq!(config.micro_tile.k, 1);
}

#[test]
fn test_cpu_avx512_matmul_all_fields() {
    let config = TilingConfig::cpu_avx512_matmul();
    assert_eq!(config.name, "MatMul_AVX512");
    assert_eq!(config.backend, TilingBackend::CpuAvx512);

    assert_eq!(config.macro_tile.m, 512);
    assert_eq!(config.macro_tile.n, 512);
    assert_eq!(config.macro_tile.k, 512);
    assert_eq!(config.macro_tile.alignment, 64);

    assert_eq!(config.midi_tile.m, 128);
    assert_eq!(config.midi_tile.n, 128);
    assert_eq!(config.midi_tile.k, 128);

    assert_eq!(config.micro_tile.m, 4);
    assert_eq!(config.micro_tile.n, 16);
    assert_eq!(config.micro_tile.k, 128);
}

#[test]
fn test_cpu_avx512_q4k_matvec_all_fields() {
    let config = TilingConfig::cpu_avx512_q4k_matvec();
    assert_eq!(config.name, "Q4K_MatVec_AVX512");
    assert_eq!(config.backend, TilingBackend::CpuAvx512);

    assert_eq!(config.macro_tile.m, 4096);
    assert_eq!(config.macro_tile.n, 1);
    assert_eq!(config.macro_tile.k, 4096);
    assert_eq!(config.macro_tile.alignment, 64);

    assert_eq!(config.midi_tile.m, 64);
    assert_eq!(config.midi_tile.n, 1);
    assert_eq!(config.midi_tile.k, 256);

    assert_eq!(config.micro_tile.m, 4);
    assert_eq!(config.micro_tile.n, 1);
    assert_eq!(config.micro_tile.k, 256);
}

#[test]
fn test_cpu_avx512_vnni_all_fields() {
    let config = TilingConfig::cpu_avx512_vnni_q4k_q8k();
    assert_eq!(config.name, "Q4K_Q8K_VNNI");
    assert_eq!(config.backend, TilingBackend::CpuAvx512);

    assert_eq!(config.macro_tile.m, 4096);
    assert_eq!(config.macro_tile.n, 1);
    assert_eq!(config.macro_tile.k, 4096);
    assert_eq!(config.macro_tile.alignment, 64);

    assert_eq!(config.midi_tile.m, 64);
    assert_eq!(config.midi_tile.n, 1);
    assert_eq!(config.midi_tile.k, 256);

    assert_eq!(config.micro_tile.m, 4);
    assert_eq!(config.micro_tile.n, 1);
    assert_eq!(config.micro_tile.k, 256);
}

#[test]
fn test_cpu_avx2_matmul_all_fields() {
    let config = TilingConfig::cpu_avx2_matmul();
    assert_eq!(config.name, "MatMul_AVX2");
    assert_eq!(config.backend, TilingBackend::CpuAvx2);

    assert_eq!(config.macro_tile.m, 256);
    assert_eq!(config.macro_tile.n, 256);
    assert_eq!(config.macro_tile.k, 256);
    assert_eq!(config.macro_tile.alignment, 32);

    assert_eq!(config.midi_tile.m, 64);
    assert_eq!(config.midi_tile.n, 64);
    assert_eq!(config.midi_tile.k, 64);
    assert_eq!(config.midi_tile.alignment, 32);

    assert_eq!(config.micro_tile.m, 4);
    assert_eq!(config.micro_tile.n, 8);
    assert_eq!(config.micro_tile.k, 64);
    assert_eq!(config.micro_tile.alignment, 32);
}

#[test]
fn test_cpu_avx2_q4k_matvec_all_fields() {
    let config = TilingConfig::cpu_avx2_q4k_matvec();
    assert_eq!(config.name, "Q4K_MatVec_AVX2");
    assert_eq!(config.backend, TilingBackend::CpuAvx2);

    assert_eq!(config.macro_tile.m, 4096);
    assert_eq!(config.macro_tile.n, 1);
    assert_eq!(config.macro_tile.k, 4096);
    assert_eq!(config.macro_tile.alignment, 32);

    assert_eq!(config.midi_tile.m, 64);
    assert_eq!(config.midi_tile.n, 1);
    assert_eq!(config.midi_tile.k, 256);

    assert_eq!(config.micro_tile.m, 4);
    assert_eq!(config.micro_tile.n, 1);
    assert_eq!(config.micro_tile.k, 256);
}

#[test]
fn test_cpu_rmsnorm_all_fields() {
    let config = TilingConfig::cpu_rmsnorm();
    assert_eq!(config.name, "RMSNorm_CPU");
    assert_eq!(config.backend, TilingBackend::CpuAvx512);

    assert_eq!(config.macro_tile.m, 1);
    assert_eq!(config.macro_tile.n, 4096);
    assert_eq!(config.macro_tile.k, 1);
    assert_eq!(config.macro_tile.alignment, 32);

    assert_eq!(config.midi_tile.m, 1);
    assert_eq!(config.midi_tile.n, 256);
    assert_eq!(config.midi_tile.k, 1);

    assert_eq!(config.micro_tile.m, 1);
    assert_eq!(config.micro_tile.n, 16);
    assert_eq!(config.micro_tile.k, 1);
}

// All factory methods validate successfully
#[test]
fn test_all_factory_configs_validate() {
    let configs = [
        TilingConfig::gpu_q4k_matvec(),
        TilingConfig::gpu_q4k_matmul(),
        TilingConfig::gpu_softmax(),
        TilingConfig::cpu_avx512_matmul(),
        TilingConfig::cpu_avx512_q4k_matvec(),
        TilingConfig::cpu_avx512_vnni_q4k_q8k(),
        TilingConfig::cpu_avx2_matmul(),
        TilingConfig::cpu_avx2_q4k_matvec(),
        TilingConfig::cpu_rmsnorm(),
    ];
    for config in &configs {
        assert!(config.validate().is_ok(), "Factory config '{}' failed validation", config.name);
    }
}

// Serde round-trip for all factory configs
#[test]
fn test_all_factory_configs_serde_roundtrip() {
    let configs = [
        TilingConfig::gpu_q4k_matvec(),
        TilingConfig::gpu_q4k_matmul(),
        TilingConfig::gpu_softmax(),
        TilingConfig::cpu_avx512_matmul(),
        TilingConfig::cpu_avx512_q4k_matvec(),
        TilingConfig::cpu_avx512_vnni_q4k_q8k(),
        TilingConfig::cpu_avx2_matmul(),
        TilingConfig::cpu_avx2_q4k_matvec(),
        TilingConfig::cpu_rmsnorm(),
    ];
    for config in &configs {
        let json = serde_json::to_string(config).unwrap();
        let decoded: TilingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.name, decoded.name);
        assert_eq!(config.backend, decoded.backend);
        assert_eq!(config.macro_tile, decoded.macro_tile);
        assert_eq!(config.midi_tile, decoded.midi_tile);
        assert_eq!(config.micro_tile, decoded.micro_tile);
    }
}

// Test num_macro_tiles for different configs
#[test]
fn test_num_macro_tiles_various_configs() {
    // GPU Q4K MatVec: macro is 1x4096
    let config = TilingConfig::gpu_q4k_matvec();
    // 4096 rows, 4096 cols with 1x4096 tiles -> ceil(4096/1)*ceil(4096/4096) = 4096*1 = 4096
    assert_eq!(config.num_macro_tiles(4096, 4096), 4096);

    // GPU Softmax: macro is 1x32000
    let config = TilingConfig::gpu_softmax();
    // 32 rows, 32000 cols with 1x32000 tiles -> ceil(32/1)*ceil(32000/32000) = 32*1 = 32
    assert_eq!(config.num_macro_tiles(32, 32000), 32);

    // CPU AVX-512 MatMul: macro is 512x512
    let config = TilingConfig::cpu_avx512_matmul();
    // 1024x1024 with 512x512 tiles -> 2*2 = 4
    assert_eq!(config.num_macro_tiles(1024, 1024), 4);
}

// Validation passes when tiles are exactly equal at boundaries
#[test]
fn test_validate_equal_tile_sizes() {
    let geom = TcbGeometry::with_alignment(16, 16, 16, 16);
    let config = TilingConfig {
        name: "Equal".into(),
        macro_tile: geom,
        midi_tile: geom,
        micro_tile: geom,
        backend: TilingBackend::Scalar,
    };
    assert!(config.validate().is_ok());
}

// Validation passes with minimal tile hierarchy
#[test]
fn test_validate_minimal_tiles() {
    let config = TilingConfig {
        name: "Minimal".into(),
        macro_tile: TcbGeometry::with_alignment(1, 1, 1, 16),
        midi_tile: TcbGeometry::with_alignment(1, 1, 1, 16),
        micro_tile: TcbGeometry::with_alignment(1, 1, 1, 16),
        backend: TilingBackend::Scalar,
    };
    assert!(config.validate().is_ok());
    assert_eq!(config.num_macro_tiles(1, 1), 1);
    assert_eq!(config.midi_tiles_per_macro(), 1);
    assert_eq!(config.micro_tiles_per_midi(), 1);
}

// Large problem size num_macro_tiles
#[test]
fn test_num_macro_tiles_large_problem() {
    let config = TilingConfig::cpu_avx2_matmul();
    // 10000 x 10000 with 256x256 tiles -> ceil(10000/256) * ceil(10000/256) = 40*40 = 1600
    let tiles = config.num_macro_tiles(10000, 10000);
    assert_eq!(tiles, 40 * 40);
}

// TilingConfig with CpuNeon backend validates
#[test]
fn test_tiling_config_neon_backend_validate() {
    let config = TilingConfig {
        name: "NEON_MatMul".into(),
        macro_tile: TcbGeometry::with_alignment(128, 128, 128, 16),
        midi_tile: TcbGeometry::with_alignment(32, 32, 32, 16),
        micro_tile: TcbGeometry::with_alignment(4, 4, 32, 16),
        backend: TilingBackend::CpuNeon,
    };
    assert!(config.validate().is_ok());
    assert_eq!(config.backend, TilingBackend::CpuNeon);
    assert_eq!(config.midi_tiles_per_macro(), 16);
    assert_eq!(config.micro_tiles_per_midi(), 64);
}

// TilingConfig with Scalar backend validates
#[test]
fn test_tiling_config_scalar_backend_validate() {
    let config = TilingConfig {
        name: "Scalar_MatMul".into(),
        macro_tile: TcbGeometry::with_alignment(64, 64, 64, 16),
        midi_tile: TcbGeometry::with_alignment(16, 16, 16, 16),
        micro_tile: TcbGeometry::with_alignment(4, 4, 16, 16),
        backend: TilingBackend::Scalar,
    };
    assert!(config.validate().is_ok());
    assert_eq!(config.backend, TilingBackend::Scalar);
}
