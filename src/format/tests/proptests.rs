//! Core property-based tests for APR format.

use super::super::*;
use proptest::prelude::*;

// ================================================================
// Arbitrary Strategies
// ================================================================

/// Generate arbitrary ModelType
fn arb_model_type() -> impl Strategy<Value = ModelType> {
    prop_oneof![
        Just(ModelType::LinearRegression),
        Just(ModelType::LogisticRegression),
        Just(ModelType::DecisionTree),
        Just(ModelType::RandomForest),
        Just(ModelType::GradientBoosting),
        Just(ModelType::KMeans),
        Just(ModelType::Pca),
        Just(ModelType::NaiveBayes),
        Just(ModelType::Knn),
        Just(ModelType::Svm),
        Just(ModelType::NgramLm),
        Just(ModelType::Tfidf),
        Just(ModelType::CountVectorizer),
        Just(ModelType::NeuralSequential),
        Just(ModelType::NeuralCustom),
        Just(ModelType::ContentRecommender),
        Just(ModelType::Custom),
    ]
}

/// Generate arbitrary Compression
fn arb_compression() -> impl Strategy<Value = Compression> {
    prop_oneof![
        Just(Compression::None),
        Just(Compression::ZstdDefault),
        Just(Compression::ZstdMax),
        Just(Compression::Lz4),
    ]
}

/// Generate arbitrary valid flags (6 bits)
fn arb_flags() -> impl Strategy<Value = Flags> {
    (0u8..64).prop_map(Flags::from_bits)
}

/// Generate arbitrary Header with valid values
fn arb_header() -> impl Strategy<Value = Header> {
    (
        arb_model_type(),
        0u32..1_000_000,             // metadata_size
        0u32..100_000_000,           // payload_size
        0u32..MAX_UNCOMPRESSED_SIZE, // uncompressed_size
        arb_compression(),
        arb_flags(),
    )
        .prop_map(
            |(model_type, metadata_size, payload_size, uncompressed_size, compression, flags)| {
                Header {
                    magic: MAGIC,
                    version: FORMAT_VERSION,
                    model_type,
                    metadata_size,
                    payload_size,
                    uncompressed_size,
                    compression,
                    flags,
                    quality_score: 0, // APR-POKA-001: default for tests
                }
            },
        )
}

// ================================================================
// Header Roundtrip Property Tests
// ================================================================

proptest! {
    /// Property: Header serialization always produces exactly 32 bytes
    #[test]
    fn prop_header_size_always_32(header in arb_header()) {
        let bytes = header.to_bytes();
        prop_assert_eq!(bytes.len(), HEADER_SIZE);
    }

    /// Property: Header always starts with magic "APRN"
    #[test]
    fn prop_header_has_magic(header in arb_header()) {
        let bytes = header.to_bytes();
        prop_assert_eq!(&bytes[0..4], &MAGIC);
    }

    /// Property: Header roundtrip preserves model_type
    #[test]
    fn prop_header_roundtrip_model_type(header in arb_header()) {
        let bytes = header.to_bytes();
        let parsed = Header::from_bytes(&bytes).expect("valid header");
        prop_assert_eq!(header.model_type, parsed.model_type);
    }

    /// Property: Header roundtrip preserves metadata_size
    #[test]
    fn prop_header_roundtrip_metadata_size(header in arb_header()) {
        let bytes = header.to_bytes();
        let parsed = Header::from_bytes(&bytes).expect("valid header");
        prop_assert_eq!(header.metadata_size, parsed.metadata_size);
    }

    /// Property: Header roundtrip preserves payload_size
    #[test]
    fn prop_header_roundtrip_payload_size(header in arb_header()) {
        let bytes = header.to_bytes();
        let parsed = Header::from_bytes(&bytes).expect("valid header");
        prop_assert_eq!(header.payload_size, parsed.payload_size);
    }

    /// Property: Header roundtrip preserves uncompressed_size
    #[test]
    fn prop_header_roundtrip_uncompressed_size(header in arb_header()) {
        let bytes = header.to_bytes();
        let parsed = Header::from_bytes(&bytes).expect("valid header");
        prop_assert_eq!(header.uncompressed_size, parsed.uncompressed_size);
    }

    /// Property: Header roundtrip preserves compression
    #[test]
    fn prop_header_roundtrip_compression(header in arb_header()) {
        let bytes = header.to_bytes();
        let parsed = Header::from_bytes(&bytes).expect("valid header");
        prop_assert_eq!(header.compression, parsed.compression);
    }

    /// Property: Header roundtrip preserves flags
    #[test]
    fn prop_header_roundtrip_flags(header in arb_header()) {
        let bytes = header.to_bytes();
        let parsed = Header::from_bytes(&bytes).expect("valid header");
        prop_assert_eq!(header.flags.bits(), parsed.flags.bits());
    }

    // ================================================================
    // ModelType Property Tests
    // ================================================================

    /// Property: ModelType from_u16 roundtrip
    #[test]
    fn prop_model_type_roundtrip(model_type in arb_model_type()) {
        let value = model_type as u16;
        let parsed = ModelType::from_u16(value);
        prop_assert_eq!(parsed, Some(model_type));
    }

    /// Property: Invalid model type returns None
    #[test]
    fn prop_invalid_model_type_none(value in 0x0100u16..0x1000) {
        // Values outside defined range should return None
        let parsed = ModelType::from_u16(value);
        prop_assert!(parsed.is_none());
    }

    // ================================================================
    // Compression Property Tests
    // ================================================================

    /// Property: Compression from_u8 roundtrip
    #[test]
    fn prop_compression_roundtrip(compression in arb_compression()) {
        let value = compression as u8;
        let parsed = Compression::from_u8(value);
        prop_assert_eq!(parsed, Some(compression));
    }

    /// Property: Invalid compression returns None
    #[test]
    fn prop_invalid_compression_none(value in 4u8..255) {
        let parsed = Compression::from_u8(value);
        prop_assert!(parsed.is_none());
    }

    // ================================================================
    // Flags Property Tests
    // ================================================================

    /// Property: Flags from_bits masks reserved bits
    #[test]
    fn prop_flags_masks_reserved(raw in any::<u8>()) {
        let flags = Flags::from_bits(raw);
        // Bit 7 should be masked off (bit 6 is now HAS_MODEL_CARD)
        prop_assert!(flags.bits() < 128);
    }

    /// Property: with_encrypted sets ENCRYPTED bit
    #[test]
    fn prop_flags_with_encrypted(_seed in any::<u8>()) {
        let flags = Flags::new().with_encrypted();
        prop_assert!(flags.is_encrypted());
        prop_assert_eq!(flags.bits() & Flags::ENCRYPTED, Flags::ENCRYPTED);
    }

    /// Property: with_signed sets SIGNED bit
    #[test]
    fn prop_flags_with_signed(_seed in any::<u8>()) {
        let flags = Flags::new().with_signed();
        prop_assert!(flags.is_signed());
        prop_assert_eq!(flags.bits() & Flags::SIGNED, Flags::SIGNED);
    }

    /// Property: with_streaming sets STREAMING bit
    #[test]
    fn prop_flags_with_streaming(_seed in any::<u8>()) {
        let flags = Flags::new().with_streaming();
        prop_assert!(flags.is_streaming());
        prop_assert_eq!(flags.bits() & Flags::STREAMING, Flags::STREAMING);
    }

    /// Property: with_licensed sets LICENSED bit
    #[test]
    fn prop_flags_with_licensed(_seed in any::<u8>()) {
        let flags = Flags::new().with_licensed();
        prop_assert!(flags.is_licensed());
        prop_assert_eq!(flags.bits() & Flags::LICENSED, Flags::LICENSED);
    }

    /// Property: with_quantized sets QUANTIZED bit
    #[test]
    fn prop_flags_with_quantized(_seed in any::<u8>()) {
        let flags = Flags::new().with_quantized();
        prop_assert!(flags.is_quantized());
        prop_assert_eq!(flags.bits() & Flags::QUANTIZED, Flags::QUANTIZED);
    }

    /// Property: with_model_card sets HAS_MODEL_CARD bit
    #[test]
    fn prop_flags_with_model_card(_seed in any::<u8>()) {
        let flags = Flags::new().with_model_card();
        prop_assert!(flags.has_model_card());
        prop_assert_eq!(flags.bits() & Flags::HAS_MODEL_CARD, Flags::HAS_MODEL_CARD);
    }

    /// Property: Flag chaining is commutative (order doesn't matter)
    #[test]
    fn prop_flags_chaining_commutative(a in any::<bool>(), b in any::<bool>()) {
        let mut flags1 = Flags::new();
        let mut flags2 = Flags::new();

        if a {
            flags1 = flags1.with_encrypted();
        }
        if b {
            flags1 = flags1.with_signed();
        }

        // Reverse order
        if b {
            flags2 = flags2.with_signed();
        }
        if a {
            flags2 = flags2.with_encrypted();
        }

        prop_assert_eq!(flags1.bits(), flags2.bits());
    }

    /// Property: Flag bits are independent (setting one doesn't affect others)
    #[test]
    fn prop_flags_independent(flags in arb_flags()) {
        // Check each flag independently
        let encrypted = flags.is_encrypted();
        let signed = flags.is_signed();
        let streaming = flags.is_streaming();
        let licensed = flags.is_licensed();
        let quantized = flags.is_quantized();

        // Reconstruct from individual bits
        let reconstructed = (if encrypted { Flags::ENCRYPTED } else { 0 })
            | (if signed { Flags::SIGNED } else { 0 })
            | (if streaming { Flags::STREAMING } else { 0 })
            | (if licensed { Flags::LICENSED } else { 0 })
            | (if quantized { Flags::QUANTIZED } else { 0 })
            | (flags.bits() & Flags::TRUENO_NATIVE); // Don't forget trueno_native

        prop_assert_eq!(flags.bits(), reconstructed);
    }

    // ================================================================
    // CRC32 Checksum Property Tests
    // ================================================================

    /// Property: CRC32 of same data is always identical
    #[test]
    fn prop_crc32_deterministic(data in proptest::collection::vec(any::<u8>(), 0..1000)) {
        let crc1 = crc32(&data);
        let crc2 = crc32(&data);
        prop_assert_eq!(crc1, crc2);
    }

    /// Property: CRC32 changes when data changes (avalanche property)
    #[test]
    fn prop_crc32_avalanche(
        data in proptest::collection::vec(any::<u8>(), 1..100),
        flip_pos in 0usize..100,
        flip_bit in 0u8..8
    ) {
        if flip_pos >= data.len() {
            return Ok(());
        }

        let crc_original = crc32(&data);

        let mut modified = data.clone();
        modified[flip_pos] ^= 1 << flip_bit;

        let crc_modified = crc32(&modified);

        // Single bit flip should change CRC
        prop_assert_ne!(crc_original, crc_modified);
    }

    /// Property: Empty data has consistent CRC (IEEE polynomial)
    #[test]
    fn prop_crc32_empty(_seed in any::<u8>()) {
        let crc = crc32(&[]);
        // CRC32 of empty data is 0 for our implementation
        prop_assert_eq!(crc, 0);
    }
}
