    // ============================================================================
    // load_mmap and load_auto Tests (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_load_mmap_roundtrip() {
        let model = TestModel {
            name: "mmap_test".to_string(),
            values: vec![10.0, 20.0, 30.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("mmap_test.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        let loaded: TestModel =
            load_mmap(&path, ModelType::LinearRegression).expect("load mmap");
        assert_eq!(model, loaded);
    }

    #[test]
    fn test_load_auto_small_file() {
        let model = TestModel {
            name: "auto_small".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("auto_small.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        // Small file should use standard load path
        let loaded: TestModel =
            load_auto(&path, ModelType::LinearRegression).expect("load auto");
        assert_eq!(model, loaded);
    }

    // ============================================================================
    // ZstdMax Compression (GH-219 coverage)
    // ============================================================================

    #[cfg(feature = "format-compression")]
    #[test]
    fn test_compress_decompress_zstd_max() {
        let data = b"test data that should compress well with zstd max compression level";
        let (compressed, compression) =
            compress_payload(data, Compression::ZstdMax).expect("compress");
        assert_eq!(compression, Compression::ZstdMax);
        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed, data);
    }

    // ============================================================================
    // Header Error Paths (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_header_roundtrip() {
        use super::super::{Header, ModelType};
        let mut header = Header::new(ModelType::RandomForest);
        header.metadata_size = 100;
        header.payload_size = 5000;
        header.uncompressed_size = 10000;
        header.compression = Compression::None;
        header.quality_score = 85;

        let bytes = header.to_bytes();
        let parsed = Header::from_bytes(&bytes).expect("parse header");

        assert_eq!(parsed.model_type, ModelType::RandomForest);
        assert_eq!(parsed.metadata_size, 100);
        assert_eq!(parsed.payload_size, 5000);
        assert_eq!(parsed.uncompressed_size, 10000);
        assert_eq!(parsed.compression, Compression::None);
        assert_eq!(parsed.quality_score, 85);
    }

    #[test]
    fn test_header_from_bytes_bad_magic() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"BAAD");
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_too_short() {
        let bytes = [0u8; 10];
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_unsupported_version() {
        use super::super::MAGIC;
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 255; // Future major version
        bytes[5] = 0;
        // Set a valid model type
        bytes[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_unknown_model_type() {
        use super::super::{FORMAT_VERSION, MAGIC};
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION.0;
        bytes[5] = FORMAT_VERSION.1;
        bytes[6..8].copy_from_slice(&0xFFFEu16.to_le_bytes()); // Unknown model type
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_unknown_compression() {
        use super::super::{FORMAT_VERSION, MAGIC};
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION.0;
        bytes[5] = FORMAT_VERSION.1;
        bytes[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        bytes[20] = 0xEE; // Unknown compression
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_compression_bomb_protection() {
        use super::super::{FORMAT_VERSION, MAGIC, MAX_UNCOMPRESSED_SIZE};
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION.0;
        bytes[5] = FORMAT_VERSION.1;
        bytes[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        // Set uncompressed_size to exceed max
        let bomb_size = MAX_UNCOMPRESSED_SIZE + 1;
        bytes[16..20].copy_from_slice(&bomb_size.to_le_bytes());
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    // ============================================================================
    // Flags Tests (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_flags_all_individual() {
        use super::super::Flags;

        let f = Flags::new()
            .with_encrypted()
            .with_signed()
            .with_streaming()
            .with_licensed()
            .with_trueno_native()
            .with_quantized()
            .with_model_card();

        assert!(f.is_encrypted());
        assert!(f.is_signed());
        assert!(f.is_streaming());
        assert!(f.is_licensed());
        assert!(f.is_trueno_native());
        assert!(f.is_quantized());
        assert!(f.has_model_card());
    }

    #[test]
    fn test_flags_default_empty() {
        use super::super::Flags;

        let f = Flags::new();
        assert!(!f.is_encrypted());
        assert!(!f.is_signed());
        assert!(!f.is_streaming());
        assert!(!f.is_licensed());
        assert!(!f.is_trueno_native());
        assert!(!f.is_quantized());
        assert!(!f.has_model_card());
        assert_eq!(f.bits(), 0);
    }

    #[test]
    fn test_flags_from_bits_masks_reserved() {
        use super::super::Flags;
        // Bit 7 is reserved and should be masked
        let f = Flags::from_bits(0xFF);
        assert_eq!(f.bits(), 0x7F);
    }

    // ============================================================================
    // ModelType / Compression from_u16/from_u8 (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_model_type_from_u16_all_variants() {
        assert_eq!(ModelType::from_u16(0x0001), Some(ModelType::LinearRegression));
        assert_eq!(ModelType::from_u16(0x0002), Some(ModelType::LogisticRegression));
        assert_eq!(ModelType::from_u16(0x0003), Some(ModelType::DecisionTree));
        assert_eq!(ModelType::from_u16(0x0004), Some(ModelType::RandomForest));
        assert_eq!(ModelType::from_u16(0x0005), Some(ModelType::GradientBoosting));
        assert_eq!(ModelType::from_u16(0x0006), Some(ModelType::KMeans));
        assert_eq!(ModelType::from_u16(0x0007), Some(ModelType::Pca));
        assert_eq!(ModelType::from_u16(0x0008), Some(ModelType::NaiveBayes));
        assert_eq!(ModelType::from_u16(0x0009), Some(ModelType::Knn));
        assert_eq!(ModelType::from_u16(0x000A), Some(ModelType::Svm));
        assert_eq!(ModelType::from_u16(0x0010), Some(ModelType::NgramLm));
        assert_eq!(ModelType::from_u16(0x0011), Some(ModelType::Tfidf));
        assert_eq!(ModelType::from_u16(0x0012), Some(ModelType::CountVectorizer));
        assert_eq!(ModelType::from_u16(0x0020), Some(ModelType::NeuralSequential));
        assert_eq!(ModelType::from_u16(0x0021), Some(ModelType::NeuralCustom));
        assert_eq!(ModelType::from_u16(0x0030), Some(ModelType::ContentRecommender));
        assert_eq!(ModelType::from_u16(0x0040), Some(ModelType::MixtureOfExperts));
        assert_eq!(ModelType::from_u16(0x00FF), Some(ModelType::Custom));
        assert_eq!(ModelType::from_u16(0x9999), None);
    }

    #[test]
    fn test_compression_from_u8_all_variants() {
        assert_eq!(Compression::from_u8(0x00), Some(Compression::None));
        assert_eq!(Compression::from_u8(0x01), Some(Compression::ZstdDefault));
        assert_eq!(Compression::from_u8(0x02), Some(Compression::ZstdMax));
        assert_eq!(Compression::from_u8(0x03), Some(Compression::Lz4));
        assert_eq!(Compression::from_u8(0xFF), None);
    }

    // ============================================================================
    // SaveOptions Builder (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_save_options_builder() {
        let opts = SaveOptions::new()
            .with_name("test-model")
            .with_description("A description")
            .with_compression(Compression::None)
            .with_quality_score(95);

        assert_eq!(opts.metadata.model_name, Some("test-model".to_string()));
        assert_eq!(
            opts.metadata.description,
            Some("A description".to_string())
        );
        assert_eq!(opts.compression, Compression::None);
        assert_eq!(opts.quality_score, Some(95));
    }

    #[test]
    fn test_save_options_with_license() {
        use super::super::{LicenseInfo, LicenseTier};

        let license = LicenseInfo {
            uuid: "uuid-123".to_string(),
            hash: "hash-abc".to_string(),
            expiry: Some("2027-01-01".to_string()),
            seats: Some(10),
            licensee: Some("PAIML".to_string()),
            tier: LicenseTier::Team,
        };
        let opts = SaveOptions::new().with_license(license);
        assert!(opts.metadata.license.is_some());
        let lic = opts.metadata.license.unwrap();
        assert_eq!(lic.tier, LicenseTier::Team);
        assert_eq!(lic.seats, Some(10));
    }

    // ============================================================================
    // inspect_bytes metadata boundary (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_inspect_bytes_metadata_beyond_boundary() {
        use super::super::{FORMAT_VERSION, MAGIC};
        // Create a header claiming huge metadata_size
        let mut bytes = vec![0u8; HEADER_SIZE + 4]; // minimal: header + checksum
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION.0;
        bytes[5] = FORMAT_VERSION.1;
        bytes[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        // metadata_size = 10000 (way beyond the data)
        bytes[8..12].copy_from_slice(&10000u32.to_le_bytes());
        let result = inspect_bytes(&bytes);
        assert!(result.is_err());
    }

    // ============================================================================
    // load payload_end boundary check (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_load_payload_beyond_boundary() {
        use super::super::{FORMAT_VERSION, MAGIC};
        // Create a file with header claiming payload extends beyond file
        let mut content = vec![0u8; HEADER_SIZE + 10];
        content[0..4].copy_from_slice(&MAGIC);
        content[4] = FORMAT_VERSION.0;
        content[5] = FORMAT_VERSION.1;
        content[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        // metadata_size = 0, payload_size = 99999
        content[12..16].copy_from_slice(&99999u32.to_le_bytes());
        // Append valid checksum
        let checksum = crc32(&content);
        content.extend_from_slice(&checksum.to_le_bytes());

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("bad_boundary.apr");
        std::fs::write(&path, &content).expect("write");

        let result: Result<TestModel> = load(&path, ModelType::Custom);
        assert!(result.is_err());
    }

    // ============================================================================
    // Metadata default (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_metadata_default() {
        let m = Metadata::default();
        assert!(m.model_name.is_none());
        assert!(m.description.is_none());
        assert!(m.training.is_none());
        assert!(m.hyperparameters.is_empty());
        assert!(m.metrics.is_empty());
        assert!(m.custom.is_empty());
        assert!(m.distillation.is_none());
        assert!(m.distillation_info.is_none());
        assert!(m.license.is_none());
        assert!(m.model_card.is_none());
        assert!(!m.created_at.is_empty());
    }
