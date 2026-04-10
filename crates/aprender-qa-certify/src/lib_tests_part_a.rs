/// Verify parse_csv extracts model data including status, gates, and scores
#[test]
fn test_parse_csv_valid() {
    let models = parse_csv(SAMPLE_CSV).expect("should parse");
    assert_eq!(models.len(), 3);

    assert_eq!(models[0].model_id, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
    assert_eq!(models[0].family, "qwen-coder");
    assert_eq!(models[0].parameters, "0.5B");
    assert!(matches!(models[0].status, CertificationStatus::Pending));

    assert_eq!(models[1].mqs_score, 920);
    assert!(matches!(models[1].status, CertificationStatus::Certified));
    assert!(models[1].g1);
    assert!(models[1].g2);

    assert!(matches!(models[2].status, CertificationStatus::Blocked));
    assert!(models[2].g1);
    assert!(!models[2].g2);
}

/// Verify parse_csv returns empty vec for empty input
#[test]
fn test_parse_csv_empty() {
    let models = parse_csv("").expect("should parse empty");
    assert!(models.is_empty());
}

/// Verify parse_csv returns empty vec for header-only input
#[test]
fn test_parse_csv_header_only() {
    let csv = "model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4";
    let models = parse_csv(csv).expect("should parse header only");
    assert!(models.is_empty());
}

/// Verify parse_csv returns error for CSV with wrong column count
#[test]
fn test_parse_csv_invalid_fields() {
    let csv = "a,b,c\n1,2,3";
    let result = parse_csv(csv);
    assert!(result.is_err());
}

/// Verify CertificationStatus::parse handles all variants case-insensitively
#[test]
fn test_certification_status_parse() {
    assert_eq!(
        CertificationStatus::parse("CERTIFIED"),
        Some(CertificationStatus::Certified)
    );
    assert_eq!(
        CertificationStatus::parse("certified"),
        Some(CertificationStatus::Certified)
    );
    assert_eq!(
        CertificationStatus::parse("PROVISIONAL"),
        Some(CertificationStatus::Provisional)
    );
    assert_eq!(
        CertificationStatus::parse("BLOCKED"),
        Some(CertificationStatus::Blocked)
    );
    assert_eq!(
        CertificationStatus::parse("PENDING"),
        Some(CertificationStatus::Pending)
    );
    // Unknown values return None (Jidoka: fail loud)
    assert_eq!(CertificationStatus::parse("unknown"), None);
}

/// Verify badge() returns correct color for each certification status
#[test]
fn test_certification_status_badge() {
    assert!(
        CertificationStatus::Certified
            .badge()
            .contains("brightgreen")
    );
    assert!(CertificationStatus::Provisional.badge().contains("yellow"));
    assert!(CertificationStatus::Blocked.badge().contains("red"));
    assert!(CertificationStatus::Pending.badge().contains("lightgray"));
}

/// Verify short_name extracts model name after the org slash
#[test]
fn test_model_short_name() {
    let model = ModelCertification {
        model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
        family: "qwen-coder".to_string(),
        parameters: "1.5B".to_string(),
        size_category: SizeCategory::Small,
        status: CertificationStatus::Pending,
        mqs_score: 0,
        grade: "-".to_string(),
        certified_tier: "none".to_string(),
        last_certified: None,
        g1: false,
        g2: false,
        g3: false,
        g4: false,
        tps_gguf_cpu: None,
        tps_gguf_gpu: None,
        tps_apr_cpu: None,
        tps_apr_gpu: None,
        tps_st_cpu: None,
        tps_st_gpu: None,
        provenance_verified: false,
        kernel_proof_ref: None,
    };
    assert_eq!(model.short_name(), "Qwen2.5-Coder-1.5B-Instruct");
}

/// Verify hf_url constructs correct HuggingFace URL from model_id
#[test]
fn test_model_hf_url() {
    let model = ModelCertification {
        model_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
        family: String::new(),
        parameters: String::new(),
        size_category: SizeCategory::Small,
        status: CertificationStatus::Pending,
        mqs_score: 0,
        grade: String::new(),
        certified_tier: String::new(),
        last_certified: None,
        g1: false,
        g2: false,
        g3: false,
        g4: false,
        tps_gguf_cpu: None,
        tps_gguf_gpu: None,
        tps_apr_cpu: None,
        tps_apr_gpu: None,
        tps_st_cpu: None,
        tps_st_gpu: None,
        provenance_verified: false,
        kernel_proof_ref: None,
    };
    assert_eq!(
        model.hf_url(),
        "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct"
    );
}

/// Verify param_count parses numeric values from parameter strings like "1.5B"
#[test]
fn test_model_param_count() {
    let mut model = ModelCertification {
        model_id: String::new(),
        family: String::new(),
        parameters: "1.5B".to_string(),
        size_category: SizeCategory::Small,
        status: CertificationStatus::Pending,
        mqs_score: 0,
        grade: String::new(),
        certified_tier: String::new(),
        last_certified: None,
        g1: false,
        g2: false,
        g3: false,
        g4: false,
        tps_gguf_cpu: None,
        tps_gguf_gpu: None,
        tps_apr_cpu: None,
        tps_apr_gpu: None,
        tps_st_cpu: None,
        tps_st_gpu: None,
        provenance_verified: false,
        kernel_proof_ref: None,
    };
    assert!((model.param_count() - 1.5).abs() < f64::EPSILON);

    model.parameters = "32B".to_string();
    assert!((model.param_count() - 32.0).abs() < f64::EPSILON);

    model.parameters = "invalid".to_string();
    assert!((model.param_count() - 0.0).abs() < f64::EPSILON);
}

/// Verify gateway_symbol returns check/cross marks for certified and dash for pending
#[test]
fn test_gateway_symbol() {
    assert_eq!(
        ModelCertification::gateway_symbol(true, CertificationStatus::Certified),
        "\u{2713}"
    );
    assert_eq!(
        ModelCertification::gateway_symbol(false, CertificationStatus::Certified),
        "\u{2717}"
    );
    assert_eq!(
        ModelCertification::gateway_symbol(true, CertificationStatus::Pending),
        "-"
    );
    assert_eq!(
        ModelCertification::gateway_symbol(false, CertificationStatus::Pending),
        "-"
    );
}

/// Verify generate_summary includes status counts and timestamp
#[test]
fn test_generate_summary() {
    let models = parse_csv(SAMPLE_CSV).expect("should parse");
    let summary = generate_summary(&models, "2026-01-31 12:00 UTC");

    assert!(summary.contains("Certified | 1/3"));
    assert!(summary.contains("Blocked | 1/3"));
    assert!(summary.contains("Pending | 1/3"));
    assert!(summary.contains("2026-01-31 12:00 UTC"));
}

/// Verify generate_table produces markdown with model names and status badges
#[test]
fn test_generate_table() {
    let models = parse_csv(SAMPLE_CSV).expect("should parse");
    let table = generate_table(&models);

    assert!(table.contains("| Model | Family |"));
    assert!(table.contains("Qwen2.5-Coder-0.5B-Instruct"));
    assert!(table.contains("qwen-coder"));
    assert!(table.contains("CERTIFIED-brightgreen"));
    assert!(table.contains("BLOCKED-red"));
}

/// Verify generate_table sorts models by family then by parameter count
#[test]
fn test_generate_table_sorting() {
    let models = parse_csv(SAMPLE_CSV).expect("should parse");
    let table = generate_table(&models);
    let lines: Vec<&str> = table.lines().collect();

    // Should be sorted by family, then by param count
    // llama (1B) should come before qwen-coder (0.5B, 1.5B)
    let llama_idx = lines
        .iter()
        .position(|l| l.contains("Llama"))
        .expect("llama found");
    let qwen_05_idx = lines
        .iter()
        .position(|l| l.contains("0.5B"))
        .expect("qwen 0.5 found");

    assert!(
        llama_idx < qwen_05_idx,
        "llama should come before qwen-coder"
    );
}

/// Verify update_readme replaces content between certification markers
#[test]
fn test_update_readme_success() {
    let readme = r"# Title

Some content

<!-- CERTIFICATION_TABLE_START -->
old table
<!-- CERTIFICATION_TABLE_END -->

More content";

    let new_table = "new table content";
    let result = update_readme(readme, new_table).expect("should update");

    assert!(result.contains("new table content"));
    assert!(!result.contains("old table"));
    assert!(result.contains("# Title"));
    assert!(result.contains("More content"));
}

/// Verify update_readme returns MarkerNotFound when start marker is absent
#[test]
fn test_update_readme_missing_start_marker() {
    let readme = "no markers here <!-- CERTIFICATION_TABLE_END -->";
    let result = update_readme(readme, "table");
    assert!(matches!(result, Err(CertifyError::MarkerNotFound(_))));
}

/// Verify update_readme returns MarkerNotFound when end marker is absent
#[test]
fn test_update_readme_missing_end_marker() {
    let readme = "<!-- CERTIFICATION_TABLE_START --> no end marker";
    let result = update_readme(readme, "table");
    assert!(matches!(result, Err(CertifyError::MarkerNotFound(_))));
}

/// Verify SizeCategory::parse handles all variants and rejects unknown
#[test]
fn test_size_category_parse() {
    assert_eq!(SizeCategory::parse("tiny"), Some(SizeCategory::Tiny));
    assert_eq!(SizeCategory::parse("SMALL"), Some(SizeCategory::Small));
    assert_eq!(SizeCategory::parse("Medium"), Some(SizeCategory::Medium));
    assert_eq!(SizeCategory::parse("large"), Some(SizeCategory::Large));
    assert_eq!(SizeCategory::parse("xlarge"), Some(SizeCategory::XLarge));
    // Unknown values return None (Jidoka: fail loud)
    assert_eq!(SizeCategory::parse("unknown"), None);
}

/// Verify CertificationStatus Display outputs uppercase label
#[test]
fn test_certification_status_display() {
    assert_eq!(format!("{}", CertificationStatus::Certified), "CERTIFIED");
    assert_eq!(
        format!("{}", CertificationStatus::Provisional),
        "PROVISIONAL"
    );
    assert_eq!(format!("{}", CertificationStatus::Blocked), "BLOCKED");
    assert_eq!(format!("{}", CertificationStatus::Pending), "PENDING");
}

/// Verify short_name returns full model_id when there is no org slash
#[test]
fn test_short_name_no_slash() {
    let model = ModelCertification {
        model_id: "model-without-org".to_string(),
        family: String::new(),
        parameters: String::new(),
        size_category: SizeCategory::Small,
        status: CertificationStatus::Pending,
        mqs_score: 0,
        grade: String::new(),
        certified_tier: String::new(),
        last_certified: None,
        g1: false,
        g2: false,
        g3: false,
        g4: false,
        tps_gguf_cpu: None,
        tps_gguf_gpu: None,
        tps_apr_cpu: None,
        tps_apr_gpu: None,
        tps_st_cpu: None,
        tps_st_gpu: None,
        provenance_verified: false,
        kernel_proof_ref: None,
    };
    assert_eq!(model.short_name(), "model-without-org");
}

