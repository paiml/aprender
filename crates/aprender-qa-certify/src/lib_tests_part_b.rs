#[test]
fn test_markdown_link() {
    let model = ModelCertification {
        model_id: "Org/Model".to_string(),
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
        model.markdown_link(),
        "[Model](https://huggingface.co/Org/Model)"
    );
}

#[test]
fn test_write_csv_roundtrip() {
    let models = parse_csv(SAMPLE_CSV).expect("should parse");
    let csv_output = write_csv(&models);

    // Parse it back
    let reparsed = parse_csv(&csv_output).expect("should reparse");
    assert_eq!(reparsed.len(), models.len());

    // Check first model
    assert_eq!(reparsed[0].model_id, models[0].model_id);
    assert_eq!(reparsed[0].family, models[0].family);
    assert_eq!(reparsed[0].mqs_score, models[0].mqs_score);
}

#[test]
fn test_write_csv_has_header() {
    let models = parse_csv(SAMPLE_CSV).expect("should parse");
    let csv_output = write_csv(&models);
    assert!(csv_output.starts_with("model_id,family,"));
}

#[test]
fn test_status_from_score_certified() {
    assert!(matches!(
        status_from_score(900_u32, false),
        CertificationStatus::Certified
    ));
    assert!(matches!(
        status_from_score(850_u32, false),
        CertificationStatus::Certified
    ));
    // MVP_PASS_SCORE (800) is the certified threshold
    assert!(matches!(
        status_from_score(800_u32, false),
        CertificationStatus::Certified
    ));
}

#[test]
fn test_status_from_score_provisional() {
    // 700-799: provisional range
    assert!(matches!(
        status_from_score(799_u32, false),
        CertificationStatus::Provisional
    ));
    assert!(matches!(
        status_from_score(700_u32, false),
        CertificationStatus::Provisional
    ));
}

#[test]
fn test_status_from_score_blocked() {
    assert!(matches!(
        status_from_score(699_u32, false),
        CertificationStatus::Blocked
    ));
    assert!(matches!(
        status_from_score(0_u32, false),
        CertificationStatus::Blocked
    ));
}

#[test]
fn test_status_from_score_p0_failure() {
    // P0 failure always results in BLOCKED regardless of score
    assert!(matches!(
        status_from_score(950_u32, true),
        CertificationStatus::Blocked
    ));
    assert!(matches!(
        status_from_score(900_u32, true),
        CertificationStatus::Blocked
    ));
}

#[test]
fn test_grade_from_score() {
    assert_eq!(grade_from_score(1000_u32), "A+");
    assert_eq!(grade_from_score(950_u32), "A+");
    assert_eq!(grade_from_score(920_u32), "A");
    assert_eq!(grade_from_score(900_u32), "A");
    assert_eq!(grade_from_score(880_u32), "B+");
    assert_eq!(grade_from_score(850_u32), "B+");
    assert_eq!(grade_from_score(820_u32), "B");
    assert_eq!(grade_from_score(800_u32), "B");
    assert_eq!(grade_from_score(750_u32), "C");
    assert_eq!(grade_from_score(700_u32), "C");
    assert_eq!(grade_from_score(699_u32), "F");
    assert_eq!(grade_from_score(0_u32), "F");
}

#[test]
fn test_mvp_tier_pass() {
    // MVP tier with 90%+ pass rate should get B grade (800 score)
    let status = status_from_tier(CertificationTier::Mvp, 0.95, false);
    assert!(matches!(status, CertificationStatus::Provisional));

    let score = score_from_tier(CertificationTier::Mvp, 0.95, false);
    assert_eq!(score, 800);

    let grade = grade_from_tier(CertificationTier::Mvp, 0.95, false);
    assert_eq!(grade, "B");
}

#[test]
fn test_mvp_tier_exactly_90_percent() {
    // MVP tier at exactly 90% should pass
    let status = status_from_tier(CertificationTier::Mvp, 0.90, false);
    assert!(matches!(status, CertificationStatus::Provisional));

    let score = score_from_tier(CertificationTier::Mvp, 0.90, false);
    assert_eq!(score, 800);
}

#[test]
fn test_mvp_tier_fail() {
    // MVP tier below 90% should fail
    let status = status_from_tier(CertificationTier::Mvp, 0.85, false);
    assert!(matches!(status, CertificationStatus::Blocked));

    let score = score_from_tier(CertificationTier::Mvp, 0.85, false);
    assert!(score < 700); // F grade
}

#[test]
fn test_mvp_tier_p0_failure() {
    // MVP tier with P0 failure should always block
    let status = status_from_tier(CertificationTier::Mvp, 0.99, true);
    assert!(matches!(status, CertificationStatus::Blocked));

    let score = score_from_tier(CertificationTier::Mvp, 0.99, true);
    assert!(score < 700); // F grade even with high pass rate
}

#[test]
fn test_full_tier_pass() {
    // Full tier with 95%+ should get A+ (950+ score)
    let status = status_from_tier(CertificationTier::Full, 0.98, false);
    assert!(matches!(status, CertificationStatus::Certified));

    let score = score_from_tier(CertificationTier::Full, 0.98, false);
    assert!(score >= 950);

    let grade = grade_from_tier(CertificationTier::Full, 0.98, false);
    assert_eq!(grade, "A+");
}

#[test]
fn test_full_tier_provisional() {
    // Full tier between 90% and 95% should get PROVISIONAL
    let status = status_from_tier(CertificationTier::Full, 0.92, false);
    assert!(matches!(status, CertificationStatus::Provisional));

    let score = score_from_tier(CertificationTier::Full, 0.92, false);
    assert!((800..900).contains(&score)); // B to B+ range
}

#[test]
fn test_full_tier_fail() {
    // Full tier below 90% should fail
    let status = status_from_tier(CertificationTier::Full, 0.85, false);
    assert!(matches!(status, CertificationStatus::Blocked));

    let score = score_from_tier(CertificationTier::Full, 0.85, false);
    assert!(score < 700);
}

#[test]
fn test_certification_tier_default() {
    let tier = CertificationTier::default();
    assert!(matches!(tier, CertificationTier::Mvp));
}

#[test]
fn test_parse_csv_with_empty_lines() {
    let csv = r"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4

Qwen/Qwen2.5-Coder-0.5B-Instruct,qwen-coder,0.5B,tiny,PENDING,0,-,none,2026-01-31T00:00:00Z,false,false,false,false

";
    let models = parse_csv(csv).expect("should parse with empty lines");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].model_id, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
}

#[test]
fn test_parse_csv_insufficient_fields_in_line() {
    let csv = r"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4
only,a,few,fields";
    let result = parse_csv(csv);
    assert!(result.is_err());
    let err = result.expect_err("Should be an error");
    assert!(
        matches!(err, CertifyError::CsvParse { line: 2, .. }),
        "Error should indicate line 2"
    );
}

#[test]
fn test_write_csv_all_size_categories() {
    // Test that write_csv correctly handles all size categories
    let models = vec![
        ModelCertification {
            model_id: "tiny-model".to_string(),
            family: "test".to_string(),
            parameters: "0.5B".to_string(),
            size_category: SizeCategory::Tiny,
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
        },
        ModelCertification {
            model_id: "medium-model".to_string(),
            family: "test".to_string(),
            parameters: "7B".to_string(),
            size_category: SizeCategory::Medium,
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
        },
        ModelCertification {
            model_id: "large-model".to_string(),
            family: "test".to_string(),
            parameters: "34B".to_string(),
            size_category: SizeCategory::Large,
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
        },
        ModelCertification {
            model_id: "xlarge-model".to_string(),
            family: "test".to_string(),
            parameters: "70B".to_string(),
            size_category: SizeCategory::XLarge,
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
        },
    ];

    let csv_output = write_csv(&models);
    assert!(csv_output.contains(",tiny,"));
    assert!(csv_output.contains(",medium,"));
    assert!(csv_output.contains(",large,"));
    assert!(csv_output.contains(",xlarge,"));
}

#[test]
fn test_csv_quote_carriage_return() {
    // RFC 4180: fields containing CR must be quoted
    let field = "line1\rline2";
    let quoted = csv_quote(field);
    assert_eq!(quoted, "\"line1\rline2\"");
}

#[test]
fn test_csv_quote_crlf() {
    // RFC 4180: fields containing CRLF must be quoted
    let field = "line1\r\nline2";
    let quoted = csv_quote(field);
    assert_eq!(quoted, "\"line1\r\nline2\"");
}

#[test]
fn test_csv_quote_plain() {
    assert_eq!(csv_quote("hello"), "hello");
}

#[test]
fn test_csv_quote_comma() {
    assert_eq!(csv_quote("a,b"), "\"a,b\"");
}

#[test]
fn test_csv_quote_double_quote() {
    assert_eq!(csv_quote("say \"hi\""), "\"say \"\"hi\"\"\"");
}

#[test]
fn test_csv_split_basic() {
    let fields = csv_split("a,b,c");
    assert_eq!(fields, vec!["a", "b", "c"]);
}

#[test]
fn test_csv_split_quoted_comma() {
    let fields = csv_split("\"a,b\",c");
    assert_eq!(fields, vec!["a,b", "c"]);
}

#[test]
fn test_csv_split_escaped_quote() {
    let fields = csv_split("\"say \"\"hi\"\"\",done");
    assert_eq!(fields, vec!["say \"hi\"", "done"]);
}

#[test]
fn test_csv_quote_roundtrip_with_cr() {
    // Round-trip: quote then split should preserve the field
    let original = "has\rcarriage\rreturn";
    let quoted_line = format!("{},{}", csv_quote(original), csv_quote("normal"));
    let fields = csv_split(&quoted_line);
    assert_eq!(fields[0], original);
    assert_eq!(fields[1], "normal");
}

#[test]
fn test_update_readme_duplicate_end_marker() {
    // If END marker appears twice, only the first is used — content between
    // START and first END gets replaced, second END stays in the "after" section
    let readme = format!(
        "before\n{START_MARKER}\nold table\n{END_MARKER}\nmiddle\n{END_MARKER}\nafter",
    );
    let result = update_readme(&readme, "new table").expect("should succeed");
    assert!(result.contains("new table"));
    assert!(result.contains("middle"));
    assert!(result.contains("after"));
    // The old table should be gone
    assert!(!result.contains("old table"));
}

