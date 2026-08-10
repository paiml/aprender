use super::*;

// ==================== Magic Byte Detection ====================

#[test]
fn test_is_valid_magic_aprn() {
    assert!(is_valid_magic(&MAGIC_APRN));
}

#[test]
fn test_is_valid_magic_apr1() {
    assert!(is_valid_magic(&MAGIC_APR1));
}

#[test]
fn test_is_valid_magic_apr2() {
    assert!(is_valid_magic(&MAGIC_APR2));
}

#[test]
fn test_is_valid_magic_apr0() {
    assert!(is_valid_magic(&MAGIC_APR0));
}

#[test]
fn test_is_valid_magic_gguf() {
    assert!(is_valid_magic(&MAGIC_GGUF));
}

#[test]
fn test_is_valid_magic_unknown() {
    assert!(!is_valid_magic(&[0x00, 0x00, 0x00, 0x00]));
}

#[test]
fn test_is_valid_magic_too_short() {
    assert!(!is_valid_magic(&[0x41, 0x50]));
}

#[test]
fn test_is_valid_magic_empty() {
    assert!(!is_valid_magic(&[]));
}

#[test]
fn test_is_valid_magic_extra_bytes_ignored() {
    let mut bytes = MAGIC_GGUF.to_vec();
    bytes.extend_from_slice(&[0xFF, 0xFF]);
    assert!(is_valid_magic(&bytes));
}

// ==================== Format Name ====================

#[test]
fn test_format_name_aprn() {
    assert_eq!(format_name(&MAGIC_APRN), "APRN (aprender v1)");
}

#[test]
fn test_format_name_apr1() {
    assert_eq!(format_name(&MAGIC_APR1), "APR1 (whisper.apr)");
}

#[test]
fn test_format_name_apr2() {
    assert_eq!(format_name(&MAGIC_APR2), "APR2 (aprender v2)");
}

#[test]
fn test_format_name_apr0() {
    assert_eq!(format_name(&MAGIC_APR0), "APR v2 (ONE TRUE format)");
}

#[test]
fn test_format_name_gguf() {
    assert_eq!(format_name(&MAGIC_GGUF), "GGUF (llama.cpp)");
}

#[test]
fn test_format_name_unknown() {
    assert_eq!(format_name(&[0x00, 0x00, 0x00, 0x00]), "Unknown");
}

#[test]
fn test_format_name_too_short() {
    assert_eq!(format_name(&[0x41]), "Unknown");
}

#[test]
fn test_format_name_empty() {
    assert_eq!(format_name(&[]), "Unknown");
}

// ==================== Output Formatting (no-panic) ====================

#[test]
fn test_section_does_not_panic() {
    section("Test Section");
}

#[test]
fn test_kv_does_not_panic() {
    kv("key", "value");
}

#[test]
fn test_kv_with_number() {
    kv("count", 42);
}

#[test]
fn test_success_does_not_panic() {
    success("operation completed");
}

#[test]
fn test_warning_does_not_panic() {
    warning("something may be wrong");
}

#[test]
fn test_fail_does_not_panic() {
    fail("operation failed");
}

#[test]
fn test_info_does_not_panic() {
    info("informational message");
}

#[test]
fn test_error_does_not_panic() {
    error("error message");
}

#[test]
fn test_warn_alias_does_not_panic() {
    warn("warning via alias");
}

// ==================== Format Size ====================

#[test]
fn test_format_size_zero() {
    let s = format_size(0);
    assert!(s.contains('0'));
}

#[test]
fn test_format_size_bytes() {
    let s = format_size(512);
    assert!(s.contains("512"));
}

#[test]
fn test_format_size_kib() {
    let s = format_size(1024);
    assert!(s.contains("KiB") || s.contains("1"));
}

#[test]
fn test_format_size_mib() {
    let s = format_size(1024 * 1024);
    assert!(s.contains("MiB") || s.contains("1"));
}

#[test]
fn test_format_size_gib() {
    let s = format_size(1024 * 1024 * 1024);
    assert!(s.contains("GiB") || s.contains("1"));
}

// ==================== Magic Byte Constants ====================

#[test]
fn test_magic_constants_are_4_bytes() {
    assert_eq!(MAGIC_APRN.len(), 4);
    assert_eq!(MAGIC_APR1.len(), 4);
    assert_eq!(MAGIC_APR2.len(), 4);
    assert_eq!(MAGIC_APR0.len(), 4);
    assert_eq!(MAGIC_GGUF.len(), 4);
}

#[test]
fn test_magic_aprn_is_ascii_aprn() {
    assert_eq!(&MAGIC_APRN, b"APRN");
}

#[test]
fn test_magic_apr1_is_ascii_apr1() {
    assert_eq!(&MAGIC_APR1, b"APR1");
}

#[test]
fn test_magic_apr2_is_ascii_apr2() {
    assert_eq!(&MAGIC_APR2, b"APR2");
}

#[test]
fn test_magic_gguf_is_ascii_gguf() {
    assert_eq!(&MAGIC_GGUF, b"GGUF");
}

// ==================== Rich Output Primitives ====================

#[test]
fn test_header_does_not_panic() {
    header("Test Header");
}

#[test]
fn test_subheader_does_not_panic() {
    subheader("Test Subheader");
}

#[test]
fn test_badge_pass() {
    let s = badge_pass("OK");
    assert!(s.contains("OK"));
}

#[test]
fn test_badge_fail() {
    let s = badge_fail("ERROR");
    assert!(s.contains("ERROR"));
}

#[test]
fn test_badge_warn() {
    let s = badge_warn("WARNING");
    assert!(s.contains("WARNING"));
}

#[test]
fn test_badge_skip() {
    let s = badge_skip("SKIPPED");
    assert!(s.contains("SKIPPED"));
}

#[test]
fn test_badge_info() {
    let s = badge_info("INFO");
    assert!(s.contains("INFO"));
}

#[test]
fn test_metric_does_not_panic() {
    metric("Throughput", 42.5, "tok/s");
}

#[test]
fn test_metric_empty_unit() {
    metric("Count", 100, "");
}

#[test]
fn test_progress_bar_full() {
    let bar = progress_bar(10, 10, 20);
    assert_eq!(bar.chars().count(), 22); // 20 fill chars + 2 brackets
    assert!(bar.contains("█"));
}

#[test]
fn test_progress_bar_empty() {
    let bar = progress_bar(0, 10, 20);
    assert!(bar.contains("░"));
    assert!(!bar.contains("█"));
}

#[test]
fn test_progress_bar_half() {
    let bar = progress_bar(5, 10, 20);
    assert!(bar.contains("█"));
    assert!(bar.contains("░"));
}

#[test]
fn test_progress_bar_zero_total() {
    let bar = progress_bar(0, 0, 10);
    assert_eq!(bar, "[          ]");
}

#[test]
fn test_duration_fmt_milliseconds() {
    assert_eq!(duration_fmt(42), "42ms");
    assert_eq!(duration_fmt(999), "999ms");
}

#[test]
fn test_duration_fmt_seconds() {
    assert_eq!(duration_fmt(1_000), "1.0s");
    assert_eq!(duration_fmt(1_500), "1.5s");
    assert_eq!(duration_fmt(59_999), "60.0s");
}

#[test]
fn test_duration_fmt_minutes() {
    assert_eq!(duration_fmt(60_000), "1.0m");
    assert_eq!(duration_fmt(90_000), "1.5m");
}

#[test]
fn test_count_fmt_small() {
    assert_eq!(count_fmt(0), "0");
    assert_eq!(count_fmt(42), "42");
    assert_eq!(count_fmt(999), "999");
}

#[test]
fn test_count_fmt_thousands() {
    assert_eq!(count_fmt(1_000), "1,000");
    assert_eq!(count_fmt(1_234), "1,234");
    assert_eq!(count_fmt(12_345), "12,345");
    assert_eq!(count_fmt(123_456), "123,456");
    assert_eq!(count_fmt(1_234_567), "1,234,567");
}

#[test]
fn test_dim_does_not_panic() {
    let _ = dim("text");
}

#[test]
fn test_highlight_does_not_panic() {
    let _ = highlight("text");
}

#[test]
fn test_filepath_does_not_panic() {
    let _ = filepath(std::path::Path::new("/tmp/model.apr"));
}

#[test]
fn test_dtype_color_f32() {
    let s = dtype_color("F32");
    assert!(s.to_string().contains("F32"));
}

#[test]
fn test_dtype_color_f16() {
    let s = dtype_color("F16");
    assert!(s.to_string().contains("F16"));
}

#[test]
fn test_dtype_color_q4k() {
    let s = dtype_color("Q4_K");
    assert!(s.to_string().contains("Q4_K"));
}

#[test]
fn test_dtype_color_q6k() {
    let s = dtype_color("Q6_K");
    assert!(s.to_string().contains("Q6_K"));
}

#[test]
fn test_dtype_color_q8_0() {
    let s = dtype_color("Q8_0");
    assert!(s.to_string().contains("Q8_0"));
}

#[test]
fn test_dtype_color_unknown() {
    let s = dtype_color("INT8");
    assert!(s.to_string().contains("INT8"));
}

#[test]
fn test_grade_color_all_grades() {
    for grade in &["A+", "A", "B+", "B", "C+", "C", "D", "F"] {
        let s = grade_color(grade);
        assert!(s.to_string().contains(grade));
    }
}

#[test]
fn test_table_empty_rows() {
    let result = table(&["A", "B"], &[]);
    assert!(result.is_empty());
}

#[test]
fn test_table_with_data() {
    let rows = vec![
        vec!["hello".to_string(), "world".to_string()],
        vec!["foo".to_string(), "bar".to_string()],
    ];
    let result = table(&["Col1", "Col2"], &rows);
    assert!(result.contains("Col1"));
    assert!(result.contains("hello"));
    assert!(result.contains("bar"));
}

#[test]
fn test_kv_table_empty() {
    let result = kv_table(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_kv_table_with_data() {
    let pairs = vec![
        ("Format", "APR v2".to_string()),
        ("Size", "4.2 GiB".to_string()),
    ];
    let result = kv_table(&pairs);
    assert!(result.contains("Format"));
    assert!(result.contains("APR v2"));
    assert!(result.contains("Size"));
}

#[test]
fn test_pipeline_stage_all_statuses() {
    pipeline_stage("Load", StageStatus::Pending);
    pipeline_stage("Parse", StageStatus::Running);
    pipeline_stage("Validate", StageStatus::Done);
    pipeline_stage("Export", StageStatus::Failed);
    pipeline_stage("Optional", StageStatus::Skipped);
}

// ==================== #2392 finding 5: unterminated single-row kv_table ======

/// Does the rendered table close with a bottom border?
fn closes_the_box(rendered: &str) -> bool {
    rendered
        .lines()
        .last()
        .is_some_and(|last| last.starts_with('╰') && last.ends_with('╯'))
}

/// #2392 finding 5: `apr convert model.safetensors --quantize int8` prints a
/// single-pair config table, and it rendered as
///
/// ```text
/// ╭──────────────┬──────╮
/// │ Quantization │ Int8 │
/// ├──────────────┼──────┤
/// ```
///
/// — top border, one row, a header separator, then nothing. The box never
/// closed. It reproduced for all four quantization schemes and at every
/// single-pair `kv_table` call site in the CLI; adding `--compress` gave a
/// second row and the box closed, which is what identified row count (not
/// content) as the discriminator.
#[test]
fn kv_table_single_pair_closes_its_box() {
    let rendered = kv_table(&[("Quantization", "Int8".to_string())]);
    assert!(
        closes_the_box(&rendered),
        "#2392 finding 5: a one-pair kv_table must terminate with ╰───┴───╯.\nGot:\n{rendered}"
    );
}

/// The same must hold at every row count — the shipped behaviour was correct
/// for 2+ pairs only by accident of the first pair being drawn as a header.
#[test]
fn kv_table_closes_its_box_at_every_row_count() {
    let all: Vec<(&str, String)> = vec![
        ("Input", "model.safetensors".to_string()),
        ("Output", "model.apr".to_string()),
        ("Quantization", "Int8".to_string()),
        ("Compression", "ZstdDefault".to_string()),
    ];
    for n in 1..=all.len() {
        let rendered = kv_table(&all[..n]);
        assert!(
            closes_the_box(&rendered),
            "#2392 finding 5: kv_table with {n} pair(s) left the box open.\nGot:\n{rendered}"
        );
    }
}

/// A key-value table has no header, so it must not draw a header separator that
/// implies the first pair is one. This is the specific rule the fix installs;
/// without it the single-pair case cannot close.
#[test]
fn kv_table_draws_no_header_separator() {
    let rendered = kv_table(&[
        ("Quantization", "None (copy)".to_string()),
        ("Compression", "ZstdDefault".to_string()),
    ]);
    assert!(
        !rendered.contains('├'),
        "#2392 finding 5: kv_table pairs are all data — none is a header.\nGot:\n{rendered}"
    );
}

/// Guard the honest negative: `table()` DOES have a real header row and must
/// keep its separator. A blanket `remove_horizontals()` would break it.
#[test]
fn table_with_real_headers_keeps_its_header_separator() {
    let rows = vec![vec!["a".to_string(), "b".to_string()]];
    let rendered = table(&["Col1", "Col2"], &rows);
    assert!(
        rendered.contains('├'),
        "a headed table must still separate its header from the body.\nGot:\n{rendered}"
    );
    assert!(
        closes_the_box(&rendered),
        "and must still close.\nGot:\n{rendered}"
    );
}
