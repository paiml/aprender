//! Tests for ContentType and CourseLevel

use crate::content::*;
use std::str::FromStr;

// ============================================================================
// ContentType Tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_CONTENT_001_content_type_codes() {
    assert_eq!(ContentType::HighLevelOutline.code(), "HLO");
    assert_eq!(ContentType::DetailedOutline.code(), "DLO");
    assert_eq!(ContentType::BookChapter.code(), "BCH");
    assert_eq!(ContentType::BlogPost.code(), "BLP");
    assert_eq!(ContentType::PresentarDemo.code(), "PDM");
}

#[test]
#[allow(non_snake_case)]
fn test_CONTENT_002_content_type_names() {
    assert_eq!(ContentType::HighLevelOutline.name(), "High-Level Outline");
    assert_eq!(ContentType::DetailedOutline.name(), "Detailed Outline");
    assert_eq!(ContentType::BookChapter.name(), "Book Chapter");
    assert_eq!(ContentType::BlogPost.name(), "Blog Post");
    assert_eq!(ContentType::PresentarDemo.name(), "Presentar Demo");
}

#[test]
#[allow(non_snake_case)]
fn test_CONTENT_003_content_type_from_str() {
    assert_eq!(ContentType::from_str("hlo").unwrap(), ContentType::HighLevelOutline);
    assert_eq!(ContentType::from_str("DLO").unwrap(), ContentType::DetailedOutline);
    assert_eq!(ContentType::from_str("book-chapter").unwrap(), ContentType::BookChapter);
    assert_eq!(ContentType::from_str("blog").unwrap(), ContentType::BlogPost);
    assert_eq!(ContentType::from_str("demo").unwrap(), ContentType::PresentarDemo);
}

#[test]
#[allow(non_snake_case)]
fn test_CONTENT_004_content_type_from_str_invalid() {
    let result = ContentType::from_str("invalid");
    assert!(result.is_err());
    assert!(matches!(result, Err(ContentError::InvalidContentType(_))));
}

#[test]
#[allow(non_snake_case)]
fn test_CONTENT_005_content_type_output_formats() {
    assert_eq!(ContentType::HighLevelOutline.output_format(), "YAML/Markdown");
    assert_eq!(ContentType::BookChapter.output_format(), "Markdown (mdBook)");
    assert_eq!(ContentType::BlogPost.output_format(), "Markdown + TOML");
    assert_eq!(ContentType::PresentarDemo.output_format(), "HTML + YAML");
}

#[test]
#[allow(non_snake_case)]
fn test_CONTENT_006_content_type_all() {
    let all = ContentType::all();
    assert_eq!(all.len(), 5);
    assert!(all.contains(&ContentType::HighLevelOutline));
    assert!(all.contains(&ContentType::PresentarDemo));
}

#[test]
#[allow(non_snake_case)]
fn test_CONTENT_007_content_type_target_length() {
    assert_eq!(ContentType::HighLevelOutline.target_length(), 50..200);
    assert_eq!(ContentType::BookChapter.target_length(), 2000..8000);
    assert_eq!(ContentType::BlogPost.target_length(), 500..3000);
}

// ============================================================================
// CourseLevel Tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_001_course_level_short_config() {
    let level = CourseLevel::Short;
    assert_eq!(level.weeks(), 1);
    assert_eq!(level.modules(), 2);
    assert_eq!(level.videos_per_module(), 3);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_002_course_level_standard_config() {
    let level = CourseLevel::Standard;
    assert_eq!(level.weeks(), 3);
    assert_eq!(level.modules(), 3);
    assert_eq!(level.videos_per_module(), 5);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_003_course_level_extended_config() {
    let level = CourseLevel::Extended;
    assert_eq!(level.weeks(), 6);
    assert_eq!(level.modules(), 6);
    assert_eq!(level.videos_per_module(), 5);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_004_course_level_custom_config() {
    let level = CourseLevel::Custom { weeks: 4, modules: 8, videos_per_module: 4 };
    assert_eq!(level.weeks(), 4);
    assert_eq!(level.modules(), 8);
    assert_eq!(level.videos_per_module(), 4);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_005_course_level_from_str_short() {
    let level: CourseLevel = "short".parse().unwrap();
    assert_eq!(level, CourseLevel::Short);

    let level: CourseLevel = "s".parse().unwrap();
    assert_eq!(level, CourseLevel::Short);

    let level: CourseLevel = "1".parse().unwrap();
    assert_eq!(level, CourseLevel::Short);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_006_course_level_from_str_standard() {
    let level: CourseLevel = "standard".parse().unwrap();
    assert_eq!(level, CourseLevel::Standard);

    let level: CourseLevel = "std".parse().unwrap();
    assert_eq!(level, CourseLevel::Standard);

    let level: CourseLevel = "3".parse().unwrap();
    assert_eq!(level, CourseLevel::Standard);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_007_course_level_from_str_extended() {
    let level: CourseLevel = "extended".parse().unwrap();
    assert_eq!(level, CourseLevel::Extended);

    let level: CourseLevel = "ext".parse().unwrap();
    assert_eq!(level, CourseLevel::Extended);

    let level: CourseLevel = "6".parse().unwrap();
    assert_eq!(level, CourseLevel::Extended);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_008_course_level_from_str_case_insensitive() {
    let level: CourseLevel = "SHORT".parse().unwrap();
    assert_eq!(level, CourseLevel::Short);

    let level: CourseLevel = "Standard".parse().unwrap();
    assert_eq!(level, CourseLevel::Standard);

    let level: CourseLevel = "EXTENDED".parse().unwrap();
    assert_eq!(level, CourseLevel::Extended);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_009_course_level_from_str_invalid() {
    let result: Result<CourseLevel, _> = "invalid".parse();
    assert!(result.is_err());
    assert!(matches!(result, Err(ContentError::InvalidContentType(_))));
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_010_course_level_default() {
    let level = CourseLevel::default();
    assert_eq!(level, CourseLevel::Standard);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_018_course_level_equality() {
    assert_eq!(CourseLevel::Short, CourseLevel::Short);
    assert_eq!(CourseLevel::Standard, CourseLevel::Standard);
    assert_eq!(CourseLevel::Extended, CourseLevel::Extended);
    assert_ne!(CourseLevel::Short, CourseLevel::Standard);

    let custom1 = CourseLevel::Custom { weeks: 4, modules: 4, videos_per_module: 4 };
    let custom2 = CourseLevel::Custom { weeks: 4, modules: 4, videos_per_module: 4 };
    assert_eq!(custom1, custom2);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_019_course_level_debug() {
    let level = CourseLevel::Standard;
    let debug_str = format!("{:?}", level);
    assert!(debug_str.contains("Standard"));
}

#[test]
fn test_level_020_course_level_clone() {
    let level = CourseLevel::Extended;
    let cloned = level; // Copy instead of clone since CourseLevel is Copy
    assert_eq!(level, cloned);
}
