//! Tests for PromptEmitter and EmitConfig

use crate::content::*;
use std::path::PathBuf;

#[test]
#[allow(non_snake_case)]
fn test_EMIT_001_emitter_new() {
    let emitter = PromptEmitter::new();
    assert!(!emitter.toyota_constraints().is_empty());
    assert!(!emitter.quality_gates().is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_EMIT_002_emit_hlo() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::HighLevelOutline)
        .with_title("Rust for Data Engineers")
        .with_audience("Python developers");
    let prompt = emitter.emit(&config).unwrap();
    assert!(prompt.contains("High-Level Outline"));
    assert!(prompt.contains("Rust for Data Engineers"));
    assert!(prompt.contains("Python developers"));
    assert!(prompt.contains("Toyota Way"));
}

#[test]
#[allow(non_snake_case)]
fn test_EMIT_003_emit_bch() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::BookChapter)
        .with_title("Error Handling")
        .with_word_count(4000);
    let prompt = emitter.emit(&config).unwrap();
    assert!(prompt.contains("Book Chapter"));
    assert!(prompt.contains("4000 words"));
    assert!(prompt.contains("mdBook"));
}

#[test]
#[allow(non_snake_case)]
fn test_EMIT_004_emit_blp() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::BlogPost).with_title("Rust Performance Tips");
    let prompt = emitter.emit(&config).unwrap();
    assert!(prompt.contains("Blog Post"));
    assert!(prompt.contains("TOML Frontmatter"));
    assert!(prompt.contains("SEO"));
}

#[test]
#[allow(non_snake_case)]
fn test_EMIT_005_emit_pdm() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::PresentarDemo).with_title("Shell Autocomplete");
    let prompt = emitter.emit(&config).unwrap();
    assert!(prompt.contains("Presentar Demo"));
    assert!(prompt.contains("wasm")); // lowercase in template
    assert!(prompt.contains("Accessibility"));
}

#[test]
#[allow(non_snake_case)]
fn test_EMIT_006_emit_missing_content_type() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::default();
    let result = emitter.emit(&config);
    assert!(result.is_err());
    assert!(matches!(result, Err(ContentError::MissingRequiredField(_))));
}

#[test]
#[allow(non_snake_case)]
fn test_EMIT_007_emit_with_budget() {
    let emitter = PromptEmitter::new();
    let mut config = EmitConfig::new(ContentType::BookChapter);
    config.show_budget = true;
    config.word_count = Some(4000);
    let prompt = emitter.emit(&config).unwrap();
    assert!(prompt.contains("Token Budget"));
    assert!(prompt.contains("claude-sonnet"));
}

#[test]
#[allow(non_snake_case)]
fn test_EMIT_008_toyota_constraints_content() {
    let emitter = PromptEmitter::new();
    assert!(emitter.toyota_constraints().contains("Jidoka"));
    assert!(emitter.toyota_constraints().contains("Poka-Yoke"));
    assert!(emitter.toyota_constraints().contains("Genchi Genbutsu"));
    assert!(emitter.toyota_constraints().contains("Heijunka"));
    assert!(emitter.toyota_constraints().contains("Kaizen"));
}

#[test]
#[allow(non_snake_case)]
fn test_EMIT_009_quality_gates_content() {
    let emitter = PromptEmitter::new();
    assert!(emitter.quality_gates().contains("Andon"));
    assert!(emitter.quality_gates().contains("meta-commentary"));
    assert!(emitter.quality_gates().contains("code blocks"));
    assert!(emitter.quality_gates().contains("Heading hierarchy"));
}

#[test]
#[allow(non_snake_case)]
fn test_EMIT_010_emit_config_builder() {
    let config = EmitConfig::new(ContentType::BookChapter)
        .with_title("Test")
        .with_audience("Developers")
        .with_word_count(5000)
        .with_source_context(PathBuf::from("/src"));
    assert_eq!(config.title, Some("Test".to_string()));
    assert_eq!(config.audience, Some("Developers".to_string()));
    assert_eq!(config.word_count, Some(5000));
    assert_eq!(config.source_context_paths.len(), 1);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_011_emit_config_with_course_level() {
    let config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Test Course")
        .with_course_level(CourseLevel::Extended);
    assert_eq!(config.course_level, CourseLevel::Extended);
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_012_emit_dlo_short_no_weekly_objectives() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Quick Start")
        .with_course_level(CourseLevel::Short);
    let prompt = emitter.emit(&config).unwrap();

    // Short courses should NOT have weekly learning objectives section
    assert!(!prompt.contains("weeks:\n"));
    // But should have course-level objectives
    assert!(prompt.contains("learning_objectives:"));
    // Should have correct duration
    assert!(prompt.contains("1 week"));
    assert!(prompt.contains("2 modules"));
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_013_emit_dlo_standard_has_weekly_objectives() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Complete Course")
        .with_course_level(CourseLevel::Standard);
    let prompt = emitter.emit(&config).unwrap();

    // Standard courses SHOULD have weekly learning objectives
    assert!(prompt.contains("weeks:\n"));
    assert!(prompt.contains("- week: 1"));
    assert!(prompt.contains("- week: 2"));
    assert!(prompt.contains("- week: 3"));
    // Should have correct duration
    assert!(prompt.contains("3 weeks"));
    assert!(prompt.contains("3 modules"));
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_014_emit_dlo_extended_has_six_weeks() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Masterclass")
        .with_course_level(CourseLevel::Extended);
    let prompt = emitter.emit(&config).unwrap();

    // Extended courses should have 6 weeks
    assert!(prompt.contains("6 weeks"));
    assert!(prompt.contains("6 modules"));
    assert!(prompt.contains("- week: 1"));
    assert!(prompt.contains("- week: 6"));
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_015_emit_dlo_course_description() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Test Course")
        .with_course_level(CourseLevel::Standard);
    let prompt = emitter.emit(&config).unwrap();

    // All courses should have description and learning objectives
    assert!(prompt.contains("description: string"));
    assert!(prompt.contains("learning_objectives:"));

    // Check the course section has 3 learning objectives before weeks section
    // The format is: course: ... learning_objectives: ... weeks:
    let course_start = prompt.find("course:").expect("Should have course section");
    let weeks_start = prompt.find("\nweeks:").expect("Should have weeks section");
    let course_section = &prompt[course_start..weeks_start];
    let objective_count = course_section.matches("- objective: string").count();
    assert_eq!(objective_count, 3, "Course should have 3 learning objectives");
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_016_emit_dlo_structure_requirements() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Test")
        .with_course_level(CourseLevel::Standard);
    let prompt = emitter.emit(&config).unwrap();

    // Check structure requirements are documented
    assert!(prompt.contains("**Duration**"));
    assert!(prompt.contains("**Modules**"));
    assert!(prompt.contains("**Per Module**"));
    assert!(prompt.contains("**Learning Objectives**"));
    assert!(prompt.contains("3 for course, 3 per week"));
}

#[test]
#[allow(non_snake_case)]
fn test_LEVEL_017_emit_dlo_short_learning_objectives_text() {
    let emitter = PromptEmitter::new();
    let config = EmitConfig::new(ContentType::DetailedOutline)
        .with_title("Quick Start")
        .with_course_level(CourseLevel::Short);
    let prompt = emitter.emit(&config).unwrap();

    // Short courses should say "3 for course" without ", 3 per week"
    assert!(prompt.contains("**Learning Objectives**: 3 for course\n"));
}
