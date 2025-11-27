//! Suggestion quality validation and enhancement for aprender-shell
//!
//! Follows Toyota Way principle *Kaizen* (Continuous improvement):
//! Small, incremental improvements in suggestion quality.

/// Known typo corrections (learned from common mistakes)
const CORRECTIONS: &[(&str, &str)] = &[
    ("git stauts", "git status"),
    ("git comit", "git commit"),
    ("git psuh", "git push"),
    ("git pul", "git pull"),
    ("dokcer", "docker"),
    ("kubeclt", "kubectl"),
    ("carg ", "cargo "),
    ("pytohn", "python"),
    ("ndoe", "node"),
    ("yran", "yarn"),
];

/// Known command tools for quality scoring
const KNOWN_TOOLS: &[&str] = &[
    "git",
    "cargo",
    "docker",
    "kubectl",
    "npm",
    "yarn",
    "make",
    "aws",
    "gcloud",
    "az",
    "terraform",
    "ansible",
    "helm",
    "python",
    "node",
    "go",
    "rustc",
    "gcc",
    "clang",
];

/// Validate and clean a suggestion before returning to user.
///
/// Returns `None` if the suggestion is invalid or malformed.
///
/// # Arguments
/// * `suggestion` - The raw suggestion from the model
/// * `prefix` - The user's input prefix
///
/// # Example
/// ```
/// use aprender_shell::quality::validate_suggestion;
///
/// assert!(validate_suggestion("git status", "git").is_some());
/// assert!(validate_suggestion("git", "git").is_none()); // Not longer than prefix
/// assert!(validate_suggestion("git status \\", "git").is_none()); // Trailing backslash
/// ```
pub fn validate_suggestion(suggestion: &str, prefix: &str) -> Option<String> {
    // Must start with the prefix (case-sensitive for commands)
    if !suggestion.starts_with(prefix) {
        return None;
    }

    // Normalize whitespace (collapse multiple spaces)
    let normalized: String = suggestion.split_whitespace().collect::<Vec<_>>().join(" ");

    // Must be longer than prefix (actually suggesting something)
    if normalized.len() <= prefix.trim().len() {
        return None;
    }

    // Reject suggestions with obvious issues
    if normalized.ends_with('\\') || normalized.ends_with('|') || normalized.ends_with('&') {
        return None;
    }

    // Reject double spaces that weren't normalized away
    if normalized.contains("  ") {
        return None;
    }

    // Reject if it contains control characters
    if normalized.chars().any(|c| c.is_control() && c != '\t') {
        return None;
    }

    Some(normalized)
}

/// Apply common typo corrections to a suggestion.
///
/// This helps when the model has learned typos from actual history.
///
/// # Example
/// ```
/// use aprender_shell::quality::apply_typo_corrections;
///
/// assert_eq!(apply_typo_corrections("git stauts"), "git status");
/// assert_eq!(apply_typo_corrections("dokcer ps"), "docker ps");
/// ```
pub fn apply_typo_corrections(suggestion: &str) -> String {
    let mut corrected = suggestion.to_string();
    for (typo, fix) in CORRECTIONS {
        if corrected.contains(typo) {
            corrected = corrected.replace(typo, fix);
        }
    }
    corrected
}

/// Score suggestion quality (0.0 to 1.0).
///
/// Higher scores indicate better quality suggestions.
///
/// # Scoring Factors
/// - Length: Very short suggestions are penalized
/// - Characters: Unusual characters reduce score
/// - Completeness: Trailing spaces/dashes are penalized
/// - Tool recognition: Known tools boost score
///
/// # Example
/// ```
/// use aprender_shell::quality::suggestion_quality_score;
///
/// assert!(suggestion_quality_score("git status") > 0.8);
/// assert!(suggestion_quality_score("git") < 0.6);
/// ```
pub fn suggestion_quality_score(suggestion: &str) -> f32 {
    // Start with base score based on known tool
    let first_word = suggestion.split_whitespace().next().unwrap_or("");
    let mut score = if KNOWN_TOOLS.contains(&first_word) {
        1.0_f32
    } else {
        0.8_f32 // Unknown tools start lower
    };

    // Penalize very short suggestions
    if suggestion.len() < 5 {
        score *= 0.5;
    }

    // Penalize suggestions with unusual characters
    let unusual_char_count = suggestion
        .chars()
        .filter(|c| !c.is_alphanumeric() && !" -_./=:@".contains(*c))
        .count();
    score *= 1.0 - (unusual_char_count as f32 * 0.1).min(0.5);

    // Penalize incomplete-looking commands
    if suggestion.ends_with(' ') || suggestion.ends_with('-') {
        score *= 0.7;
    }

    score.clamp(0.0, 1.0)
}

/// Filter and enhance suggestions with quality checks.
///
/// This function:
/// 1. Validates each suggestion
/// 2. Applies typo corrections
/// 3. Filters by quality score threshold
///
/// # Arguments
/// * `suggestions` - Raw suggestions from the model
/// * `prefix` - The user's input prefix
/// * `min_quality` - Minimum quality score (0.0 to 1.0)
pub fn filter_quality_suggestions(
    suggestions: Vec<(String, f32)>,
    prefix: &str,
    min_quality: f32,
) -> Vec<(String, f32)> {
    suggestions
        .into_iter()
        .filter_map(|(suggestion, model_score)| {
            // Validate the suggestion
            let validated = validate_suggestion(&suggestion, prefix)?;

            // Apply typo corrections
            let corrected = apply_typo_corrections(&validated);

            // Check quality score
            let quality = suggestion_quality_score(&corrected);
            if quality < min_quality {
                return None;
            }

            // Combine model score with quality score
            let combined_score = model_score * quality;

            Some((corrected, combined_score))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Suggestion Validation Tests
    // =========================================================================

    #[test]
    fn test_valid_suggestion_accepted() {
        assert!(validate_suggestion("git status", "git").is_some());
        assert!(validate_suggestion("cargo build --release", "cargo").is_some());
    }

    #[test]
    fn test_suggestion_not_longer_than_prefix() {
        assert!(validate_suggestion("git", "git").is_none());
        assert!(validate_suggestion("git", "git ").is_none());
    }

    #[test]
    fn test_suggestion_must_start_with_prefix() {
        assert!(validate_suggestion("cargo build", "git").is_none());
    }

    #[test]
    fn test_trailing_backslash_rejected() {
        assert!(validate_suggestion("git status \\", "git").is_none());
    }

    #[test]
    fn test_trailing_pipe_rejected() {
        assert!(validate_suggestion("git status |", "git").is_none());
    }

    #[test]
    fn test_trailing_ampersand_rejected() {
        assert!(validate_suggestion("git status &", "git").is_none());
    }

    #[test]
    fn test_whitespace_normalized() {
        let result = validate_suggestion("git  status   -v", "git").unwrap();
        assert_eq!(result, "git status -v");
    }

    #[test]
    fn test_control_chars_rejected() {
        assert!(validate_suggestion("git\x07status", "git").is_none());
    }

    // =========================================================================
    // Typo Correction Tests
    // =========================================================================

    #[test]
    fn test_git_typos_corrected() {
        assert_eq!(apply_typo_corrections("git stauts"), "git status");
        assert_eq!(
            apply_typo_corrections("git comit -m 'test'"),
            "git commit -m 'test'"
        );
        assert_eq!(apply_typo_corrections("git psuh origin"), "git push origin");
    }

    #[test]
    fn test_tool_typos_corrected() {
        assert_eq!(apply_typo_corrections("dokcer ps"), "docker ps");
        assert_eq!(
            apply_typo_corrections("kubeclt get pods"),
            "kubectl get pods"
        );
    }

    #[test]
    fn test_no_false_corrections() {
        // These should not be changed
        assert_eq!(apply_typo_corrections("git status"), "git status");
        assert_eq!(apply_typo_corrections("docker run"), "docker run");
    }

    // =========================================================================
    // Quality Scoring Tests
    // =========================================================================

    #[test]
    fn test_known_tool_higher_score() {
        let git_score = suggestion_quality_score("git status");
        let unknown_score = suggestion_quality_score("xyz status");
        assert!(git_score > unknown_score);
    }

    #[test]
    fn test_short_suggestion_lower_score() {
        let short = suggestion_quality_score("git");
        let long = suggestion_quality_score("git status --verbose");
        assert!(short < long);
    }

    #[test]
    fn test_unusual_chars_lower_score() {
        let normal = suggestion_quality_score("git status");
        let unusual = suggestion_quality_score("git !@#$%");
        assert!(normal > unusual);
    }

    #[test]
    fn test_incomplete_lower_score() {
        let complete = suggestion_quality_score("git status");
        let incomplete = suggestion_quality_score("git status ");
        assert!(complete > incomplete);
    }

    #[test]
    fn test_score_bounded_zero_to_one() {
        assert!(suggestion_quality_score("git status") <= 1.0);
        assert!(suggestion_quality_score("git status") >= 0.0);
        assert!(suggestion_quality_score("!@#$%^&*") <= 1.0);
        assert!(suggestion_quality_score("!@#$%^&*") >= 0.0);
    }

    // =========================================================================
    // Filter Quality Tests
    // =========================================================================

    #[test]
    fn test_filter_quality_suggestions() {
        let suggestions = vec![
            ("git status".to_string(), 0.9),
            ("git stauts".to_string(), 0.8), // typo
            ("git".to_string(), 0.7),        // too short
            ("git commit".to_string(), 0.6),
        ];

        let filtered = filter_quality_suggestions(suggestions, "git", 0.3);

        // Should include status and commit (typo corrected)
        assert!(filtered.iter().any(|(s, _)| s == "git status"));
        assert!(filtered.iter().any(|(s, _)| s == "git commit"));

        // "git" alone should be filtered (not longer than prefix)
        assert!(!filtered.iter().any(|(s, _)| s == "git"));
    }
}
