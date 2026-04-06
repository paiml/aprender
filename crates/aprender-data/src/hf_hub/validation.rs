//! Dataset card validation for HuggingFace Hub.

use crate::error::{Error, Result};

/// Valid HuggingFace task categories as of 2024.
/// Source: https://huggingface.co/docs/hub/datasets-cards#task-categories
pub const VALID_TASK_CATEGORIES: &[&str] = &[
    // NLP
    "text-classification",
    "token-classification",
    "table-question-answering",
    "question-answering",
    "zero-shot-classification",
    "translation",
    "summarization",
    "feature-extraction",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "sentence-similarity",
    "text-to-speech",
    "text-to-audio",
    "automatic-speech-recognition",
    "audio-to-audio",
    "audio-classification",
    "voice-activity-detection",
    // Computer Vision
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    "image-to-text",
    "image-to-image",
    "image-to-video",
    "unconditional-image-generation",
    "video-classification",
    "reinforcement-learning",
    "robotics",
    "tabular-classification",
    "tabular-regression",
    // Multimodal
    "visual-question-answering",
    "document-question-answering",
    "zero-shot-image-classification",
    "graph-ml",
    "mask-generation",
    "zero-shot-object-detection",
    "text-to-3d",
    "image-to-3d",
    "image-feature-extraction",
    // Other
    "other",
];

/// Valid HuggingFace size categories.
pub const VALID_SIZE_CATEGORIES: &[&str] = &[
    "n<1K",
    "1K<n<10K",
    "10K<n<100K",
    "100K<n<1M",
    "1M<n<10M",
    "10M<n<100M",
    "100M<n<1B",
    "1B<n<10B",
    "10B<n<100B",
    "100B<n<1T",
    "n>1T",
];

/// Common valid SPDX license identifiers
pub const VALID_LICENSES: &[&str] = &[
    "apache-2.0",
    "mit",
    "gpl-3.0",
    "gpl-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "cc0-1.0",
    "unlicense",
    "openrail",
    "openrail++",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
    "llama2",
    "llama3",
    "llama3.1",
    "gemma",
    "other",
];

/// Validation error for dataset card metadata.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// The field that has an invalid value
    pub field: String,
    /// The invalid value
    pub value: String,
    /// Suggested valid values (if applicable)
    pub suggestions: Vec<String>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid '{}': '{}' is not valid", self.field, self.value)?;
        if !self.suggestions.is_empty() {
            write!(f, ". Did you mean: {}?", self.suggestions.join(", "))?;
        }
        Ok(())
    }
}

/// Validator for HuggingFace dataset card YAML metadata.
///
/// Validates common fields against HuggingFace's official accepted values.
///
/// # Example
///
/// ```
/// use alimentar::hf_hub::DatasetCardValidator;
///
/// let readme = r"---
/// license: mit
/// task_categories:
///   - translation
/// ---
/// # My Dataset
/// ";
///
/// let errors = DatasetCardValidator::validate_readme(readme);
/// assert!(errors.is_empty());
/// ```
#[derive(Debug, Default)]
pub struct DatasetCardValidator;

impl DatasetCardValidator {
    /// Validates a README.md content and returns any validation errors.
    ///
    /// Parses the YAML frontmatter and validates:
    /// - `task_categories`: Must be from the official HuggingFace list
    /// - `size_categories`: Must match the HuggingFace format
    pub fn validate_readme(content: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Extract YAML frontmatter (between --- markers)
        let Some(yaml_content) = Self::extract_frontmatter(content) else {
            return errors;
        };

        // Parse YAML
        let Ok(yaml) = serde_yaml::from_str::<serde_yaml::Value>(&yaml_content) else {
            return errors;
        };

        // Validate task_categories
        if let Some(categories) = yaml.get("task_categories") {
            if let Some(arr) = categories.as_sequence() {
                for cat in arr {
                    if let Some(cat_str) = cat.as_str() {
                        if !VALID_TASK_CATEGORIES.contains(&cat_str) {
                            errors.push(ValidationError {
                                field: "task_categories".to_string(),
                                value: cat_str.to_string(),
                                suggestions: Self::suggest_similar(cat_str, VALID_TASK_CATEGORIES),
                            });
                        }
                    }
                }
            }
        }

        // Validate size_categories
        if let Some(sizes) = yaml.get("size_categories") {
            if let Some(arr) = sizes.as_sequence() {
                for size in arr {
                    if let Some(size_str) = size.as_str() {
                        if !VALID_SIZE_CATEGORIES.contains(&size_str) {
                            errors.push(ValidationError {
                                field: "size_categories".to_string(),
                                value: size_str.to_string(),
                                suggestions: Self::suggest_similar(size_str, VALID_SIZE_CATEGORIES),
                            });
                        }
                    }
                }
            }
        }

        errors
    }

    /// Validates a README file and returns a Result.
    ///
    /// Returns Ok(()) if valid, or Err with combined error messages.
    pub fn validate_readme_strict(content: &str) -> Result<()> {
        let errors = Self::validate_readme(content);
        if errors.is_empty() {
            Ok(())
        } else {
            let msg = errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            Err(Error::invalid_config(msg))
        }
    }

    /// Extracts YAML frontmatter from markdown content.
    fn extract_frontmatter(content: &str) -> Option<String> {
        let content = content.trim_start();
        if !content.starts_with("---") {
            return None;
        }

        let rest = &content[3..];
        let end_idx = rest.find("\n---")?;
        Some(rest[..end_idx].to_string())
    }

    /// Suggests similar valid values using simple substring matching.
    pub(crate) fn suggest_similar(value: &str, valid: &[&str]) -> Vec<String> {
        let value_lower = value.to_lowercase();
        valid
            .iter()
            .filter(|v| {
                let v_lower = v.to_lowercase();
                v_lower.contains(&value_lower)
                    || value_lower.contains(&v_lower)
                    || Self::levenshtein(&value_lower, &v_lower) <= 3
            })
            .take(3)
            .map(|s| (*s).to_string())
            .collect()
    }

    /// Simple Levenshtein distance for fuzzy matching.
    pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        let mut dp = vec![vec![0; n + 1]; m + 1];

        for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
            row[0] = i;
        }
        for (j, cell) in dp[0].iter_mut().enumerate().take(n + 1) {
            *cell = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[m][n]
    }

    /// Check if a task category is valid.
    #[must_use]
    pub fn is_valid_task_category(category: &str) -> bool {
        VALID_TASK_CATEGORIES.contains(&category)
    }

    /// Check if a license is valid (case-insensitive).
    #[must_use]
    pub fn is_valid_license(license: &str) -> bool {
        let lower = license.to_lowercase();
        VALID_LICENSES.contains(&lower.as_str())
    }

    /// Check if a size category is valid.
    #[must_use]
    pub fn is_valid_size_category(size: &str) -> bool {
        VALID_SIZE_CATEGORIES.contains(&size)
    }

    /// Suggest a similar valid task category for common mistakes.
    #[must_use]
    pub fn suggest_task_category(invalid: &str) -> Option<&'static str> {
        match invalid {
            "text2text-generation" => Some("text2text-generation"), // This is actually valid
            "code-generation" | "code" => Some("text-generation"),
            "qa" | "QA" => Some("question-answering"),
            "ner" | "NER" => Some("token-classification"),
            "sentiment" => Some("text-classification"),
            _ => VALID_TASK_CATEGORIES
                .iter()
                .find(|c| c.starts_with(invalid) || invalid.starts_with(*c))
                .copied(),
        }
    }
}
