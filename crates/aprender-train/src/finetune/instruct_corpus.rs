//! Instruction-following corpus loader for generative fine-tuning (GH-371)
//!
//! Loads JSONL files with `{"instruction": "...", "response": "..."}` format
//! for causal language model fine-tuning.
//!
//! # Contract
//!
//! - F-INST-001: Each sample must have non-empty instruction and response
//! - F-INST-002: Total token count (prompt + response) must fit max_seq_len

use serde::Deserialize;
use std::path::Path;

/// A single instruction-response training sample.
#[derive(Debug, Clone, Deserialize)]
pub struct InstructSample {
    /// The instruction/prompt text
    pub instruction: String,
    /// The expected response/completion
    pub response: String,
    /// Optional system prompt override
    #[serde(default)]
    pub system: Option<String>,
    /// Optional metadata (source corpus, complexity, etc.)
    #[serde(default)]
    pub metadata: Option<InstructMetadata>,
}

/// Optional metadata for an instruction sample.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct InstructMetadata {
    /// Source corpus name
    #[serde(default)]
    pub source: Option<String>,
    /// Libraries used in the response
    #[serde(default)]
    pub libraries: Vec<String>,
    /// Estimated complexity (1-10)
    #[serde(default)]
    pub complexity: Option<u32>,
}

/// Format an instruction sample as a Qwen chat prompt.
///
/// Uses the `<|im_start|>` / `<|im_end|>` template that Qwen2.5 models expect.
///
/// Returns (prompt_text, response_text) where:
/// - prompt_text includes system + user + assistant prefix
/// - response_text is the completion + `<|im_end|>`
#[must_use]
pub fn format_chat_prompt(sample: &InstructSample) -> (String, String) {
    let system = sample.system.as_deref().unwrap_or(
        "You are a helpful programming assistant. Write clean, correct, well-documented code.",
    );

    let prompt = format!(
        "<|im_start|>system\n{system}<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        sample.instruction
    );

    let response = format!("{}<|im_end|>", sample.response);

    (prompt, response)
}

/// Corpus statistics for instruction samples.
#[derive(Debug, Clone)]
pub struct InstructCorpusStats {
    /// Total number of samples
    pub total: usize,
    /// Average instruction length (chars)
    pub avg_instruction_len: usize,
    /// Average response length (chars)
    pub avg_response_len: usize,
    /// Samples with system prompt override
    pub with_system: usize,
    /// Unique source corpora
    pub sources: Vec<String>,
}

/// Load instruction corpus from JSONL file.
///
/// Each line is `{"instruction": "...", "response": "..."}`.
///
/// # Contract (F-INST-001)
/// All samples must have non-empty instruction and response.
///
/// # Errors
/// Returns error if file cannot be read or contains invalid samples.
pub fn load_instruct_corpus(path: &Path) -> crate::Result<Vec<InstructSample>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| crate::Error::Io(format!("Corpus file not found: {}: {e}", path.display())))?;

    let mut samples = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let sample: InstructSample = serde_json::from_str(line).map_err(|e| {
            crate::Error::ConfigError(format!("Invalid JSONL at line {}: {e}", line_num + 1))
        })?;

        // F-INST-001: non-empty validation
        if sample.instruction.trim().is_empty() {
            return Err(crate::Error::ConfigError(format!(
                "F-INST-001: empty instruction at line {}",
                line_num + 1,
            )));
        }
        if sample.response.trim().is_empty() {
            return Err(crate::Error::ConfigError(format!(
                "F-INST-001: empty response at line {}",
                line_num + 1,
            )));
        }

        samples.push(sample);
    }

    Ok(samples)
}

/// Compute corpus statistics.
pub fn instruct_corpus_stats(samples: &[InstructSample]) -> InstructCorpusStats {
    if samples.is_empty() {
        return InstructCorpusStats {
            total: 0,
            avg_instruction_len: 0,
            avg_response_len: 0,
            with_system: 0,
            sources: Vec::new(),
        };
    }

    let total_inst_len: usize = samples.iter().map(|s| s.instruction.len()).sum();
    let total_resp_len: usize = samples.iter().map(|s| s.response.len()).sum();
    let with_system = samples.iter().filter(|s| s.system.is_some()).count();

    let mut sources: Vec<String> =
        samples.iter().filter_map(|s| s.metadata.as_ref()?.source.clone()).collect();
    sources.sort();
    sources.dedup();

    InstructCorpusStats {
        total: samples.len(),
        avg_instruction_len: total_inst_len / samples.len(),
        avg_response_len: total_resp_len / samples.len(),
        with_system,
        sources,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_instruct_corpus() {
        let mut f = NamedTempFile::new().expect("valid");
        writeln!(
            f,
            r#"{{"instruction": "Write hello world", "response": "print('hello world')"}}"#
        )
        .expect("valid");
        writeln!(f, r#"{{"instruction": "Sort a list", "response": "sorted(lst)"}}"#)
            .expect("valid");

        let samples = load_instruct_corpus(f.path()).expect("valid");
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].instruction, "Write hello world");
        assert_eq!(samples[1].response, "sorted(lst)");
    }

    #[test]
    fn test_empty_instruction_rejected() {
        let mut f = NamedTempFile::new().expect("valid");
        writeln!(f, r#"{{"instruction": "", "response": "some code"}}"#).expect("valid");

        let result = load_instruct_corpus(f.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("F-INST-001"));
    }

    #[test]
    fn test_empty_response_rejected() {
        let mut f = NamedTempFile::new().expect("valid");
        writeln!(f, r#"{{"instruction": "Do something", "response": "  "}}"#).expect("valid");

        let result = load_instruct_corpus(f.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("F-INST-001"));
    }

    #[test]
    fn test_format_chat_prompt() {
        let sample = InstructSample {
            instruction: "Write a sort function".to_string(),
            response: "def sort(lst):\n    return sorted(lst)".to_string(),
            system: None,
            metadata: None,
        };

        let (prompt, response) = format_chat_prompt(&sample);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Write a sort function"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        assert!(response.contains("def sort(lst)"));
        assert!(response.ends_with("<|im_end|>"));
    }

    #[test]
    fn test_format_chat_prompt_custom_system() {
        let sample = InstructSample {
            instruction: "test".to_string(),
            response: "ok".to_string(),
            system: Some("You are a Python expert.".to_string()),
            metadata: None,
        };

        let (prompt, _) = format_chat_prompt(&sample);
        assert!(prompt.contains("You are a Python expert."));
    }

    #[test]
    fn test_instruct_corpus_stats() {
        let samples = vec![
            InstructSample {
                instruction: "hello".to_string(),
                response: "world".to_string(),
                system: Some("sys".to_string()),
                metadata: Some(InstructMetadata {
                    source: Some("test".to_string()),
                    ..Default::default()
                }),
            },
            InstructSample {
                instruction: "foo".to_string(),
                response: "bar".to_string(),
                system: None,
                metadata: None,
            },
        ];

        let stats = instruct_corpus_stats(&samples);
        assert_eq!(stats.total, 2);
        assert_eq!(stats.with_system, 1);
        assert_eq!(stats.sources, vec!["test".to_string()]);
    }

    #[test]
    fn test_skip_empty_lines() {
        let mut f = NamedTempFile::new().expect("valid");
        writeln!(f, r#"{{"instruction": "a", "response": "b"}}"#).expect("valid");
        writeln!(f).expect("valid"); // empty line
        writeln!(f, r#"{{"instruction": "c", "response": "d"}}"#).expect("valid");

        let samples = load_instruct_corpus(f.path()).expect("valid");
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn test_invalid_json_rejected() {
        let mut f = NamedTempFile::new().expect("valid");
        writeln!(f, "not json").expect("valid");

        let result = load_instruct_corpus(f.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_corpus_stats_empty() {
        let stats = instruct_corpus_stats(&[]);
        assert_eq!(stats.total, 0);
        assert_eq!(stats.avg_instruction_len, 0);
    }
}

/// A DPO preference training sample (prompt + chosen + rejected).
/// Contract: dpo-alignment-v1 / preference_data_valid
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PreferenceSample {
    /// The prompt/instruction text
    pub prompt: String,
    /// The preferred (chosen) response
    pub chosen: String,
    /// The rejected response
    pub rejected: String,
}

/// Load preference pairs from JSONL file.
pub fn load_preference_pairs(path: &std::path::Path) -> Result<Vec<PreferenceSample>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Open {}: {e}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut samples = Vec::new();
    for (i, line) in std::io::BufRead::lines(reader).enumerate() {
        let line = line.map_err(|e| format!("Line {i}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let sample: PreferenceSample =
            serde_json::from_str(&line).map_err(|e| format!("Line {i}: {e}"))?;
        // FALSIFY-DPO-002: validate non-empty fields
        if sample.prompt.is_empty() || sample.chosen.is_empty() || sample.rejected.is_empty() {
            return Err(format!("Line {i}: empty prompt/chosen/rejected"));
        }
        samples.push(sample);
    }
    Ok(samples)
}
