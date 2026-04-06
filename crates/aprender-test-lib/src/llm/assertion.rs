//! LLM output assertions: structural validation, content checks, and latency budgets.

use super::client::{ChatResponse, TimedChatResponse};
use std::time::Duration;

/// Result of an LLM assertion check.
#[derive(Debug, Clone)]
pub struct LlmAssertionResult {
    /// Name of the assertion.
    pub name: String,
    /// Whether the assertion passed.
    pub passed: bool,
    /// Human-readable detail on failure.
    pub detail: Option<String>,
}

/// Errors from LLM assertions.
#[derive(Debug, thiserror::Error)]
pub enum LlmAssertionError {
    /// Response structure is invalid.
    #[error("Invalid response structure: {0}")]
    InvalidStructure(String),
    /// Content assertion failed.
    #[error("Content assertion failed: {0}")]
    ContentMismatch(String),
    /// Latency exceeded budget.
    #[error("Latency {actual_ms}ms exceeds budget {budget_ms}ms")]
    LatencyExceeded {
        /// Actual latency in milliseconds.
        actual_ms: u64,
        /// Budget in milliseconds.
        budget_ms: u64,
    },
    /// Regex compilation failed.
    #[error("Invalid regex pattern: {0}")]
    InvalidRegex(String),
}

/// Collection of assertions to run against LLM responses.
#[derive(Debug, Default)]
pub struct LlmAssertion {
    checks: Vec<Box<dyn AssertionCheck>>,
}

trait AssertionCheck: std::fmt::Debug + Send + Sync {
    fn check(&self, response: &TimedChatResponse) -> LlmAssertionResult;
}

// --- Built-in checks ---

#[derive(Debug)]
struct ResponseValidCheck;

impl AssertionCheck for ResponseValidCheck {
    fn check(&self, timed: &TimedChatResponse) -> LlmAssertionResult {
        let r = &timed.response;
        let mut issues = Vec::new();

        if r.id.is_empty() {
            issues.push("missing id");
        }
        if r.choices.is_empty() {
            issues.push("no choices");
        }
        if let Some(choice) = r.choices.first() {
            if choice.message.content.is_empty() {
                issues.push("empty content in first choice");
            }
        }

        if issues.is_empty() {
            LlmAssertionResult {
                name: "response_valid".to_string(),
                passed: true,
                detail: None,
            }
        } else {
            LlmAssertionResult {
                name: "response_valid".to_string(),
                passed: false,
                detail: Some(issues.join(", ")),
            }
        }
    }
}

#[derive(Debug)]
struct ContainsCheck {
    substring: String,
    case_insensitive: bool,
}

impl AssertionCheck for ContainsCheck {
    fn check(&self, timed: &TimedChatResponse) -> LlmAssertionResult {
        let content = first_content(&timed.response);
        let (hay, needle) = if self.case_insensitive {
            (content.to_lowercase(), self.substring.to_lowercase())
        } else {
            (content.clone(), self.substring.clone())
        };

        if hay.contains(&needle) {
            LlmAssertionResult {
                name: "contains".to_string(),
                passed: true,
                detail: None,
            }
        } else {
            LlmAssertionResult {
                name: "contains".to_string(),
                passed: false,
                detail: Some(format!(
                    "expected output to contain {:?}, got: {:?}",
                    self.substring,
                    truncate(&content, 200)
                )),
            }
        }
    }
}

#[derive(Debug)]
struct PatternCheck {
    pattern: String,
}

impl AssertionCheck for PatternCheck {
    fn check(&self, timed: &TimedChatResponse) -> LlmAssertionResult {
        let content = first_content(&timed.response);
        match regex::Regex::new(&self.pattern) {
            Ok(re) => {
                if re.is_match(&content) {
                    LlmAssertionResult {
                        name: "matches_pattern".to_string(),
                        passed: true,
                        detail: None,
                    }
                } else {
                    LlmAssertionResult {
                        name: "matches_pattern".to_string(),
                        passed: false,
                        detail: Some(format!(
                            "pattern {:?} did not match: {:?}",
                            self.pattern,
                            truncate(&content, 200)
                        )),
                    }
                }
            }
            Err(e) => LlmAssertionResult {
                name: "matches_pattern".to_string(),
                passed: false,
                detail: Some(format!("invalid regex: {e}")),
            },
        }
    }
}

#[derive(Debug)]
struct LatencyCheck {
    budget: Duration,
}

impl AssertionCheck for LatencyCheck {
    fn check(&self, timed: &TimedChatResponse) -> LlmAssertionResult {
        if timed.latency <= self.budget {
            LlmAssertionResult {
                name: "latency_under".to_string(),
                passed: true,
                detail: None,
            }
        } else {
            LlmAssertionResult {
                name: "latency_under".to_string(),
                passed: false,
                detail: Some(format!(
                    "latency {}ms exceeds budget {}ms",
                    timed.latency.as_millis(),
                    self.budget.as_millis()
                )),
            }
        }
    }
}

#[derive(Debug)]
struct TokenCountCheck {
    min: Option<u32>,
    max: Option<u32>,
}

impl AssertionCheck for TokenCountCheck {
    fn check(&self, timed: &TimedChatResponse) -> LlmAssertionResult {
        let tokens = timed
            .response
            .usage
            .as_ref()
            .map_or(0, |u| u.completion_tokens);

        let passed =
            self.min.map_or(true, |m| tokens >= m) && self.max.map_or(true, |m| tokens <= m);

        if passed {
            LlmAssertionResult {
                name: "token_count".to_string(),
                passed: true,
                detail: None,
            }
        } else {
            LlmAssertionResult {
                name: "token_count".to_string(),
                passed: false,
                detail: Some(format!(
                    "completion_tokens={tokens}, expected range [{}, {}]",
                    self.min.map_or("*".to_string(), |m| m.to_string()),
                    self.max.map_or("*".to_string(), |m| m.to_string()),
                )),
            }
        }
    }
}

impl LlmAssertion {
    /// Create a new empty assertion builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Assert that the response has valid structure (id, choices, non-empty content).
    pub fn assert_response_valid(mut self) -> Self {
        self.checks.push(Box::new(ResponseValidCheck));
        self
    }

    /// Assert the first choice's content contains the given substring.
    pub fn assert_contains(mut self, substring: impl Into<String>) -> Self {
        self.checks.push(Box::new(ContainsCheck {
            substring: substring.into(),
            case_insensitive: false,
        }));
        self
    }

    /// Assert the first choice's content contains the given substring (case-insensitive).
    pub fn assert_contains_ignore_case(mut self, substring: impl Into<String>) -> Self {
        self.checks.push(Box::new(ContainsCheck {
            substring: substring.into(),
            case_insensitive: true,
        }));
        self
    }

    /// Assert the first choice's content matches the given regex pattern.
    pub fn assert_matches_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.checks.push(Box::new(PatternCheck {
            pattern: pattern.into(),
        }));
        self
    }

    /// Assert total latency is under the given duration.
    pub fn assert_latency_under(mut self, budget: Duration) -> Self {
        self.checks.push(Box::new(LatencyCheck { budget }));
        self
    }

    /// Assert completion token count is within the given range.
    pub fn assert_token_count(mut self, min: Option<u32>, max: Option<u32>) -> Self {
        self.checks.push(Box::new(TokenCountCheck { min, max }));
        self
    }

    /// Run all assertions against a timed response, returning results for each.
    pub fn run(&self, response: &TimedChatResponse) -> Vec<LlmAssertionResult> {
        self.checks.iter().map(|c| c.check(response)).collect()
    }

    /// Run all assertions and return true only if all passed.
    pub fn run_all_pass(&self, response: &TimedChatResponse) -> bool {
        self.run(response).iter().all(|r| r.passed)
    }
}

/// Extract the content string from the first choice, or empty string.
fn first_content(response: &ChatResponse) -> String {
    response
        .choices
        .first()
        .map_or_else(String::new, |c| c.message.content.clone())
}

/// Truncate a string for display purposes.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

/// Check determinism: given multiple responses to the same prompt (temp=0),
/// verify they all produce the same output.
pub fn assert_deterministic(responses: &[ChatResponse]) -> LlmAssertionResult {
    if responses.len() < 2 {
        return LlmAssertionResult {
            name: "deterministic".to_string(),
            passed: true,
            detail: Some("fewer than 2 responses, vacuously true".to_string()),
        };
    }

    let first = first_content(&responses[0]);
    for (i, resp) in responses.iter().enumerate().skip(1) {
        let content = first_content(resp);
        if content != first {
            return LlmAssertionResult {
                name: "deterministic".to_string(),
                passed: false,
                detail: Some(format!(
                    "response[0] != response[{i}]: {:?} vs {:?}",
                    truncate(&first, 100),
                    truncate(&content, 100)
                )),
            };
        }
    }

    LlmAssertionResult {
        name: "deterministic".to_string(),
        passed: true,
        detail: None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::client::*;
    use super::*;

    fn make_timed(content: &str, latency_ms: u64) -> TimedChatResponse {
        TimedChatResponse {
            response: ChatResponse {
                id: "test-id".to_string(),
                object: "chat.completion".to_string(),
                created: 1_700_000_000,
                model: "test-model".to_string(),
                choices: vec![ChatResponseChoice {
                    index: 0,
                    message: ChatMessage {
                        role: Role::Assistant,
                        content: content.to_string(),
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 20,
                    total_tokens: 30,
                }),
                brick_trace: None,
            },
            latency: Duration::from_millis(latency_ms),
            ttfb: Duration::from_millis(latency_ms / 2),
            brick_trace: None,
        }
    }

    fn make_response(content: &str) -> ChatResponse {
        ChatResponse {
            id: "test".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatResponseChoice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: content.to_string(),
                },
                finish_reason: None,
            }],
            usage: None,
            brick_trace: None,
        }
    }

    #[test]
    fn test_response_valid_pass() {
        let timed = make_timed("Hello!", 100);
        let results = LlmAssertion::new().assert_response_valid().run(&timed);
        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
    }

    #[test]
    fn test_response_valid_empty_id() {
        let mut timed = make_timed("Hi", 100);
        timed.response.id = String::new();
        let results = LlmAssertion::new().assert_response_valid().run(&timed);
        assert!(!results[0].passed);
        assert!(results[0].detail.as_ref().unwrap().contains("missing id"));
    }

    #[test]
    fn test_response_valid_no_choices() {
        let mut timed = make_timed("Hi", 100);
        timed.response.choices.clear();
        let results = LlmAssertion::new().assert_response_valid().run(&timed);
        assert!(!results[0].passed);
        assert!(results[0].detail.as_ref().unwrap().contains("no choices"));
    }

    #[test]
    fn test_contains_pass() {
        let timed = make_timed("The answer is 56.", 100);
        let results = LlmAssertion::new().assert_contains("56").run(&timed);
        assert!(results[0].passed);
    }

    #[test]
    fn test_contains_fail() {
        let timed = make_timed("The answer is 42.", 100);
        let results = LlmAssertion::new().assert_contains("56").run(&timed);
        assert!(!results[0].passed);
    }

    #[test]
    fn test_contains_ignore_case() {
        let timed = make_timed("def Fibonacci(n):", 100);
        let results = LlmAssertion::new()
            .assert_contains_ignore_case("def fibonacci")
            .run(&timed);
        assert!(results[0].passed);
    }

    #[test]
    fn test_matches_pattern_pass() {
        let timed = make_timed("fn main() { println!(\"Hello\"); }", 100);
        let results = LlmAssertion::new()
            .assert_matches_pattern("fn main")
            .run(&timed);
        assert!(results[0].passed);
    }

    #[test]
    fn test_matches_pattern_fail() {
        let timed = make_timed("def main():", 100);
        let results = LlmAssertion::new()
            .assert_matches_pattern("fn main")
            .run(&timed);
        assert!(!results[0].passed);
    }

    #[test]
    fn test_matches_pattern_regex() {
        let timed = make_timed("doubled values: [2, 4, 6]", 100);
        let results = LlmAssertion::new()
            .assert_matches_pattern("(?i)(double|multiply)")
            .run(&timed);
        assert!(results[0].passed);
    }

    #[test]
    fn test_latency_under_pass() {
        let timed = make_timed("ok", 100);
        let results = LlmAssertion::new()
            .assert_latency_under(Duration::from_millis(200))
            .run(&timed);
        assert!(results[0].passed);
    }

    #[test]
    fn test_latency_under_fail() {
        let timed = make_timed("ok", 500);
        let results = LlmAssertion::new()
            .assert_latency_under(Duration::from_millis(200))
            .run(&timed);
        assert!(!results[0].passed);
    }

    #[test]
    fn test_token_count_in_range() {
        let timed = make_timed("ok", 100);
        // usage has completion_tokens=20
        let results = LlmAssertion::new()
            .assert_token_count(Some(10), Some(50))
            .run(&timed);
        assert!(results[0].passed);
    }

    #[test]
    fn test_token_count_below_min() {
        let timed = make_timed("ok", 100);
        let results = LlmAssertion::new()
            .assert_token_count(Some(50), None)
            .run(&timed);
        assert!(!results[0].passed);
    }

    #[test]
    fn test_multiple_assertions() {
        let timed = make_timed("The answer is 56.", 100);
        let results = LlmAssertion::new()
            .assert_response_valid()
            .assert_contains("56")
            .assert_latency_under(Duration::from_millis(200))
            .run(&timed);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.passed));
    }

    #[test]
    fn test_run_all_pass() {
        let timed = make_timed("56", 100);
        let assertion = LlmAssertion::new()
            .assert_response_valid()
            .assert_contains("56");
        assert!(assertion.run_all_pass(&timed));
    }

    #[test]
    fn test_run_all_pass_fails() {
        let timed = make_timed("42", 100);
        let assertion = LlmAssertion::new()
            .assert_response_valid()
            .assert_contains("56");
        assert!(!assertion.run_all_pass(&timed));
    }

    #[test]
    fn test_deterministic_pass() {
        let r1 = make_response("hello");
        let r2 = make_response("hello");
        let result = assert_deterministic(&[r1, r2]);
        assert!(result.passed);
    }

    #[test]
    fn test_deterministic_fail() {
        let r1 = make_response("hello");
        let r2 = make_response("world");
        let result = assert_deterministic(&[r1, r2]);
        assert!(!result.passed);
    }

    #[test]
    fn test_deterministic_single_response() {
        let r1 = make_response("hello");
        let result = assert_deterministic(&[r1]);
        assert!(result.passed);
    }

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_long() {
        let s = "a".repeat(300);
        let t = truncate(&s, 200);
        assert_eq!(t.len(), 203); // 200 + "..."
        assert!(t.ends_with("..."));
    }

    #[test]
    fn test_first_content_empty_choices() {
        let resp = ChatResponse {
            id: "x".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![],
            usage: None,
            brick_trace: None,
        };
        assert_eq!(first_content(&resp), "");
    }

    #[test]
    fn test_invalid_regex_pattern() {
        let timed = make_timed("hello", 100);
        let results = LlmAssertion::new()
            .assert_matches_pattern("[invalid")
            .run(&timed);
        assert!(!results[0].passed);
        assert!(results[0]
            .detail
            .as_ref()
            .unwrap()
            .contains("invalid regex"));
    }
}
