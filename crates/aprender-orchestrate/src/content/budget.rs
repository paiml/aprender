//! Token budgeting (Heijunka)
//!
//! Token budget calculation (spec section 5.4) for load leveling.

use super::ContentError;
use serde::{Deserialize, Serialize};

/// Model context window sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelContext {
    /// Claude Sonnet/Opus (200K) - Default
    #[default]
    Claude200K,
    /// Claude Haiku (200K)
    ClaudeHaiku,
    /// Gemini Pro (1M)
    GeminiPro,
    /// Gemini Flash (1M)
    GeminiFlash,
    /// GPT-4 Turbo (128K)
    Gpt4Turbo,
    /// Custom context window
    Custom(usize),
}

impl ModelContext {
    /// Get the context window size in tokens
    pub fn window_size(&self) -> usize {
        match self {
            ModelContext::Claude200K => 200_000,
            ModelContext::ClaudeHaiku => 200_000,
            ModelContext::GeminiPro => 1_000_000,
            ModelContext::GeminiFlash => 1_000_000,
            ModelContext::Gpt4Turbo => 128_000,
            ModelContext::Custom(size) => *size,
        }
    }

    /// Get the model name
    pub fn name(&self) -> &'static str {
        match self {
            ModelContext::Claude200K => "claude-sonnet",
            ModelContext::ClaudeHaiku => "claude-haiku",
            ModelContext::GeminiPro => "gemini-pro",
            ModelContext::GeminiFlash => "gemini-flash",
            ModelContext::Gpt4Turbo => "gpt-4-turbo",
            ModelContext::Custom(_) => "custom",
        }
    }
}

/// Token budget calculation (spec section 5.4)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Target model context window
    pub context_window: usize,
    /// Reserved for system prompt
    pub system_reserve: usize,
    /// Reserved for source context (--source-context)
    pub source_context: usize,
    /// Reserved for RAG context (--rag-context)
    pub rag_context: usize,
    /// Reserved for few-shot examples
    pub few_shot: usize,
    /// Target output tokens
    pub output_target: usize,
}

impl TokenBudget {
    /// Create a new token budget for a model
    pub fn new(model: ModelContext) -> Self {
        Self {
            context_window: model.window_size(),
            system_reserve: 2_000,
            source_context: 0,
            rag_context: 0,
            few_shot: 1_500,
            output_target: 4_000,
        }
    }

    /// Set source context tokens
    pub fn with_source_context(mut self, tokens: usize) -> Self {
        self.source_context = tokens;
        self
    }

    /// Set RAG context tokens
    pub fn with_rag_context(mut self, tokens: usize) -> Self {
        self.rag_context = tokens;
        self
    }

    /// Set output target tokens
    pub fn with_output_target(mut self, tokens: usize) -> Self {
        self.output_target = tokens;
        self
    }

    /// Calculate total prompt tokens (excluding output)
    pub fn prompt_tokens(&self) -> usize {
        self.system_reserve + self.source_context + self.rag_context + self.few_shot
    }

    /// Calculate available margin
    pub fn available_margin(&self) -> usize {
        let used = self.prompt_tokens() + self.output_target;
        self.context_window.saturating_sub(used)
    }

    /// Validate the budget fits within context window
    pub fn validate(&self) -> Result<(), ContentError> {
        let total = self.prompt_tokens() + self.output_target;
        if total > self.context_window {
            Err(ContentError::TokenBudgetExceeded { used: total, limit: self.context_window })
        } else {
            Ok(())
        }
    }

    /// Estimate tokens from word count (rough: 1 word ≈ 1.3 tokens)
    pub fn words_to_tokens(words: usize) -> usize {
        (words as f64 * 1.3).ceil() as usize
    }

    /// Estimate words from token count
    pub fn tokens_to_words(tokens: usize) -> usize {
        (tokens as f64 / 1.3).floor() as usize
    }

    /// Format budget as display string
    pub fn format_display(&self, model_name: &str) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Token Budget for {} ({}K context):\n",
            model_name,
            self.context_window / 1000
        ));
        output.push_str(&format!("├── System prompt:     {:>6} tokens\n", self.system_reserve));
        output.push_str(&format!("├── Source context:    {:>6} tokens\n", self.source_context));
        output.push_str(&format!("├── RAG context:       {:>6} tokens\n", self.rag_context));
        output.push_str(&format!("├── Few-shot examples: {:>6} tokens\n", self.few_shot));
        output.push_str(&format!(
            "├── Output reserved:   {:>6} tokens (~{} words)\n",
            self.output_target,
            Self::tokens_to_words(self.output_target)
        ));
        let margin = self.available_margin();
        let status = if margin > 0 { "✓" } else { "✗" };
        output.push_str(&format!("└── Available margin:  {:>6} tokens {}\n", margin, status));
        output
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self::new(ModelContext::Claude200K)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ModelContext Tests
    // =========================================================================

    #[test]
    fn test_model_context_default() {
        let ctx = ModelContext::default();
        assert_eq!(ctx, ModelContext::Claude200K);
    }

    #[test]
    fn test_model_context_window_sizes() {
        assert_eq!(ModelContext::Claude200K.window_size(), 200_000);
        assert_eq!(ModelContext::ClaudeHaiku.window_size(), 200_000);
        assert_eq!(ModelContext::GeminiPro.window_size(), 1_000_000);
        assert_eq!(ModelContext::GeminiFlash.window_size(), 1_000_000);
        assert_eq!(ModelContext::Gpt4Turbo.window_size(), 128_000);
        assert_eq!(ModelContext::Custom(50_000).window_size(), 50_000);
    }

    #[test]
    fn test_model_context_names() {
        assert_eq!(ModelContext::Claude200K.name(), "claude-sonnet");
        assert_eq!(ModelContext::ClaudeHaiku.name(), "claude-haiku");
        assert_eq!(ModelContext::GeminiPro.name(), "gemini-pro");
        assert_eq!(ModelContext::GeminiFlash.name(), "gemini-flash");
        assert_eq!(ModelContext::Gpt4Turbo.name(), "gpt-4-turbo");
        assert_eq!(ModelContext::Custom(1000).name(), "custom");
    }

    #[test]
    fn test_model_context_serialization() {
        let ctx = ModelContext::GeminiPro;
        let json = serde_json::to_string(&ctx).expect("json serialize failed");
        let deserialized: ModelContext =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized, ctx);
    }

    #[test]
    fn test_model_context_custom_serialization() {
        let ctx = ModelContext::Custom(75_000);
        let json = serde_json::to_string(&ctx).expect("json serialize failed");
        let deserialized: ModelContext =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized, ctx);
        assert_eq!(deserialized.window_size(), 75_000);
    }

    // =========================================================================
    // TokenBudget Tests
    // =========================================================================

    #[test]
    fn test_token_budget_new() {
        let budget = TokenBudget::new(ModelContext::Claude200K);
        assert_eq!(budget.context_window, 200_000);
        assert_eq!(budget.system_reserve, 2_000);
        assert_eq!(budget.source_context, 0);
        assert_eq!(budget.rag_context, 0);
        assert_eq!(budget.few_shot, 1_500);
        assert_eq!(budget.output_target, 4_000);
    }

    #[test]
    fn test_token_budget_default() {
        let budget = TokenBudget::default();
        assert_eq!(budget.context_window, 200_000);
    }

    #[test]
    fn test_token_budget_with_source_context() {
        let budget = TokenBudget::new(ModelContext::Claude200K).with_source_context(10_000);
        assert_eq!(budget.source_context, 10_000);
    }

    #[test]
    fn test_token_budget_with_rag_context() {
        let budget = TokenBudget::new(ModelContext::Claude200K).with_rag_context(5_000);
        assert_eq!(budget.rag_context, 5_000);
    }

    #[test]
    fn test_token_budget_with_output_target() {
        let budget = TokenBudget::new(ModelContext::Claude200K).with_output_target(8_000);
        assert_eq!(budget.output_target, 8_000);
    }

    #[test]
    fn test_token_budget_prompt_tokens() {
        let budget = TokenBudget::new(ModelContext::Claude200K)
            .with_source_context(10_000)
            .with_rag_context(5_000);
        // system_reserve(2000) + source(10000) + rag(5000) + few_shot(1500)
        assert_eq!(budget.prompt_tokens(), 18_500);
    }

    #[test]
    fn test_token_budget_available_margin() {
        let budget = TokenBudget::new(ModelContext::Claude200K);
        // context(200000) - prompt(3500) - output(4000)
        let margin = budget.available_margin();
        assert_eq!(margin, 200_000 - 3_500 - 4_000);
    }

    #[test]
    fn test_token_budget_validate_ok() {
        let budget = TokenBudget::new(ModelContext::Claude200K);
        assert!(budget.validate().is_ok());
    }

    #[test]
    fn test_token_budget_validate_exceeded() {
        let budget = TokenBudget::new(ModelContext::Custom(1_000)).with_output_target(2_000);
        assert!(budget.validate().is_err());
    }

    #[test]
    fn test_words_to_tokens() {
        // 100 words * 1.3 = 130 tokens
        assert_eq!(TokenBudget::words_to_tokens(100), 130);
        assert_eq!(TokenBudget::words_to_tokens(0), 0);
    }

    #[test]
    fn test_tokens_to_words() {
        // 130 tokens / 1.3 = 100 words
        assert_eq!(TokenBudget::tokens_to_words(130), 100);
        assert_eq!(TokenBudget::tokens_to_words(0), 0);
    }

    #[test]
    fn test_token_budget_format_display() {
        let budget = TokenBudget::new(ModelContext::Claude200K);
        let output = budget.format_display("claude-sonnet");
        assert!(output.contains("Token Budget for claude-sonnet"));
        assert!(output.contains("200K context"));
        assert!(output.contains("System prompt"));
        assert!(output.contains("Available margin"));
        assert!(output.contains("✓")); // Should have margin available
    }

    #[test]
    fn test_token_budget_format_display_exceeded() {
        let budget = TokenBudget::new(ModelContext::Custom(1_000)).with_output_target(2_000);
        let output = budget.format_display("custom");
        // When margin is 0 or negative, should show ✗
        assert!(output.contains("Available margin"));
    }

    #[test]
    fn test_token_budget_serialization() {
        let budget = TokenBudget::new(ModelContext::GeminiPro)
            .with_source_context(5_000)
            .with_rag_context(3_000);
        let json = serde_json::to_string(&budget).expect("json serialize failed");
        let deserialized: TokenBudget =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized, budget);
    }

    #[test]
    fn test_token_budget_builder_chain() {
        let budget = TokenBudget::new(ModelContext::Gpt4Turbo)
            .with_source_context(10_000)
            .with_rag_context(8_000)
            .with_output_target(6_000);

        assert_eq!(budget.context_window, 128_000);
        assert_eq!(budget.source_context, 10_000);
        assert_eq!(budget.rag_context, 8_000);
        assert_eq!(budget.output_target, 6_000);
    }
}
