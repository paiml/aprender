//! Tests for TokenBudget and ModelContext

use crate::content::*;

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_001_token_budget_new() {
    let budget = TokenBudget::new(ModelContext::Claude200K);
    assert_eq!(budget.context_window, 200_000);
    assert_eq!(budget.system_reserve, 2_000);
}

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_002_token_budget_with_source_context() {
    let budget = TokenBudget::new(ModelContext::Claude200K).with_source_context(10_000);
    assert_eq!(budget.source_context, 10_000);
}

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_003_token_budget_prompt_tokens() {
    let budget = TokenBudget::new(ModelContext::Claude200K)
        .with_source_context(5_000)
        .with_rag_context(3_000);
    // system_reserve (2000) + source (5000) + rag (3000) + few_shot (1500)
    assert_eq!(budget.prompt_tokens(), 11_500);
}

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_004_token_budget_available_margin() {
    let budget = TokenBudget::new(ModelContext::Claude200K)
        .with_source_context(5_000)
        .with_output_target(10_000);
    // 200000 - (2000 + 5000 + 0 + 1500 + 10000) = 181500
    assert_eq!(budget.available_margin(), 181_500);
}

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_005_token_budget_validate_ok() {
    let budget = TokenBudget::new(ModelContext::Claude200K).with_output_target(10_000);
    assert!(budget.validate().is_ok());
}

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_006_token_budget_validate_exceeded() {
    let budget = TokenBudget {
        context_window: 1000,
        system_reserve: 500,
        source_context: 500,
        rag_context: 500,
        few_shot: 500,
        output_target: 500,
    };
    let result = budget.validate();
    assert!(result.is_err());
    assert!(matches!(result, Err(ContentError::TokenBudgetExceeded { .. })));
}

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_007_words_to_tokens() {
    assert_eq!(TokenBudget::words_to_tokens(1000), 1300);
    assert_eq!(TokenBudget::words_to_tokens(4000), 5200);
}

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_008_tokens_to_words() {
    assert_eq!(TokenBudget::tokens_to_words(1300), 1000);
    assert_eq!(TokenBudget::tokens_to_words(5200), 4000);
}

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_009_model_context_window_sizes() {
    assert_eq!(ModelContext::Claude200K.window_size(), 200_000);
    assert_eq!(ModelContext::GeminiPro.window_size(), 1_000_000);
    assert_eq!(ModelContext::Gpt4Turbo.window_size(), 128_000);
    assert_eq!(ModelContext::Custom(50_000).window_size(), 50_000);
}

#[test]
#[allow(non_snake_case)]
fn test_BUDGET_010_format_display() {
    let budget = TokenBudget::new(ModelContext::Claude200K)
        .with_source_context(8_000)
        .with_rag_context(4_000)
        .with_output_target(15_000);
    let display = budget.format_display("claude-sonnet");
    assert!(display.contains("claude-sonnet"));
    assert!(display.contains("200K"));
    assert!(display.contains("8000"));
    assert!(display.contains("✓"));
}
