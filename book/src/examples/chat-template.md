# Case Study: Chat Templates for LLM Inference

This case study demonstrates how to use chat templates to format conversations for large language model (LLM) inference. Chat templates handle the model-specific formatting required by different LLM architectures.

## Overview

Different LLMs expect conversations in specific formats:
- **ChatML**: Used by Qwen2, Yi, and OpenAI-style models
- **LLaMA2**: Used by TinyLlama, Vicuna, and Meta's LLaMA2
- **Mistral**: Used by Mistral AI models
- **Phi**: Used by Microsoft Phi models
- **Alpaca**: Instruction-following format
- **Raw**: No special formatting

The chat template module provides:
1. Pre-built templates for popular formats
2. Auto-detection from model names
3. Custom Jinja2 template support (HuggingFace compatible)

## Basic ChatML Usage

ChatML is the most common format, used by Qwen2 and many chat models:

```rust
use aprender::text::chat_template::{ChatMLTemplate, ChatMessage, ChatTemplateEngine};

let template = ChatMLTemplate::new();
let messages = vec![
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("Hello!"),
];

let output = template.format_conversation(&messages).unwrap();

// Output format:
// <|im_start|>system
// You are a helpful assistant.<|im_end|>
// <|im_start|>user
// Hello!<|im_end|>
// <|im_start|>assistant
```

## LLaMA2 Format with System Prompt

LLaMA2 uses a distinct format with `<<SYS>>` tags:

```rust
use aprender::text::chat_template::{Llama2Template, ChatMessage, ChatTemplateEngine};

let template = Llama2Template::new();
let messages = vec![
    ChatMessage::system("You are a coding assistant."),
    ChatMessage::user("Write hello world"),
];

let output = template.format_conversation(&messages).unwrap();

// Output starts with <s> and includes <<SYS>> block
assert!(output.starts_with("<s>"));
assert!(output.contains("<<SYS>>"));
assert!(output.contains("You are a coding assistant."));
```

## Mistral Format (No System Prompt)

Mistral models don't support system prompts - they are silently ignored:

```rust
use aprender::text::chat_template::{MistralTemplate, ChatMessage, ChatTemplateEngine};

let template = MistralTemplate::new();

// Check system prompt support
assert!(!template.supports_system_prompt());

let messages = vec![
    ChatMessage::system("This will be ignored"),
    ChatMessage::user("Hello Mistral!"),
];

let output = template.format_conversation(&messages).unwrap();

// System prompt does NOT appear in output
assert!(!output.contains("This will be ignored"));
assert!(output.contains("[INST]"));
assert!(output.contains("Hello Mistral!"));
```

## Auto-Detection from Model Name

The module can automatically detect the correct format from model names:

```rust
use aprender::text::chat_template::{detect_format_from_name, TemplateFormat};

// TinyLlama -> LLaMA2 format
assert_eq!(
    detect_format_from_name("TinyLlama-1.1B-Chat"),
    TemplateFormat::Llama2
);

// Qwen -> ChatML format
assert_eq!(
    detect_format_from_name("Qwen2-0.5B-Instruct"),
    TemplateFormat::ChatML
);

// Mistral -> Mistral format
assert_eq!(
    detect_format_from_name("Mistral-7B-Instruct"),
    TemplateFormat::Mistral
);

// Phi -> Phi format
assert_eq!(detect_format_from_name("phi-2"), TemplateFormat::Phi);
```

## Creating Templates from Format Enum

Create templates programmatically using the format enum:

```rust
use aprender::text::chat_template::{create_template, TemplateFormat};

let template = create_template(TemplateFormat::ChatML);
assert_eq!(template.format(), TemplateFormat::ChatML);
assert!(template.supports_system_prompt());

let template = create_template(TemplateFormat::Mistral);
assert_eq!(template.format(), TemplateFormat::Mistral);
assert!(!template.supports_system_prompt());
```

## Multi-Turn Conversations

Templates correctly handle multi-turn conversations with user/assistant exchanges:

```rust
use aprender::text::chat_template::{ChatMLTemplate, ChatMessage, ChatTemplateEngine};

let template = ChatMLTemplate::new();
let messages = vec![
    ChatMessage::system("You are helpful."),
    ChatMessage::user("What is 2+2?"),
    ChatMessage::assistant("4"),
    ChatMessage::user("And 3+3?"),
];

let output = template.format_conversation(&messages).unwrap();

// All messages appear in correct order
let sys_pos = output.find("You are helpful.").unwrap();
let user1_pos = output.find("What is 2+2?").unwrap();
let asst_pos = output.find("4").unwrap();
let user2_pos = output.find("And 3+3?").unwrap();

assert!(sys_pos < user1_pos);
assert!(user1_pos < asst_pos);
assert!(asst_pos < user2_pos);
```

## Custom Jinja2 Templates

For HuggingFace models with custom `chat_template` fields, use `HuggingFaceTemplate`:

```rust
use aprender::text::chat_template::{
    HuggingFaceTemplate, ChatMessage, ChatTemplateEngine,
    SpecialTokens, TemplateFormat
};

let template_str = r#"{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}"#;

let template = HuggingFaceTemplate::new(
    template_str.to_string(),
    SpecialTokens::default(),
    TemplateFormat::Custom,
).expect("Template creation failed");

let messages = vec![
    ChatMessage::user("Hello"),
    ChatMessage::assistant("Hi there"),
];

let output = template.format_conversation(&messages).unwrap();
assert!(output.contains("user: Hello"));
assert!(output.contains("assistant: Hi there"));
```

## Auto-Detect and Create in One Step

The `auto_detect_template` function combines detection and creation:

```rust
use aprender::text::chat_template::{auto_detect_template, TemplateFormat};

let template = auto_detect_template("tinyllama-chat");
assert_eq!(template.format(), TemplateFormat::Llama2);

let template = auto_detect_template("qwen2-instruct");
assert_eq!(template.format(), TemplateFormat::ChatML);
```

## Supported Formats Reference

| Format | Models | System Prompt | BOS Token |
|--------|--------|---------------|-----------|
| ChatML | Qwen2, Yi, OpenAI | Yes | `<\|im_start\|>` |
| LLaMA2 | TinyLlama, Vicuna, LLaMA2 | Yes | `<s>` |
| Mistral | Mistral-7B-Instruct | No | `<s>` |
| Phi | phi-2, phi-3 | Yes | None |
| Alpaca | Alpaca-based | Yes | None |
| Raw | Any | Pass-through | None |

## Security Considerations

The Jinja2 templates are sandboxed via `minijinja`:
- No filesystem access
- No network access
- No arbitrary code execution
- Safe for processing untrusted templates

## Integration with Realizar

When using `realizar` for inference, chat templates are applied automatically:

```bash
# Chat with auto-detected template
realizar chat qwen2-0.5b.gguf --prompt "Hello!"

# Explicit template override
realizar chat model.gguf --template chatml --prompt "Hello!"
```

## Running the Example

```bash
cargo run --example chat_template
```

## Test Coverage

All examples in this chapter are validated by tests in:
- `tests/book/case_studies/chat_template_usage.rs`

Run the tests:
```bash
cargo test --test book chat_template
```
