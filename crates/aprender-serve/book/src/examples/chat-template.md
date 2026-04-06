# Chat Templates for LLM Inference

This example demonstrates how to use chat templates to format conversations for large language model inference. Chat templates handle model-specific formatting required by different LLM architectures.

## Overview

Different LLMs expect conversations in specific formats:
- **ChatML**: Qwen2, Yi, OpenHermes, OpenAI-style
- **LLaMA2**: TinyLlama, Vicuna, Meta LLaMA2
- **Mistral**: Mistral AI models (no system prompt support)
- **Phi**: Microsoft Phi-2, Phi-3
- **Alpaca**: Alpaca instruction format
- **Raw**: No formatting (passthrough)

## Auto-Detection from Model Name

The template system can automatically detect the correct format from model names:

```rust
use realizar::chat_template::{detect_format_from_name, TemplateFormat};

let models = [
    "TinyLlama-1.1B-Chat-v1.0.Q4_K_M",  // -> Llama2
    "Qwen2-0.5B-Instruct",               // -> ChatML
    "Mistral-7B-Instruct-v0.2",          // -> Mistral
    "phi-2",                              // -> Phi
    "alpaca-7b",                          // -> Alpaca
    "unknown-model",                      // -> Raw
];

for model in models {
    let format = detect_format_from_name(model);
    println!("{} -> {:?}", model, format);
}
```

## ChatML Format (Qwen2, OpenHermes, Yi)

ChatML is the most widely supported format:

```rust
use realizar::chat_template::{ChatMLTemplate, ChatMessage, ChatTemplateEngine};

let chatml = ChatMLTemplate::new();
let messages = vec![
    ChatMessage::system("You are a helpful assistant."),
    ChatMessage::user("What is 2+2?"),
    ChatMessage::assistant("4"),
    ChatMessage::user("And 3+3?"),
];

let output = chatml.format_conversation(&messages)?;
// Output:
// <|im_start|>system
// You are a helpful assistant.<|im_end|>
// <|im_start|>user
// What is 2+2?<|im_end|>
// <|im_start|>assistant
// 4<|im_end|>
// <|im_start|>user
// And 3+3?<|im_end|>
// <|im_start|>assistant
```

## LLaMA2 Format (TinyLlama, Vicuna)

LLaMA2 uses `<<SYS>>` tags for system prompts:

```rust
use realizar::chat_template::{Llama2Template, ChatMessage, ChatTemplateEngine};

let llama2 = Llama2Template::new();
let messages = vec![
    ChatMessage::system("You are a coding assistant."),
    ChatMessage::user("Write hello world in Python"),
];

let output = llama2.format_conversation(&messages)?;
// Output:
// <s>[INST] <<SYS>>
// You are a coding assistant.
// <</SYS>>
//
// Write hello world in Python [/INST]
```

## Mistral Format (No System Prompt)

Mistral models don't support system prompts - they are silently ignored:

```rust
use realizar::chat_template::{MistralTemplate, ChatMessage, ChatTemplateEngine};

let mistral = MistralTemplate::new();
println!("Supports system prompt: {}", mistral.supports_system_prompt());
// false

let messages = vec![
    ChatMessage::system("This will be ignored"),
    ChatMessage::user("Hello Mistral!"),
];

let output = mistral.format_conversation(&messages)?;
// Output:
// <s>[INST] Hello Mistral! [/INST]
// (system message is NOT included)
```

## Phi Format (Phi-2, Phi-3)

Microsoft Phi models use Instruct/Output tags:

```rust
use realizar::chat_template::{PhiTemplate, ChatMessage, ChatTemplateEngine};

let phi = PhiTemplate::new();
let messages = vec![ChatMessage::user("Explain quantum computing")];

let output = phi.format_conversation(&messages)?;
// Output:
// Instruct: Explain quantum computing
// Output:
```

## Alpaca Format

Alpaca instruction-following format:

```rust
use realizar::chat_template::{AlpacaTemplate, ChatMessage, ChatTemplateEngine};

let alpaca = AlpacaTemplate::new();
let messages = vec![
    ChatMessage::system("You are a helpful AI assistant."),
    ChatMessage::user("Summarize this text"),
];

let output = alpaca.format_conversation(&messages)?;
// Output:
// ### Instruction:
// You are a helpful AI assistant.
// Summarize this text
//
// ### Response:
```

## Raw Format (Fallback)

For unknown models, raw format passes through content unchanged:

```rust
use realizar::chat_template::{RawTemplate, ChatMessage, ChatTemplateEngine};

let raw = RawTemplate::new();
let messages = vec![ChatMessage::user("Just pass this through")];

let output = raw.format_conversation(&messages)?;
// Output: Just pass this through
```

## Custom HuggingFace Template (Jinja2)

For HuggingFace models with custom `chat_template` fields:

```rust
use realizar::chat_template::{
    HuggingFaceTemplate, ChatMessage, ChatTemplateEngine,
    SpecialTokens, TemplateFormat
};

let template_str = r#"{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}Assistant:"#;

let hf_template = HuggingFaceTemplate::new(
    template_str.to_string(),
    SpecialTokens::default(),
    TemplateFormat::Custom,
)?;

let messages = vec![
    ChatMessage::user("Hello!"),
    ChatMessage::assistant("Hi there!"),
    ChatMessage::user("How are you?"),
];

let output = hf_template.format_conversation(&messages)?;
// Output:
// user: Hello!
// assistant: Hi there!
// user: How are you?
// Assistant:
```

## Auto-Detect and Create

The `auto_detect_template` function combines detection and creation:

```rust
use realizar::chat_template::{auto_detect_template, ChatMessage, ChatTemplateEngine};

let template = auto_detect_template("tinyllama-1.1b-chat");
println!("Auto-detected format: {:?}", template.format());

let messages = vec![
    ChatMessage::system("Be concise."),
    ChatMessage::user("What is Rust?"),
];

let output = template.format_conversation(&messages)?;
```

## Create from Format Enum

Create templates programmatically:

```rust
use realizar::chat_template::{create_template, TemplateFormat};

for format in [
    TemplateFormat::ChatML,
    TemplateFormat::Llama2,
    TemplateFormat::Mistral,
    TemplateFormat::Phi,
    TemplateFormat::Alpaca,
    TemplateFormat::Raw,
] {
    let template = create_template(format);
    println!("{:?}: supports_system={}", format, template.supports_system_prompt());
}
// ChatML: supports_system=true
// Llama2: supports_system=true
// Mistral: supports_system=false
// Phi: supports_system=true
// Alpaca: supports_system=true
// Raw: supports_system=true
```

## Integration with GGUF Models

When loading GGUF files, templates are automatically applied based on model metadata:

```bash
# Chat with auto-detected template
realizar chat model.gguf --prompt "Hello!"

# Override template
realizar chat model.gguf --template chatml --prompt "Hello!"
```

## Running the Example

```bash
cargo run --example chat_template
```

## Security

Templates are sandboxed via `minijinja`:
- No filesystem access
- No network access
- No arbitrary code execution
- Safe for untrusted templates
