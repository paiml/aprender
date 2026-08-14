<!-- PCU: lib-text | contract: contracts/apr-page-lib-text-v1.yaml -->

# Module: `aprender::text`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/text/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/text/mod.rs) or directory.

## Example

<!-- example-cost: skip -->
```rust
use aprender::text::{Tokenizer, ChatTemplateEngine, ChatMessage};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::text` is the NLP toolkit. It owns the `Tokenizer` trait, BPE and
Llama-style tokenizers, the chat-template engine that turns
`Vec<ChatMessage>` into prompt strings for ChatML / Llama2 / Mistral / Phi /
Alpaca, plus classical NLP utilities — stop words, stemming, sentiment,
similarity, IDF, summarisation, vectorisation, topic modelling, and a small
RAG retrieval submodule. Anything that converts characters to tokens or
templates a conversation runs through here.

## Key types

| Type | Description |
|------|-------------|
| `Tokenizer` | Trait. `encode(&str) -> Vec<u32>`, `decode(&[u32]) -> String`. |
| `ChatTemplateEngine` | Multi-format chat-template renderer (minijinja under the hood). |
| `ChatMessage`, `SpecialTokens`, `TemplateFormat` | Building blocks for templating. |
| `ChatMLTemplate`, `Llama2Template`, `MistralTemplate`, `PhiTemplate`, `AlpacaTemplate`, `HuggingFaceTemplate`, `RawTemplate` | Concrete template implementations. |
| `auto_detect_template`, `detect_format_from_name`, `create_template` | Convenience constructors. |
| `text::bpe`, `text::llama_tokenizer`, `text::stem`, `text::similarity`, `text::vectorize`, `text::rag` | Sub-modules for specific tasks. |

## Usage patterns

### Pattern 1: Detect a chat template by model name

<!-- example-cost: skip -->
```rust
use aprender::text::{detect_format_from_name, TemplateFormat};

let fmt = detect_format_from_name("Qwen/Qwen2.5-Coder-7B-Instruct");
assert!(matches!(fmt, Some(TemplateFormat::ChatML) | Some(TemplateFormat::HuggingFace)));

let other = detect_format_from_name("meta-llama/Llama-2-7b-chat-hf");
assert!(matches!(other, Some(TemplateFormat::Llama2)));
```

### Pattern 2: Render a multi-turn conversation

<!-- example-cost: skip -->
```rust
use aprender::text::{ChatMessage, ChatMLTemplate, ChatTemplateEngine};

let template = ChatMLTemplate::default();
let messages = vec![
    ChatMessage::system("You are a helpful coding assistant."),
    ChatMessage::user("What is 2 + 2?"),
    ChatMessage::assistant("4"),
    ChatMessage::user("And 3 + 3?"),
];

let prompt = template.render(&messages, true).expect("render");
println!("--- prompt ---\n{}", prompt);
assert!(prompt.contains("system"));
assert!(prompt.contains("user"));
```

## See also

- [`models`](models.md) — Qwen2 / BERT consume tokens produced here
- [`embed`](embed.md) — vectorisation / embedding pipelines built on top
- [`code`](code.md) — code-aware tokenisation and parsing
- [`stack`](stack.md) — higher-level orchestration that bundles text + model + template

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
