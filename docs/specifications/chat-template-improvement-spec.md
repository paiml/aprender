# APR Chat Template Specification v1.4.0

**Document Status:** Draft
**Created:** 2026-01-06
**Authors:** aprender maintainers
**Issue:** GH-XXX (Chat Template Support)

---

## Executive Summary

This specification establishes a generic, model-agnostic chat template system for the aprender ecosystem. It addresses the critical gap where models like TinyLlama and Phi cannot properly format chat conversations because chat templates are currently hardcoded for Qwen2 only.

| Aspect | Current State | Target State |
|--------|--------------|--------------|
| **Template Source** | Hardcoded in `src/text/bpe.rs` | Loaded from `tokenizer_config.json` |
| **Format Support** | Qwen2 ChatML only | 6+ formats (ChatML, LLaMA2, Mistral, Alpaca, Phi, Raw) |
| **Auto-Detection** | None | Pattern-based + metadata-based |
| **HF Integration** | Downloads `model.safetensors` only | Downloads full tokenizer config |
| **Engine** | Manual string formatting | `minijinja` crate (Jinja2 compatible) |

---

## Part I: Theoretical Framework

### 1.1 The Problem: Chat Template Fragmentation

Large Language Models fine-tuned for chat use diverse templating formats:

```
# Qwen2 / ChatML
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant

# TinyLlama / LLaMA 2
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello [/INST]

# Mistral
<s>[INST] Hello [/INST]

# Phi-2
Instruct: Hello
Output:
```

Without correct templating, models produce degraded output quality as observed in TinyLlama testing (garbled special tokens in output).

### 1.2 Toyota Way Principles Applied

| Principle | Application |
|-----------|-------------|
| **Jidoka** (Autonomation) | Auto-detect template format; stop on invalid template |
| **Standardized Work** | Use industry-standard `minijinja` crate instead of custom parsing |
| **Poka-Yoke** (Error-Proofing) | Validate templates before application; escape special tokens |
| **Genchi Genbutsu** (Go and See) | Test against real HuggingFace model files |
| **Kaizen** (Continuous Improvement) | Extensible format registry for new models |
| **Heijunka** (Level Loading) | Template parsing < 1ms via optimized compiled templates |
| **Muda Elimination** | Avoid re-implementing a Jinja2 parser; leverage existing robust tools |

### 1.3 Academic Foundations

This specification draws from peer-reviewed research in ML systems engineering, tokenization, LLM deployment, and formal verification methodologies:

**ML Systems Engineering:**
- Sculley et al. (2015) - "Hidden Technical Debt in Machine Learning Systems" (NeurIPS)
- Breck et al. (2017) - "The ML Test Score: A Rubric for ML Production Readiness" (NeurIPS)
- Paleyes et al. (2022) - "Challenges in Deploying Machine Learning" (ACM Computing Surveys)

**Tokenization & NLP:**
- Sennrich et al. (2016) - "Neural Machine Translation of Rare Words with Subword Units" (ACL)
- Kudo & Richardson (2018) - "SentencePiece" (EMNLP)

**LLM Chat Formats:**
- Touvron et al. (2023) - "Llama 2: Open Foundation and Fine-Tuned Chat Models" (arXiv:2307.09288)
- Bai et al. (2023) - "Qwen Technical Report" (arXiv:2309.16609)
- Li et al. (2024) - "Phi-2: The surprising power of small language models" (Microsoft Research)
- Jiang et al. (2023) - "Mistral 7B" (arXiv:2310.06825)

**Quality & Methodology:**
- Liker, J. K. (2004) - "The Toyota Way: 14 Management Principles"
- Popper, K. (1959) - "The Logic of Scientific Discovery"

**Formal Verification & Falsification Testing:**
- Claessen & Hughes (2000) - "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs" (ICFP) - Foundation for property-based testing
- Groce et al. (2014) - "Cause Reduction for Quick Testing" (ICST) - Minimization techniques for test failures
- Paraskevopoulou et al. (2015) - "Foundational Property-Based Testing" (ITP) - Formal verification of test generators
- Lampropoulos et al. (2017) - "Generating Good Generators for Inductive Relations" (POPL) - Coverage-guided testing
- Hughes (2016) - "Experiences with QuickCheck: Testing the Hard Stuff and Staying Sane" (PADL) - Industrial falsification methodology

**GUI & State Machine Testing:**
- Memon et al. (2003) - "GUI Testing: Pitfalls and Process" (IEEE Computer) - GUI coverage methodology
- Utting & Legeard (2007) - "Practical Model-Based Testing: A Tools Approach" (Morgan Kaufmann) - State machine verification
- Tretmans (2008) - "Model Based Testing with Labelled Transition Systems" (Springer LNCS) - Formal state transitions

---

## Part II: Technical Architecture

### 2.1 Chat Template File Format

HuggingFace models store chat templates in `tokenizer_config.json`:

```json
{
  "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
  "bos_token": "<|endoftext|>",
  "eos_token": "<|im_end|>",
  "unk_token": "<|endoftext|>",
  "pad_token": "<|endoftext|>"
}
```

### 2.2 Template Engine Design

#### 2.2.1 Supported Template Formats

| Format | Models | Special Tokens | Pattern |
|--------|--------|----------------|---------|
| **ChatML** | Qwen2, OpenHermes, Yi | `<\|im_start\|>`, `<\|im_end\|>` | `<\|im_start\|>{role}\n{content}<\|im_end\|>` |
| **LLaMA2** | TinyLlama, Vicuna, LLaMA 2 | `<s>`, `</s>`, `[INST]`, `[/INST]` | `<s>[INST] {content} [/INST]` |
| **Mistral** | Mistral, Mixtral | `<s>`, `</s>`, `[INST]`, `[/INST]` | `<s>[INST] {content} [/INST]` (no system) |
| **Alpaca** | Alpaca, GPT4All | None | `### Instruction:\n{content}\n### Response:` |
| **Phi** | Phi-2, Phi-3 | None | `Instruct: {content}\nOutput:` |
| **Raw** | Fallback | None | `{content}` |

#### 2.2.2 Jinja2 Compatibility via `minijinja`

Instead of implementing a custom parser, we utilize the [`minijinja`](https://crates.io/crates/minijinja) crate. This ensures full compatibility with the Jinja2 subset used by HuggingFace `transformers` while maintaining high performance and security.

**Rationale:**
- **Standardization**: `minijinja` is the de-facto standard for Jinja2 in Rust.
- **Safety**: Robust, fuzzed parser prevents injection attacks and crashes.
- **Maintenance**: Offloads parser maintenance to the ecosystem.
- **Performance**: Compiles templates to bytecode for fast execution (<100μs).

**Configuration:**
- Enable `minijinja/loader` for template management.
- Enable `minijinja/serde` for context passing.
- Disable filesystem access (sandbox).

```rust
// Implementation Concept
use minijinja::{Environment, context};

let mut env = Environment::new();
env.add_template_owned("chat", template_str)?;
let tmpl = env.get_template("chat")?;
let prompt = tmpl.render(context!(
    messages => messages,
    add_generation_prompt => true,
    bos_token => special_tokens.bos_token,
    eos_token => special_tokens.eos_token
))?;
```

#### 2.2.3 Environment Management

To optimize performance and handle lifetimes correctly:
- The `minijinja::Environment` should be owned by the `ChatTemplateEngine` implementation.
- Templates should be added using `add_template_owned` to avoid lifetime issues with local strings.
- The environment should be configured with a recursion limit and other safety guards.

#### 2.2.4 Error Handling Integration

`minijinja` errors must be mapped to `AprenderError`.
- `AprenderError::Io` for filesystem issues.
- `AprenderError::Other` for template syntax and rendering errors.

```rust
implement From<minijinja::Error> for AprenderError {
    fn from(err: minijinja::Error) -> Self {
        AprenderError::Other(format!("Template error: {}", err))
    }
}
```

---

## Part III: Implementation Specification

### 3.1 Required Files for Chat Support

| File | Purpose | Required |
|------|---------|----------|
| `tokenizer_config.json` | Chat template, special tokens | **Yes** (for proper chat) |
| `tokenizer.json` | Vocabulary, merges | Yes |
| `config.json` | Model architecture | Yes |
| `special_tokens_map.json` | Token ID mappings | Optional |

### 3.2 Template Format Examples

#### 3.2.1 ChatML (Qwen2, OpenHermes)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

#### 3.2.2 LLaMA 2 (TinyLlama, Vicuna)

```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is 2+2? [/INST]
```

#### 3.2.3 Mistral/Mixtral

```
<s>[INST] What is 2+2? [/INST]
```

*Note: Mistral does not support system prompts in the standard template.*

#### 3.2.4 Alpaca

```
### Instruction:
What is 2+2?

### Response:
```

#### 3.2.5 Phi-2/Phi-3

```
Instruct: What is 2+2?
Output:
```

### 3.3 API Design

#### 3.3.1 Core Trait

```rust
/// Chat template engine trait (Toyota Way: Standardized Work)
pub trait ChatTemplateEngine {
    /// Format a single message with role and content
    fn format_message(&self, role: &str, content: &str) -> Result<String, AprenderError>;

    /// Format a complete conversation
    fn format_conversation(&self, messages: &[ChatMessage]) -> Result<String, AprenderError>;

    /// Get special tokens for this template
    fn special_tokens(&self) -> &SpecialTokens;

    /// Get the detected template format
    fn format(&self) -> TemplateFormat;

    /// Check if this template supports system prompts
    fn supports_system_prompt(&self) -> bool;
}

/// Chat message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,      // "system", "user", "assistant"
    pub content: String,
}

/// Template format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TemplateFormat {
    ChatML,      // Qwen2, OpenHermes, Yi
    Llama2,      // LLaMA 2, TinyLlama, Vicuna
    Mistral,     // Mistral, Mixtral
    Alpaca,      // Alpaca instruction format
    Phi,         // Phi-2, Phi-3
    Custom,      // Arbitrary Jinja2 template
    Raw,         // Fallback - no template
}
```

#### 3.3.2 HuggingFace Template Loader

```rust
/// Load chat template from HuggingFace tokenizer_config.json
pub struct HuggingFaceTemplate {
    env: Environment<'static>,
    template_str: String,
    special_tokens: SpecialTokens,
    format: TemplateFormat,
}

impl HuggingFaceTemplate {
    /// Load from tokenizer_config.json file path
    pub fn from_tokenizer_config(path: &Path) -> Result<Self, AprenderError>;

    /// Parse from JSON string
    pub fn from_json(json: &str) -> Result<Self, AprenderError>;

    /// Auto-detect format from template content
    pub fn auto_detect_format(&self) -> TemplateFormat;
}

impl ChatTemplateEngine for HuggingFaceTemplate {
    // ... implementation
}
```

---

## Part IV: 100-Point Popperian Falsification Checklist

Each checklist item follows the structure:
- **Claim**: What the system should do
- **Rejection Criterion**: Evidence that would falsify the claim
- **Evidence Required**: How to test
- **Toyota Principle**: Which principle applies

### Section 1: Template Loading & Parsing (CTL-01 to CTL-10)

#### CTL-01: tokenizer_config.json File Discovery
- **Claim**: System locates `tokenizer_config.json` adjacent to model file
- **Rejection Criterion**: File not found when present, or wrong file loaded
- **Evidence**: Test with TinyLlama, Phi, Qwen models from HuggingFace
- **Toyota Principle**: Genchi Genbutsu (go and see the actual file)

#### CTL-02: JSON Parsing Correctness
- **Claim**: Valid `tokenizer_config.json` parsed without errors
- **Rejection Criterion**: Parser fails on valid HuggingFace format
- **Evidence**: Parse 20+ real tokenizer_config.json files from HuggingFace
- **Toyota Principle**: Jidoka (stop on parse error)

#### CTL-03: chat_template Field Extraction
- **Claim**: `chat_template` string extracted from JSON correctly
- **Rejection Criterion**: Field present but not extracted, or extracted incorrectly
- **Evidence**: Validate against OpenAI, Meta, Alibaba, Mistral configs
- **Toyota Principle**: Standardized Work (consistent field access)

#### CTL-04: Jinja2 Syntax Validation
- **Claim**: Template contains valid Jinja2 subset
- **Rejection Criterion**: Invalid syntax accepted or valid syntax rejected
- **Evidence**: Parse templates from top 50 HuggingFace chat models
- **Toyota Principle**: Poka-Yoke (error-proof input validation)

#### CTL-05: Special Token Mapping
- **Claim**: BOS/EOS/special tokens correctly identified from config
- **Rejection Criterion**: Tokens misidentified or missing
- **Evidence**: Compare extracted tokens with HuggingFace tokenizer output
- **Toyota Principle**: Standardized Work

#### CTL-06: Missing File Graceful Handling
- **Claim**: System falls back gracefully when `tokenizer_config.json` missing
- **Rejection Criterion**: Crash or panic on missing file
- **Evidence**: Test with model directory missing tokenizer_config.json
- **Toyota Principle**: Jidoka (controlled failure)

#### CTL-07: Malformed JSON Handling
- **Claim**: Malformed JSON produces clear error message
- **Rejection Criterion**: Panic, unclear error, or silent failure
- **Evidence**: Test with truncated, invalid, and edge-case JSON
- **Toyota Principle**: Jidoka

#### CTL-08: Unicode in Templates
- **Claim**: Templates with Unicode characters parsed correctly
- **Rejection Criterion**: Unicode mangled or rejected
- **Evidence**: Test with CJK, emoji, RTL characters in templates
- **Toyota Principle**: Standardized Work

#### CTL-09: Large Template Handling
- **Claim**: Templates up to 10KB parsed in < 10ms
- **Rejection Criterion**: Timeout or excessive memory use
- **Evidence**: Benchmark with synthetic large templates
- **Toyota Principle**: Heijunka (level loading)

#### CTL-10: Template Caching
- **Claim**: Parsed templates cached to avoid re-parsing
- **Rejection Criterion**: Re-parsing on every message format call
- **Evidence**: Profile template application over 1000 messages
- **Toyota Principle**: Muda elimination (waste reduction)

### Section 2: Special Token Handling (CTS-01 to CTS-10)

#### CTS-01: BOS Token Identification
- **Claim**: BOS token correctly identified from `bos_token` field
- **Rejection Criterion**: Wrong token or missing when present
- **Evidence**: Test with LLaMA (`<s>`), Qwen (`<|endoftext|>`) 
- **Toyota Principle**: Standardized Work

#### CTS-02: EOS Token Identification
- **Claim**: EOS token correctly identified from `eos_token` field
- **Rejection Criterion**: Wrong token or missing when present
- **Evidence**: Test with LLaMA (`</s>`), Qwen (`<|im_end|>`) 
- **Toyota Principle**: Standardized Work

#### CTS-03: ChatML Token Recognition
- **Claim**: `<|im_start|>` and `<|im_end|>` recognized as ChatML markers
- **Rejection Criterion**: Tokens not recognized, wrong format inferred
- **Evidence**: Test with Qwen2, OpenHermes tokenizer configs
- **Toyota Principle**: Poka-Yoke

#### CTS-04: INST Token Recognition
- **Claim**: `[INST]` and `[/INST]` recognized as LLaMA2/Mistral markers
- **Rejection Criterion**: Tokens not recognized, wrong format inferred
- **Evidence**: Test with TinyLlama, Mistral tokenizer configs
- **Toyota Principle**: Poka-Yoke

#### CTS-05: System Prompt Tokens
- **Claim**: `<<SYS>>` and `<</SYS>>` correctly handled for LLaMA2
- **Rejection Criterion**: System prompt malformed or missing
- **Evidence**: Test multi-turn with system prompt on TinyLlama
- **Toyota Principle**: Standardized Work

#### CTS-06: Token ID Resolution
- **Claim**: Special tokens resolved to correct token IDs
- **Rejection Criterion**: Token string present but ID wrong
- **Evidence**: Compare with `tokenizer.encode()` output
- **Toyota Principle**: Genchi Genbutsu

#### CTS-07: Added Tokens Handling
- **Claim**: `added_tokens` from tokenizer.json integrated
- **Rejection Criterion**: Added tokens not available for encoding
- **Evidence**: Test encoding of `<|im_start|>` returns single token
- **Toyota Principle**: Standardized Work

#### CTS-08: Token Collision Prevention
- **Claim**: Special tokens don't collide with content
- **Rejection Criterion**: User content containing `<|im_start|>` misinterpreted
- **Evidence**: Test with adversarial content containing special tokens
- **Toyota Principle**: Poka-Yoke

#### CTS-09: Missing Token Fallback
- **Claim**: Missing special tokens produce clear error, not crash
- **Rejection Criterion**: Panic or silent corruption
- **Evidence**: Test with tokenizer config missing expected tokens
- **Toyota Principle**: Jidoka

#### CTS-10: Token Consistency
- **Claim**: Same special tokens used throughout conversation
- **Rejection Criterion**: Different tokens used in different turns
- **Evidence**: Format 10-turn conversation, verify token consistency
- **Toyota Principle**: Standardized Work

### Section 3: Format Auto-Detection (CTA-01 to CTA-10)

#### CTA-01: ChatML Detection from Tokens
- **Claim**: ChatML format detected when `<|im_start|>` in vocabulary
- **Rejection Criterion**: Wrong format when ChatML tokens present
- **Evidence**: Test with Qwen2, Yi, OpenHermes vocabularies
- **Toyota Principle**: Jidoka

#### CTA-02: LLaMA2 Detection from Tokens
- **Claim**: LLaMA2 format detected when `[INST]` in vocabulary (non-Mistral)
- **Rejection Criterion**: Wrong format when INST tokens present
- **Evidence**: Test with TinyLlama, Vicuna vocabularies
- **Toyota Principle**: Jidoka

#### CTA-03: Mistral Detection from Model Name
- **Claim**: Mistral format detected for models named "mistral" or "mixtral"
- **Rejection Criterion**: LLaMA2 format incorrectly applied to Mistral
- **Evidence**: Test with Mistral-7B, Mixtral-8x7B
- **Toyota Principle**: Poka-Yoke

#### CTA-04: Phi Detection from Model Name
- **Claim**: Phi format detected for models named "phi-2", "phi-3"
- **Rejection Criterion**: Wrong format for Phi models
- **Evidence**: Test with microsoft/phi-2, phi-3-mini
- **Toyota Principle**: Poka-Yoke

#### CTA-05: Alpaca Detection
- **Claim**: Alpaca format detected for models named "alpaca"
- **Rejection Criterion**: Wrong format for Alpaca models
- **Evidence**: Test with alpaca-7b, gpt4all-alpaca
- **Toyota Principle**: Poka-Yoke

#### CTA-06: Explicit Template Precedence
- **Claim**: Explicit `chat_template` takes precedence over auto-detection
- **Rejection Criterion**: Auto-detection overrides explicit template
- **Evidence**: Test with model having both template and ChatML tokens
- **Toyota Principle**: Standardized Work

#### CTA-07: Raw Fallback
- **Claim**: Raw format used when no template information available
- **Rejection Criterion**: Crash or wrong format when no info
- **Evidence**: Test with minimal model missing all template info
- **Toyota Principle**: Jidoka

#### CTA-08: Detection Stability
- **Claim**: Same format detected on repeated calls
- **Rejection Criterion**: Non-deterministic detection
- **Evidence**: Call auto_detect_format() 100 times, verify identical
- **Toyota Principle**: Standardized Work

#### CTA-09: Detection Performance
- **Claim**: Auto-detection completes in < 1ms
- **Rejection Criterion**: Detection takes > 1ms
- **Evidence**: Benchmark auto-detection on various models
- **Toyota Principle**: Heijunka

#### CTA-10: Detection Logging
- **Claim**: Detected format logged for debugging
- **Rejection Criterion**: No visibility into detection decision
- **Evidence**: Enable debug logging, verify format decision logged
- **Toyota Principle**: Visual Control

### Section 4: Multi-Turn Conversation (CTM-01 to CTM-10)

#### CTM-01: System Prompt Positioning
- **Claim**: System prompt appears first in formatted output
- **Rejection Criterion**: System prompt mispositioned or duplicated
- **Evidence**: Format conversation with system prompt, verify position
- **Toyota Principle**: Standardized Work

#### CTM-02: User/Assistant Alternation
- **Claim**: User and assistant messages alternate correctly
- **Rejection Criterion**: Messages out of order or duplicated
- **Evidence**: Format 5-turn conversation, verify alternation
- **Toyota Principle**: Standardized Work

#### CTM-03: Role Markers Correct
- **Claim**: Each role has correct markers (e.g., `<|im_start|>user`)
- **Rejection Criterion**: Wrong role marker or missing
- **Evidence**: Test each role with each format
- **Toyota Principle**: Poka-Yoke

#### CTM-04: Generation Prompt Appended
- **Claim**: Assistant prompt appended for generation (e.g., `<|im_start|>assistant\n`)
- **Rejection Criterion**: Missing generation prompt causes model confusion
- **Evidence**: Verify output ends with assistant start marker
- **Toyota Principle**: Standardized Work

#### CTM-05: No System Prompt Handling
- **Claim**: Conversation without system prompt formatted correctly
- **Rejection Criterion**: Error or corruption when no system prompt
- **Evidence**: Format user-only and user-assistant conversations
- **Toyota Principle**: Poka-Yoke

#### CTM-06: Multiple System Prompts
- **Claim**: Multiple system prompts handled (use first, warn)
- **Rejection Criterion**: Crash or silent use of wrong prompt
- **Evidence**: Test with 2+ system messages
- **Toyota Principle**: Jidoka

#### CTM-07: Empty Message Handling
- **Claim**: Empty content messages produce valid output
- **Rejection Criterion**: Crash or invalid template on empty content
- **Evidence**: Test with empty string content
- **Toyota Principle**: Poka-Yoke

#### CTM-08: Long Conversation
- **Claim**: Conversations with 50+ turns formatted correctly
- **Rejection Criterion**: Memory issues or corruption on long conversations
- **Evidence**: Format 100-turn conversation, verify structure
- **Toyota Principle**: Heijunka

#### CTM-09: Newline Handling
- **Claim**: Newlines in content preserved, not confused with template
- **Rejection Criterion**: Newlines stripped or cause template corruption
- **Evidence**: Test content with embedded newlines
- **Toyota Principle**: Poka-Yoke

#### CTM-10: Content Escaping
- **Claim**: Content containing template markers escaped/preserved
- **Rejection Criterion**: Content `<|im_end|>` breaks template
- **Evidence**: Test adversarial content with all special tokens
- **Toyota Principle**: Poka-Yoke (security)

### Section 5: Model-Specific Validation (CTX-01 to CTX-10)

#### CTX-01: Qwen2-0.5B-Instruct
- **Claim**: Qwen2 ChatML format produces correct output
- **Rejection Criterion**: Output doesn't match HuggingFace transformers output
- **Evidence**: Compare with `transformers.apply_chat_template()`
- **Toyota Principle**: Genchi Genbutsu

#### CTX-02: TinyLlama-1.1B-Chat
- **Claim**: TinyLlama LLaMA2 format produces correct output
- **Rejection Criterion**: Output doesn't match reference implementation
- **Evidence**: Compare with llama.cpp chat formatting
- **Toyota Principle**: Genchi Genbutsu

#### CTX-03: Mistral-7B-Instruct
- **Claim**: Mistral format correctly omits system prompt
- **Rejection Criterion**: System prompt incorrectly included
- **Evidence**: Verify no `<<SYS>>` markers in Mistral output
- **Toyota Principle**: Genchi Genbutsu

#### CTX-04: Phi-2
- **Claim**: Phi-2 `Instruct:/Output:` format correct
- **Rejection Criterion**: Wrong format applied to Phi
- **Evidence**: Test with microsoft/phi-2 model
- **Toyota Principle**: Genchi Genbutsu

#### CTX-05: Vicuna-7B
- **Claim**: Vicuna uses LLaMA2 format correctly
- **Rejection Criterion**: Wrong format or markers
- **Evidence**: Test with lmsys/vicuna-7b
- **Toyota Principle**: Genchi Genbutsu

#### CTX-06: OpenHermes-2.5
- **Claim**: OpenHermes uses ChatML format
- **Rejection Criterion**: Wrong format inferred
- **Evidence**: Test with teknium/OpenHermes-2.5
- **Toyota Principle**: Genchi Genbutsu

#### CTX-07: Yi-6B-Chat
- **Claim**: Yi uses ChatML format with correct tokens
- **Rejection Criterion**: Format or tokens wrong
- **Evidence**: Test with 01-ai/Yi-6B-Chat
- **Toyota Principle**: Genchi Genbutsu

#### CTX-08: Llama-2-7B-Chat
- **Claim**: LLaMA 2 reference model formats correctly
- **Rejection Criterion**: Output differs from Meta reference
- **Evidence**: Compare with official LLaMA 2 formatting code
- **Toyota Principle**: Genchi Genbutsu

#### CTX-09: Mixtral-8x7B-Instruct
- **Claim**: Mixtral uses Mistral format (no system)
- **Rejection Criterion**: System prompt incorrectly included
- **Evidence**: Test with mistralai/Mixtral-8x7B-Instruct
- **Toyota Principle**: Genchi Genbutsu

#### CTX-10: Regression Testing
- **Claim**: All previously working models continue to work
- **Rejection Criterion**: Any model that worked before now fails
- **Evidence**: CI test suite with all supported models
- **Toyota Principle**: Kaizen

### Section 6: Edge Cases (CTE-01 to CTE-10)

#### CTE-01: Empty Conversation
- **Claim**: Empty message array produces minimal valid output
- **Rejection Criterion**: Crash or invalid output on empty array
- **Evidence**: Test `format_conversation(&[])`
- **Toyota Principle**: Poka-Yoke

#### CTE-02: Unicode/Emoji Content
- **Claim**: Unicode and emoji preserved in formatted output
- **Rejection Criterion**: Characters mangled or lost
- **Evidence**: Test with CJK, emoji, combining characters
- **Toyota Principle**: Standardized Work

#### CTE-03: Very Long Content
- **Claim**: Messages with 100K+ characters handled
- **Rejection Criterion**: Crash or truncation without warning
- **Evidence**: Test with 100KB message content
- **Toyota Principle**: Heijunka

#### CTE-04: Binary Content
- **Claim**: Binary/null bytes in content don't crash
- **Rejection Criterion**: Panic on binary content
- **Evidence**: Test with `\x00`, `\xff` in content
- **Toyota Principle**: Poka-Yoke

#### CTE-05: RTL Text
- **Claim**: Right-to-left text (Arabic, Hebrew) preserved
- **Rejection Criterion**: RTL markers stripped or text reversed
- **Evidence**: Test with Arabic and Hebrew content
- **Toyota Principle**: Standardized Work

#### CTE-06: Mixed Roles
- **Claim**: Non-standard roles (e.g., "tool") handled gracefully
- **Rejection Criterion**: Crash on unknown role
- **Evidence**: Test with "tool", "function", custom roles
- **Toyota Principle**: Poka-Yoke

#### CTE-07: Whitespace Preservation
- **Claim**: Leading/trailing whitespace in content preserved
- **Rejection Criterion**: Whitespace stripped unexpectedly
- **Evidence**: Test with `"  content  "` vs `"content"`
- **Toyota Principle**: Standardized Work

#### CTE-08: Template Injection
- **Claim**: Malicious content cannot break template structure
- **Rejection Criterion**: Content `{% for %}` interpreted as template
- **Evidence**: Test with Jinja2 syntax in content
- **Toyota Principle**: Poka-Yoke (security)

#### CTE-09: Nested Quotes
- **Claim**: Quotes in content handled correctly
- **Rejection Criterion**: Quotes break JSON or template parsing
- **Evidence**: Test with `"He said \"hello\""`
- **Toyota Principle**: Poka-Yoke

#### CTE-10: Control Characters
- **Claim**: Control characters (tab, newline, etc.) handled
- **Rejection Criterion**: Control characters cause corruption
- **Evidence**: Test with `\t`, `\r`, `\n`, `\b`
- **Toyota Principle**: Poka-Yoke

### Section 7: Performance (CTP-01 to CTP-10)

#### CTP-01: Template Application Latency
- **Claim**: Single message formatting completes in < 100μs
- **Rejection Criterion**: Formatting takes > 100μs
- **Evidence**: Benchmark `format_message()` with criterion
- **Toyota Principle**: Heijunka

#### CTP-02: Conversation Formatting Latency
- **Claim**: 10-turn conversation formatted in < 1ms
- **Rejection Criterion**: Formatting takes > 1ms
- **Evidence**: Benchmark `format_conversation()` with 10 messages
- **Toyota Principle**: Heijunka

#### CTP-03: Memory Allocation
- **Claim**: Formatting allocates < 2x input size
- **Rejection Criterion**: Memory usage > 2x input
- **Evidence**: Profile memory with large conversations
- **Toyota Principle**: Muda elimination

#### CTP-04: No Heap Allocation on Hot Path
- **Claim**: Pre-allocated buffers reused
- **Rejection Criterion**: Allocation on every format call
- **Evidence**: Profile with allocation tracking
- **Toyota Principle**: Muda elimination

#### CTP-05: Template Parsing Once
- **Claim**: Template parsed once at load, not per-message
- **Rejection Criterion**: Re-parsing on each format call
- **Evidence**: Profile parse vs format timing
- **Toyota Principle**: Muda elimination

#### CTP-06: String Building Efficiency
- **Claim**: Output string built with minimal copies
- **Rejection Criterion**: O(n²) string concatenation
- **Evidence**: Profile with 1000+ messages
- **Toyota Principle**: Muda elimination

#### CTP-07: Concurrent Access
- **Claim**: Template engine is thread-safe (Send + Sync)
- **Rejection Criterion**: Data race or deadlock
- **Evidence**: Concurrent formatting from multiple threads
- **Toyota Principle**: Standardized Work

#### CTP-08: Cache Hit Rate
- **Claim**: Template cache hit rate > 99% in steady state
- **Rejection Criterion**: Frequent cache misses
- **Evidence**: Monitor cache metrics during chat session
- **Toyota Principle**: Kaizen

#### CTP-09: Startup Latency
- **Claim**: Template loading adds < 10ms to model load
- **Rejection Criterion**: Template loading > 10ms
- **Evidence**: Benchmark model load with/without template
- **Toyota Principle**: Heijunka

#### CTP-10: Scaling Behavior
- **Claim**: Performance scales linearly with message count
- **Rejection Criterion**: Superlinear scaling (O(n²) or worse)
- **Evidence**: Benchmark 1, 10, 100, 1000 messages
- **Toyota Principle**: Heijunka

### Section 8: APR Format Integration (CTA-01 to CTA-10)

#### CTA-01: chat_template in APR Metadata
- **Claim**: `chat_template` stored in APR metadata section
- **Rejection Criterion**: Field not written or not readable
- **Evidence**: Create APR with template, reload and verify
- **Toyota Principle**: Standardized Work

#### CTA-02: Backward Compatibility
- **Claim**: APR files without `chat_template` still load
- **Rejection Criterion**: Error on old APR files
- **Evidence**: Load APR files created before this feature
- **Toyota Principle**: Kaizen

#### CTA-03: Template Format in Metadata
- **Claim**: `chat_format` field indicates detected format
- **Rejection Criterion**: Format not stored or wrong
- **Evidence**: Verify `chat_format: "chatml"` in metadata
- **Toyota Principle**: Visual Control

#### CTA-04: Special Tokens in Metadata
- **Claim**: Special tokens stored in `special_tokens` object
- **Rejection Criterion**: Tokens not stored or incomplete
- **Evidence**: Verify all special tokens in APR metadata
- **Toyota Principle**: Standardized Work

#### CTA-05: apr import Preserves Template
- **Claim**: `apr import hf://...` includes chat template
- **Rejection Criterion**: Template lost during import
- **Evidence**: Import model, verify template in APR
- **Toyota Principle**: Standardized Work

#### CTA-06: apr export Preserves Template
- **Claim**: Exporting to GGUF/SafeTensors notes template
- **Rejection Criterion**: Template information lost on export
- **Evidence**: Export APR to GGUF, verify metadata preserved
- **Toyota Principle**: Standardized Work

#### CTA-07: apr inspect Shows Template
- **Claim**: `apr inspect` displays chat template info
- **Rejection Criterion**: Template not shown in inspect output
- **Evidence**: Run `apr inspect model.apr`, verify template shown
- **Toyota Principle**: Visual Control

#### CTA-08: apr validate Checks Template
- **Claim**: `apr validate` validates template syntax
- **Rejection Criterion**: Invalid template not detected
- **Evidence**: Create APR with malformed template, run validate
- **Toyota Principle**: Jidoka

#### CTA-09: Template Override via CLI
- **Claim**: `apr chat --template <format>` overrides auto-detect
- **Rejection Criterion**: CLI flag ignored
- **Evidence**: Test `--template chatml` on LLaMA model
- **Toyota Principle**: Flexibility

#### CTA-10: Template Discovery Order
- **Claim**: APR metadata > tokenizer_config.json > auto-detect
- **Rejection Criterion**: Wrong precedence order
- **Evidence**: Test with conflicting template sources
- **Toyota Principle**: Standardized Work

### Section 9: Security & Privacy (CTC-01 to CTC-10)

#### CTC-01: No Prompt Injection via Template
- **Claim**: Malicious template cannot execute code
- **Rejection Criterion**: Template can access filesystem or network
- **Evidence**: Test with `{% include '/etc/passwd' %}` in template
- **Toyota Principle**: Poka-Yoke

#### CTC-02: Content Escaping
- **Claim**: User content cannot break template structure
- **Rejection Criterion**: Content `<|im_end|>` terminates message early
- **Evidence**: Test with content containing all special tokens
- **Toyota Principle**: Poka-Yoke

#### CTC-03: Template Size Limit
- **Claim**: Templates > 100KB rejected
- **Rejection Criterion**: Arbitrarily large templates accepted
- **Evidence**: Test with 1MB template
- **Toyota Principle**: Poka-Yoke

#### CTC-04: Recursion Limit
- **Claim**: Recursive templates terminate
- **Rejection Criterion**: Stack overflow on recursive template
- **Evidence**: Test with deeply nested `{% for %}` loops
- **Toyota Principle**: Jidoka

#### CTC-05: Loop Iteration Limit
- **Claim**: Template loops limited to 10,000 iterations
- **Rejection Criterion**: Infinite loop possible
- **Evidence**: Test with `{% for i in range(999999999) %}`
- **Toyota Principle**: Jidoka

#### CTC-06: No Filesystem Access
- **Claim**: Templates cannot read files
- **Rejection Criterion**: Template can read local files
- **Evidence**: Test `{% include %}`, `{{ open().read() }}`
- **Toyota Principle**: Poka-Yoke

#### CTC-07: No Network Access
- **Claim**: Templates cannot make network requests
- **Rejection Criterion**: Template can fetch URLs
- **Evidence**: Test with `{{ fetch('http://...') }}`
- **Toyota Principle**: Poka-Yoke

#### CTC-08: No Environment Access
- **Claim**: Templates cannot read environment variables
- **Rejection Criterion**: Template can access `os.environ`
- **Evidence**: Test with `{{ env.SECRET_KEY }}`
- **Toyota Principle**: Poka-Yoke

#### CTC-09: Audit Logging
- **Claim**: Template application logged for security audit
- **Rejection Criterion**: No audit trail of template use
- **Evidence**: Verify log entries for template operations
- **Toyota Principle**: Visual Control

#### CTC-10: Input Sanitization
- **Claim**: Dangerous inputs sanitized before template
- **Rejection Criterion**: XSS-like attacks possible in output
- **Evidence**: Test with `<script>` in content
- **Toyota Principle**: Poka-Yoke

### Section 10: Toyota Way Compliance (CTT-01 to CTT-10)

#### CTT-01: Jidoka - Stop on Invalid Template
- **Claim**: Invalid templates cause immediate clear failure
- **Rejection Criterion**: Silent corruption or unclear error
- **Evidence**: Test with various malformed templates
- **Toyota Principle**: Jidoka

#### CTT-02: Genchi Genbutsu - Real Model Testing
- **Claim**: All supported models tested with real weights
- **Rejection Criterion**: Only tested with mocks
- **Evidence**: CI tests with real HuggingFace models
- **Toyota Principle**: Genchi Genbutsu

#### CTT-03: Kaizen - Extensible Format Registry
- **Claim**: New formats can be added without core changes
- **Rejection Criterion**: Adding format requires modifying enum
- **Evidence**: Document format registration API
- **Toyota Principle**: Kaizen

#### CTT-04: Standardized Work - Consistent API
- **Claim**: All formats implement same `ChatTemplateEngine` trait
- **Rejection Criterion**: Format-specific APIs required
- **Evidence**: Verify trait coverage for all formats
- **Toyota Principle**: Standardized Work

#### CTT-05: Poka-Yoke - Error-Proof Design
- **Claim**: Common mistakes prevented by design
- **Rejection Criterion**: Easy to misuse API
- **Evidence**: Review API ergonomics, test with new users
- **Toyota Principle**: Poka-Yoke

#### CTT-06: Visual Control - Observable State
- **Claim**: Current template format visible in logs/UI
- **Rejection Criterion**: Template state hidden
- **Evidence**: Verify format shown in `apr chat` output
- **Toyota Principle**: Visual Control

#### CTT-07: Heijunka - Level Performance
- **Claim**: Performance consistent across formats
- **Rejection Criterion**: Some formats 10x slower
- **Evidence**: Benchmark all formats, verify within 2x
- **Toyota Principle**: Heijunka

#### CTT-08: Muda Elimination - No Waste
- **Claim**: No unnecessary allocations or copies
- **Rejection Criterion**: Profiler shows waste
- **Evidence**: Profile hot paths, eliminate waste
- **Toyota Principle**: Muda elimination

#### CTT-09: Hansei - Reflection on Failures
- **Claim**: Failed template loads produce actionable diagnostics
- **Rejection Criterion**: Cryptic error messages
- **Evidence**: Review error messages for clarity
- **Toyota Principle**: Hansei

#### CTT-10: Customer Focus - Works Out of Box
- **Claim**: Common models work without configuration
- **Rejection Criterion**: Manual template config required
- **Evidence**: Test `apr chat` with popular models
- **Toyota Principle**: Customer Focus

---

## Part V: Implementation Roadmap

### Phase 1: Core Template Engine (Week 1-2)

1. **Dependency Management**:
   - Add `minijinja = { version = "2.0", features = ["loader", "serde"] }` to `Cargo.toml`

2. Create `src/text/chat_template.rs` with:
   - `ChatTemplateEngine` trait
   - `TemplateFormat` enum
   - `minijinja::Environment` setup (using `add_template_owned`)
   - Format-specific implementations

3. Add unit tests for each format

### Phase 2: HuggingFace Integration (Week 2-3)

1. Update `src/format/converter.rs`:
   - Download `tokenizer_config.json` in HF import
   - Parse `chat_template` field
   - Extract special tokens

2. Update `crates/apr-cli/src/commands/chat.rs`:
   - Load template from tokenizer_config.json
   - Use `ChatTemplateEngine` instead of hardcoded format

### Phase 3: APR Format Extension (Week 3-4)

1. Add `chat_template` to APR metadata schema
2. Update `apr import` to store template
3. Update `apr inspect` to show template
4. Update `apr validate` to check template

### Phase 4: Testing & Validation (Week 4-5)

1. Implement 100-point falsification checklist as tests
2. Test with real models: TinyLlama, Phi-2, Qwen2, Mistral
3. Performance benchmarking
4. Security audit

### Phase 5: Documentation & Release (Week 5-6)

1. Update CLAUDE.md with chat template info
2. Add examples to documentation
3. Release notes
4. Changelog

---

## Part VI: Files to Create/Modify

### New Files

| File | Purpose | LOC Est. |
|------|---------|----------|
| `src/text/chat_template.rs` | Chat template engine (minijinja wrapper) | 400-600 |
| `tests/chat_template_tests.rs` | Comprehensive tests | 300-500 |

### Modified Files

| File | Changes |
|------|---------|
| `Cargo.toml` | Add `minijinja` dependency |
| `src/text/mod.rs` | Add `pub mod chat_template;` |
| `src/format/converter.rs` | Download tokenizer_config.json |
| `crates/apr-cli/src/commands/chat.rs` | Use ChatTemplateEngine |
| `crates/apr-cli/src/commands/import.rs` | Download tokenizer_config.json |
| `docs/specifications/APR-SPEC.md` | Document chat_template metadata |
| `CLAUDE.md` | Document chat template support |

---

## Part VII: Final Verification Matrix

### 7.1 Model Support Matrix

The following matrix defines all supported models with verified working commands. This matrix is validated by `scripts/verify-chat-models.sh` using bashrs methodology.

<!-- BEGIN MODEL MATRIX - Auto-validated by scripts/verify-chat-models.sh -->

| Model | Format | Template | Size | Command | Status |
|-------|--------|----------|------|---------|--------|
| TinyLlama-1.1B-Chat | GGUF Q4_K_M | LLaMA2 | 638MB | `apr chat ~/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | |
| Qwen2-0.5B-Instruct | SafeTensors | ChatML | 1.1GB | `apr chat ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/*/model.safetensors` | |
| Phi-2 | SafeTensors | Phi | 5.6GB | `apr chat ~/.cache/huggingface/hub/models--microsoft--phi-2/snapshots/*/model.safetensors` | |
| Mistral-7B-Instruct | GGUF Q4_K_M | Mistral | 4.1GB | `apr chat ~/.apr/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` | |
| OpenHermes-2.5-Mistral | GGUF Q4_K_M | ChatML | 4.1GB | `apr chat ~/.apr/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf` | |
| Vicuna-7B | GGUF Q4_K_M | LLaMA2 | 4.1GB | `apr chat ~/.apr/models/vicuna-7b-v1.5.Q4_K_M.gguf` | |
| Yi-6B-Chat | GGUF Q4_K_M | ChatML | 3.5GB | `apr chat ~/.apr/models/yi-6b-chat.Q4_K_M.gguf` | |

<!-- END MODEL MATRIX -->

### 7.2 bashrs Verification Script

Create `scripts/verify-chat-models.sh` following bashrs purified standards:

```bash
#!/usr/bin/env bash
# scripts/verify-chat-models.sh
# Verified with: bashrs lint && bashrs purify
#
# Model Chat Template Verification Matrix
# Toyota Way: Genchi Genbutsu - test with real models

set -euo pipefail

# ============================================================================ 
# Configuration
# ============================================================================ 

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly APR_BIN="${PROJECT_ROOT}/target/release/apr"
readonly MODELS_DIR="${HOME}/.apr/models"
readonly RESULTS_FILE="${PROJECT_ROOT}/target/model-verification-results.json"
readonly README_FILE="${PROJECT_ROOT}/README.md"

# Test prompt (simple, deterministic)
readonly TEST_PROMPT="What is 2+2?"
readonly MAX_TOKENS=16
readonly TIMEOUT_SECS=30

# ============================================================================ 
# Model Registry (Source of Truth)
# ============================================================================ 

declare -A MODEL_URLS=(
    ["tinyllama"]="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    ["mistral"]="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    ["openhermes"]="https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
    ["vicuna"]="https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF/resolve/main/vicuna-7b-v1.5.Q4_K_M.gguf"
    ["yi"]="https://huggingface.co/TheBloke/Yi-6B-Chat-GGUF/resolve/main/yi-6b-chat.Q4_K_M.gguf"
)

declare -A MODEL_FILES=(
    ["tinyllama"]="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    ["mistral"]="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    ["openhermes"]="openhermes-2.5-mistral-7b.Q4_K_M.gguf"
    ["vicuna"]="vicuna-7b-v1.5.Q4_K_M.gguf"
    ["yi"]="yi-6b-chat.Q4_K_M.gguf"
)

declare -A MODEL_TEMPLATES=(
    ["tinyllama"]="llama2"
    ["mistral"]="mistral"
    ["openhermes"]="chatml"
    ["vicuna"]="llama2"
    ["yi"]="chatml"
)

declare -A MODEL_SIZES=(
    ["tinyllama"]="638M"
    ["mistral"]="4.1G"
    ["openhermes"]="4.1G"
    ["vicuna"]="4.1G"
    ["yi"]="3.5G"
)

# ============================================================================ 
# Utility Functions
# ============================================================================ 

log_info() {
    printf "\033[0;34m[INFO]\033[0m %s\n" "$1"
}

log_success() {
    printf "\033[0;32m[PASS]\033[0m %s\n" "$1"
}

log_failure() {
    printf "\033[0;31m[FAIL]\033[0m %s\n" "$1"
}

log_skip() {
    printf "\033[0;33m[SKIP]\033[0m %s\n" "$1"
}

# ============================================================================ 
# Verification Functions
# ============================================================================ 

verify_apr_binary() {
    if [[ ! -x "${APR_BIN}" ]]; then
        log_info "Building apr-cli with inference feature..."
        cargo build --release -p apr-cli --features inference \
            --manifest-path "${PROJECT_ROOT}/Cargo.toml"
    fi

    if [[ ! -x "${APR_BIN}" ]]; then
        log_failure "apr binary not found at ${APR_BIN}"
        return 1
    fi

    log_success "apr binary ready: ${APR_BIN}"
}

download_model() {
    local model_key="$1"
    local url="${MODEL_URLS[$model_key]}"
    local filename="${MODEL_FILES[$model_key]}"
    local filepath="${MODELS_DIR}/${filename}"

    if [[ -f "${filepath}" ]]; then
        log_info "Model already cached: ${filename}"
        return 0
    fi

    log_info "Downloading ${model_key} (${MODEL_SIZES[$model_key]})..."
    mkdir -p "${MODELS_DIR}"

    if curl -L -o "${filepath}" "${url}" 2>/dev/null; then
        log_success "Downloaded: ${filename}"
        return 0
    else
        log_failure "Failed to download: ${filename}"
        return 1
    fi
}

verify_model() {
    local model_key="$1"
    local filename="${MODEL_FILES[$model_key]}"
    local filepath="${MODELS_DIR}/${filename}"
    local template="${MODEL_TEMPLATES[$model_key]}"

    if [[ ! -f "${filepath}" ]]; then
        log_skip "${model_key}: model file not present"
        echo "skip"
        return 0
    fi

    log_info "Testing ${model_key} (template: ${template})..."

    # Run chat with timeout and capture output
    local output
    local exit_code=0

    output=$(echo -e "${TEST_PROMPT}\n/quit" | \
        timeout "${TIMEOUT_SECS}" "${APR_BIN}" chat "${filepath}" \
            --max-tokens "${MAX_TOKENS}" \
            --temperature 0.1 2>&1) || exit_code=$?

    # Check for success indicators
    if [[ ${exit_code} -eq 0 ]] && \
       echo "${output}" | grep -q "Loaded.*format" && \
       echo "${output}" | grep -q "Goodbye"; then
        log_success "${model_key}: chat completed successfully"
        echo "pass"
        return 0
    else
        log_failure "${model_key}: chat failed (exit=${exit_code})"
        echo "fail"
        return 1
    fi
}

# ============================================================================ 
# Matrix Generation
# ============================================================================ 

generate_results_json() {
    local results_json="{"
    results_json+='"timestamp": "'$(date -Iseconds)'",'
    results_json+='"models": {'

    local first=true
    for model_key in "${!MODEL_FILES[@]}"; do
        local result
        result=$(verify_model "${model_key}")

        if [[ "${first}" == "true" ]]; then
            first=false
        else
            results_json+=','
        fi

        results_json+='"'"${model_key}"'": {"
        results_json+='"file": "'"${MODEL_FILES[$model_key]}""",','
        results_json+='"template": "'"${MODEL_TEMPLATES[$model_key]}""",','
        results_json+='"size": "'"${MODEL_SIZES[$model_key]}""",','
        results_json+='"status": "'"${result}"""'
        results_json+='}'
    done

    results_json+='}}'

    mkdir -p "$(dirname "${RESULTS_FILE}")"
    echo "${results_json}" > "${RESULTS_FILE}"
    log_success "Results written to: ${RESULTS_FILE}"
}

update_readme_matrix() {
    if [[ ! -f "${RESULTS_FILE}" ]]; then
        log_failure "Results file not found: ${RESULTS_FILE}"
        return 1
    fi

    log_info "Updating README.md model matrix..."

    # Generate markdown table from results
    local table="| Model | Format | Template | Size | Status |\n"
table+="|-------|--------|----------|------|--------|\n"

    for model_key in tinyllama mistral openhermes vicuna yi; do
        local status
        status=$(jq -r ".models.${model_key}.status // \"unknown\"" "${RESULTS_FILE}")
        local template="${MODEL_TEMPLATES[$model_key]}"
        local size="${MODEL_SIZES[$model_key]}"
        local filename="${MODEL_FILES[$model_key]}"

        local status_emoji
        case "${status}" in
            pass) status_emoji="✅ Pass" ;;
            fail) status_emoji="❌ Fail" ;;
            skip) status_emoji="⏭️ Skip" ;;
            *)    status_emoji="❓ Unknown" ;;
        esac

        table+="| ${model_key} | GGUF | ${template} | ${size} | ${status_emoji} |\n"
    done

    # Update README between markers (if markers exist)
    if grep -q "<!-- MODEL_MATRIX_START -->" "${README_FILE}"; then
        sed -i '/<!-- MODEL_MATRIX_START -->/,/<!-- MODEL_MATRIX_END -->/c\<!-- MODEL_MATRIX_START -->\n'"$(echo -e "${table}")"'\n<!-- MODEL_MATRIX_END -->" "${README_FILE}"
        log_success "README.md matrix updated"
    else
        log_info "README.md markers not found, skipping update"
    fi
}

# ============================================================================ 
# Main Entry Point
# ============================================================================ 

main() {
    local cmd="${1:-verify}"

    case "${cmd}" in
        verify)
            verify_apr_binary
            generate_results_json
            ;; 
        download)
            local model="${2:-all}"
            if [[ "${model}" == "all" ]]; then
                for key in "${!MODEL_URLS[@]}"; do
                    download_model "${key}" || true
                done
            else
                download_model "${model}"
            fi
            ;; 
        update-readme)
            update_readme_matrix
            ;; 
        full)
            verify_apr_binary
            for key in tinyllama; do  # Start with smallest
                download_model "${key}" || true
            done
            generate_results_json
            update_readme_matrix
            ;; 
        *)
            echo "Usage: $0 {verify|download [model]|update-readme|full}"
            exit 1
            ;; 
    esac
}

main "$@"
```

### 7.3 README.md Integration

Add the following to `README.md` for automatic matrix updates:

```markdown
## Supported Chat Models

The following models have been verified with `apr chat`:

<!-- MODEL_MATRIX_START -->
| Model | Format | Template | Size | Status |
|-------|--------|----------|------|--------|
| tinyllama | GGUF | llama2 | 638M | ✅ Pass |
| mistral | GGUF | mistral | 4.1G | ⏭️ Skip |
| openhermes | GGUF | chatml | 4.1G | ⏭️ Skip |
| vicuna | GGUF | llama2 | 4.1G | ⏭️ Skip |
| yi | GGUF | chatml | 3.5G | ⏭️ Skip |
<!-- MODEL_MATRIX_END -->

### Quick Start

```bash
# Download and test TinyLlama (smallest, recommended for testing)
./scripts/verify-chat-models.sh download tinyllama
./scripts/verify-chat-models.sh verify

# Interactive chat
apr chat ~/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# With options
apr chat ~/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --temperature 0.7 \
    --max-tokens 128 \
    --inspect
```
```

## 7.4 CI Integration

Add to `.github/workflows/ci.yml`:

```yaml
  model-verification:
    name: Model Chat Verification
    runs-on: ubuntu-latest
    needs: [test]
    steps:
      - uses: actions/checkout@v4

      - name: Install bashrs
        run: cargo install bashrs

      - name: Lint verification script
        run: bashrs lint scripts/verify-chat-models.sh

      - name: Build apr-cli
        run: cargo build --release -p apr-cli --features inference

      - name: Download TinyLlama (CI model)
        run: ./scripts/verify-chat-models.sh download tinyllama

      - name: Verify model chat
        run: ./scripts/verify-chat-models.sh verify

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: model-verification-results
          path: target/model-verification-results.json
          retention-days: 30
```

### 7.5 Falsification Test Integration

Each model in the matrix maps to falsification tests:

| Model | Checklist Items | Test File |
|-------|-----------------|-----------|
| TinyLlama | CTX-02, CTM-01-10 | `tests/chat_template_tinyllama.rs` |
| Qwen2 | CTX-01, CTS-03 | `tests/chat_template_qwen2.rs` |
| Mistral | CTX-03, CTM-05 | `tests/chat_template_mistral.rs` |
| Phi-2 | CTX-04 | `tests/chat_template_phi.rs` |
| OpenHermes | CTX-06, CTS-03 | `tests/chat_template_chatml.rs` |

### 7.6 Verification Commands Reference

```bash
# ============================================================================ 
# Model Download Commands
# ============================================================================ 

# TinyLlama 1.1B Chat (638MB) - Recommended for testing
curl -L -o ~/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Mistral 7B Instruct (4.1GB)
curl -L -o ~/.apr/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# OpenHermes 2.5 (4.1GB) - ChatML format
curl -L -o ~/.apr/models/openhermes-2.5-mistral-7b.Q4_K_M.gguf \
    "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf"

# ============================================================================ 
# Chat Commands
# ============================================================================ 

# Basic chat
apr chat ~/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# With parameters
apr chat ~/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-tokens 128

# With inspection (shows tok/s, debug info)
apr chat ~/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --inspect

# With system prompt
apr chat ~/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --system "You are a helpful coding assistant."

# ============================================================================ 
# Verification Commands
# ============================================================================ 

# Lint the verification script
bashrs lint scripts/verify-chat-models.sh

# Run full verification
./scripts/verify-chat-models.sh full

# Verify specific model
./scripts/verify-chat-models.sh verify

# Update README matrix
./scripts/verify-chat-models.sh update-readme
```

---

## Part VIII: bashrs Probar Playbook Verification

### 8.1 Overview

All model formats, chat templates, and CLI commands **MUST** achieve 100% coverage through bashrs probar playbook testing. This ensures Toyota Way compliance (Genchi Genbutsu) by testing with real models and real state transitions.

**Verification Checklist:** [`docs/model-verification-checklist.md`](../model-verification-checklist.md)

### 8.2 Probar Playbook Architecture

The verification system uses bashrs probar's three-tier testing methodology:

| Tier | Test Suite | Coverage Target | Purpose |
|------|------------|-----------------|---------|
| **Parser** | `parser_probar_testing.rs` | ≥80% per category | Syntax coverage for all template formats |
| **Simulation** | `simulation_probar_testing.rs` | 100 S-codes | Edge case discovery (unicode, boundaries, nesting) |
| **Falsification** | `falsification_probar_testing.rs` | 130 F-codes | Popper falsification (no false positives) |

### 8.3 Template Format Playbook Coverage

Each template format requires a dedicated playbook with state machine verification:

```yaml
# playbooks/chat_template.yaml
version: "1.0"
name: "Chat Template State Machine"

machine:
  id: "chat_template_engine"
  initial: "idle"

  states:
    idle:
      description: "Template engine ready"
    detecting:
      description: "Auto-detecting template format"
    loading:
      description: "Loading template from config"
    formatting:
      description: "Formatting conversation"
    complete:
      final_state: true
    error:
      final_state: true

  transitions:
    - from: "idle"
      to: "detecting"
      event: "model_loaded"
    - from: "detecting"
      to: "loading"
      event: "format_detected"
    - from: "loading"
      to: "formatting"
      event: "template_compiled"
    - from: "formatting"
      to: "complete"
      event: "conversation_formatted"

coverage:
  features:
    templates:
      - chatml_basic           # ChatML single message
      - chatml_multiturn       # ChatML conversation
      - chatml_system          # ChatML with system prompt
      - llama2_basic           # LLaMA2 single message
      - llama2_system          # LLaMA2 with <<SYS>>
      - mistral_basic          # Mistral (no system)
      - phi_basic              # Phi Instruct/Output
      - alpaca_basic           # Alpaca ### format
      - raw_passthrough        # Raw fallback
      - custom_jinja2          # Custom HF template

    auto_detection:
      - detect_chatml_tokens   # <|im_start|> detection
      - detect_llama2_tokens   # [INST] detection
      - detect_mistral_name    # Model name detection
      - detect_phi_name        # Phi model detection
      - detect_fallback_raw    # Unknown model fallback

    edge_cases:
      - empty_conversation     # [] input
      - unicode_content        # CJK, emoji, RTL
      - special_token_escape   # Content with <|im_end|>
      - long_conversation      # 100+ turns
      - binary_content         # Null bytes

  thresholds:
    minimum_coverage: 95
    target_coverage: 100
```

### 8.4 CLI Command Coverage Matrix

Every `apr` CLI command that interacts with chat templates requires probar verification:

| Command | Probar Test | Coverage Requirement |
|---------|-------------|---------------------|
| `apr chat` | `CTM-*` (Multi-turn) | 100% |
| `apr import` | `CTA-05` (Template preservation) | 100% |
| `apr export` | `CTA-06` (Template export) | 100% |
| `apr inspect` | `CTA-07` (Template display) | 100% |
| `apr validate` | `CTA-08` (Template validation) | 100% |
| `apr convert` | `CTA-*` (Format conversion) | 100% |

### 8.5 Probar Test Implementation

```rust
// tests/chat_template_probar_testing.rs
use jugar_probar::{gui_coverage, TuiSnapshot};

#[test]
fn test_chat_template_format_coverage() {
    let mut gui = gui_coverage! {
        buttons: [
            "chatml_basic", "chatml_multiturn", "chatml_system",
            "llama2_basic", "llama2_system",
            "mistral_basic", "phi_basic", "alpaca_basic",
            "raw_passthrough", "custom_jinja2"
        ],
        screens: ["formatted", "error"]
    };

    let tests = [
        (ChatMLTemplate::new(), "Hello", "chatml_basic"),
        (Llama2Template::new(), "Hello", "llama2_basic"),
        (MistralTemplate::new(), "Hello", "mistral_basic"),
        (PhiTemplate::new(), "Hello", "phi_basic"),
        (AlpacaTemplate::new(), "Hello", "alpaca_basic"),
        (RawTemplate::new(), "Hello", "raw_passthrough"),
    ];

    for (template, input, feature) in tests {
        let messages = vec![ChatMessage::user(input)];
        let result = template.format_conversation(&messages);

        gui.click(feature);
        gui.visit(if result.is_ok() { "formatted" } else { "error" });

        // Snapshot for determinism verification
        let frame = format!("{}:{}", feature, result.is_ok());
        let snapshot = TuiSnapshot::from_frame(feature, &frame);
        assert!(!snapshot.hash.is_empty());
    }

    println!("\nTemplate Format Coverage: {:.1}%", gui.percent());
    assert!(gui.meets(95.0), "Template coverage >= 95%");
}

#[test]
fn test_chat_template_auto_detection_coverage() {
    let mut gui = gui_coverage! {
        buttons: [
            "detect_chatml_tokens", "detect_llama2_tokens",
            "detect_mistral_name", "detect_phi_name", "detect_fallback_raw"
        ],
        screens: ["detected", "fallback"]
    };

    let tests = [
        ("Qwen2-0.5B-Instruct", TemplateFormat::ChatML, "detect_chatml_tokens"),
        ("TinyLlama-1.1B-Chat", TemplateFormat::Llama2, "detect_llama2_tokens"),
        ("Mistral-7B-Instruct", TemplateFormat::Mistral, "detect_mistral_name"),
        ("phi-2", TemplateFormat::Phi, "detect_phi_name"),
        ("unknown-model", TemplateFormat::Raw, "detect_fallback_raw"),
    ];

    for (model_name, expected_format, feature) in tests {
        let detected = detect_format_from_name(model_name);
        gui.click(feature);
        gui.visit(if detected == expected_format { "detected" } else { "fallback" });

        assert_eq!(detected, expected_format, "Format detection for {}", model_name);
    }

    println!("\nAuto-Detection Coverage: {:.1}%", gui.percent());
    assert!(gui.meets(100.0), "Auto-detection coverage must be 100%");
}
```

### 8.6 Falsification Test Mapping

Each falsification checklist item (CTL-*, CTS-*, CTA-*, etc.) maps to probar F-codes:

| Checklist | F-Code Range | Description |
|-----------|--------------|-------------|
| CTL-01 to CTL-10 | F001-F010 | Template Loading |
| CTS-01 to CTS-10 | F011-F020 | Special Tokens |
| CTA-01 to CTA-10 | F021-F030 | Auto-Detection |
| CTM-01 to CTM-10 | F031-F040 | Multi-Turn |
| CTX-01 to CTX-10 | F041-F050 | Model-Specific |
| CTE-01 to CTE-10 | F051-F060 | Edge Cases |
| CTP-01 to CTP-10 | F061-F070 | Performance |
| CTA-01 to CTA-10 | F071-F080 | APR Integration |
| CTC-01 to CTC-10 | F081-F090 | Security |
| CTT-01 to CTT-10 | F091-F100 | Toyota Way |

### 8.7 CI Integration with bashrs

```yaml
# .github/workflows/ci.yml
chat-template-probar:
  name: Chat Template Probar Verification
  runs-on: ubuntu-latest
  needs: [test]
  steps:
    - uses: actions/checkout@v4

    - name: Install bashrs
      run: cargo install bashrs

    - name: Lint playbooks
      run: |
        bashrs lint playbooks/chat_template.yaml
        bashrs lint scripts/verify-chat-models.sh

    - name: Run probar tests
      run: |
        cargo test --test chat_template_probar_testing -- --nocapture

    - name: Verify 100% coverage
      run: |
        # Extract coverage from test output
        cargo test --test chat_template_probar_testing 2>&1 | \
          grep "Coverage:" | \
          awk '{if ($NF < 100) exit 1}'

    - name: Generate verification checklist
      run: |
        ./scripts/verify-chat-models.sh verify
        ./scripts/verify-chat-models.sh update-readme

    - name: Upload probar results
      uses: actions/upload-artifact@v4
      with:
        name: probar-verification
        path: |
          target/model-verification-results.json
          target/probar-coverage-report.json
```

### 8.8 Verification Output Format

The verification system outputs results in a standardized JSON format:

```json
{
  "timestamp": "2026-01-06T12:00:00Z",
  "spec_version": "1.2.0",
  "probar_version": "6.49.0",
  "coverage": {
    "templates": {
      "chatml": { "tests": 10, "passed": 10, "coverage": 100.0 },
      "llama2": { "tests": 10, "passed": 10, "coverage": 100.0 },
      "mistral": { "tests": 8, "passed": 8, "coverage": 100.0 },
      "phi": { "tests": 6, "passed": 6, "coverage": 100.0 },
      "alpaca": { "tests": 6, "passed": 6, "coverage": 100.0 },
      "raw": { "tests": 4, "passed": 4, "coverage": 100.0 }
    },
    "auto_detection": { "tests": 5, "passed": 5, "coverage": 100.0 },
    "edge_cases": { "tests": 10, "passed": 10, "coverage": 100.0 },
    "falsification": { "tests": 100, "passed": 100, "coverage": 100.0 }
  },
  "models": {
    "tinyllama": { "status": "pass", "template": "llama2" },
    "qwen2": { "status": "pass", "template": "chatml" },
    "mistral": { "status": "pass", "template": "mistral" },
    "phi2": { "status": "pass", "template": "phi" }
  },
  "overall_status": "PASS",
  "overall_coverage": 100.0
}
```

---

## Part IX: Reliable Demo Best Practices

To ensure `apr` provides a world-class developer experience, we adopt demo best practices from industry leaders like Hugging Face, Ollama, llama.cpp, and llamafile. Every demo and example in the `aprender` ecosystem must adhere to the following standards to guarantee reliability, reproducibility, and ease of use.

### 9.1 The "Zero-Config" Guarantee

**Inspiration:** llamafile, Ollama

**Principle:** A user should be able to run a demo with a single command, without manual prerequisite installation or complex configuration.

**Checklist Item (RDB-01):**
- [ ] Demos must use `cargo run --example <name>` or `apr run <url>` as the primary entry point.
- [ ] Dependencies (models, datasets) must be automatically downloaded and cached if missing.
- [ ] No manual `pip install` or system package requirements unless absolutely unavoidable (and then checked for).

### 9.2 Clear Prerequisites & Environment Isolation

**Inspiration:** Hugging Face Spaces (Dockerfiles), Python `venv`

**Principle:** Avoid "it works on my machine" by strictly defining the execution environment.

**Checklist Item (RDB-02):**
- [ ] All examples must have a corresponding entry in `Cargo.toml` with specific versions.
- [ ] If external tools (e.g., `ffmpeg`) are required, the demo must check for their presence and provide a clear, actionable error message if missing (e.g., "Please install ffmpeg: `sudo apt install ffmpeg`").
- [ ] Use `apr.lock` or `Cargo.lock` to pin model versions and library dependencies.

### 9.3 Interactive & Non-Interactive Modes

**Inspiration:** llama.cpp (`main` vs `server`), Unix CLI philosophy

**Principle:** Demos should be usable both by humans (interactive) and scripts (non-interactive/pipes).

**Checklist Item (RDB-03):**
- [ ] Detect TTY: If running in a terminal, offer an interactive TUI or prompt.
- [ ] Support Stdin/Stdout: If piped, accept input from stdin and output raw data to stdout (JSON/text) for composition.
- [ ] Provide CLI flags to force specific modes (`--interactive`, `--batch`).

### 9.4 Robust Error Recovery & informative Feedback

**Inspiration:** Rust `Result` type, Elm/Elixir compiler messages

**Principle:** Errors should be helpful, not scary. Don't just crash; explain *why* and *how to fix it*.

**Checklist Item (RDB-04):**
- [ ] No `unwrap()` or `expect()` in demo main loops. Handle errors gracefully.
- [ ] If a model fails to load (checksum mismatch, corruption), offer to re-download it.
- [ ] If running on CPU when GPU was requested, log a warning but fallback gracefully (or fail with a clear "GPU required" if strict).

### 9.5 Performance Transparency

**Inspiration:** llama.cpp output stats

**Principle:** Users need to know if the system is performing as expected.

**Checklist Item (RDB-05):**
- [ ] Display key metrics: Load time, Tokens/sec, Memory usage.
- [ ] Log the active backend (e.g., "Using AVX2", "Using Metal/M1", "Using CUDA").
- [ ] Provide a `--verbose` flag for deep debugging (layer timings, tensor shapes).

### 9.6 Model Provenance & Licensing

**Inspiration:** Hugging Face Model Cards

**Principle:** Be transparent about what model is running and its license.

**Checklist Item (RDB-06):**
- [ ] Demos must print the Model Name, Version/Revision, and License (e.g., Apache 2.0, Llama Community) at startup.
- [ ] Link to the original model card or paper in the demo output or help text.

---

## Appendix A: Reference Implementations

### Hugging Face transformers

```python
# Reference: transformers.apply_chat_template()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
messages = [{"role": "user", "content": "Hello"}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

### llama.cpp

```cpp
// Reference: llama_chat_apply_template()
// https://github.com/ggerganov/llama.cpp/blob/master/common/common.cpp
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **BOS** | Beginning of Sequence token |
| **EOS** | End of Sequence token |
| **ChatML** | Chat Markup Language format (`<\|im_start\|>...<\|im_end\|>`) |
| **Jinja2** | Python templating language used by HuggingFace |
| **GQA** | Grouped Query Attention (TinyLlama uses 4 KV heads) |
| **tokenizer_config.json** | HuggingFace file containing chat_template |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.4.0 | 2026-01-06 | Add Part IX: Reliable Demo Best Practices inspired by HF/Ollama/llama.cpp. |
| 1.3.0 | 2026-01-06 | Add Part VIII: bashrs probar playbook verification, peer-reviewed citations for formal verification, and model-verification-checklist.md |
| 1.2.0 | 2026-01-06 | Refine minijinja configuration, correct error handling, and add Model Verification Matrix. |
| 1.1.0 | 2026-01-06 | Mandate minijinja for Jinja2 compatibility and add Custom template support. |
| 1.0.0 | 2026-01-06 | Initial specification |