#![allow(clippy::disallowed_methods)]
//! Chat Template Probar Testing
//!
//! This module implements 100% probar playbook coverage for the chat template system.
//! Following bashrs probar methodology with three testing tiers:
//!
//! 1. **Parser Coverage** - All template formats and auto-detection
//! 2. **Simulation** - Edge cases (unicode, boundaries, nesting)
//! 3. **Falsification** - Popper methodology (no false positives)
//!
//! Reference: docs/specifications/chat-template-improvement-spec.md v1.3.0
//! Playbook: playbooks/chat_template.yaml

use aprender::text::chat_template::{
    auto_detect_template, create_template, detect_format_from_name, AlpacaTemplate, ChatMLTemplate,
    ChatMessage, ChatTemplateEngine, HuggingFaceTemplate, Llama2Template, MistralTemplate,
    PhiTemplate, RawTemplate, SpecialTokens, TemplateFormat,
};
use jugar_probar::gui_coverage;

include!("includes/chat_template_probar.rs");
include!("includes/chat_template_probar_part_02.rs");
