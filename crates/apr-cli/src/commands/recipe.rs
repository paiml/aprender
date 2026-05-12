//! Modelfile-style `apr create` recipe parser + message renderer (CRUX-A-16).
//!
//! Contract: `contracts/crux-A-16-v1.yaml`.
//!
//! Pure classifier — no I/O, no network. Two algorithm-level claims we
//! discharge here:
//!
//! 1. Strict-schema validator rejects unknown top-level keys and wrong
//!    types. This is the algorithm-level sub-claim of the ollama-parity
//!    "Recipe is a closed grammar" expectation.
//!
//! 2. `render_messages(recipe, user_prompt, cli_system_override)` produces
//!    a message vector with role=system at position 0 whose content is:
//!    - `cli_system_override` if present (last-writer wins), else
//!    - `recipe.system` if present, else
//!    - no system message is emitted and messages[0] is the user prompt.
//!    This is the full algorithm-level discharge of FALSIFY-CRUX-A-16-002
//!    and FALSIFY-CRUX-A-16-003.
//!
//! Full discharge of FALSIFY-CRUX-A-16-001 (apr create tag persists in
//! `apr ls`) and FALSIFY-CRUX-A-16-004 (apr/ollama token-level parity)
//! both require integration harnesses and are left open.

use serde::Deserialize;

/// Top-level recipe document.
///
/// `#[serde(deny_unknown_fields)]` enforces the strict-schema invariant:
/// any unknown top-level key produces a parse error. This matches
/// ollama's Modelfile strictness and is the first line of defense
/// against typo-drift across recipes.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// Base model reference. Required. Mirrors `FROM` in a Modelfile.
    /// Resolution to an actual base model is the `apr create` caller's
    /// concern — this classifier only checks the string is non-empty.
    pub from: String,

    /// Default SYSTEM prompt. Injected at messages[0] on every chat
    /// invocation unless the caller provides `--system` on the CLI.
    #[serde(default)]
    pub system: Option<String>,

    /// Chat-template override (Jinja2 / minijinja). Opaque to this
    /// classifier — validated at render time in the chat template
    /// engine, not here.
    #[serde(default)]
    pub template: Option<String>,

    /// Inference-time parameter overrides.
    #[serde(default)]
    pub parameters: Option<RecipeParameters>,
}

/// Structured inference parameters. `deny_unknown_fields` applies here
/// too so typos like `temprature` fail loud.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeParameters {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub num_ctx: Option<u32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// Errors the Recipe parser/validator can emit.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum RecipeError {
    #[error("recipe YAML parse error: {0}")]
    Parse(String),
    #[error("recipe.from is required and must be a non-empty string")]
    MissingFrom,
}

/// Parse a recipe from a YAML (or JSON, which is a subset) string.
///
/// Returns `Err` on:
/// - YAML syntax errors
/// - Unknown top-level keys (strict schema)
/// - Missing or empty `from`
pub fn parse_recipe(src: &str) -> Result<Recipe, RecipeError> {
    let r: Recipe = serde_yaml::from_str(src).map_err(|e| RecipeError::Parse(e.to_string()))?;
    if r.from.trim().is_empty() {
        return Err(RecipeError::MissingFrom);
    }
    Ok(r)
}

/// Minimal chat message with a role and content. This mirrors the OpenAI
/// chat schema without carrying tool-calls or name fields; those are
/// added downstream by the template engine.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedMessage {
    pub role: String,
    pub content: String,
}

/// Render the initial message vector for a chat turn.
///
/// Precedence for the leading system message (last-writer wins):
///   1. `cli_system_override` if `Some(...)` and non-empty
///   2. `recipe.system` if `Some(...)` and non-empty
///   3. none — in which case no system message is emitted
///
/// The user prompt is always appended as role="user" at the end.
/// This is the FULL algorithm discharge of FALSIFY-CRUX-A-16-002 and
/// FALSIFY-CRUX-A-16-003: the returned vector has exactly the claimed
/// structure, by construction.
pub fn render_messages(
    recipe: &Recipe,
    user_prompt: &str,
    cli_system_override: Option<&str>,
) -> Vec<RenderedMessage> {
    let mut out = Vec::with_capacity(2);

    let effective_system = cli_system_override
        .filter(|s| !s.is_empty())
        .or(recipe.system.as_deref())
        .filter(|s| !s.is_empty());

    if let Some(system) = effective_system {
        out.push(RenderedMessage {
            role: "system".to_string(),
            content: system.to_string(),
        });
    }
    out.push(RenderedMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Schema parser tests =====

    #[test]
    fn minimal_recipe_parses() {
        let r = parse_recipe("from: hf://Qwen/Qwen2.5-Coder-7B").unwrap();
        assert_eq!(r.from, "hf://Qwen/Qwen2.5-Coder-7B");
        assert_eq!(r.system, None);
        assert_eq!(r.template, None);
        assert_eq!(r.parameters, None);
    }

    #[test]
    fn full_recipe_parses() {
        let src = r#"
from: "hf://unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
system: "You are a Rust expert. Always answer in one sentence."
template: "{{ messages }}"
parameters:
  temperature: 0.2
  top_p: 0.9
  top_k: 40
  num_ctx: 8192
  stop:
    - "</s>"
    - "<|im_end|>"
"#;
        let r = parse_recipe(src).unwrap();
        assert_eq!(r.from, "hf://unsloth/Qwen2.5-Coder-7B-Instruct-GGUF");
        assert_eq!(
            r.system.as_deref(),
            Some("You are a Rust expert. Always answer in one sentence.")
        );
        let p = r.parameters.unwrap();
        assert_eq!(p.temperature, Some(0.2));
        assert_eq!(p.top_p, Some(0.9));
        assert_eq!(p.top_k, Some(40));
        assert_eq!(p.num_ctx, Some(8192));
        assert_eq!(
            p.stop.unwrap(),
            vec!["</s>".to_string(), "<|im_end|>".to_string()]
        );
    }

    #[test]
    fn unknown_top_level_key_rejected() {
        // CRUX-A-16 INV-A-16-001: strict-schema validator rejects
        // unknown top-level keys. `temprature` (typo at top-level) must
        // fail at parse time.
        let src = "from: x\ntemprature: 0.5\n";
        let err = parse_recipe(src).unwrap_err();
        assert!(
            matches!(err, RecipeError::Parse(_)),
            "expected Parse err for unknown key, got {err:?}"
        );
    }

    #[test]
    fn unknown_parameters_key_rejected() {
        let src = "from: x\nparameters:\n  temprature: 0.5\n";
        let err = parse_recipe(src).unwrap_err();
        assert!(
            matches!(err, RecipeError::Parse(_)),
            "expected Parse err for unknown parameters key, got {err:?}"
        );
    }

    #[test]
    fn missing_from_rejected() {
        let src = "system: You are helpful.\n";
        assert!(matches!(
            parse_recipe(src),
            Err(RecipeError::Parse(_)) | Err(RecipeError::MissingFrom)
        ));
    }

    #[test]
    fn empty_from_rejected() {
        let src = "from: ''\nsystem: x\n";
        assert_eq!(parse_recipe(src).unwrap_err(), RecipeError::MissingFrom);
    }

    #[test]
    fn wrong_type_for_temperature_rejected() {
        // temperature must be a number, not a string.
        let src = "from: x\nparameters:\n  temperature: hot\n";
        assert!(matches!(parse_recipe(src), Err(RecipeError::Parse(_))));
    }

    // ===== Message-render tests =====

    fn recipe_with_system(system: Option<&str>) -> Recipe {
        Recipe {
            from: "hf://x/y".to_string(),
            system: system.map(str::to_string),
            template: None,
            parameters: None,
        }
    }

    #[test]
    fn system_prompt_injected_at_position_zero() {
        // FALSIFY-CRUX-A-16-002: recipe.system becomes messages[0]
        // with role=system.
        let r = recipe_with_system(Some("You are a Rust expert."));
        let msgs = render_messages(&r, "What is ownership?", None);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "You are a Rust expert.");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "What is ownership?");
    }

    #[test]
    fn no_system_means_no_system_message() {
        let r = recipe_with_system(None);
        let msgs = render_messages(&r, "hi", None);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "hi");
    }

    #[test]
    fn empty_recipe_system_treated_as_none() {
        let r = recipe_with_system(Some(""));
        let msgs = render_messages(&r, "hi", None);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
    }

    #[test]
    fn cli_override_wins_over_recipe_system() {
        // FALSIFY-CRUX-A-16-003: explicit --system wins over recipe.system.
        let r = recipe_with_system(Some("recipe says be verbose"));
        let msgs = render_messages(&r, "hi", Some("CLI says be terse"));
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "CLI says be terse");
    }

    #[test]
    fn empty_cli_override_falls_back_to_recipe_system() {
        // `--system ""` should not clobber recipe.system — empty is not
        // "last writer wins", it's "no override".
        let r = recipe_with_system(Some("recipe system"));
        let msgs = render_messages(&r, "hi", Some(""));
        assert_eq!(msgs[0].content, "recipe system");
    }

    #[test]
    fn cli_override_with_no_recipe_system_still_injects() {
        let r = recipe_with_system(None);
        let msgs = render_messages(&r, "hi", Some("cli system"));
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "cli system");
    }

    #[test]
    fn rendered_messages_preserve_user_prompt_verbatim() {
        let r = recipe_with_system(Some("sys"));
        let weird = "line1\nline2\twith tab\n🎉 emoji";
        let msgs = render_messages(&r, weird, None);
        assert_eq!(msgs.last().unwrap().content, weird);
    }

    #[test]
    fn render_is_deterministic() {
        let r = recipe_with_system(Some("sys"));
        let a = render_messages(&r, "hi", None);
        let b = render_messages(&r, "hi", None);
        assert_eq!(a, b);
    }

    #[test]
    fn render_never_emits_more_than_two_messages() {
        // The classifier emits at most [system, user]. Anything past that
        // is the template engine's concern, not this layer.
        let cases = [
            (None, None),
            (Some("sys"), None),
            (None, Some("cli")),
            (Some("sys"), Some("cli")),
        ];
        for (recipe_sys, cli) in cases {
            let r = recipe_with_system(recipe_sys);
            let msgs = render_messages(&r, "hi", cli);
            assert!(msgs.len() <= 2, "too many messages: {msgs:?}");
        }
    }

    #[test]
    fn system_message_always_precedes_user_message_when_present() {
        for sys in [Some("s"), None] {
            let r = recipe_with_system(sys);
            let msgs = render_messages(&r, "u", None);
            if let Some(system_pos) = msgs.iter().position(|m| m.role == "system") {
                let user_pos = msgs
                    .iter()
                    .position(|m| m.role == "user")
                    .expect("user message always present");
                assert!(system_pos < user_pos, "system must precede user: {msgs:?}");
            }
        }
    }
}
