//! chat_template_qwen4_exp: the Qwen3.5-family ChatML + thinking template as applied to
//! qwen4_exp (Qwen3.8-Flash-Next).
//!
//! Source of the expected strings: the published `chat_template.jinja` at
//! <https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/chat_template.jinja>
//! (sha256 af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375), rendered with
//! jinja2 3.0.3 (`{% generation %}` treated as a transparent block). The renderer script is
//! documented in `golden/qwen4_exp/README`. The goldens are NOT produced by this crate.
use super::*;

fn msg(role: &str, content: &str, think: Option<&str>) -> ThinkingMessage {
    ThinkingMessage {
        role: role.into(),
        content: content.into(),
        reasoning_content: think.map(str::to_string),
    }
}

fn render(m: &[ThinkingMessage], o: &Qwen35ThinkingOptions) -> String {
    Qwen35ThinkingTemplate.render(m, o).expect("render")
}

macro_rules! golden {
    ($n:literal) => {
        include_str!(concat!("golden/qwen4_exp/", $n, ".txt"))
    };
}

#[test]
fn chat_template_qwen4_exp_default_system_user() {
    let m = [msg("system", "Be brief.", None), msg("user", "Hi", None)];
    assert_eq!(
        render(&m, &Qwen35ThinkingOptions::default()),
        golden!("default_system_user")
    );
}

#[test]
fn chat_template_qwen4_exp_no_thinking_closes_empty_block() {
    let m = [msg("system", "Be brief.", None), msg("user", "Hi", None)];
    let o = Qwen35ThinkingOptions {
        enable_thinking: false,
        ..Default::default()
    };
    assert_eq!(render(&m, &o), golden!("no_thinking"));
}

#[test]
fn chat_template_qwen4_exp_effort_low_without_system() {
    let o = Qwen35ThinkingOptions {
        reasoning_effort: ReasoningEffort::Low,
        ..Default::default()
    };
    assert_eq!(
        render(&[msg("user", "Hi", None)], &o),
        golden!("effort_low_no_system")
    );
}

#[test]
fn chat_template_qwen4_exp_effort_medium_injects_nothing() {
    let o = Qwen35ThinkingOptions {
        reasoning_effort: ReasoningEffort::Medium,
        ..Default::default()
    };
    assert_eq!(
        render(&[msg("user", "Hi", None)], &o),
        golden!("effort_medium")
    );
}

fn two_turns() -> [ThinkingMessage; 3] {
    [
        msg("user", "2+2?", None),
        msg("assistant", "4", Some("Add the numbers.")),
        msg("user", "And 3+3?", None),
    ]
}

#[test]
fn chat_template_qwen4_exp_multi_turn_preserves_thinking() {
    assert_eq!(
        render(&two_turns(), &Qwen35ThinkingOptions::default()),
        golden!("multi_turn_preserve")
    );
}

#[test]
fn chat_template_qwen4_exp_multi_turn_drops_old_thinking() {
    let o = Qwen35ThinkingOptions {
        preserve_thinking: false,
        ..Default::default()
    };
    assert_eq!(render(&two_turns(), &o), golden!("multi_turn_drop_old"));
}

#[test]
fn chat_template_qwen4_exp_completed_turn_no_generation_prompt() {
    let o = Qwen35ThinkingOptions {
        add_generation_prompt: false,
        ..Default::default()
    };
    let m = [
        msg("user", "2+2?", None),
        msg("assistant", "4", Some("Add the numbers.")),
    ];
    assert_eq!(render(&m, &o), golden!("completed_turn"));
}

#[test]
fn chat_template_qwen4_exp_thinking_block_round_trips() {
    // The render of a completed turn, taken back apart, yields the same reasoning and answer.
    let rendered = golden!("completed_turn");
    let asst = rendered.rsplit("<|im_start|>assistant\n").next().unwrap();
    let body = asst.strip_suffix("<|im_end|>\n").unwrap();
    let (think, answer) = split_thinking(body);
    assert_eq!(think.as_deref(), Some("Add the numbers."));
    assert_eq!(answer, "4");
    // And re-rendering from the parsed parts reproduces the golden byte for byte.
    let m = [
        msg("user", "2+2?", None),
        msg("assistant", &answer, think.as_deref()),
    ];
    let o = Qwen35ThinkingOptions {
        add_generation_prompt: false,
        ..Default::default()
    };
    assert_eq!(render(&m, &o), rendered);
}

#[test]
fn chat_template_qwen4_exp_generation_prompt_is_split_after_open_tag() {
    // Model output begins inside <think>; a closed-only continuation still splits.
    let (t, a) = split_thinking("step one\n</think>\n\nanswer");
    assert_eq!(t.as_deref(), Some("step one"));
    assert_eq!(a, "answer");
    let (t, a) = split_thinking("plain");
    assert_eq!(t, None);
    assert_eq!(a, "plain");
}

#[test]
fn chat_template_qwen4_exp_rejects_what_the_template_rejects() {
    let o = Qwen35ThinkingOptions::default();
    let t = Qwen35ThinkingTemplate;
    assert!(t.render(&[], &o).is_err(), "No messages provided.");
    let late_sys = [msg("user", "a", None), msg("system", "b", None)];
    assert!(t.render(&late_sys, &o).is_err(), "System must be first");
    assert!(t.render(&[msg("system", "only", None)], &o).is_err());
    assert!(t.render(&[msg("user", "a", None), msg("bogus", "b", None)], &o).is_err());
}

#[test]
fn chat_template_qwen4_exp_auto_detects_chatml_family() {
    assert_eq!(
        detect_format_from_name("Qwen3.8-Flash-Next"),
        TemplateFormat::ChatML
    );
}
