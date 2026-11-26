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
        ..Default::default()
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

/// Lane-2 finding 3 (PR #7356) claimed the published jinja splits inline `<think>` out of
/// assistant `content` when `reasoning_content` is absent. It does not: the template's only
/// `</think>` literals are lines 119 and 170; line 103 only trims `content`, and lines 110-122
/// either wrap it in an empty reasoning block (preserve) or emit it verbatim (drop). Both
/// goldens were rendered from the published jinja, not from this crate.
fn inline_think_turns() -> [ThinkingMessage; 3] {
    [
        msg("user", "2+2?", None),
        msg("assistant", "<think>\nAdd the numbers.\n</think>\n\n4", None),
        msg("user", "And 3+3?", None),
    ]
}

#[test]
fn chat_template_qwen4_exp_inline_think_is_not_split_when_preserving() {
    assert_eq!(
        render(&inline_think_turns(), &Qwen35ThinkingOptions::default()),
        golden!("inline_think_preserve")
    );
}

#[test]
fn chat_template_qwen4_exp_inline_think_is_kept_verbatim_when_dropping() {
    let o = Qwen35ThinkingOptions {
        preserve_thinking: false,
        ..Default::default()
    };
    assert_eq!(render(&inline_think_turns(), &o), golden!("inline_think_drop_old"));
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

// ---------------------------------------------------------------------------------------
// #7336: tools and vision branches, checked against the published jinja itself.
//
// `oracle` renders `golden/qwen4_exp/chat_template.jinja` (the published file, verbatim)
// with minijinja. Three things the engine does not ship are supplied here and nowhere else:
// `raise_exception`, `str.startswith/endswith`, and `tojson` (Python `json.dumps` spelling,
// see `qwen35_tojson`). `{% generation %}` is a transparent block and is removed.
// Every golden below is asserted equal to BOTH the oracle and the Rust renderer, and the nine
// jinja2-rendered goldens of #7356 are replayed through the oracle to prove it reads the
// template the way jinja2 does.
// ---------------------------------------------------------------------------------------

const PUBLISHED_JINJA: &str = include_str!("golden/qwen4_exp/chat_template.jinja");
const PUBLISHED_SHA256: &str = "af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375";

fn part_json(p: &ThinkingContentPart) -> serde_json::Value {
    match p {
        ThinkingContentPart::Text(t) => serde_json::json!({"type": "text", "text": t}),
        ThinkingContentPart::Image => serde_json::json!({"type": "image", "image": "a.png"}),
        ThinkingContentPart::Video => serde_json::json!({"type": "video", "video": "a.mp4"}),
    }
}

/// The message as an OpenAI-shaped JSON object, the form the jinja is written for.
fn msg_json(m: &ThinkingMessage) -> serde_json::Value {
    let mut o = serde_json::Map::new();
    o.insert("role".into(), m.role.clone().into());
    let content = if m.parts.is_empty() {
        serde_json::Value::String(m.content.clone())
    } else {
        m.parts.iter().map(part_json).collect()
    };
    o.insert("content".into(), content);
    if let Some(r) = &m.reasoning_content {
        o.insert("reasoning_content".into(), r.clone().into());
    }
    if !m.tool_calls.is_empty() {
        let calls: Vec<serde_json::Value> = m
            .tool_calls
            .iter()
            .map(|c| {
                let args: serde_json::Map<String, serde_json::Value> =
                    c.arguments.iter().cloned().collect();
                serde_json::json!({"type": "function", "function": {"name": c.name, "arguments": args}})
            })
            .collect();
        o.insert("tool_calls".into(), calls.into());
    }
    o.into()
}

fn oracle_err(m: String) -> minijinja::Error {
    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m)
}

fn oracle(
    m: &[ThinkingMessage],
    tools: &[serde_json::Value],
    o: &Qwen35ThinkingOptions,
) -> Result<String, String> {
    let src = PUBLISHED_JINJA
        .replace("{%- generation %}", "")
        .replace("{%- endgeneration %}", "");
    let mut env = minijinja::Environment::new();
    env.add_function(
        "raise_exception",
        |m: String| -> Result<String, minijinja::Error> { Err(oracle_err(m)) },
    );
    env.add_filter(
        "tojson",
        |v: minijinja::Value| -> Result<String, minijinja::Error> {
            let j = serde_json::to_value(&v).map_err(|e| oracle_err(e.to_string()))?;
            Ok(qwen35_tojson(&j))
        },
    );
    env.set_unknown_method_callback(|_state, value, method, args| {
        let s = value.as_str();
        let a = args.first().and_then(minijinja::Value::as_str);
        match (s, a, method) {
            (Some(s), Some(a), "startswith") => Ok(s.starts_with(a).into()),
            (Some(s), Some(a), "endswith") => Ok(s.ends_with(a).into()),
            _ => Err(oracle_err(format!("no method {method}"))),
        }
    });
    env.add_template("published", &src).map_err(|e| e.to_string())?;
    let effort = match o.reasoning_effort {
        ReasoningEffort::XHigh => "xhigh",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::Low => "low",
    };
    let ctx = serde_json::json!({
        "messages": m.iter().map(msg_json).collect::<Vec<_>>(),
        "tools": tools,
        "enable_thinking": o.enable_thinking,
        "reasoning_effort": effort,
        "preserve_thinking": o.preserve_thinking,
        "add_generation_prompt": o.add_generation_prompt,
        "add_vision_id": o.add_vision_id,
    });
    env.get_template("published")
        .and_then(|t| t.render(&ctx))
        .map_err(|e| e.detail().unwrap_or("").to_string())
}

/// Golden == published jinja (oracle) == Rust renderer. With `QWEN4EXP_BLESS_DIR` set the
/// oracle's output is also written there (how the tools_* / vision_* goldens were produced).
fn check(
    name: &str,
    golden: &str,
    m: &[ThinkingMessage],
    tools: &[serde_json::Value],
    o: &Qwen35ThinkingOptions,
) {
    let published = oracle(m, tools, o).expect("the published jinja renders this input");
    if let Ok(dir) = std::env::var("QWEN4EXP_BLESS_DIR") {
        std::fs::write(format!("{dir}/{name}.txt"), &published).expect("bless");
    }
    assert_eq!(published, golden, "{name}: golden is not what the published jinja renders");
    let ours = Qwen35ThinkingTemplate
        .render_with_tools(m, tools, o)
        .expect("render");
    assert_eq!(ours, golden, "{name}: Rust renderer differs from the published jinja");
}

// Tool fixtures are written with their object keys in sorted order, at every depth. The
// oracle's `tojson` reads a minijinja map, which is sorted; the Rust renderer reads the
// `serde_json::Value`, which is sorted only when serde_json's `preserve_order` is off. That
// feature is off when aprender-core is built alone and ON in a workspace build (unified in
// by aprender-ci-tools, pmcp and llguidance), where a fixture written in another order made
// the renderer disagree with the sorted goldens. Sorted fixtures hold the same order in
// both builds, so the same goldens are checked in both.
fn weather_tool() -> serde_json::Value {
    serde_json::json!({"function": {
        "description": "Current weather for a city (°C).",
        "name": "get_weather",
        "parameters": {
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
            "type": "object"}},
        "type": "function"})
}

fn clock_tool() -> serde_json::Value {
    serde_json::json!({"function": {"name": "now", "parameters": {}}, "type": "function"})
}

fn call(name: &str, args: &[(&str, serde_json::Value)]) -> ThinkingToolCall {
    ThinkingToolCall {
        name: name.into(),
        arguments: args.iter().map(|(k, v)| ((*k).to_string(), v.clone())).collect(),
    }
}

fn asst_calls(content: &str, think: Option<&str>, calls: Vec<ThinkingToolCall>) -> ThinkingMessage {
    ThinkingMessage {
        tool_calls: calls,
        ..msg("assistant", content, think)
    }
}

fn user_parts(parts: Vec<ThinkingContentPart>) -> ThinkingMessage {
    ThinkingMessage {
        role: "user".into(),
        parts,
        ..Default::default()
    }
}

fn text(t: &str) -> ThinkingContentPart {
    ThinkingContentPart::Text(t.into())
}

#[test]
fn chat_template_qwen4_exp_published_jinja_is_the_recorded_file() {
    use sha2::Digest;
    let got = sha2::Sha256::digest(PUBLISHED_JINJA.as_bytes());
    let hex: String = got.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, PUBLISHED_SHA256);
}

/// The oracle is only worth something if it reads the template as jinja2 does: replay the
/// nine goldens #7356 rendered with jinja2 3.0.3 (not with this crate, not with minijinja).
#[test]
fn chat_template_qwen4_exp_oracle_reproduces_the_jinja2_goldens() {
    let d = Qwen35ThinkingOptions::default();
    let su = [msg("system", "Be brief.", None), msg("user", "Hi", None)];
    let u = [msg("user", "Hi", None)];
    let done = [msg("user", "2+2?", None), msg("assistant", "4", Some("Add the numbers."))];
    let no_think = Qwen35ThinkingOptions { enable_thinking: false, ..d };
    let low = Qwen35ThinkingOptions { reasoning_effort: ReasoningEffort::Low, ..d };
    let medium = Qwen35ThinkingOptions { reasoning_effort: ReasoningEffort::Medium, ..d };
    let drop_old = Qwen35ThinkingOptions { preserve_thinking: false, ..d };
    let no_gen = Qwen35ThinkingOptions { add_generation_prompt: false, ..d };
    let cases: [(&str, &str, &[ThinkingMessage], Qwen35ThinkingOptions); 9] = [
        ("default_system_user", golden!("default_system_user"), &su, d),
        ("no_thinking", golden!("no_thinking"), &su, no_think),
        ("effort_low_no_system", golden!("effort_low_no_system"), &u, low),
        ("effort_medium", golden!("effort_medium"), &u, medium),
        ("multi_turn_preserve", golden!("multi_turn_preserve"), &two_turns(), d),
        ("multi_turn_drop_old", golden!("multi_turn_drop_old"), &two_turns(), drop_old),
        ("inline_think_preserve", golden!("inline_think_preserve"), &inline_think_turns(), d),
        ("inline_think_drop_old", golden!("inline_think_drop_old"), &inline_think_turns(), drop_old),
        ("completed_turn", golden!("completed_turn"), &done, no_gen),
    ];
    for (name, golden, m, o) in cases {
        assert_eq!(oracle(m, &[], &o).expect("oracle"), golden, "{name}");
    }
}

#[test]
fn chat_template_qwen4_exp_tojson_is_python_json_dumps() {
    // json.dumps(x, ensure_ascii=False): ", " and ": " separators, raw non-ASCII, \n escaped.
    let v = serde_json::json!({"a": [1, true, null, 1.5], "b": {"c": "é\n\"q\"\\"}, "d": {}, "e": []});
    assert_eq!(
        qwen35_tojson(&v),
        "{\"a\": [1, true, null, 1.5], \"b\": {\"c\": \"é\\n\\\"q\\\"\\\\\"}, \"d\": {}, \"e\": []}"
    );
    assert_eq!(qwen35_tojson(&serde_json::json!("\u{1}\t")), "\"\\u0001\\t\"");
}

#[test]
fn chat_template_qwen4_exp_tools_system_user() {
    let m = [msg("system", "Be brief.", None), msg("user", "Weather in Paris?", None)];
    check(
        "tools_system_user",
        golden!("tools_system_user"),
        &m,
        &[weather_tool(), clock_tool()],
        &Qwen35ThinkingOptions::default(),
    );
}

#[test]
fn chat_template_qwen4_exp_tools_without_system_or_instruction() {
    let o = Qwen35ThinkingOptions {
        reasoning_effort: ReasoningEffort::Medium,
        ..Default::default()
    };
    let m = [msg("user", "Weather in Paris?", None)];
    check("tools_no_system_medium", golden!("tools_no_system_medium"), &m, &[weather_tool()], &o);
}

#[test]
fn chat_template_qwen4_exp_tool_call_then_tool_response() {
    let m = [
        msg("system", "Be brief.", None),
        msg("user", "Weather in Paris?", None),
        asst_calls(
            "",
            Some("Need the tool."),
            vec![call(
                "get_weather",
                &[("city", "Paris".into()), ("days", 3.into())],
            )],
        ),
        msg("tool", "{\"temp_c\": 18}", None),
    ];
    check(
        "tools_call_and_response",
        golden!("tools_call_and_response"),
        &m,
        &[weather_tool()],
        &Qwen35ThinkingOptions::default(),
    );
}

#[test]
fn chat_template_qwen4_exp_two_calls_two_responses_old_thinking_dropped() {
    let o = Qwen35ThinkingOptions {
        preserve_thinking: false,
        ..Default::default()
    };
    let m = [
        msg("user", "Weather and time?", None),
        asst_calls(
            "Checking.",
            Some("Two tools."),
            vec![
                call(
                    "get_weather",
                    &[
                        ("city", "Paris\nFrance".into()),
                        ("opts", serde_json::json!({"days": 2, "units": ["c", "f"], "wind": true})),
                    ],
                ),
                call("now", &[]),
            ],
        ),
        msg("tool", "18", None),
        msg("tool", "12:00", None),
        msg("assistant", "18 C at noon.", Some("Combine.")),
        msg("user", "Thanks", None),
    ];
    check(
        "tools_two_calls_drop_old",
        golden!("tools_two_calls_drop_old"),
        &m,
        &[weather_tool(), clock_tool()],
        &o,
    );
}

#[test]
fn chat_template_qwen4_exp_vision_parts_render_pad_tokens() {
    let m = [user_parts(vec![ThinkingContentPart::Image, text("What is this?")])];
    check("vision_image_text", golden!("vision_image_text"), &m, &[], &Qwen35ThinkingOptions::default());
}

#[test]
fn chat_template_qwen4_exp_vision_ids_count_across_turns() {
    let o = Qwen35ThinkingOptions {
        add_vision_id: true,
        ..Default::default()
    };
    let m = [
        msg("system", "Be brief.", None),
        user_parts(vec![ThinkingContentPart::Image, ThinkingContentPart::Video, text("Compare.")]),
        msg("assistant", "Done.", None),
        user_parts(vec![text("And these? "), ThinkingContentPart::Image, ThinkingContentPart::Video]),
    ];
    check("vision_ids_two_turns", golden!("vision_ids_two_turns"), &m, &[], &o);
}

/// Shapes with no golden: the Rust renderer must still say exactly what the published jinja
/// says, including its errors and its quirks (a leading tool turn gets no `user` header).
#[test]
fn chat_template_qwen4_exp_agrees_with_the_published_jinja_on_edge_shapes() {
    let d = Qwen35ThinkingOptions::default();
    let no_gen = Qwen35ThinkingOptions { add_generation_prompt: false, ..d };
    let ids = Qwen35ThinkingOptions { add_vision_id: true, ..d };
    let sys_image = ThinkingMessage {
        role: "system".into(),
        parts: vec![ThinkingContentPart::Image],
        ..Default::default()
    };
    let sys_video = ThinkingMessage {
        parts: vec![ThinkingContentPart::Video],
        ..sys_image.clone()
    };
    let sys_text_parts = ThinkingMessage {
        parts: vec![text(" Be "), text("brief. ")],
        ..sys_image.clone()
    };
    let wrapped = msg("user", "<tool_response>\n18\n</tool_response>", None);
    let shapes: Vec<(&str, Vec<ThinkingMessage>, Vec<serde_json::Value>, Qwen35ThinkingOptions)> = vec![
        ("tool turn first", vec![msg("tool", "x", None), msg("user", "q", None)], vec![], d),
        ("tool turn last, no prompt", vec![msg("user", "q", None), asst_calls("", None, vec![call("now", &[])]), msg("tool", "12:00", None)], vec![clock_tool()], no_gen),
        ("tool after system", vec![msg("system", "s", None), msg("tool", "x", None), msg("user", "q", None)], vec![clock_tool()], d),
        ("wrapped tool response is not the query", vec![msg("user", "q", None), msg("assistant", "a", Some("t")), wrapped.clone()], vec![], Qwen35ThinkingOptions { preserve_thinking: false, ..d }),
        ("only a wrapped tool response", vec![wrapped], vec![], d),
        ("system image", vec![sys_image.clone(), msg("user", "q", None)], vec![], d),
        ("system image with tools", vec![sys_image, msg("user", "q", None)], vec![clock_tool()], d),
        ("system video", vec![sys_video, msg("user", "q", None)], vec![], d),
        ("system text parts", vec![sys_text_parts.clone(), msg("user", "q", None)], vec![], d),
        ("system text parts with tools", vec![sys_text_parts, msg("user", "q", None)], vec![weather_tool()], ids),
        ("late system", vec![msg("user", "q", None), msg("system", "s", None)], vec![clock_tool()], d),
        ("unknown role", vec![msg("user", "q", None), msg("bogus", "b", None)], vec![], d),
        ("assistant image with ids", vec![user_parts(vec![ThinkingContentPart::Image]), ThinkingMessage { parts: vec![ThinkingContentPart::Image, text(" seen")], tool_calls: vec![call("now", &[])], ..msg("assistant", "", Some("r")) }, msg("tool", "1", None)], vec![clock_tool()], ids),
        ("string and non-string arguments", vec![msg("user", "q", None), asst_calls(" ", None, vec![call("f", &[("a", "".into()), ("b", serde_json::Value::Null), ("c", serde_json::json!([1, "x"])), ("d", 1.5.into())])])], vec![], no_gen),
    ];
    for (name, m, tools, o) in shapes {
        let published = oracle(&m, &tools, &o);
        let ours = Qwen35ThinkingTemplate
            .render_with_tools(&m, &tools, &o)
            .map_err(|e| e.to_string());
        match (&published, &ours) {
            (Ok(p), Ok(r)) => assert_eq!(r, p, "{name}"),
            (Err(p), Err(r)) => assert!(r.contains(p.as_str()) && !p.is_empty(), "{name}: jinja raised {p:?}, Rust {r:?}"),
            _ => panic!("{name}: jinja {published:?} vs Rust {ours:?}"),
        }
    }
}
