
/// The golden gate's stop token for THIS model, read from the model's own
/// metadata.
///
/// #3870. This used to be `SpecialTokens::qwen2().eos_id` — the literal 151645 —
/// passed as the stop token for EVERY model the gate was ever pointed at. That
/// made `apr qa`'s golden gate structurally Qwen-only:
///
/// | model | arch | vocab | its real eos | 151645 reachable? |
/// |---|---|---|---|---|
/// | `Qwen3-1.7B-Q4_K_M` | qwen3 | 151936 | 151645 | yes |
/// | `Qwen3-Coder-30B-A3B` | qwen3moe | 151936 | 151645 | yes |
/// | `tinyllama-1.1b-chat-v1.0` | llama | **32000** | **2** | **NO** |
///
/// 151645 is not a token id that EXISTS in a 32000-entry vocabulary, so for
/// tinyllama the stop condition could not fire — not late, not on the wrong
/// token, unreachable. The model ran its full 512-token budget and drifted into
/// chat scaffolding, which is precisely what #1864's comment at the call sites
/// says the stop token exists to prevent. Both the CPU and GPU legs hardcoded
/// the same wrong constant, which is why their output was byte-identical, and
/// that identity was read as "the backends agree" rather than as the signature
/// of a shared upstream cause.
///
/// **There is deliberately no fallback guess.** A model whose GGUF carries no
/// eos gets NO stop token rather than an architecture default, because a guess
/// is what this defect was. `TransformerConfig::from_apr` faces the same
/// question and answers it the same way under OBLIG-SPECIAL-TOKEN-WITHIN-VOCAB
/// (PMAT-908): it takes an architecture default "only … when it is a reachable
/// logit (< vocab_size); a small-vocab model must not inherit a large
/// arch-default eos" (`aprender-serve/src/gguf/config.rs:544`). That ladder is
/// the better long-term answer here too, but `default_eos_for_architecture` is
/// `pub(crate)` to aprender-serve and unreachable from this crate — a
/// cross-crate visibility job, filed rather than forced. Note it would not have
/// saved tinyllama anyway: its llama default is 128001 (Llama-3's), and the
/// vocab filter is what rejects it.
#[cfg(feature = "inference")]
fn golden_stop_tokens(gguf: &realizar::gguf::GGUFModel) -> Vec<u32> {
    gguf.eos_token_id().into_iter().collect()
}

/// The golden prompt's tokens, with a BOS taken from the model rather than
/// assumed.
///
/// #3870, same cause as [`golden_stop_tokens`]: the fallback was
/// `vec![SpecialTokens::qwen2().bos_id, 9707]`, and 151643 is no more reachable
/// in a 32000-entry vocabulary than 151645 was.
///
/// This path is only taken when the model's own tokenizer cannot encode the
/// prompt at all. `9707` is left as-is and is still a Qwen token id: there is no
/// architecture-neutral "second token", and inventing one would be the same kind
/// of guess this function exists to remove. A gate that reaches this line is
/// already not measuring what it thinks it is.
#[cfg(feature = "inference")]
fn golden_prompt_tokens(gguf: &realizar::gguf::GGUFModel, prompt: &str) -> Vec<u32> {
    gguf.encode(prompt).unwrap_or_else(|| {
        let bos = gguf
            .bos_token_id()
            .unwrap_or_else(|| aprender::demo::SpecialTokens::qwen2().bos_id);
        vec![bos, 9707]
    })
}

/// Run golden output test for SafeTensors format models. Returns None if tokenizer missing.
#[cfg(feature = "inference")]
fn golden_output_safetensors(
    path: &Path,
    prompt: &str,
    max_tokens: usize,
) -> Result<Option<(Vec<u32>, String)>> {
    use aprender::text::bpe::{load_from_json, BpeTokenizer};
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    let tokenizer_path = realizar::safetensors::find_sibling_file(path, "tokenizer.json");
    let tokenizer: Option<BpeTokenizer> = tokenizer_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|json| load_from_json(&json).ok());

    let Some(tokenizer) = tokenizer else {
        return Ok(None);
    };

    let transformer = SafetensorsToAprConverter::convert(path)
        .map_err(|e| CliError::ValidationFailed(format!("SafeTensors convert failed: {e}")))?;

    let prompt_tokens = tokenizer.encode(prompt);
    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    let tokens = transformer
        .generate_with_cache(&prompt_tokens, &gen_config)
        .map_err(|e| CliError::ValidationFailed(format!("Generation failed: {e}")))?;
    let text = tokenizer.decode(&tokens);
    Ok(Some((tokens, text)))
}

/// Run golden output CPU generation for GGUF format.
#[cfg(feature = "inference")]
fn golden_output_gguf_cpu(
    mapped: &realizar::gguf::MappedGGUFModel,
    gguf: &realizar::gguf::GGUFModel,
    prompt: &str,
    max_tokens: usize,
) -> Result<(Vec<u32>, String)> {
    use realizar::gguf::{OwnedQuantizedModel, QuantizedGenerateConfig};

    let prompt_tokens = golden_prompt_tokens(gguf, prompt);
    // #1864: without stop_tokens, the model runs for the full max_tokens
    // budget and starts emitting in-distribution chat-template tokens like
    // `<|im_start|>` from accumulated drift — which `verify_output` then
    // (correctly) flags as gibberish. The actual underlying inference path
    // (`apr serve` /v1/chat/completions) sets stop_tokens to EOS — so user
    // traffic was never affected. This aligns the gate's gen_config with the
    // production path. See `cuda_chat_backend.rs:113` for the symmetric setup.
    let gen_config = QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: golden_stop_tokens(gguf),
        ..Default::default()
    };
    let model = OwnedQuantizedModel::from_mapped(mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;
    let tokens = model
        .generate_with_cache(&prompt_tokens, &gen_config)
        .map_err(|e| CliError::ValidationFailed(format!("CPU generation failed: {e}")))?;
    let text = gguf.decode(&tokens);
    Ok((tokens, text))
}

/// Gate 1: Golden Output Test
///
/// Runs the model with a known prompt and verifies the output contains expected patterns.
/// Uses verify_output() for structured validation (PMAT-QA-PROTOCOL-001 §7.4).
/// Golden test cases: ChatML prompt + expected output patterns.
///
/// SELECTION RULE (#2350): a golden prompt must have a WIDE argmax margin, so its
/// greedy continuation is the same on every backend. These assert on exact
/// generated content under `temperature 0.0 / top_k 1`, which turns any near-tie
/// into a coin flip — and CPU and CUDA kernels legitimately differ by small
/// numerics well inside tolerance (F2 measures full prefill parity at cosine
/// 0.9937 with zero argmax mismatches on this model).
///
/// The removed case was `"Hello"` alone, expecting a greeting back. Measured on
/// one binary, one model, `max_tokens 512`, greedy:
///
///   prompt                                   CPU                        CUDA
///   ---------------------------------------  -------------------------  -------------------------
///   "Hello"                                  "Hello! How can I ..."     "I'm sorry, but I'm not
///                                                                        sure what you're asking"
///   "Hi"                                     "I'm sorry, but I'm not    "I'm here to help! ..."
///                                             sure what you're asking"
///   "What is 2+2?"                           "2+2 equals 4."            "2 + 2 equals 4."
///   "Hello there, how are you doing today    "Hello! I'm doing well,    same
///    my friend?"                              thank you."
///   "What is the capital of France?"         "The capital of France     same
///                                             is Paris."
///
/// The decisive row is the second: the SAME evasive completion appears on **CPU**,
/// just for `"Hi"` rather than `"Hello"`. Both backends answer bare one-word
/// greetings evasively; they only disagree about which one tips over. So the old
/// case was not detecting a GPU defect — it was sampling a knife-edge, and it
/// flipped when #2323 made the CUDA path reachable on sm_89.
///
/// The three cases below were all verified to produce identical continuations on
/// CPU and CUDA. Keep it that way: if you add a case, run it on both backends
/// first (`crates/apr-cli/tests/golden_prompt_tokenization.rs` is the harness).
/// The golden QUESTIONS — what is asked, with no opinion about how it is wrapped.
///
/// #3724 split this from the rendering. The gate used to carry three ChatML
/// strings, so every model was asked in ChatML whatever production sends it: on
/// `qwen3` that left the model in thinking mode, a mode no production path uses,
/// and on lambda its greedy reasoning for "2+2" overran the 512-token budget and
/// the gate reported "Empty output" for a model that answers "2 + 2 = 4." through
/// `apr serve`.
fn golden_questions() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("What is 2+2?", vec!["4"]),
        // Replaces the bare "Hello". Still exercises a conversational turn, but
        // with enough context that the first generated token is not a near-tie.
        (
            "Hello there, how are you doing today my friend?",
            // #3782: `"!"` removed. Substring-any made it satisfiable by
            // `"!!!!!!!!"` — token id 0 in the Qwen vocab, i.e. exactly what a
            // model with dead logits emits. A bare exclamation mark was never
            // evidence of a greeting anyway. `gibberish_dominant_character`
            // (output_verification.rs) is the general guard; this is the
            // case-local half, so neither alone has to carry it.
            vec!["Hello", "Hi", "hey", "hello", "well"],
        ),
        // Factual recall: a wide-margin argmax and a check that the model is
        // actually reasoning over its weights rather than emitting boilerplate.
        ("What is the capital of France?", vec!["Paris"]),
    ]
}

/// Format-distinctive markers. A template that renders one of these is making a claim
/// about the model's chat format that the model's own declared template can refute.
#[cfg(feature = "inference")]
const FORMAT_MARKERS: &[&str] = &["[INST]", "<|im_start|>", "<|user|>", "<<SYS>>", "### Instruction"];

/// Does `rendered` use a format marker that `declared` never mentions?
#[cfg(feature = "inference")]
fn contradicts_declared(rendered: &str, declared: &str) -> bool {
    FORMAT_MARKERS
        .iter()
        .any(|m| rendered.contains(m) && !declared.contains(m))
}

/// Which string the template detector is keyed on (#3914).
///
/// `general.architecture` names the model's SHAPE; it does not name its chat format.
/// TinyLlama-1.1B-Chat is a `llama`-architecture model fine-tuned on the Zephyr
/// `<|user|>` format, and `detect_format_from_name` ALREADY has a `tinyllama` rule
/// ordered ahead of `llama`, carrying the comment "check BEFORE llama!", for exactly
/// this reason. That rule is UNREACHABLE when the key is an architecture, because
/// "llama" does not contain "tinyllama". So the gate sent `[INST]` to a model never
/// trained on it and got back "France is the capital of France." — no "Paris".
///
/// The model's own `tokenizer.chat_template` is the authority on its chat format, and
/// it is used here as a REFEREE rather than as a renderer: only when the architecture's
/// template renders a marker the declared template never mentions is the architecture
/// treated as contradicted, and only then is the model NAME tried instead.
///
/// KEYING ON THE NAME UNCONDITIONALLY IS NOT SAFE, and was measured not to be.
/// `Qwen3-Coder-30B-A3B-Instruct` has architecture `qwen3moe` (-> ChatML) and a name
/// containing "qwen3" (-> Qwen3NoThink); `detect_format_from_name`'s own comment records
/// that pre-injecting a think block makes that model emit `<|endoftext|>` immediately.
/// Its declared template mentions `<|im_start|>` and nothing else, so it agrees with its
/// architecture and the referee leaves it alone. Measured over the local inventory, this
/// rule is inert for all three qwen models and fires only on tinyllama.
#[cfg(feature = "inference")]
fn template_key(
    architecture: Option<&str>,
    name: Option<&str>,
    declared: Option<&str>,
) -> Option<String> {
    use realizar::chat_template::{format_messages, ChatMessage};

    let arch = architecture.map(str::trim).filter(|a| !a.is_empty())?;
    let declared = declared.map(str::trim).filter(|d| !d.is_empty());
    let name = name.map(str::trim).filter(|n| !n.is_empty());
    let (Some(declared), Some(name)) = (declared, name) else {
        return Some(arch.to_string());
    };
    let render = |k: &str| {
        format_messages(&[ChatMessage::user("x")], Some(k)).unwrap_or_default()
    };
    if !contradicts_declared(&render(arch), declared) {
        return Some(arch.to_string());
    }
    // The architecture is contradicted. The name is only an improvement if it is not
    // ALSO contradicted — otherwise keep the architecture rather than trade one wrong
    // template for another.
    if contradicts_declared(&render(name), declared) {
        return Some(arch.to_string());
    }
    Some(name.to_string())
}

/// Render one golden question the way PRODUCTION renders it for this architecture
/// (#3724 done_when 2): the same detector and the same template `apr serve` and
/// `apr run` use, keyed on the GGUF's `general.architecture`.
///
/// `None` — a format that declares no architecture — keeps the ChatML the gate has
/// always sent. For every architecture the detector maps to `ChatML` this is
/// byte-identical to the old hardcoded strings, because `ChatMLTemplate`'s
/// `format_conversation` of a single user message is exactly
/// `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n`; that identity is
/// pinned by `chatml_architectures_render_the_legacy_prompt_byte_for_byte`.
/// `qwen3`/`qwen35` map to `Qwen3NoThink`, which appends `<think>\n</think>\n` —
/// the production prompt, and the fix.
#[cfg(feature = "inference")]
fn golden_prompt_for(architecture: Option<&str>, question: &str) -> String {
    use realizar::chat_template::{format_messages, ChatMessage};

    let legacy = || format!("<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n");
    let Some(architecture) = architecture.map(str::trim).filter(|a| !a.is_empty()) else {
        return legacy();
    };
    // A rendering failure must not silently change what is asked: fall back to the
    // prompt the gate has always sent rather than to an empty or partial one.
    format_messages(&[ChatMessage::user(question)], Some(architecture)).unwrap_or_else(|_| legacy())
}

// #3907 wiring: `const THINKING_ON_BUDGET: usize = 2048` was DELETED here, not
// merely stopped-being-used. While it existed a third site could reach for it, and
// a named constant reads as authoritative to the next reviewer — which is exactly
// how the hybrid leg at output_verification.rs came to pass it to both its
// generation and its judging while the dense leg used the resolver. The resolver
// `thinking_on_budget_for` is now the ONLY way to obtain an ON-leg budget, and its
// 2048 lives in contracts/thinking-budgets-v1.yaml's `default`, WITH its basis.

/// The per-model budgets, embedded from the packaged mirror (#3907).
///
/// `crates/apr-cli/contracts/` and not the workspace root, for the reason
/// `capability.rs` gives: `include_str!` cannot escape the crate directory at package
/// time. `tests/thinking_budgets_mirror.rs` asserts the two are byte-identical.
#[cfg(feature = "inference")]
const THINKING_BUDGETS: &str = include_str!("../../contracts/thinking-budgets-v1.yaml");

/// `*`-glob match, the same semantics the ladder's `fnmatch` uses on `inventory.deferred`.
#[cfg(feature = "inference")]
fn glob_match(pat: &str, name: &str) -> bool {
    match pat.split_once('*') {
        None => pat == name,
        Some((head, tail)) => {
            name.len() >= head.len() + tail.len()
                && name.starts_with(head)
                && name.ends_with(tail)
        },
    }
}

/// The ON-leg budget for this model, with the provenance of the number.
///
/// FAIL-CLOSED by construction, copying `evidence/parity/thresholds.yaml`:
///   * a model LISTED in `models:` takes its entry and never falls back to `default`,
///     so an entry with no `budget` is a refusal rather than an inherited 8B figure;
///   * a `default` with no `basis` is refused too — a budget without a measurement
///     behind it is not a budget, and that is the defect this replaced;
///   * a parse failure is an error, never a default. An unreadable table rendered as
///     2048 would be the old constant wearing a contract's clothes.
///
/// `APR_THINKING_ON_BUDGET` overrides for a probe, because the `const` had no seam and
/// a threshold you cannot vary without a `--features cuda,inference` rebuild is one
/// nobody re-measures. The override is named in the returned provenance so a probed
/// number cannot be mistaken for a measured one.
#[cfg(feature = "inference")]
/// The probe override, or `None` when the variable is unset.
///
/// Split out of `thinking_on_budget_for` because
/// `check_complexity_ratchet.sh` measured the combined function at **cognitive
/// 32** against a ceiling of 25 and refused it as NEW against the merge-base.
/// Every message below is byte-identical to the single-function version: the
/// split is for the nesting, and the strings are the contract (`#3907`).
fn thinking_on_budget_override() -> std::result::Result<Option<(usize, String)>, String> {
    let Ok(raw) = std::env::var("APR_THINKING_ON_BUDGET") else {
        return Ok(None);
    };
    let n: usize = raw
        .trim()
        .parse()
        .map_err(|_| format!("APR_THINKING_ON_BUDGET={raw:?} is not a token count (#3907)"))?;
    Ok(Some((
        n,
        format!("PROBE OVERRIDE APR_THINKING_ON_BUDGET={n}, not a measurement"),
    )))
}

/// One matched row of the `models` table. A row that declares no `budget`
/// REFUSES rather than inheriting the default, which was measured on a
/// different model (#3907).
fn thinking_on_budget_row(
    v: &serde_yaml::Value,
    pat: &str,
    model_file: &str,
) -> std::result::Result<(usize, String), String> {
    let Some(b) = v.get("budget").and_then(serde_yaml::Value::as_u64) else {
        let why = v
            .get("why_unmeasured")
            .and_then(|x| x.as_str())
            .unwrap_or("no reason recorded");
        return Err(format!(
            "no measured thinking budget for `{model_file}` (matches `{pat}` in \
             contracts/thinking-budgets-v1.yaml, which declares no `budget`). \
             Refusing rather than inheriting the default, which was measured on a \
             different model: {} (#3907)",
            why.trim()
        ));
    };
    let basis = v.get("basis").and_then(|x| x.as_str()).unwrap_or("").trim();
    if basis.is_empty() {
        return Err(format!(
            "`{pat}` declares budget {b} with no `basis` — a budget without the \
             measurement behind it is not a budget (#3907)"
        ));
    }
    Ok((usize::try_from(b).unwrap_or(0), format!("{pat}: {basis}")))
}

/// The first `models` pattern that matches `model_file`, or `None` when the
/// model is unlisted and the `default` row applies.
fn thinking_on_budget_match(
    doc: &serde_yaml::Value,
    model_file: &str,
) -> Option<std::result::Result<(usize, String), String>> {
    let models = doc.get("models").and_then(|m| m.as_mapping())?;
    for (k, v) in models {
        let Some(pat) = k.as_str() else { continue };
        if glob_match(pat, model_file) {
            return Some(thinking_on_budget_row(v, pat, model_file));
        }
    }
    None
}

/// The `default` row, for a model no pattern names.
fn thinking_on_budget_default(
    doc: &serde_yaml::Value,
) -> std::result::Result<(usize, String), String> {
    let d = doc.get("default").ok_or_else(|| {
        "contracts/thinking-budgets-v1.yaml has no `default` and this model is unlisted (#3907)"
            .to_string()
    })?;
    let b = d
        .get("budget")
        .and_then(serde_yaml::Value::as_u64)
        .ok_or_else(|| "`default` declares no `budget` (#3907)".to_string())?;
    let basis = d.get("basis").and_then(|x| x.as_str()).unwrap_or("").trim();
    if basis.is_empty() {
        return Err("`default` declares a budget with no `basis` (#3907)".to_string());
    }
    Ok((usize::try_from(b).unwrap_or(0), format!("default: {basis}")))
}

fn thinking_on_budget_for(model_file: &str) -> std::result::Result<(usize, String), String> {
    if let Some(over) = thinking_on_budget_override()? {
        return Ok(over);
    }
    let doc: serde_yaml::Value = serde_yaml::from_str(THINKING_BUDGETS)
        .map_err(|e| format!("the embedded thinking-budget table did not parse: {e} (#3907)"))?;
    if let Some(hit) = thinking_on_budget_match(&doc, model_file) {
        return hit;
    }
    thinking_on_budget_default(&doc)
}

/// The same conversation with production's THINKING SUPPRESSION removed, or
/// `None` when production does not suppress thinking for this architecture.
///
/// Derived from production's own rendered prompt rather than from a template
/// name: a prompt that ends in an EMPTY `<think></think>` block is one that
/// pre-closes the model's reasoning so it answers directly. Removing that block
/// is the same conversation in the other mode. Reading it off the rendering means
/// a replacement template (#3755) is followed automatically instead of silently
/// bypassed.
#[cfg(feature = "inference")]
fn without_thinking_prefill(prompt: &str) -> Option<String> {
    let start = prompt.rfind("<think>")?;
    let tail = &prompt[start + "<think>".len()..];
    let close = tail.find("</think>")?;
    // Only an EMPTY block is a suppression prefill; a block with reasoning in it
    // is content, and cutting it would change the conversation.
    if !tail[..close].trim().is_empty() {
        return None;
    }
    if !tail[close + "</think>".len()..].trim().is_empty() {
        return None;
    }
    Some(prompt[..start].to_string())
}

/// #3724 done_when 3: judge a thinking-capable model in the OTHER mode too.
///
/// OFF is the production prompt, already judged by the normal cases. ON is the
/// same conversation with the suppression removed and a budget that overdoes it:
/// the think block must CLOSE and the answer after it must be correct. An
/// unclosed block is reported by name with its budget — never stripped to an
/// empty answer, which is what turned this defect into "Empty output".
///
/// One question, not three: the ON leg exists to prove the model still answers
/// correctly when it reasons, and a 2048-token CPU generation per case would
/// quadruple the gate for every Qwen3 model to prove the same thing three times.
#[cfg(feature = "inference")]
fn thinking_on_case(architecture: Option<&str>) -> Option<(String, Vec<&'static str>)> {
    let (question, patterns) = golden_questions().into_iter().next()?;
    let production = golden_prompt_for(architecture, question);
    let on_prompt = without_thinking_prefill(&production)?;
    Some((on_prompt, patterns))
}

/// Judge one generated ON-mode output: the block closed, and the answer is right.
///
/// `generated` is the model's continuation with the prompt echo already removed.
#[cfg(feature = "inference")]
fn judge_thinking_on_output(generated: &str, patterns: &[&str], budget: usize) -> Option<String> {
    match split_thinking_blocks(generated) {
        ThinkingSplit::Unclosed => Some(unclosed_think_reason(
            "golden_output_thinking_on",
            budget,
            generated.len(),
        )),
        ThinkingSplit::Answer(answer) => {
            if !generated.contains("<think>") {
                // Nothing was suppressed and nothing was thought: the ON leg
                // proved nothing, and saying so is better than a green.
                return Some(format!(
                    "golden_output_thinking_on: no <think> block was produced within {budget} tokens \
                     — the thinking mode this leg exists to judge was never entered"
                ));
            }
            // #3948: a block that opened and closed with nothing in it is the same
            // outcome as no block at all — the model skipped the reasoning. Judging
            // only the answer after it scored "skipped reasoning, answered" as
            // "reasoned, answered", so the leg could not fail on the case it exists
            // to catch (Qwen3.5-0.8B Q4_K_M answered at budget 8).
            if !any_think_block_has_content(generated) {
                return Some(format!(
                    "golden_output_thinking_on: the <think> block closed EMPTY within {budget} tokens \
                     — the model skipped the reasoning, so this leg judged an answer given without \
                     thinking (#3948)"
                ));
            }
            match verify_output(&answer, "golden_output_thinking_on", patterns) {
                OutputVerification::Fail { reason } => Some(reason),
                OutputVerification::Pass => None,
            }
        }
    }
}

/// #3948: does any closed `<think>` block hold more than whitespace?
#[cfg(feature = "inference")]
fn any_think_block_has_content(generated: &str) -> bool {
    let mut rest = generated;
    while let Some(start) = rest.find("<think>") {
        let body = &rest[start + "<think>".len()..];
        let Some(end) = body.find("</think>") else {
            return false;
        };
        if !body[..end].trim().is_empty() {
            return true;
        }
        rest = &body[end + "</think>".len()..];
    }
    false
}

/// The golden cases as (prompt, expected patterns) for one architecture.
#[cfg(feature = "inference")]
fn golden_test_cases_for(architecture: Option<&str>) -> Vec<(String, Vec<&'static str>)> {
    golden_questions()
        .into_iter()
        .map(|(question, patterns)| (golden_prompt_for(architecture, question), patterns))
        .collect()
}

/// Generate output for a single test case based on model format.
/// GH-239: GGUF objects are Optional — only provided when format is GGUF.
#[cfg(feature = "inference")]
fn generate_golden_for_format(
    path: &Path,
    prompt: &str,
    max_tokens: usize,
    format: realizar::format::ModelFormat,
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
    gguf_model: Option<&realizar::gguf::GGUFModel>,
) -> Result<Option<(Vec<u32>, String)>> {
    use realizar::format::ModelFormat;

    match format {
        ModelFormat::Gguf => {
            let mapped = mapped.ok_or_else(|| {
                CliError::ValidationFailed("GGUF mapped model required".to_string())
            })?;
            let gguf_model = gguf_model
                .ok_or_else(|| CliError::ValidationFailed("GGUF model required".to_string()))?;
            Ok(Some(golden_output_gguf_cpu(
                mapped, gguf_model, prompt, max_tokens,
            )?))
        }
        ModelFormat::Apr => Ok(Some(golden_output_apr(path, prompt, max_tokens)?)),
        ModelFormat::SafeTensors => golden_output_safetensors(path, prompt, max_tokens),
    }
}

/// Is there a CUDA device this build can use? Both golden rungs ask (#3711).
#[cfg(feature = "cuda")]
fn cuda_device_present() -> bool {
    use realizar::cuda::CudaExecutor;
    CudaExecutor::is_available() && CudaExecutor::num_devices() > 0
}

/// Without the cuda feature there is no device this build can use.
#[cfg(not(feature = "cuda"))]
fn cuda_device_present() -> bool {
    false
}

/// One golden case, judged (#3711).
#[cfg(feature = "inference")]
enum GoldenCaseOutcome {
    /// The CPU answer is right, and so is the GPU's when its leg ran; the leg says which.
    Passed(GpuGoldenLeg),
    /// The gate's verdict for this case: a failure, or the tokenizer skip.
    Verdict(GateResult),
}

/// The GPU half of one golden case (#3711): run and judged when a cuda build has a device
/// and the model is a GGUF, else `NotRun` naming which of those it lacks.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn gpu_golden_leg(
    prompt: &str,
    expected_patterns: &[&str],
    config: &QaConfig,
    format: realizar::format::ModelFormat,
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
    gguf_model: Option<&realizar::gguf::GGUFModel>,
    cuda_available: bool,
    golden_max_tokens: usize,
) -> Result<GpuGoldenLeg> {
    use realizar::format::ModelFormat;
    use realizar::gguf::QuantizedGenerateConfig;

    if let Some(why) = gpu_golden_not_run(true, cuda_available, format == ModelFormat::Gguf) {
        return Ok(GpuGoldenLeg::NotRun(why));
    }
    // Safe: format==Gguf guarantees these are Some
    let gguf_ref = gguf_model.expect("GGUF model required for GPU golden output");
    let mapped_ref = mapped.expect("GGUF mapped model required for GPU golden output");
    let prompt_tokens = golden_prompt_tokens(gguf_ref, prompt);
    // #1864: GPU path mirrors the CPU gate's fix above — set stop_tokens
    // to EOS so generation terminates at end-of-turn rather than running
    // the full 512-token budget and drifting into `<|im_start|>` repeats.
    let gen_config = QuantizedGenerateConfig {
        max_tokens: golden_max_tokens, // GH-279-4: match CPU budget
        temperature: 0.0,
        top_k: 1,
        stop_tokens: golden_stop_tokens(gguf_ref),
        ..Default::default()
    };
    validate_gpu_golden_output(
        mapped_ref,
        &prompt_tokens,
        &gen_config,
        gguf_ref,
        expected_patterns,
        config,
    )
}

/// Without the cuda feature the GPU leg never starts, and says so.
#[cfg(all(feature = "inference", not(feature = "cuda")))]
fn gpu_golden_leg(
    prompt: &str,
    expected_patterns: &[&str],
    config: &QaConfig,
    format: realizar::format::ModelFormat,
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
    gguf_model: Option<&realizar::gguf::GGUFModel>,
    cuda_available: bool,
    golden_max_tokens: usize,
) -> Result<GpuGoldenLeg> {
    let _ = (prompt, expected_patterns, config, mapped, gguf_model, golden_max_tokens);
    let gguf = format == realizar::format::ModelFormat::Gguf;
    Ok(GpuGoldenLeg::NotRun(
        gpu_golden_not_run(false, cuda_available, gguf).unwrap_or("this build has no cuda feature"),
    ))
}

/// Validate a single golden test case: generate output, check GPU parity, verify patterns.
#[cfg(feature = "inference")]
fn validate_golden_test_case(
    path: &Path,
    prompt: &str,
    expected_patterns: &[&str],
    config: &QaConfig,
    format: realizar::format::ModelFormat,
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
    gguf_model: Option<&realizar::gguf::GGUFModel>,
    cuda_available: bool,
    start: Instant,
) -> Result<GoldenCaseOutcome> {
    // GH-279-4: Thinking models (Qwen3) need extra tokens for <think>...</think>
    // chain-of-thought before the answer. 32 tokens is not enough — the model
    // exhausts the budget on reasoning and never emits the answer. Qwen3's
    // thinking can be verbose (~100-200 tokens for simple math), so 512 gives
    // ample room for reasoning + answer.
    let golden_max_tokens = config.max_tokens.max(512);

    let Some((_, output_text)) =
        generate_golden_for_format(path, prompt, golden_max_tokens, format, mapped, gguf_model)?
    else {
        return Ok(GoldenCaseOutcome::Verdict(GateResult::skipped(
            "golden_output",
            "SafeTensors: tokenizer.json not found",
        )));
    };

    // #3870: judge the CPU leg BEFORE the GPU message is composed.
    //
    // This block used to sit AFTER the GPU early return, so on a GPU failure the
    // CPU answer was never judged — while the GPU message hardcoded "(CPU
    // passed)". tinyllama was reported as a CUDA correctness defect blocking the
    // tag on that basis, and it fails identically on CPU.
    let cpu = CpuGoldenVerdict::judge(&output_text, prompt, expected_patterns, golden_max_tokens);

    // #3711: a GPU leg that errored is a FAIL naming the error, never a skip.
    let gpu_leg = gpu_golden_leg(
        prompt,
        expected_patterns,
        config,
        format,
        mapped,
        gguf_model,
        cuda_available,
        golden_max_tokens,
    )?;
    if let Some(failure) = gpu_leg.failure_given_cpu(&cpu) {
        return Ok(GoldenCaseOutcome::Verdict(GateResult::failed(
            "golden_output",
            &failure,
            None,
            None,
            start.elapsed(),
        )));
    }

    // The CPU verdict, already computed above, now decides. Same messages as
    // before; only the ORDER changed.
    match cpu {
        CpuGoldenVerdict::Unclosed { budget, generated_chars } => {
            return Ok(GoldenCaseOutcome::Verdict(GateResult::failed(
                "golden_output",
                &unclosed_think_reason("golden_output", budget, generated_chars),
                None,
                None,
                start.elapsed(),
            )));
        },
        CpuGoldenVerdict::WrongAnswer(reason) => {
            return Ok(GoldenCaseOutcome::Verdict(GateResult::failed(
                "golden_output",
                &reason,
                None,
                None,
                start.elapsed(),
            )));
        },
        CpuGoldenVerdict::Passed => {},
    }

    Ok(GoldenCaseOutcome::Passed(gpu_leg))
}

fn run_golden_output_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!("{}", "Running golden output test...".yellow());
    }

    #[cfg(feature = "inference")]
    {
        use realizar::format::{detect_format, ModelFormat};
        use realizar::gguf::{GGUFModel, MappedGGUFModel};

        let cuda_available = cuda_device_present();
        let model_bytes = std::fs::read(path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;
        let format = detect_format(&model_bytes[..8.min(model_bytes.len())])
            .map_err(|e| CliError::ValidationFailed(format!("Failed to detect format: {e}")))?;

        // GH-239: Only create GGUF objects when format is actually GGUF
        let (mapped, gguf_model) = if format == ModelFormat::Gguf {
            let m = MappedGGUFModel::from_path(path)
                .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
            let g = GGUFModel::from_bytes(&model_bytes)
                .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;
            (Some(m), Some(g))
        } else {
            (None, None)
        };

        // #3711: the pass message names the GPU leg, so a skipped leg says which skip it was
        let mut gpu_leg = GpuGoldenLeg::NotRun("no golden case ran");
        // #3724: ask the model the way production asks it. `general.architecture`
        // is the key the detector takes; a format that declares none keeps ChatML.
        let architecture = mapped
            .as_ref()
            .and_then(|m| m.model.architecture())
            .map(String::from);
        // #3914: the architecture is the model's SHAPE, not its chat format. Let the
        // model's own declared template referee that choice.
        let meta_str = |key: &str| match mapped.as_ref().and_then(|m| m.model.metadata.get(key)) {
            Some(realizar::gguf::GGUFValue::String(s)) => Some(s.clone()),
            _ => None,
        };
        let declared = meta_str("tokenizer.chat_template");
        let model_name = meta_str("general.name");
        let key = template_key(
            architecture.as_deref(),
            model_name.as_deref(),
            declared.as_deref(),
        );
        let test_cases = golden_test_cases_for(key.as_deref());

        for (prompt, expected_patterns) in &test_cases {
            match validate_golden_test_case(
                path,
                prompt.as_str(),
                expected_patterns,
                config,
                format,
                mapped.as_ref(),
                gguf_model.as_ref(),
                cuda_available,
                start,
            )? {
                GoldenCaseOutcome::Verdict(result) => return Ok(result),
                GoldenCaseOutcome::Passed(leg) => gpu_leg = leg,
            }
        }

        // #3724 done_when 3: a thinking-capable model is judged in BOTH modes.
        if let Some((on_prompt, on_patterns)) = thinking_on_case(key.as_deref()) {
            // #3907: the budget is per-model with a basis, and a model with no measured
            // budget REFUSES here rather than inheriting an 8B's number. The refusal is
            // reported as a budget gap, not as "the model was still reasoning" — the two
            // are different findings and only one of them is about the model.
            let model_file = path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default();
            let (on_budget, budget_basis) = match thinking_on_budget_for(&model_file) {
                Ok(v) => v,
                Err(reason) => {
                    return Ok(GateResult::failed(
                        "golden_output",
                        &format!("golden_output_thinking_on: {reason}"),
                        None,
                        None,
                        start.elapsed(),
                    ))
                },
            };
            if let Some((_, on_text)) = generate_golden_for_format(
                path,
                &on_prompt,
                on_budget,
                format,
                mapped.as_ref(),
                gguf_model.as_ref(),
            )? {
                let generated = on_text.strip_prefix(on_prompt.as_str()).unwrap_or(&on_text);
                if let Some(reason) = judge_thinking_on_output(generated, &on_patterns, on_budget)
                    .map(|r| format!("{r} [budget basis — {budget_basis}]"))
                {
                    return Ok(GateResult::failed(
                        "golden_output",
                        &reason,
                        None,
                        None,
                        start.elapsed(),
                    ));
                }
            }
        }

        Ok(GateResult::passed(
            "golden_output",
            &format!(
                "{} golden test cases passed ({})",
                test_cases.len(),
                gpu_leg.describe()
            ),
            Some(test_cases.len() as f64),
            Some(test_cases.len() as f64),
            start.elapsed(),
        ))
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (path, config);
        Ok(GateResult::skipped(
            "golden_output",
            "Requires 'inference' feature",
        ))
    }
}

/// Run warmup+measure loop for throughput benchmarking.
///
/// Calls `generate_fn` for `warmup` iterations (discarding results), then
/// measures `iterations` runs using `BrickTracer` for syscall-level diagnostics.
/// Returns (tokens_per_second, measurement_duration).
#[cfg(feature = "inference")]
fn measure_generate_throughput(
    warmup: usize,
    iterations: usize,
    prompt_len: usize,
    tracer: &TracerImpl,
    brick_name: &str,
    budget_us: u64,
    verbose: bool,
    mut generate_fn: impl FnMut() -> Vec<u32>,
) -> (f64, Duration) {
    // Warmup (untraced)
    for _ in 0..warmup {
        let _ = generate_fn();
    }

    // Measurement (traced via BrickTracer for syscall breakdown)
    let traced = tracer.trace(brick_name, budget_us, || {
        let mut tokens = 0usize;
        for _ in 0..iterations {
            let output = generate_fn();
            tokens += output.len().saturating_sub(prompt_len);
        }
        tokens
    });
    let total_tokens = traced.result;
    let measure_secs = traced.duration_us as f64 / 1_000_000.0;
    let tps = if measure_secs > 0.0 {
        total_tokens as f64 / measure_secs
    } else {
        0.0
    };

    if verbose {
        let bd = &traced.syscall_breakdown;
        eprintln!(
            "  BrickTracer [{brick_name}]: {:.1} tok/s, {}us total",
            tps, traced.duration_us
        );
        eprintln!(
            "    compute: {}us  mmap: {}us  futex: {}us  ioctl: {}us",
            bd.compute_us, bd.mmap_us, bd.futex_us, bd.ioctl_us
        );
        eprintln!(
            "    overhead: {:.1}%  dominant: {}",
            bd.syscall_overhead_percent(),
            bd.dominant_syscall()
        );
        if let Some(ref meta) = traced.metadata {
            eprintln!(
                "    budget: {}us  actual: {}us  efficiency: {:.1}%",
                meta.budget_us,
                meta.actual_us,
                meta.efficiency * 100.0
            );
        }
    }

    let duration = Duration::from_micros(traced.duration_us);
    (tps, duration)
}

/// Measure throughput for a GGUF model (GPU or CPU path).
#[cfg(feature = "inference")]
fn throughput_gguf(
    path: &Path,
    model_bytes: &[u8],
    config: &QaConfig,
    cuda_available: bool,
    tracer: &TracerImpl,
    prompt: &str,
) -> Result<(f64, Duration)> {
    use realizar::gguf::{
        GGUFModel, MappedGGUFModel, OwnedQuantizedModel,
        QuantizedGenerateConfig,
    };

    let gguf = GGUFModel::from_bytes(model_bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;
    let prompt_tokens = golden_prompt_tokens(&gguf, prompt);
    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let budget_us = config.max_tokens as u64 * config.iterations as u64 * 100_000;

    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Map failed: {e}")))?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Model failed: {e}")))?;

    // GH-284: Try CUDA, fall back to CPU on capability mismatch (e.g. missing QkNorm kernel)
    #[cfg(feature = "cuda")]
    if cuda_available {
        use realizar::gguf::OwnedQuantizedModelCuda;
        match OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048) {
            Ok(mut cuda_model) => {
                return Ok(measure_generate_throughput(
                    config.warmup,
                    config.iterations,
                    prompt_tokens.len(),
                    tracer,
                    "qa_throughput_gguf_gpu",
                    budget_us,
                    config.verbose,
                    || {
                        cuda_model
                            .generate_gpu_resident(&prompt_tokens, &gen_config)
                            .unwrap_or_default()
                    },
                ));
            }
            Err(e) => {
                let model = e.into_model();
                return Ok(measure_generate_throughput(
                    config.warmup,
                    config.iterations,
                    prompt_tokens.len(),
                    tracer,
                    "qa_throughput_gguf_cpu_fallback",
                    budget_us,
                    config.verbose,
                    || {
                        model
                            .generate_with_cache(&prompt_tokens, &gen_config)
                            .unwrap_or_default()
                    },
                ));
            }
        }
    }
    Ok(measure_generate_throughput(
        config.warmup,
        config.iterations,
        prompt_tokens.len(),
        tracer,
        "qa_throughput_gguf_cpu",
        budget_us,
        config.verbose,
        || {
            model
                .generate_with_cache(&prompt_tokens, &gen_config)
                .unwrap_or_default()
        },
    ))
}

/// Measure throughput for an APR model.
#[cfg(feature = "inference")]
fn throughput_apr(
    path: &Path,
    config: &QaConfig,
    tracer: &TracerImpl,
    prompt: &str,
) -> Result<(f64, Duration)> {
    use realizar::apr::AprV2Model;
    use realizar::apr_transformer::{AprTransformer, GenerateConfig};

    let apr_model = AprV2Model::load(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;
    let tokenizer = apr_model
        .load_embedded_bpe_tokenizer()
        .ok_or_else(|| CliError::ValidationFailed("APR missing embedded tokenizer".to_string()))?;
    let transformer = AprTransformer::from_apr_file(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR transformer: {e}")))?;

    let prompt_tokens = tokenizer.encode(prompt);
    let gen_config = GenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };
    let budget_us = config.max_tokens as u64 * config.iterations as u64 * 100_000;

    Ok(measure_generate_throughput(
        config.warmup,
        config.iterations,
        prompt_tokens.len(),
        tracer,
        "qa_throughput_apr",
        budget_us,
        config.verbose,
        || {
            transformer
                .generate_with_cache(&prompt_tokens, &gen_config)
                .unwrap_or_default()
        },
    ))
}

// =============================================================================
// PMAT-125 B4: golden_test_cases fixture coverage
// =============================================================================

#[cfg(all(test, feature = "inference"))]
mod thinking_budget_resolution {
    use super::{glob_match, thinking_on_budget_for};

    /// The listed model with NO budget must REFUSE, not inherit `default`. This is the
    /// whole mechanism: inheriting 2048 would republish an 8B's measurement as a 0.8B's.
    #[test]
    fn a_listed_model_without_a_budget_is_refused_not_defaulted() {
        let err = thinking_on_budget_for("Qwen3.5-0.8B-IQ4_XS.gguf")
            .expect_err("a model listed with no budget must refuse");
        assert!(err.contains("no measured thinking budget"), "{err}");
        assert!(err.contains("Refusing rather than inheriting"), "{err}");
        assert!(err.contains("8,901"), "the refusal must carry what was measured: {err}");
    }

    /// An unlisted model takes `default` — and the returned provenance SAYS it is the
    /// default, so a reader of a failure can tell an inherited number from a measured one.
    #[test]
    fn an_unlisted_model_takes_the_default_and_says_so() {
        let (budget, basis) =
            thinking_on_budget_for("some-other-model-q4km.gguf").expect("default applies");
        assert_eq!(budget, 2048);
        assert!(basis.starts_with("default:"), "{basis}");
        assert!(basis.contains("qwen3-8b-q4km"), "the basis must name its one model: {basis}");
    }

    /// The env seam exists because the old `const` had none, and a threshold that needs a
    /// `--features cuda,inference` rebuild to vary is one nobody re-measures. A probed
    /// number must never be mistakable for a measured one.
    #[test]
    fn the_probe_override_is_labelled_as_a_probe() {
        std::env::set_var("APR_THINKING_ON_BUDGET", "8192");
        let (budget, basis) = thinking_on_budget_for("Qwen3.5-0.8B-IQ4_XS.gguf")
            .expect("the override applies even to a refused model, so it can be probed");
        std::env::remove_var("APR_THINKING_ON_BUDGET");
        assert_eq!(budget, 8192);
        assert!(basis.contains("PROBE OVERRIDE"), "{basis}");
        assert!(basis.contains("not a measurement"), "{basis}");
    }

    #[test]
    fn glob_match_is_the_fnmatch_the_ladder_uses() {
        assert!(glob_match("Qwen3.5-0.8B-*", "Qwen3.5-0.8B-IQ4_XS.gguf"));
        assert!(glob_match("exact.gguf", "exact.gguf"));
        assert!(!glob_match("Qwen3.5-0.8B-*", "Qwen3.5-4B-UD-Q4_K_XL.gguf"));
        // a pattern must not match a name shorter than its own literal halves
        assert!(!glob_match("aaaa*bbbb", "aaaabbb"));
    }
}

#[cfg(test)]
mod golden_output_tests {
    use super::*;

    #[test]
    fn test_golden_test_cases_nonempty_and_structured() {
        let cases = golden_test_cases_for(None);
        assert!(!cases.is_empty());
        for (prompt, patterns) in &cases {
            assert!(prompt.contains("<|im_start|>assistant"));
            assert!(prompt.contains("<|im_start|>user"));
            assert!(!patterns.is_empty());
        }
    }

    #[test]
    fn test_golden_test_cases_arithmetic_case_present() {
        let cases = golden_test_cases_for(None);
        let arith = cases
            .iter()
            .find(|(p, _)| p.contains("2+2"))
            .expect("arithmetic golden case must exist");
        assert!(arith.1.contains(&"4"));
    }

    #[test]
    fn test_golden_test_cases_deterministic() {
        let a = golden_test_cases_for(None);
        let b = golden_test_cases_for(None);
        assert_eq!(a.len(), b.len());
        for ((pa, _), (pb, _)) in a.iter().zip(b.iter()) {
            assert_eq!(pa, pb);
        }
    }

    /// Poka-yoke for #2350: no golden prompt may be a bare one-or-two-word user
    /// message.
    ///
    /// These cases assert on exact greedy-decoded content, so a prompt whose
    /// first generated token is a near-tie flips between backends. The removed
    /// case was a single word ("Hello") and produced
    /// "Hello! How can I assist you today?" on CPU but
    /// "I'm sorry, but I'm not sure what you're asking" on CUDA — while "Hi"
    /// produced the evasive answer on CPU instead. The model answers bare
    /// greetings on a knife edge; the backends merely disagree about which side.
    ///
    /// Word count is a crude proxy for "wide argmax margin", but it is the one
    /// that is checkable without a GPU in CI, and it blocks the specific shape
    /// that actually cost 24 days of a red nightly.
    #[test]
    fn golden_prompts_are_not_bare_one_word_messages() {
        for (prompt, _) in golden_test_cases_for(None) {
            let user_msg = prompt
                .split("<|im_start|>user\n")
                .nth(1)
                .and_then(|s| s.split("<|im_end|>").next())
                .unwrap_or("");
            let words = user_msg.split_whitespace().count();
            assert!(
                words >= 3,
                "golden prompt user message {user_msg:?} has {words} word(s). \
                 Bare greetings sit on a near-tie under greedy decoding and flip \
                 between CPU and CUDA (#2350) — use a prompt with a wide argmax \
                 margin and verify it on BOTH backends before adding it."
            );
        }
    }

    // =========================================================================
    // #3711: the GPU leg's verdicts. An ERROR is a FAIL naming the error, never
    // a skip; a skip is only a leg that never started, and it names which.
    // =========================================================================

    const TWO_PLUS_TWO: &[&str] = &["4"];

    #[test]
    fn gpu_generation_error_fails_the_gate_naming_the_error() {
        let leg = GpuGoldenLeg::judge(
            Err("GPU generation: CUDA_ERROR_ILLEGAL_ADDRESS".to_string()),
            TWO_PLUS_TWO,
            512,
        );
        assert_eq!(
            leg,
            GpuGoldenLeg::Errored("GPU generation: CUDA_ERROR_ILLEGAL_ADDRESS".to_string())
        );
        let failure = leg.failure_given_cpu(&CpuGoldenVerdict::Passed).expect("a GPU generation error must FAIL the gate");
        assert!(failure.contains("CUDA_ERROR_ILLEGAL_ADDRESS"), "{failure}");
        assert!(failure.contains("not a skip"), "{failure}");
    }

    #[test]
    fn cuda_init_error_on_a_host_with_a_device_fails_the_gate() {
        let leg = GpuGoldenLeg::judge(
            Err("CUDA init on device 0: CUDA_ERROR_OUT_OF_MEMORY".to_string()),
            TWO_PLUS_TWO,
            512,
        );
        let failure = leg.failure_given_cpu(&CpuGoldenVerdict::Passed).expect("a CUDA init error must FAIL the gate");
        assert!(failure.contains("CUDA_ERROR_OUT_OF_MEMORY"), "{failure}");
    }

    #[test]
    fn gpu_wrong_answer_fails_the_gate() {
        let leg = GpuGoldenLeg::judge(Ok("2 + 2 = 5".to_string()), TWO_PLUS_TWO, 512);
        assert!(matches!(leg, GpuGoldenLeg::WrongAnswer(_)), "{leg:?}");
        let failure = leg.failure_given_cpu(&CpuGoldenVerdict::Passed).expect("a wrong GPU answer must FAIL the gate");
        assert!(failure.starts_with("GPU output failed (CPU passed)"), "{failure}");
    }

    // =========================================================================
    // #3870: the stop token comes from the MODEL, not from a Qwen constant
    //
    // The gate passed SpecialTokens::qwen2().eos_id — 151645 — as the stop token
    // for every model. For tinyllama (vocab 32000, eos 2) that id does not exist
    // in the vocabulary, so the stop condition could not fire and the model ran
    // its full budget into chat-scaffolding drift. Both legs hardcoded it, which
    // is why GPU and CPU output were byte-identical.
    // =========================================================================

    /// Every GGUF the box has, so this reads as a matrix and not as one anecdote.
    #[cfg(feature = "inference")]
    fn local_ggufs() -> Vec<std::path::PathBuf> {
        let root = std::env::var("APR_MODELS").ok().filter(|v| !v.trim().is_empty()).map_or_else(
            || {
                std::env::var("HOME").map_or_else(
                    |_| std::path::PathBuf::from("/nonexistent"),
                    |h| std::path::PathBuf::from(h).join(".apr/models"),
                )
            },
            std::path::PathBuf::from,
        );
        let Ok(rd) = std::fs::read_dir(&root) else {
            return Vec::new();
        };
        let mut v: Vec<_> = rd
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
            .collect();
        v.sort();
        v
    }

    /// OBLIG-SPECIAL-TOKEN-WITHIN-VOCAB (PMAT-908), applied to the golden gate.
    ///
    /// The general invariant, stated so it holds for models this box has never
    /// seen: whatever the gate passes as a stop token must be a token that
    /// EXISTS in the model being gated. A stop token the model cannot emit is
    /// not a stop token.
    #[cfg(feature = "inference")]
    #[test]
    fn the_golden_stop_token_is_reachable_in_every_model_we_have() {
        let models = local_ggufs();
        if models.is_empty() {
            println!("SKIP: no GGUFs under $APR_MODELS or $HOME/.apr/models");
            return;
        }
        let mut checked = 0usize;
        for path in &models {
            let Ok(bytes) = std::fs::read(path) else { continue };
            let Ok(gguf) = realizar::gguf::GGUFModel::from_bytes(&bytes) else { continue };
            let Some(vocab) = gguf.vocabulary().map(|v| v.len()) else { continue };
            let name = path.file_name().unwrap_or_default().to_string_lossy();

            for tok in golden_stop_tokens(&gguf) {
                assert!(
                    (tok as usize) < vocab,
                    "#3870: {name} has vocab {vocab}, so stop token {tok} is not a token this \
                     model can EVER emit — the stop condition cannot fire and the gate will \
                     score the model's full-budget drift instead of its answer"
                );
                checked += 1;
            }
            // And it must be the model's OWN eos, not a constant that happens to
            // be in range.
            assert_eq!(
                golden_stop_tokens(&gguf),
                gguf.eos_token_id().into_iter().collect::<Vec<_>>(),
                "{name}: the gate must stop on the model's own eos"
            );
        }
        assert!(checked > 0, "no model yielded a stop token; this test proved nothing");
        println!("#3870: {checked} stop token(s) checked across {} GGUFs", models.len());
    }

    /// The control that makes the row above non-vacuous, and the one the cop
    /// asked for: the fix must not be "stop honouring stop tokens".
    ///
    /// A Qwen model's stop token must still be exactly 151645 — the value the
    /// old hardcoded constant supplied — so Qwen behaviour is provably
    /// unchanged by this fix. If this row and the one above are both green, the
    /// gate reads the model without having lost the thing #1864 added.
    #[cfg(feature = "inference")]
    #[test]
    fn a_qwen_model_still_stops_on_151645_and_a_llama_model_does_not() {
        let mut saw_qwen = false;
        let mut saw_non_qwen = false;
        for path in local_ggufs() {
            let Ok(bytes) = std::fs::read(&path) else { continue };
            let Ok(gguf) = realizar::gguf::GGUFModel::from_bytes(&bytes) else { continue };
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
            let stops = golden_stop_tokens(&gguf);
            if name.contains("qwen3") || name.contains("qwen2") {
                assert_eq!(
                    stops,
                    vec![151_645],
                    "{name}: Qwen must still stop where SpecialTokens::qwen2() said it did — \
                     losing that would make this fix a regression dressed as a repair"
                );
                saw_qwen = true;
            } else if name.contains("tinyllama") {
                assert_eq!(
                    stops,
                    vec![2],
                    "{name}: a Llama-2 model must stop on ITS eos, not on Qwen's 151645"
                );
                saw_non_qwen = true;
            }
        }
        if !saw_qwen && !saw_non_qwen {
            println!("SKIP: neither a Qwen nor a tinyllama GGUF is present");
        }
    }

    // =========================================================================
    // #3870: a GPU failure message may not ASSERT the CPU leg's verdict
    //
    // `failure()` hardcoded "(CPU passed)" and `validate_golden_test_case`
    // returned on a GPU failure BEFORE the CPU pattern check ran, so the claim
    // was made about a leg that had never been judged. That inverted a release
    // verdict once already (tinyllama, below), so both directions are asserted:
    // the false claim must be impossible, and the TRUE one must survive.
    // =========================================================================

    /// The golden capital case's patterns — the case tinyllama failed.
    const CAPITAL_OF_FRANCE: &[&str] = &["Paris"];

    /// What `tinyllama-1.1b-chat-v1.0.Q4_K_M` actually produced for the capital
    /// case: a `[S][INST]` loop with no "Paris" anywhere. Measured 2026-09-22 on
    /// `apr 0.69.1 (87d9d5484)` and produced IDENTICALLY by the GPU leg and by
    /// the CPU leg under `CUDA_VISIBLE_DEVICES=""`. The old message reported this
    /// as "GPU output failed (CPU passed)" and it was escalated as a CUDA
    /// correctness defect blocking the 0.69.1 tag.
    const TINYLLAMA_LOOP: &str = "[S][INST][S][INST][S][INST][S][INST][S][INST]";

    #[test]
    fn a_gpu_failure_may_not_claim_a_cpu_pass_when_the_cpu_failed_too() {
        let gpu = GpuGoldenLeg::judge(Ok(TINYLLAMA_LOOP.to_string()), CAPITAL_OF_FRANCE, 512);
        assert!(matches!(gpu, GpuGoldenLeg::WrongAnswer(_)), "{gpu:?}");

        // The SAME text on the CPU leg, because that is what the model does on
        // both backends.
        let cpu = CpuGoldenVerdict::judge(TINYLLAMA_LOOP, "", CAPITAL_OF_FRANCE, 512);
        assert!(matches!(cpu, CpuGoldenVerdict::WrongAnswer(_)), "{cpu:?}");

        let failure = gpu
            .failure_given_cpu(&cpu)
            .expect("a wrong GPU answer must still FAIL the gate");
        assert!(
            !failure.contains("CPU passed"),
            "#3870: the CPU leg produced the same wrong answer, so this message \
             claims a measurement that did not happen and reads as a GPU-specific \
             defect: {failure}"
        );
        assert!(
            failure.contains("CPU failed too") && failure.contains("not a GPU-specific"),
            "the message must say what was actually measured: {failure}"
        );
    }

    #[test]
    fn a_gpu_failure_still_names_a_genuine_cpu_pass() {
        // The converse, and the reason the gate exists (#3477): the GPU is wrong
        // and the CPU is RIGHT on the same prompt. Losing this wording to fix the
        // case above would be the over-correction.
        let gpu = GpuGoldenLeg::judge(Ok(TINYLLAMA_LOOP.to_string()), CAPITAL_OF_FRANCE, 512);
        let cpu = CpuGoldenVerdict::judge(
            "The capital of France is Paris.",
            "",
            CAPITAL_OF_FRANCE,
            512,
        );
        assert_eq!(cpu, CpuGoldenVerdict::Passed, "{cpu:?}");

        let failure = gpu.failure_given_cpu(&cpu).expect("a GPU-only defect FAILS the gate");
        assert!(
            failure.starts_with("GPU output failed (CPU passed)"),
            "#3477's signature — a real GPU-specific defect — must survive #3870's fix: {failure}"
        );
    }

    #[test]
    fn a_gpu_failure_says_so_when_the_cpu_reached_no_verdict() {
        // The third CPU state: still inside an unclosed <think> at the budget.
        // There is no CPU pass AND no CPU wrong answer to compare against, and
        // claiming either would be the same defect in a different direction.
        let gpu = GpuGoldenLeg::judge(Ok(TINYLLAMA_LOOP.to_string()), CAPITAL_OF_FRANCE, 512);
        let cpu = CpuGoldenVerdict::judge(
            "<think>the capital of France is",
            "",
            CAPITAL_OF_FRANCE,
            512,
        );
        assert!(matches!(cpu, CpuGoldenVerdict::Unclosed { .. }), "{cpu:?}");

        let failure = gpu.failure_given_cpu(&cpu).expect("the GPU leg still FAILS");
        assert!(!failure.contains("CPU passed"), "{failure}");
        assert!(!failure.contains("CPU failed too"), "{failure}");
        assert!(
            failure.contains("no CPU verdict to compare against"),
            "the message must name the absence rather than pick a side: {failure}"
        );
    }

    #[test]
    fn the_cpu_verdict_cannot_be_omitted_from_the_claim() {
        // A structural row, not a behavioural one. The ordering defect was that
        // `validate_golden_test_case` composed the GPU message before the CPU
        // leg had been judged. `failure_given_cpu` takes the verdict BY ARGUMENT,
        // so that ordering is now a type error rather than a review item: there is
        // no way to obtain the message without a CpuGoldenVerdict in hand.
        //
        // This test exists to state that intent where a future edit will read it.
        // Reintroducing a `failure(&self)` that guesses would compile — and would
        // make the three rows above dead. If you are here because you added one,
        // that is what this row is objecting to.
        let gpu = GpuGoldenLeg::judge(Ok(TINYLLAMA_LOOP.to_string()), CAPITAL_OF_FRANCE, 512);
        let messages: Vec<String> = [
            CpuGoldenVerdict::Passed,
            CpuGoldenVerdict::WrongAnswer("no Paris".to_string()),
            CpuGoldenVerdict::Unclosed { budget: 512, generated_chars: 30 },
        ]
        .iter()
        .map(|cpu| gpu.failure_given_cpu(cpu).expect("all three still FAIL"))
        .collect();

        // Three CPU verdicts, three DISTINCT messages. If any two coincide, the
        // message is not carrying the CPU leg's verdict and the reader cannot
        // tell which was measured.
        assert_ne!(messages[0], messages[1]);
        assert_ne!(messages[1], messages[2]);
        assert_ne!(messages[0], messages[2]);
    }

    #[test]
    fn gpu_right_answer_passes_and_says_it_was_judged() {
        for text in ["2 + 2 = 4", "<think>two and two</think>The answer is 4."] {
            let leg = GpuGoldenLeg::judge(Ok(text.to_string()), TWO_PLUS_TWO, 512);
            assert_eq!(leg, GpuGoldenLeg::Passed, "{text:?}");
            assert_eq!(leg.failure_given_cpu(&CpuGoldenVerdict::Passed), None);
            assert_eq!(leg.describe(), "GPU leg judged on CUDA device 0");
        }
    }

    #[test]
    fn no_device_is_the_skip_and_names_it() {
        let why = gpu_golden_not_run(true, false, true).expect("no device: the leg never starts");
        assert_eq!(why, "no CUDA device on this host");
        let leg = GpuGoldenLeg::NotRun(why);
        assert_eq!(leg.failure_given_cpu(&CpuGoldenVerdict::Passed), None, "a leg that never started is not a failure");
        assert_eq!(leg.describe(), "GPU leg SKIPPED: no CUDA device on this host");
    }

    #[test]
    fn each_skip_names_which_and_a_cuda_host_runs_the_leg() {
        assert_eq!(
            gpu_golden_not_run(false, true, true),
            Some("this build has no cuda feature")
        );
        assert_eq!(
            gpu_golden_not_run(false, false, false),
            Some("this build has no cuda feature")
        );
        assert_eq!(
            gpu_golden_not_run(true, true, false),
            Some("the dense GPU leg judges GGUF only")
        );
        // a cuda build with a device and a GGUF: the leg STARTS, so no skip can be reported
        assert_eq!(gpu_golden_not_run(true, true, true), None);
    }

    // The runtime rung (the hybrid, and architectures the GPU declines) names the
    // backend the dispatch REPORTED. It never names the one the build could have used.

    #[test]
    fn runtime_cuda_build_with_no_device_is_labelled_cpu_never_gpu() {
        // the old message said "GPU hybrid forward" on every cuda build
        let not_run = gpu_golden_not_run(true, false, true);
        let label = runtime_golden_backend(false, not_run).expect("CPU was expected: a pass");
        assert!(label.starts_with("CPU: the dispatch reported used_gpu=false"), "{label}");
        assert!(label.contains("no CUDA device on this host"), "{label}");
        assert!(!label.contains("GPU:"), "{label}");
    }

    #[test]
    fn runtime_gpu_fallback_on_a_cuda_host_fails_the_gate() {
        // a cuda build, a device, an architecture the GPU runs, and the dispatch reported CPU
        let failure = runtime_golden_backend(false, gpu_golden_not_run(true, true, true))
            .expect_err("a GPU that fell back to the CPU must FAIL the gate");
        assert!(failure.contains("fell back"), "{failure}");
        assert!(failure.contains("not a pass"), "{failure}");
    }

    #[test]
    fn runtime_gpu_that_served_is_labelled_gpu_from_the_dispatch() {
        let label = runtime_golden_backend(true, gpu_golden_not_run(true, true, true))
            .expect("the GPU served: a pass");
        assert_eq!(label, "GPU: the dispatch reported used_gpu=true");
    }

    #[test]
    fn runtime_architecture_the_gpu_declines_is_a_cpu_pass_that_says_why() {
        let label = runtime_golden_backend(false, Some("the GPU backend declines this architecture"))
            .expect("CPU was expected: a pass");
        assert!(label.contains("the GPU backend declines this architecture"), "{label}");
    }

    /// THE MERGE TEST (#3711 + #3724). Both behaviours have to survive the fold,
    /// and they are the same principle applied to two failure modes: a leg that
    /// could not run is never a skip, and a model still reasoning is never an
    /// empty answer. If the merged type cannot tell `Errored`, `Unclosed` and
    /// `NotRun` apart, the merge lost something — this goes RED if any of the
    /// three collapses into another.
    #[test]
    fn errored_unclosed_and_not_run_are_three_different_verdicts() {
        let errored = GpuGoldenLeg::judge(Err("CUDA init on device 0: OOM".to_string()), TWO_PLUS_TWO, 512);
        let unclosed = GpuGoldenLeg::judge(
            Ok("<think>let me work through this carefully".to_string()),
            TWO_PLUS_TWO,
            512,
        );
        let not_run = GpuGoldenLeg::NotRun("no CUDA device on this host");
        let wrong = GpuGoldenLeg::judge(Ok("<think>2+2</think>It is five.".to_string()), TWO_PLUS_TWO, 512);
        let passed = GpuGoldenLeg::judge(Ok("<think>2+2</think>2 + 2 = 4.".to_string()), TWO_PLUS_TWO, 512);

        // All five are distinct values, so none can be silently produced for another.
        assert!(matches!(errored, GpuGoldenLeg::Errored(_)));
        assert!(matches!(unclosed, GpuGoldenLeg::Unclosed { .. }));
        assert!(matches!(not_run, GpuGoldenLeg::NotRun(_)));
        assert!(matches!(wrong, GpuGoldenLeg::WrongAnswer(_)));
        assert_eq!(passed, GpuGoldenLeg::Passed);

        // #3711: an error FAILS and names the error, and is not a skip.
        let e = errored.failure_given_cpu(&CpuGoldenVerdict::Passed).expect("an error fails the gate");
        assert!(e.contains("OOM") && e.contains("not a skip"), "{e}");

        // #3724: an unclosed block FAILS by name WITH the budget, and never reads
        // as an empty answer.
        let u = unclosed.failure_given_cpu(&CpuGoldenVerdict::Passed).expect("an unclosed block fails the gate");
        assert!(u.contains("think block unclosed within 512 tokens"), "{u}");
        assert!(!u.contains("Empty output"), "{u}");
        assert_ne!(u, e, "an unclosed block must not report as an error");

        // A leg that never started is the ONLY thing that does not fail.
        assert!(not_run.failure_given_cpu(&CpuGoldenVerdict::Passed).is_none());
        assert!(passed.failure_given_cpu(&CpuGoldenVerdict::Passed).is_none());
        assert!(not_run.describe().contains("SKIPPED"), "{}", not_run.describe());
        assert!(!unclosed.describe().contains("SKIPPED"), "an unclosed block is not a skip");
    }

    // =========================================================================
    // #3724: the gate asks the model the way PRODUCTION asks it
    // =========================================================================

    /// The legacy prompt, spelled out once so the pin is on a literal and not on
    /// the function under test.
    fn legacy_chatml(question: &str) -> String {
        format!("<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
    }

    /// done_when 2, second half: "Every other architecture keeps byte-identical
    /// ChatML, pinned by a test."
    ///
    /// The CLAIM here is derived, not hand-written: the detector decides which
    /// architectures are ChatML, and each one must render exactly what the gate
    /// hardcoded before #3724. The architecture names below are a SAMPLE drawn
    /// from `contracts/arch-constraints-v1.yaml` — they are not a registry, and
    /// nothing may be concluded from one being absent. Both partitions are
    /// asserted non-empty so the test cannot pass vacuously if the detector
    /// changes under it.
    #[test]
    fn chatml_architectures_render_the_legacy_prompt_byte_for_byte() {
        use realizar::chat_template::{detect_format_from_name, TemplateFormat};
        let sample = [
            "gpt2", "llama", "qwen2", "qwen3", "qwen3_moe", "qwen35", "mistral", "phi2", "phi",
            "gemma", "deepseek", "mamba", "stablelm", "yi", "internlm2", "smollm", "olmo",
            "granite", "starcoder", "falcon_7b",
        ];
        let (mut chatml, mut other) = (0usize, 0usize);
        for arch in sample {
            let is_chatml = detect_format_from_name(arch) == TemplateFormat::ChatML;
            for (question, _) in golden_questions() {
                let rendered = golden_prompt_for(Some(arch), question);
                if is_chatml {
                    assert_eq!(
                        rendered,
                        legacy_chatml(question),
                        "{arch} maps to ChatML, so #3724 must not have changed its prompt"
                    );
                } else {
                    // Not a failure — it is the point. Recorded so the blast
                    // radius of #3724 is visible rather than silent.
                    assert_ne!(
                        rendered,
                        legacy_chatml(question),
                        "{arch} maps to {:?}, so the gate must now send THAT, not ChatML",
                        detect_format_from_name(arch)
                    );
                }
            }
            if is_chatml {
                chatml += 1;
            } else {
                other += 1;
            }
        }
        assert!(chatml > 0, "no ChatML architecture in the sample — the byte-identity half of done_when 2 went untested");
        assert!(other > 0, "no non-ChatML architecture in the sample — the detector half went untested");
        assert_eq!(
            detect_format_from_name("qwen2"),
            TemplateFormat::ChatML,
            "qwen2 is a ladder rung and its prompt must be the unchanged one"
        );
    }

    /// done_when 2, first half, and the whole defect: `qwen3` must be asked with
    /// production's no-think template, not ChatML. Without the `<think>\n</think>`
    /// prefill the model reasons for the full budget and the gate reports "Empty
    /// output" (measured on lambda, 527 tokens for "What is 2+2?").
    #[test]
    fn qwen3_is_asked_with_the_production_no_think_prompt() {
        for arch in ["qwen3", "qwen35"] {
            let prompt = golden_prompt_for(Some(arch), "What is 2+2?");
            assert_eq!(
                prompt,
                "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n",
                "{arch} must get production's no-think prompt"
            );
            assert_ne!(
                prompt,
                legacy_chatml("What is 2+2?"),
                "{arch} must NOT get the gate's old ChatML — that is the defect"
            );
        }
    }

    /// The gate and production must not drift apart by construction: whatever the
    /// detector says for an architecture is what the gate sends. This fails if a
    /// later change re-hardcodes a template in the gate.
    #[test]
    fn the_gate_renders_through_the_same_detector_production_uses() {
        use realizar::chat_template::{format_messages, ChatMessage};
        for arch in ["qwen2", "qwen3", "qwen35", "llama", "mistral", "phi2"] {
            for (question, _) in golden_questions() {
                let production = format_messages(&[ChatMessage::user(question)], Some(arch))
                    .expect("production renders this architecture");
                assert_eq!(
                    golden_prompt_for(Some(arch), question),
                    production,
                    "{arch}: the gate's prompt must BE production's prompt"
                );
            }
        }
    }

    /// A format that declares no architecture (APR, SafeTensors) is unchanged.
    #[test]
    fn an_unknown_architecture_keeps_the_legacy_chatml() {
        for arch in [None, Some(""), Some("   ")] {
            for (question, _) in golden_questions() {
                assert_eq!(golden_prompt_for(arch, question), legacy_chatml(question));
            }
        }
    }

    // ---- done_when 3: both modes -------------------------------------------

    /// The ON prompt is production's own prompt with the suppression removed —
    /// derived from the rendering, so a replacement template is followed rather
    /// than bypassed.
    #[test]
    fn the_thinking_on_prompt_is_production_minus_the_suppression() {
        let production = golden_prompt_for(Some("qwen3"), "What is 2+2?");
        assert!(production.ends_with("<think>\n</think>\n"), "{production}");
        let (on_prompt, patterns) = thinking_on_case(Some("qwen3")).expect("qwen3 is judged in both modes");
        assert_eq!(on_prompt, legacy_chatml("What is 2+2?"));
        assert_eq!(patterns, golden_questions()[0].1);
        assert!(!on_prompt.contains("<think>"), "the suppression is gone: {on_prompt}");
    }

    /// An architecture production does NOT suppress has no second mode to judge,
    /// and the gate must not invent one.
    #[test]
    fn an_architecture_without_suppression_has_no_thinking_on_leg() {
        assert!(thinking_on_case(Some("qwen2")).is_none());
        assert!(thinking_on_case(Some("llama")).is_none());
        assert!(thinking_on_case(None).is_none());
    }

    /// Only an EMPTY prefilled block is suppression. A block with reasoning in it
    /// is content, and cutting it would change the conversation.
    #[test]
    fn a_think_block_with_content_is_not_a_suppression_prefill() {
        assert_eq!(
            without_thinking_prefill("<|im_start|>assistant\n<think>\n</think>\n").as_deref(),
            Some("<|im_start|>assistant\n")
        );
        assert_eq!(
            without_thinking_prefill("<|im_start|>assistant\n<think>hmm</think>\n"),
            None
        );
        assert_eq!(without_thinking_prefill("<|im_start|>assistant\n"), None);
        // Something after the block is not a prefill either.
        assert_eq!(
            without_thinking_prefill("<think>\n</think>\nalready answering"),
            None
        );
    }

    /// done_when 3: ON passes only when the block CLOSES and the answer is right.
    #[test]
    fn the_thinking_on_leg_judges_closure_and_the_answer() {
        let patterns = vec!["4"];
        // closed + correct → pass
        assert_eq!(
            judge_thinking_on_output("<think>2 plus 2</think>2 + 2 = 4.", &patterns, 2048),
            None
        );
        // closed + wrong answer → the answer's failure, not a think failure
        let wrong = judge_thinking_on_output("<think>2 plus 2</think>It is five.", &patterns, 2048)
            .expect("a wrong answer fails");
        assert!(wrong.contains("golden_output_thinking_on"), "{wrong}");
        assert!(!wrong.contains("unclosed"), "{wrong}");
        // unclosed → reported by name WITH the budget, never as an empty answer
        let unclosed = judge_thinking_on_output("<think>let me work through this", &patterns, 2048)
            .expect("an unclosed block fails");
        assert!(
            unclosed.contains("think block unclosed within 2048 tokens"),
            "{unclosed}"
        );
        assert!(!unclosed.contains("Empty output"), "{unclosed}");
        // never entered thinking at all → the leg proved nothing, and says so
        let absent = judge_thinking_on_output("2 + 2 = 4.", &patterns, 2048)
            .expect("no think block means the leg judged nothing");
        assert!(absent.contains("never entered"), "{absent}");
        // #3948: closed but EMPTY → the reasoning was skipped; a right answer does not save it
        for empty in ["<think></think>2 + 2 = 4.", "<think>\n\n</think>\n\n2 + 2 = 4."] {
            let skipped = judge_thinking_on_output(empty, &patterns, 2048)
                .expect("an empty think block proves no reasoning happened");
            assert!(skipped.contains("closed EMPTY within 2048 tokens"), "{skipped}");
            assert!(!skipped.contains("never entered"), "{skipped}");
        }
    }

    /// The ON budget is the ON leg's alone. #3724's ruling: the fix is the prompt,
    /// never the budget — the production leg keeps the budget it had.
    ///
    /// #3907 wiring: this used to open `assert_eq!(THINKING_ON_BUDGET, 2048)`, which
    /// pinned a constant's VALUE and said nothing about whether anything READ it. It
    /// passed while the hybrid leg bypassed the resolver entirely, and it would have
    /// passed after the bypass was fixed with the constant dead — an assertion
    /// orthogonal to the property it appeared to guard. Deleted with the constant.
    /// What remains is the claim that actually constrains something: the ON leg's
    /// budget does not become the production leg's.
    #[test]
    fn the_thinking_on_budget_does_not_touch_the_production_leg() {
        let config = QaConfig::default();
        assert_eq!(
            config.max_tokens.max(512),
            512,
            "the production leg's budget is still 512 by default"
        );
    }

    /// Whatever the architecture, the QUESTIONS and the expected patterns are the
    /// same ones — #3724 changed how the gate asks, never what it checks, and
    /// "no check is relaxed" is part of done_when 2.
    #[test]
    fn changing_the_architecture_changes_the_wrapper_and_never_the_question() {
        let chatml = golden_test_cases_for(Some("qwen2"));
        let nothink = golden_test_cases_for(Some("qwen3"));
        assert_eq!(chatml.len(), golden_questions().len());
        assert_eq!(chatml.len(), nothink.len());
        for (((c_prompt, c_pat), (n_prompt, n_pat)), (question, patterns)) in
            chatml.iter().zip(nothink.iter()).zip(golden_questions())
        {
            assert_eq!(c_pat, &patterns, "patterns are architecture-independent");
            assert_eq!(n_pat, &patterns, "patterns are architecture-independent");
            assert!(c_prompt.contains(question), "the question survives the wrapper");
            assert!(n_prompt.contains(question), "the question survives the wrapper");
        }
    }
}

include!("throughput.rs");


/// #3914: which string the template detector is keyed on.
///
/// Every triple below is MEASURED from the GGUF on disk (`general.architecture`,
/// `general.name`, `tokenizer.chat_template`), not invented, because the whole point of
/// the referee is that it must be inert for models whose declared template agrees with
/// their architecture — and "agrees" is a fact about real files.
#[cfg(all(test, feature = "inference"))]
mod template_key_3914 {
    use super::*;

    // Measured 2026-09-23 from the files in ~/.cache/apr-home/models.
    const TINYLLAMA_DECLARED: &str = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token }}{% endif %}{% endfor %}";
    const QWEN_CHATML_DECLARED: &str = "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}{%- endfor %}";
    const QWEN3_THINK_DECLARED: &str = "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}{%- endfor %}{{- '<think>' }}";

    /// THE ROW. `llama` is the architecture; the Zephyr rule keyed on "tinyllama" is
    /// unreachable from it. The model's own template never says `[INST]`, so the
    /// architecture is refuted and the name is used.
    #[test]
    fn tinyllama_is_keyed_on_its_name_because_its_own_template_refutes_inst() {
        let key = template_key(
            Some("llama"),
            Some("tinyllama_tinyllama-1.1b-chat-v1.0"),
            Some(TINYLLAMA_DECLARED),
        );
        assert_eq!(key.as_deref(), Some("tinyllama_tinyllama-1.1b-chat-v1.0"));

        let rendered = golden_prompt_for(key.as_deref(), "What is the capital of France?");
        assert!(rendered.contains("<|user|>"), "{rendered:?}");
        assert!(
            !rendered.contains("[INST]"),
            "the model was sent a format it was never trained on: {rendered:?}"
        );
    }

    /// MUST-RED CONTROL for the naive fix. Keying on the name unconditionally sends
    /// this MoE model a think block, which `detect_format_from_name`'s own comment says
    /// makes it emit `<|endoftext|>` immediately. Its declared template agrees with its
    /// architecture, so the referee must leave it ALONE.
    #[test]
    fn the_moe_architecture_survives_although_its_name_contains_qwen3() {
        assert_eq!(
            template_key(
                Some("qwen3moe"),
                Some("Qwen3-Coder-30B-A3B-Instruct"),
                Some(QWEN_CHATML_DECLARED),
            )
            .as_deref(),
            Some("qwen3moe"),
            "keying on the name would flip this model to a thinking template"
        );
        // and the name really would have chosen differently — so the assertion above is
        // discriminating, not incidentally true.
        assert_ne!(
            realizar::chat_template::detect_format_from_name("Qwen3-Coder-30B-A3B-Instruct"),
            realizar::chat_template::detect_format_from_name("qwen3moe"),
        );
    }

    #[test]
    fn a_dense_qwen3_whose_template_agrees_is_left_alone() {
        assert_eq!(
            template_key(Some("qwen3"), Some("Qwen3-1.7B"), Some(QWEN3_THINK_DECLARED)).as_deref(),
            Some("qwen3"),
        );
    }

    /// No declared template is no evidence, and no evidence is not a licence to change
    /// the key. A GGUF without `tokenizer.chat_template` keeps the architecture.
    #[test]
    fn without_a_declared_template_the_architecture_stands() {
        assert_eq!(
            template_key(Some("llama"), Some("tinyllama-1.1b-chat"), None).as_deref(),
            Some("llama"),
        );
        assert_eq!(
            template_key(Some("llama"), Some("tinyllama-1.1b-chat"), Some("   ")).as_deref(),
            Some("llama"),
        );
    }

    /// If the NAME is contradicted too, keep the architecture rather than trade one
    /// wrong template for another.
    #[test]
    fn a_name_that_is_also_refuted_does_not_displace_the_architecture() {
        // declared mentions none of the markers either template renders
        assert_eq!(
            template_key(Some("llama"), Some("qwen2-ish"), Some("{{ content }}")).as_deref(),
            Some("llama"),
        );
    }

    /// The referee only fires on a CONTRADICTION. A rendered marker the declared
    /// template does mention is agreement, whatever the names are.
    #[test]
    fn agreement_is_not_a_contradiction() {
        assert!(!contradicts_declared("<|im_start|>user\nx", QWEN_CHATML_DECLARED));
        assert!(contradicts_declared("<s>[INST] x [/INST]", TINYLLAMA_DECLARED));
    }
}
