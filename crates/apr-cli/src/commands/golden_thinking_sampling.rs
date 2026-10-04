// #4696 (GATE CHANGE): the golden gate's thinking-ON leg judged SAMPLED for a model
// whose row in contracts/thinking-budgets-v1.yaml declares `sampling`.
//
// Greedy never closes Qwen3.5-0.8B's think block at any budget from 2048 to 16384
// tokens, while Qwen's recommended sampler does in a measured fraction of seeds, in
// apr and in llama.cpp alike. One greedy run therefore judged a decoding mode the
// model does not close in. The row now names the sampler, the fixed seeds and how
// many must pass; the per-run judge (`judge_thinking_on_output`) is unchanged. Every
// model whose row has no `sampling` is judged greedy, exactly as before.

/// Qwen's thinking sampler and the closure rule, read from one `models` row.
#[cfg(feature = "inference")]
#[derive(Debug, Clone, PartialEq)]
struct ThinkingSampling {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seeds: Vec<u64>,
    min_pass: usize,
}

/// The refusal for a `sampling` row that is not a rule a gate can apply.
#[cfg(feature = "inference")]
fn sampling_refusal(pat: &str, what: &str) -> String {
    format!(
        "`{pat}` in contracts/thinking-budgets-v1.yaml declares `sampling` {what}; \
         refusing rather than judging under a rule the row does not state (#4696)"
    )
}

/// The seed list, refused when any entry is not an integer: dropping one silently
/// would shrink N and with it the rule the quorum signed.
#[cfg(feature = "inference")]
fn sampling_seeds(s: &serde_yaml::Value, pat: &str) -> std::result::Result<Vec<u64>, String> {
    let declared = s
        .get("seeds")
        .and_then(serde_yaml::Value::as_sequence)
        .ok_or_else(|| sampling_refusal(pat, "with no `seeds` list"))?;
    let seeds: Vec<u64> = declared
        .iter()
        .filter_map(serde_yaml::Value::as_u64)
        .collect();
    if seeds.is_empty() || seeds.len() != declared.len() {
        return Err(sampling_refusal(
            pat,
            "with `seeds` that is not a non-empty list of integers",
        ));
    }
    // A repeated seed is the same run counted twice: [0; 8] would be N = 1 judged as 8.
    let distinct: std::collections::BTreeSet<u64> = seeds.iter().copied().collect();
    if distinct.len() != seeds.len() {
        return Err(sampling_refusal(pat, "with a repeated seed in `seeds`"));
    }
    Ok(seeds)
}

/// Validate one `sampling` mapping. FAIL-CLOSED like the budget row: a missing or
/// out-of-range field refuses, never defaults.
#[cfg(feature = "inference")]
fn thinking_sampling_row(
    s: &serde_yaml::Value,
    pat: &str,
) -> std::result::Result<ThinkingSampling, String> {
    let num = |key: &str| s.get(key).and_then(serde_yaml::Value::as_f64);
    let temperature = num("temperature")
        .filter(|t| *t > 0.0)
        .ok_or_else(|| sampling_refusal(pat, "with no `temperature` > 0"))?;
    let top_p = num("top_p")
        .filter(|p| *p > 0.0 && *p <= 1.0)
        .ok_or_else(|| sampling_refusal(pat, "with no `top_p` in (0, 1]"))?;
    let top_k = s
        .get("top_k")
        .and_then(serde_yaml::Value::as_u64)
        .and_then(|k| usize::try_from(k).ok())
        .filter(|k| *k >= 1)
        .ok_or_else(|| sampling_refusal(pat, "with no `top_k` >= 1"))?;
    let seeds = sampling_seeds(s, pat)?;
    let min_pass = s
        .get("min_pass")
        .and_then(serde_yaml::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| (1..=seeds.len()).contains(n))
        .ok_or_else(|| sampling_refusal(pat, "with no `min_pass` between 1 and the seed count"))?;
    let basis = s.get("basis").and_then(|x| x.as_str()).unwrap_or("").trim();
    if basis.is_empty() {
        return Err(sampling_refusal(pat, "with no `basis`"));
    }
    #[allow(clippy::cast_possible_truncation)]
    let (temperature, top_p) = (temperature as f32, top_p as f32);
    Ok(ThinkingSampling {
        temperature,
        top_k,
        top_p,
        seeds,
        min_pass,
    })
}

/// The sampled rule for this model: `Ok(None)` when its row (or the default) is
/// judged greedy, an error when the row's `sampling` is not a usable rule.
#[cfg(feature = "inference")]
fn thinking_on_sampling_from(
    table: &str,
    model_file: &str,
) -> std::result::Result<Option<ThinkingSampling>, String> {
    let doc: serde_yaml::Value = serde_yaml::from_str(table)
        .map_err(|e| format!("the embedded thinking-budget table did not parse: {e} (#4696)"))?;
    let Some(models) = doc.get("models").and_then(|m| m.as_mapping()) else {
        return Ok(None);
    };
    for (k, v) in models {
        let Some(pat) = k.as_str() else { continue };
        if glob_match(pat, model_file) {
            return v
                .get("sampling")
                .map(|s| thinking_sampling_row(s, pat))
                .transpose();
        }
    }
    Ok(None)
}

/// The embedded contract's sampled rule for `model_file`.
#[cfg(feature = "inference")]
fn thinking_on_sampling_for(
    model_file: &str,
) -> std::result::Result<Option<ThinkingSampling>, String> {
    thinking_on_sampling_from(THINKING_BUDGETS, model_file)
}

/// The leg's verdict over every seed. `results` holds one `(seed, failure)` per seed
/// that RAN; a seed absent from it is a failed seed, never a skipped one.
#[cfg(feature = "inference")]
fn judge_sampled_on_seeds(
    results: &[(u64, Option<String>)],
    rule: &ThinkingSampling,
    budget: usize,
) -> Option<String> {
    let passed = results.iter().filter(|(_, r)| r.is_none()).count();
    let n = rule.seeds.len();
    if results.len() == n && passed >= rule.min_pass {
        return None;
    }
    let first = results
        .iter()
        .find_map(|(seed, r)| r.as_ref().map(|r| format!("seed {seed}: {r}")))
        .unwrap_or_else(|| format!("only {} of {n} seeds ran", results.len()));
    Some(format!(
        "golden_output_thinking_on: {passed}/{n} sampled runs passed (temperature {}, top_k {}, \
         top_p {}, budget {budget}), below the required {} (#4696); first failure — {first}",
        rule.temperature, rule.top_k, rule.top_p, rule.min_pass
    ))
}

/// The generation config for one seed: the row's sampler, never greedy. Pure, so the
/// wiring from the contract row to the sampler is tested without a model.
#[cfg(feature = "inference")]
fn sampled_gen_config(
    rule: &ThinkingSampling,
    budget: usize,
    seed: u64,
    stop_tokens: Vec<u32>,
) -> realizar::gguf::QuantizedGenerateConfig {
    realizar::gguf::QuantizedGenerateConfig {
        max_tokens: budget,
        temperature: rule.temperature,
        top_k: rule.top_k,
        top_p: rule.top_p,
        seed,
        stop_tokens,
        ..Default::default()
    }
}

/// One sampled thinking-ON leg, on the CPU, for a GGUF model.
#[cfg(feature = "inference")]
struct SampledOnLeg<'a> {
    mapped: &'a realizar::gguf::MappedGGUFModel,
    gguf: &'a realizar::gguf::GGUFModel,
    prompt: &'a str,
    patterns: &'a [&'a str],
    budget: usize,
    rule: &'a ThinkingSampling,
}

#[cfg(feature = "inference")]
impl SampledOnLeg<'_> {
    /// One seed's verdict: `None` passed. A generation error is that seed's failure.
    fn seed_verdict(&self, seed: u64) -> Option<String> {
        let gen_config =
            sampled_gen_config(self.rule, self.budget, seed, golden_stop_tokens(self.gguf));
        match golden_gguf_cpu_generate(self.mapped, self.gguf, self.prompt, &gen_config) {
            Err(e) => Some(format!("generation error: {e}")),
            Ok((_, text)) => {
                let generated = text.strip_prefix(self.prompt).unwrap_or(&text);
                let judged = on_leg_judged_text(self.prompt, generated);
                judge_thinking_on_output(&judged, self.patterns, self.budget)
            }
        }
    }

    fn judge(&self) -> Option<String> {
        let results: Vec<(u64, Option<String>)> = self
            .rule
            .seeds
            .iter()
            .map(|&seed| (seed, self.seed_verdict(seed)))
            .collect();
        judge_sampled_on_seeds(&results, self.rule, self.budget)
    }
}

/// `None` when this model's ON leg is greedy; `Some(verdict)` when its row is sampled,
/// where `verdict` is `None` for a pass and the failure (or refusal) otherwise.
#[cfg(feature = "inference")]
fn sampled_on_leg_verdict(
    model_file: &str,
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
    gguf: Option<&realizar::gguf::GGUFModel>,
    prompt: &str,
    patterns: &[&str],
    budget: usize,
) -> Option<Option<String>> {
    let rule = match thinking_on_sampling_for(model_file) {
        Ok(None) => return None,
        Ok(Some(rule)) => rule,
        Err(reason) => return Some(Some(format!("golden_output_thinking_on: {reason}"))),
    };
    let (Some(mapped), Some(gguf)) = (mapped, gguf) else {
        return Some(Some(
            "golden_output_thinking_on: this row is judged sampled (#4696), which runs on a \
             GGUF model only; this format cannot be judged by it"
                .to_string(),
        ));
    };
    let leg = SampledOnLeg {
        mapped,
        gguf,
        prompt,
        patterns,
        budget,
        rule: &rule,
    };
    Some(leg.judge())
}

#[cfg(all(test, feature = "inference"))]
mod golden_thinking_sampling_tests {
    use super::*;

    fn rule(min_pass: usize) -> ThinkingSampling {
        ThinkingSampling {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            seeds: (0..8).collect(),
            min_pass,
        }
    }

    fn results(passes: usize) -> Vec<(u64, Option<String>)> {
        (0..8u64)
            .map(|s| {
                let fail = (s as usize) >= passes;
                (s, fail.then(|| "think block unclosed".to_string()))
            })
            .collect()
    }

    /// The contract's own row is what the quorum signed: Q4_K_M, 8 seeds, 4 to pass.
    #[test]
    fn the_q4km_row_is_the_signed_rule() {
        let got = thinking_on_sampling_for("Qwen3.5-0.8B-Q4_K_M.gguf")
            .expect("the row parses")
            .expect("the Q4_K_M row is sampled");
        assert_eq!(got, rule(4));
    }

    /// Greedy stays the rule everywhere else, including the unmeasured IQ4_XS sibling.
    #[test]
    fn every_other_model_stays_greedy() {
        for m in [
            "Qwen3.5-0.8B-IQ4_XS.gguf",
            "some-other-model-q4km.gguf",
            "qwen3-8b-q4_k_m.gguf",
        ] {
            assert_eq!(thinking_on_sampling_for(m), Ok(None), "{m}");
        }
    }

    /// The threshold is inclusive at K and fails one below it; each half of the
    /// boundary is a mutation the other would not catch.
    #[test]
    fn the_leg_passes_at_k_and_fails_one_below() {
        assert_eq!(judge_sampled_on_seeds(&results(4), &rule(4), 2048), None);
        assert_eq!(judge_sampled_on_seeds(&results(8), &rule(4), 2048), None);
        let below = judge_sampled_on_seeds(&results(3), &rule(4), 2048).expect("3/8 < 4 fails");
        assert!(below.contains("3/8 sampled runs passed"), "{below}");
        assert!(below.contains("below the required 4"), "{below}");
        assert!(below.contains("seed 3: think block unclosed"), "{below}");
        assert!(judge_sampled_on_seeds(&results(0), &rule(1), 2048).is_some());
    }

    /// Lane-3 guard: a seed that never ran is a failed seed. Enough passes among the
    /// seeds that did run must not carry the leg.
    #[test]
    fn a_missing_seed_is_a_failure_not_a_skip() {
        let mut r = results(8);
        r.truncate(5);
        let got = judge_sampled_on_seeds(&r, &rule(4), 2048).expect("5 of 8 seeds ran");
        assert!(got.contains("only 5 of 8 seeds ran"), "{got}");
    }

    /// The per-run judge is unchanged, so a closed block with the wrong answer is a
    /// failed seed, and an unclosed one still fails exactly as under greedy.
    #[test]
    fn each_run_keeps_the_greedy_judge() {
        let unclosed = judge_thinking_on_output("<think>let me see", &["4"], 2048);
        assert!(unclosed.is_some(), "an unclosed run must fail under the sampled leg too");
        let wrong = judge_thinking_on_output("<think>2 plus 2 is</think>five", &["4"], 2048);
        assert!(wrong.is_some());
    }

    /// FAIL-CLOSED: every malformed field refuses, by name, instead of defaulting.
    #[test]
    fn a_malformed_sampling_row_refuses() {
        let ok = "temperature: 0.6\ntop_k: 20\ntop_p: 0.95\nseeds: [0, 1]\nmin_pass: 1\nbasis: m\n";
        let cases = [
            ("temperature: 0.6", "temperature: 0", "temperature"),
            ("top_k: 20", "top_k: 0", "top_k"),
            ("top_p: 0.95", "top_p: 1.5", "top_p"),
            ("seeds: [0, 1]", "seeds: [0, x]", "seeds"),
            ("seeds: [0, 1]", "seeds: []", "seeds"),
            ("seeds: [0, 1]", "seeds: [1, 1]", "repeated seed"),
            ("min_pass: 1", "min_pass: 3", "min_pass"),
            ("min_pass: 1", "min_pass: 0", "min_pass"),
            ("basis: m", "basis: ''", "basis"),
        ];
        let parse = |y: &str| -> serde_yaml::Value { serde_yaml::from_str(y).expect("yaml") };
        assert!(thinking_sampling_row(&parse(ok), "p").is_ok());
        for (from, to, field) in cases {
            let err = thinking_sampling_row(&parse(&ok.replace(from, to)), "p")
                .expect_err(&format!("{to} must refuse"));
            assert!(err.contains(field) && err.contains("#4696"), "{to}: {err}");
        }
    }

    /// Sign-off lane 1: the row's sampler reaches the generation config, per seed, and
    /// the config is not greedy (temperature 0 or top_k 1 would silently be greedy).
    #[test]
    fn each_seed_runs_the_rows_sampler_not_greedy() {
        let r = rule(4);
        let a = sampled_gen_config(&r, 2048, 3, vec![7]);
        let b = sampled_gen_config(&r, 2048, 4, vec![7]);
        assert_eq!((a.max_tokens, a.temperature, a.top_k, a.top_p), (2048, 0.6, 20, 0.95));
        assert_eq!((a.seed, b.seed), (3, 4), "each run carries its own seed");
        assert_eq!(a.stop_tokens, vec![7]);
        assert!(a.temperature > 0.0 && a.top_k > 1, "a sampled run must not be greedy");
    }

    /// A row's `sampling` is found by the same glob as its budget, and a table
    /// without it is greedy.
    #[test]
    fn the_sampled_rule_follows_the_row_glob() {
        let table = "models:\n  \"m-*.gguf\":\n    budget: 64\n    basis: b\n    sampling:\n      \
                     temperature: 0.6\n      top_k: 20\n      top_p: 0.95\n      seeds: [7]\n      \
                     min_pass: 1\n      basis: s\n";
        let got = thinking_on_sampling_from(table, "m-a.gguf").expect("parses");
        assert_eq!(got.map(|r| r.seeds), Some(vec![7]));
        assert_eq!(thinking_on_sampling_from(table, "n-a.gguf"), Ok(None));
    }
}
