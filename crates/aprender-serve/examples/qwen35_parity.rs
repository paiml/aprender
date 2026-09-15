//! Qwen3.5 CPU parity against llama.cpp (#3091): greedy free-running and teacher-forced.
//!
//! Reads `PROMPT_TOKENS:` / `OUTPUT_TOKENS:` pairs that llama.cpp master produced greedily
//! from the evidence file, runs the same prompts through realizar's Qwen3.5 CPU forward, and
//! writes `parity.json` next to the evidence file.
//!
//! Gate: at every teacher-forced step llama.cpp's token must be realizar's argmax, or a
//! near-tie, meaning realizar's rank 2 with a logit gap of at most `QWEN35_PARITY_TAU`
//! (default 0.25). Exact greedy equality between two backends with different accumulation
//! precision is a coin flip at a near-tie (#2359), so the gaps are recorded, never hidden.
//!
//! Usage: `cargo run --release -p aprender-serve --example qwen35_parity -- <model.gguf> [tokens.txt]`

use realizar::gguf::forward_qwen35::{Qwen35Model, Qwen35State};
use realizar::gguf::MappedGGUFModel;
use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Serialize)]
struct Disagreement {
    step: usize,
    llama: u32,
    realizar: u32,
    gap: f32,
    rank: usize,
}

#[derive(Serialize)]
struct PromptParity {
    prompt_len: usize,
    llama_tokens: Vec<u32>,
    greedy_tokens: Vec<u32>,
    greedy_divergence: Option<usize>,
    teacher_forced_agree: usize,
    teacher_forced_steps: usize,
    disagreements: Vec<Disagreement>,
    min_agreeing_margin: f32,
}

#[derive(Serialize)]
struct Report {
    model: String,
    tau: f32,
    pass: bool,
    prompts: Vec<PromptParity>,
}

/// (argmax, best logit, second-best logit)
fn top2(logits: &[f32]) -> (u32, f32, f32) {
    let (mut best, mut best_v, mut second) = (0usize, f32::MIN, f32::MIN);
    for (j, &v) in logits.iter().enumerate() {
        if v > best_v {
            second = best_v;
            best = j;
            best_v = v;
        } else if v > second {
            second = v;
        }
    }
    (u32::try_from(best).unwrap_or(u32::MAX), best_v, second)
}

fn read_pairs(path: &Path) -> std::io::Result<Vec<(Vec<u32>, Vec<u32>)>> {
    let parse = |s: &str| {
        s.split_whitespace()
            .filter_map(|t| t.parse::<u32>().ok())
            .collect::<Vec<_>>()
    };
    let (mut prompts, mut outputs) = (Vec::new(), Vec::new());
    for line in BufReader::new(std::fs::File::open(path)?).lines() {
        let line = line?;
        if let Some(rest) = line.strip_prefix("PROMPT_TOKENS:") {
            prompts.push(parse(rest));
        } else if let Some(rest) = line.strip_prefix("OUTPUT_TOKENS:") {
            outputs.push(parse(rest));
        }
    }
    Ok(prompts.into_iter().zip(outputs).collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .ok_or("usage: qwen35_parity <model.gguf> [tokens.txt]")?;
    let tokens_path = args
        .next()
        .unwrap_or_else(|| "evidence/parity/qwen35/lambda/llama_tokens.txt".to_string());
    let tau: f32 = std::env::var("QWEN35_PARITY_TAU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.25);

    let mapped = MappedGGUFModel::from_path(&model_path)?;
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data())?;
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())?;
    let new_state = |max_seq_len: usize| qwen.new_state(max_seq_len);

    let mut prompts = Vec::new();
    for (prompt, want) in read_pairs(Path::new(&tokens_path))? {
        let steps = want.len().min(32);
        let prefill = |state: &mut Qwen35State| -> realizar::error::Result<Vec<f32>> {
            let mut logits = Vec::new();
            for (pos, &token) in prompt.iter().enumerate() {
                logits = qwen.forward_single_qwen35(token, state, pos)?;
            }
            Ok(logits)
        };

        // Free-running greedy: realizar feeds back its own argmax.
        let mut state = new_state(prompt.len() + steps + 1);
        let mut logits = prefill(&mut state)?;
        let mut greedy = Vec::with_capacity(steps);
        for i in 0..steps {
            let (next, _, _) = top2(&logits);
            greedy.push(next);
            if i + 1 < steps {
                logits = qwen.forward_single_qwen35(next, &mut state, prompt.len() + i)?;
            }
        }
        let greedy_divergence = greedy.iter().zip(&want).position(|(a, b)| a != b);

        // Teacher-forced: realizar is fed llama.cpp's tokens, so every step is judged.
        let mut state = new_state(prompt.len() + steps + 1);
        let mut logits = prefill(&mut state)?;
        let (mut agree, mut min_margin, mut disagreements) = (0usize, f32::MAX, Vec::new());
        for (i, &llama) in want.iter().take(steps).enumerate() {
            let (best, best_v, second) = top2(&logits);
            if best == llama {
                agree += 1;
                min_margin = min_margin.min(best_v - second);
            } else {
                let llama_v = logits.get(llama as usize).copied().unwrap_or(f32::MIN);
                let rank = logits.iter().filter(|&&v| v > llama_v).count() + 1;
                disagreements.push(Disagreement {
                    step: i,
                    llama,
                    realizar: best,
                    gap: best_v - llama_v,
                    rank,
                });
            }
            if i + 1 < steps {
                logits = qwen.forward_single_qwen35(llama, &mut state, prompt.len() + i)?;
            }
        }
        println!(
            "prompt {} ({} tokens): greedy divergence {:?}; teacher-forced {agree}/{steps}; disagreements {:?}",
            prompts.len(),
            prompt.len(),
            greedy_divergence,
            disagreements.iter().map(|d| (d.step, d.rank, d.gap)).collect::<Vec<_>>()
        );
        prompts.push(PromptParity {
            prompt_len: prompt.len(),
            llama_tokens: want[..steps].to_vec(),
            greedy_tokens: greedy,
            greedy_divergence,
            teacher_forced_agree: agree,
            teacher_forced_steps: steps,
            disagreements,
            min_agreeing_margin: min_margin,
        });
    }

    let pass = !prompts.is_empty()
        && prompts
            .iter()
            .all(|p| p.disagreements.iter().all(|d| d.rank <= 2 && d.gap <= tau));
    let report = Report {
        model: model_path,
        tau,
        pass,
        prompts,
    };
    let out = Path::new(&tokens_path).with_file_name("parity.json");
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!(
        "{} (tau {tau}); wrote {}",
        if pass { "PASS" } else { "FAIL" },
        out.display()
    );
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
