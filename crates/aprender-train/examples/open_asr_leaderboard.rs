//! OpenASR Leaderboard: Fine-Tune Whisper and Publish to HuggingFace
//!
//! This example demonstrates the complete Entrenar workflow for ASR:
//!
//! 1. Fine-tune a Whisper model on domain-specific audio
//! 2. Evaluate with WER and RTFx metrics
//! 3. Compare against the HuggingFace Open ASR Leaderboard
//! 4. Generate a model card and publish to HuggingFace Hub
//!
//! ## Usage
//!
//! ```bash
//! # Dry run (no HF_TOKEN needed — uses mock data)
//! cargo run --example open_asr_leaderboard
//!
//! # Live mode (fetches real leaderboard, publishes to HF Hub)
//! HF_TOKEN=hf_xxx cargo run --example open_asr_leaderboard --features hub-publish -- --live
//! ```
//!
//! ## Prerequisites
//!
//! - For live mode: `HF_TOKEN` environment variable with write access
//! - Feature flag: `--features hub-publish` enables network access

use entrenar::eval::evaluator::{EvalResult, Leaderboard, Metric};
use entrenar::eval::generative::{real_time_factor_inverse, word_error_rate};

fn main() {
    let live = std::env::args().any(|a| a == "--live");

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Entrenar — Open ASR Leaderboard Example                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // ── Step 1: Simulate fine-tuning results ──────────────────────────
    //
    // In a real pipeline you would:
    //   1. Load Whisper weights via `HfModelFetcher`
    //   2. Fine-tune on your domain dataset (medical transcription, etc.)
    //   3. Run inference on a held-out test set
    //
    // Here we use pre-computed transcription pairs to demonstrate
    // the evaluation and leaderboard comparison pipeline.

    println!("── Step 1: Evaluate fine-tuned model ──────────────────────");
    println!();

    let test_pairs = vec![
        // (reference, hypothesis, audio_duration_secs, processing_secs)
        (
            "the patient presents with acute respiratory distress",
            "the patient presents with acute respiratory distress",
            4.2,
            0.028,
        ),
        (
            "administer fifty milligrams of diphenhydramine intravenously",
            "administer fifty milligrams of diphenhydramine intravenously",
            5.1,
            0.033,
        ),
        (
            "blood pressure is one forty over ninety",
            "blood pressure is one forty over ninty",
            3.8,
            0.025,
        ),
        (
            "echocardiogram shows normal left ventricular function",
            "echocardiogram shows normal left ventricular function",
            4.5,
            0.030,
        ),
        (
            "the hemoglobin a one c level is six point eight percent",
            "the hemoglobin a one c level is six point eight percent",
            5.8,
            0.038,
        ),
        (
            "recommend starting metformin five hundred milligrams twice daily",
            "recommend starting metformin five hundred milligrams twice daily",
            5.5,
            0.036,
        ),
        (
            "chest x ray reveals bilateral infiltrates consistent with pneumonia",
            "chest x ray reveals bilateral infiltrates consistant with pneumonia",
            6.2,
            0.041,
        ),
        (
            "patient denies any history of cardiac arrhythmia",
            "patient denies any history of cardiac arrhythmia",
            4.0,
            0.026,
        ),
        (
            "schedule follow up in two weeks for repeat labs",
            "schedule follow up in two weeks for repeat labs",
            3.5,
            0.023,
        ),
        (
            "the electrocardiogram shows sinus rhythm with no st changes",
            "the electrocardiogram shows sinus rythm with no st changes",
            5.3,
            0.035,
        ),
    ];

    // Compute WER for each utterance
    let mut total_wer = 0.0;
    let mut total_audio_secs = 0.0;
    let mut total_processing_secs = 0.0;

    for (i, (reference, hypothesis, audio_dur, proc_dur)) in test_pairs.iter().enumerate() {
        let wer = word_error_rate(reference, hypothesis);
        let rtfx = real_time_factor_inverse(*proc_dur, *audio_dur);

        total_wer += wer;
        total_audio_secs += audio_dur;
        total_processing_secs += proc_dur;

        if wer > 0.0 {
            println!("  [{:>2}] WER={:.1}%  RTFx={:.0}x  ← error", i + 1, wer * 100.0, rtfx);
        } else {
            println!("  [{:>2}] WER=0.0%   RTFx={:.0}x  ✓", i + 1, rtfx);
        }
    }

    let avg_wer = total_wer / test_pairs.len() as f64;
    let overall_rtfx = real_time_factor_inverse(total_processing_secs, total_audio_secs);

    println!();
    println!("  Average WER:  {:.2}%", avg_wer * 100.0);
    println!("  Overall RTFx: {overall_rtfx:.1}x real-time");
    println!();

    // ── Step 2: Build EvalResult ──────────────────────────────────────

    println!("── Step 2: Build evaluation result ────────────────────────");
    println!();

    let mut my_result = EvalResult::new("paiml/whisper-small-medical-v1");
    my_result.add_score(Metric::WER, avg_wer);
    my_result.add_score(Metric::RTFx, overall_rtfx);

    println!("  Model:  {}", my_result.model_name);
    println!("  WER:    {:.2}%", my_result.get_score(Metric::WER).unwrap_or(0.0) * 100.0);
    println!("  RTFx:   {:.1}x", my_result.get_score(Metric::RTFx).unwrap_or(0.0));
    println!();

    // ── Step 3: Compare against leaderboard ───────────────────────────

    println!("── Step 3: Compare against Open ASR Leaderboard ───────────");
    println!();

    if live {
        #[cfg(feature = "hub-publish")]
        {
            live_leaderboard_comparison(&my_result);
        }
        #[cfg(not(feature = "hub-publish"))]
        {
            eprintln!("  Error: --live requires --features hub-publish");
            std::process::exit(1);
        }
    } else {
        mock_leaderboard_comparison(&my_result);
    }

    // ── Step 4: Generate model card ───────────────────────────────────

    println!("── Step 4: Generate model card ─────────────────────────────");
    println!();

    #[cfg(feature = "hub-publish")]
    {
        use entrenar::hf_pipeline::publish::ModelCard;

        let card = ModelCard::from_eval_result(&my_result);
        let markdown = card.to_markdown();
        println!("{}", &markdown[..markdown.len().min(600)]);
        if markdown.len() > 600 {
            println!("  ... ({} more bytes)", markdown.len() - 600);
        }
    }
    #[cfg(not(feature = "hub-publish"))]
    {
        println!("  (Model card generation requires --features hub-publish)");
        println!("  Preview of what would be generated:");
        println!();
        println!("  ---");
        println!("  license: apache-2.0");
        println!("  language:");
        println!("    - en");
        println!("  tags:");
        println!("    - whisper");
        println!("    - asr");
        println!("    - medical");
        println!("    - entrenar");
        println!("  base_model: openai/whisper-small");
        println!("  model-index:");
        println!("    - name: paiml/whisper-small-medical-v1");
        println!("      results:");
        println!("        - metrics:");
        println!("            - type: wer");
        println!("              value: {avg_wer:.4}");
        println!("            - type: rtfx");
        println!("              value: {overall_rtfx:.1}");
        println!("  ---");
    }

    println!();

    // ── Step 5: Publish (live mode only) ──────────────────────────────

    if live {
        println!("── Step 5: Publish to HuggingFace Hub ───────────────────────");
        println!();

        #[cfg(feature = "hub-publish")]
        {
            live_publish(&my_result);
        }
        #[cfg(not(feature = "hub-publish"))]
        {
            eprintln!("  Error: --live requires --features hub-publish");
        }
    } else {
        println!("── Step 5: Publish (skipped — use --live to enable) ───────");
        println!();
        println!("  To publish to HuggingFace Hub:");
        println!("  HF_TOKEN=hf_xxx cargo run --example open_asr_leaderboard \\");
        println!("      --features hub-publish -- --live");
    }

    println!();
    println!("Done.");
}

/// Compare against a mock leaderboard (no network required)
fn mock_leaderboard_comparison(my_result: &EvalResult) {
    // Simulated leaderboard entries (representative of real OpenASR values)
    let competitors = [
        ("openai/whisper-large-v3", 0.0256, 160.0),
        ("openai/whisper-large-v2", 0.0318, 145.0),
        ("nvidia/canary-1b", 0.0342, 120.0),
        ("openai/whisper-medium", 0.0451, 180.0),
        ("openai/whisper-small", 0.0512, 210.0),
        ("openai/whisper-base", 0.0823, 250.0),
        ("facebook/wav2vec2-large-960h", 0.0935, 95.0),
        ("openai/whisper-tiny", 0.1240, 310.0),
    ];

    let mut leaderboard = Leaderboard::new(Metric::WER);

    for (name, wer, rtfx) in &competitors {
        let mut entry = EvalResult::new(*name);
        entry.add_score(Metric::WER, *wer);
        entry.add_score(Metric::RTFx, *rtfx);
        leaderboard.add(entry);
    }

    leaderboard.add(my_result.clone());
    leaderboard.sort();

    // Print ranked results
    let my_wer = my_result.get_score(Metric::WER).unwrap_or(f64::MAX);

    println!("  Rank  Model                             WER      RTFx");
    println!("  ────  ────────────────────────────────  ───────  ──────");

    for (i, result) in leaderboard.results.iter().enumerate() {
        let wer = result.get_score(Metric::WER).unwrap_or(0.0);
        let rtfx = result.get_score(Metric::RTFx).unwrap_or(0.0);
        let marker =
            if result.model_name == "paiml/whisper-small-medical-v1" { " ◄ YOU" } else { "" };
        println!(
            "  {:>4}  {:<34}  {:>5.2}%  {:>5.0}x{}",
            i + 1,
            result.model_name,
            wer * 100.0,
            rtfx,
            marker
        );
    }

    // Find our rank
    let my_rank = leaderboard
        .results
        .iter()
        .position(|r| (r.get_score(Metric::WER).unwrap_or(0.0) - my_wer).abs() < 1e-10)
        .map_or(0, |i| i + 1);

    println!();
    println!(
        "  Your model ranks #{} out of {} (mock leaderboard)",
        my_rank,
        leaderboard.results.len()
    );
    println!();
}

/// Live leaderboard comparison via HuggingFace API
#[cfg(feature = "hub-publish")]
fn live_leaderboard_comparison(my_result: &EvalResult) {
    use entrenar::hf_pipeline::leaderboard::{
        compare_with_leaderboard, LeaderboardClient, LeaderboardKind,
    };

    println!("  Fetching Open ASR Leaderboard from HuggingFace...");

    match LeaderboardClient::new() {
        Ok(client) => match client.fetch(LeaderboardKind::OpenASR) {
            Ok(hf) => {
                println!("  Fetched {} entries (total: {})", hf.entries.len(), hf.total_count);
                println!();

                let ranked = compare_with_leaderboard(my_result, &hf);
                println!("{}", ranked.to_markdown());
            }
            Err(e) => {
                eprintln!("  Failed to fetch leaderboard: {e}");
                eprintln!("  Falling back to mock data...");
                println!();
                mock_leaderboard_comparison(my_result);
            }
        },
        Err(e) => {
            eprintln!("  Failed to create client: {e}");
            eprintln!("  Falling back to mock data...");
            println!();
            mock_leaderboard_comparison(my_result);
        }
    }
}

/// Live publish to HuggingFace Hub
#[cfg(feature = "hub-publish")]
fn live_publish(my_result: &EvalResult) {
    use entrenar::hf_pipeline::publish::{HfPublisher, ModelCard, PublishConfig, RepoType};

    let config = PublishConfig {
        repo_id: my_result.model_name.clone(),
        repo_type: RepoType::Model,
        private: false,
        token: None, // resolved from HF_TOKEN env var
        license: Some("apache-2.0".into()),
        tags: vec!["whisper".into(), "asr".into(), "medical".into(), "entrenar".into()],
    };

    let card = ModelCard::from_eval_result(my_result);

    match HfPublisher::new(config) {
        Ok(publisher) => {
            // In a real pipeline, you'd include the model weights:
            // let files = &[
            //     (Path::new("model.safetensors"), "model.safetensors"),
            //     (Path::new("config.json"), "config.json"),
            //     (Path::new("tokenizer.json"), "tokenizer.json"),
            // ];
            // publisher.publish(files, Some(&card))

            println!(
                "  Publisher created for: {}",
                publisher
                    .create_repo()
                    .unwrap_or_else(|_err| "(dry run — repo not created)".into())
            );
            println!("  Model card preview:");
            let md = card.to_markdown();
            for line in md.lines().take(10) {
                println!("    {line}");
            }
            println!("    ...");
        }
        Err(e) => {
            eprintln!("  Publish failed: {e}");
            eprintln!("  Ensure HF_TOKEN is set with write access.");
        }
    }
}
