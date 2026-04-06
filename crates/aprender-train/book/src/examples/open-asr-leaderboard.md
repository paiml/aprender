# OpenASR Leaderboard: Evaluate and Publish

This example demonstrates the full ASR pipeline: evaluate a
fine-tuned Whisper model with WER and RTFx metrics, rank it against
the HuggingFace Open ASR Leaderboard, and publish the results.

## Overview

- Compute Word Error Rate (WER) on domain-specific transcriptions
- Compute Real-Time Factor inverse (RTFx) for speed benchmarking
- Build an `EvalResult` with typed `Metric::WER` and `Metric::RTFx`
- Compare against the Open ASR Leaderboard (mock or live)
- Generate a HuggingFace model card with YAML front matter
- Publish the model and results to HuggingFace Hub

## Running the Example

```bash
# Dry run — no network, no HF_TOKEN needed
cargo run --example open_asr_leaderboard

# Live mode — fetches real leaderboard, publishes to HF Hub
HF_TOKEN=hf_xxx cargo run --example open_asr_leaderboard \
    --features hub-publish -- --live
```

## Pipeline Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Fine-tune   │    │   Evaluate   │    │    Compare vs     │
│  Whisper on  │───►│  WER + RTFx  │───►│  OpenASR Board   │
│  domain data │    │  per utt.    │    │  (rank model)    │
└──────────────┘    └──────────────┘    └────────┬─────────┘
                                                 │
                    ┌──────────────┐    ┌─────────▼─────────┐
                    │   Publish    │◄───│  Generate Model   │
                    │   to HF Hub │    │  Card (README.md) │
                    └──────────────┘    └───────────────────┘
```

## Step-by-Step Walkthrough

### Step 1: Evaluate WER and RTFx

The core metrics for ASR evaluation:

```rust,ignore
use entrenar::eval::generative::{word_error_rate, real_time_factor_inverse};

// WER: word-level Levenshtein distance / reference length
let wer = word_error_rate(
    "blood pressure is one forty over ninety",   // reference
    "blood pressure is one forty over ninty",     // hypothesis
);
// wer = 1/8 = 0.125 (one substitution: "ninety" → "ninty")

// RTFx: audio_duration / processing_time (higher = faster)
let rtfx = real_time_factor_inverse(0.025, 3.8);
// rtfx = 152.0 (152x faster than real-time)
```

WER is computed per utterance, then averaged across the test set.
RTFx is computed as total audio duration / total processing time.

### Step 2: Build an EvalResult

```rust,ignore
use entrenar::eval::evaluator::{EvalResult, Metric};

let mut result = EvalResult::new("paiml/whisper-small-medical-v1");
result.add_score(Metric::WER, 0.0305);   // 3.05% average WER
result.add_score(Metric::RTFx, 147.3);   // 147x real-time

// Polarity is built into the Metric enum
assert!(!Metric::WER.higher_is_better());   // lower WER = better
assert!(Metric::RTFx.higher_is_better());   // higher RTFx = faster
```

### Step 3: Compare Against the Leaderboard

In dry-run mode, the example builds a mock leaderboard with
representative entries. In live mode, it fetches the real
HuggingFace Open ASR Leaderboard:

```rust,ignore
use entrenar::hf_pipeline::leaderboard::{
    LeaderboardClient, LeaderboardKind, compare_with_leaderboard,
};

let client = LeaderboardClient::new()?;
let hf = client.fetch(LeaderboardKind::OpenASR)?;

let ranked = compare_with_leaderboard(&my_result, &hf);
ranked.print();
println!("{}", ranked.to_markdown());
```

### Step 4: Generate a Model Card

The `ModelCard` generates a HuggingFace-compatible README.md with
YAML front matter containing license, tags, and metric results:

```rust,ignore
use entrenar::hf_pipeline::publish::ModelCard;

let card = ModelCard::from_eval_result(&my_result);
let markdown = card.to_markdown();
// Outputs:
// ---
// license: apache-2.0
// tags:
//   - entrenar
// model-index:
//   - name: paiml/whisper-small-medical-v1
//     results:
//       - metrics:
//           - type: wer
//             value: 0.0305
//           - type: rtfx
//             value: 147.3
// ---
// # paiml/whisper-small-medical-v1
// ...
```

### Step 5: Publish to HuggingFace Hub

```rust,ignore
use entrenar::hf_pipeline::publish::{
    HfPublisher, PublishConfig, RepoType,
};

let config = PublishConfig {
    repo_id: "paiml/whisper-small-medical-v1".into(),
    repo_type: RepoType::Model,
    private: false,
    token: None, // resolved from HF_TOKEN env var
    license: Some("apache-2.0".into()),
    tags: vec!["whisper".into(), "asr".into(), "medical".into()],
};

let publisher = HfPublisher::new(config)?;
let result = publisher.publish(
    &[
        (Path::new("model.safetensors"), "model.safetensors"),
        (Path::new("tokenizer.json"), "tokenizer.json"),
    ],
    Some(&card),
)?;

println!("Published to: {}", result.repo_url);
```

## Example Output

```
╔═══════════════════════════════════════════════════════════╗
║  Entrenar — Open ASR Leaderboard Example                 ║
╚═══════════════════════════════════════════════════════════╝

── Step 1: Evaluate fine-tuned model ──────────────────────
  [ 1] WER=0.0%   RTFx=150x  ✓
  [ 2] WER=0.0%   RTFx=155x  ✓
  [ 3] WER=12.5%  RTFx=152x  ← error
  [ 4] WER=0.0%   RTFx=150x  ✓
  [ 5] WER=0.0%   RTFx=153x  ✓
  [ 6] WER=0.0%   RTFx=153x  ✓
  [ 7] WER=10.0%  RTFx=151x  ← error
  [ 8] WER=0.0%   RTFx=154x  ✓
  [ 9] WER=0.0%   RTFx=152x  ✓
  [10] WER=10.0%  RTFx=151x  ← error

  Average WER:  3.25%
  Overall RTFx: 147.3x real-time

── Step 3: Compare against Open ASR Leaderboard ───────────
  Rank  Model                             WER      RTFx
  ────  ────────────────────────────────  ───────  ──────
     1  openai/whisper-large-v3             2.56%   160x
     2  paiml/whisper-small-medical-v1      3.25%   147x ◄ YOU
     3  openai/whisper-large-v2             3.18%   145x
     4  nvidia/canary-1b                    3.42%   120x
     5  openai/whisper-medium               4.51%   180x
     ...
```

## Metric Reference

| Metric | Formula | Direction | Range |
|--------|---------|-----------|-------|
| WER | `(S+D+I) / N` | Lower is better | `[0, ∞)` |
| RTFx | `audio_duration / processing_time` | Higher is better | `(0, ∞)` |

Where S=substitutions, D=deletions, I=insertions, N=reference words.

## Related Documentation

- [Generative AI Metrics](../eval/generative-metrics.md)
- [HuggingFace Leaderboard Integration](../hf-pipeline/leaderboard.md)
- [HuggingFace Publishing](../hf-pipeline/publishing.md)
