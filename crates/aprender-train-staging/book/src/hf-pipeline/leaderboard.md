# HuggingFace Leaderboard Integration

Entrenar can fetch and parse HuggingFace open evaluation
leaderboards, then rank your model against published results.
All types live in `entrenar::hf_pipeline::leaderboard`.
This module is feature-gated under `hub-publish`.

```toml
[dependencies]
entrenar = { version = "0.5", features = ["hub-publish"] }
```

## LeaderboardKind

Identifies which HuggingFace leaderboard to query. Each variant
maps to a dataset repository ID and a primary ranking metric.

```rust,ignore
use entrenar::hf_pipeline::leaderboard::LeaderboardKind;

let kind = LeaderboardKind::OpenASR;
assert_eq!(kind.dataset_repo_id(), "hf-audio/open_asr_leaderboard");

// Each leaderboard has a primary ranking metric
use entrenar::eval::evaluator::Metric;
assert_eq!(kind.primary_metric(), Metric::WER);
```

| Variant | Dataset Repo | Primary Metric | Tracked Metrics |
|---------|-------------|----------------|-----------------|
| `OpenASR` | `hf-audio/open_asr_leaderboard` | WER | WER, RTFx |
| `OpenLLMv2` | `open-llm-leaderboard/results` | MMLUAccuracy | MMLUAccuracy, Accuracy |
| `MTEB` | `mteb/leaderboard` | NDCG@10 | NDCG@10, Accuracy |
| `BigCodeBench` | `bigcode/bigcodebench-results` | pass@1 | pass@1, pass@10 |
| `Custom(id)` | User-specified | Accuracy | Accuracy |

## LeaderboardClient

HTTP client that fetches leaderboard data from the HuggingFace
datasets-server JSON API. Token is resolved automatically from
`HF_TOKEN` env var or `~/.cache/huggingface/token`.

```rust,ignore
use entrenar::hf_pipeline::leaderboard::{LeaderboardClient, LeaderboardKind};

// Auto-resolve token from environment
let client = LeaderboardClient::new()?;

// Or provide an explicit token
let client = LeaderboardClient::with_token("hf_abc123...")?;

// Fetch the first page (100 entries)
let hf = client.fetch(LeaderboardKind::OpenASR)?;
println!("Total models: {}", hf.total_count);
for entry in &hf.entries[..5] {
    println!("{}: WER={:.2}%",
        entry.model_id,
        entry.get_score("wer").unwrap_or(0.0) * 100.0,
    );
}

// Paginated fetch
let page2 = client.fetch_paginated(LeaderboardKind::OpenLLMv2, 100, 50)?;

// Find a specific model
let entry = client.find_model(LeaderboardKind::BigCodeBench, "deepseek-coder-v2")?;
```

The client hits the datasets-server rows endpoint
(`https://datasets-server.huggingface.co/rows`) which returns
JSON directly, avoiding Parquet parsing.

## Column-to-Metric Mapping

The `column_to_metric` function translates leaderboard-specific
column names into `Metric` enum variants. Each `LeaderboardKind`
has its own mapping table.

```rust,ignore
use entrenar::hf_pipeline::leaderboard::{column_to_metric, LeaderboardKind};
use entrenar::eval::evaluator::Metric;

// OpenASR columns
assert_eq!(
    column_to_metric(&LeaderboardKind::OpenASR, "average_wer"),
    Some(Metric::WER),
);

// BigCodeBench columns
assert_eq!(
    column_to_metric(&LeaderboardKind::BigCodeBench, "pass@1"),
    Some(Metric::PassAtK(1)),
);

// Custom leaderboards use generic best-effort mapping
assert_eq!(
    column_to_metric(&LeaderboardKind::Custom("my/board".into()), "bleu"),
    Some(Metric::BLEU),
);
```

The generic mapper (used for `Custom` leaderboards) recognizes
common column names: `accuracy`, `wer`, `bleu`, `rouge1`,
`rouge2`, `rougel`, `perplexity`, `mmlu`, `pass@1`,
and `ndcg@10`.

## Comparing Against a Leaderboard

Use `compare_with_leaderboard` to insert your model's
`EvalResult` into a fetched leaderboard. The result is a sorted
`Leaderboard` with your model ranked alongside published entries.

```rust,ignore
use entrenar::eval::evaluator::{EvalResult, Metric};
use entrenar::hf_pipeline::leaderboard::{
    LeaderboardClient, LeaderboardKind, compare_with_leaderboard,
};

let client = LeaderboardClient::new()?;
let hf = client.fetch(LeaderboardKind::OpenASR)?;

// Your model's evaluation result
let mut my_result = EvalResult::new("my-org/whisper-finetuned");
my_result.add_score(Metric::WER, 0.042);
my_result.add_score(Metric::RTFx, 156.0);

// Rank against the leaderboard (sorted by primary metric)
let ranked = compare_with_leaderboard(&my_result, &hf);

// Print as table or markdown
ranked.print();
println!("{}", ranked.to_markdown());

// Check your rank
if let Some(best) = ranked.best() {
    println!("Top model: {} (WER: {:.2}%)",
        best.model_name,
        best.get_score(Metric::WER).unwrap_or(0.0) * 100.0,
    );
}
```

The `to_leaderboard` function is also available if you want the native `Leaderboard` without inserting your own result.

## Data Types

### LeaderboardEntry

A single row from a HuggingFace leaderboard.

```rust,ignore
use entrenar::hf_pipeline::leaderboard::LeaderboardEntry;

let mut entry = LeaderboardEntry::new("openai/whisper-large-v3");
// entry.scores: HashMap<String, f64>  — raw numeric columns
// entry.metadata: HashMap<String, String> — string columns (license, etc.)
```

### HfLeaderboard

Container for fetched leaderboard data.

```rust,ignore
use entrenar::hf_pipeline::leaderboard::{HfLeaderboard, LeaderboardKind};

let hf = HfLeaderboard::new(LeaderboardKind::MTEB);
// hf.kind: LeaderboardKind
// hf.entries: Vec<LeaderboardEntry>
// hf.total_count: usize  — may exceed entries.len() (pagination)
let model = hf.find_model("BAAI/bge-large-en-v1.5");
```
