//! #4270: `apr bench` on a Qwen3.5 hybrid times the one engine.

use super::*;

const MODEL: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Every measured iteration (and every warmup) is a `Session::generate` entry for
/// exactly the bench prompt. Before #4270 the verb refused the model at load.
#[test]
fn bench_on_qwen35_times_session_generate() {
    if !Path::new(MODEL).exists() {
        eprintln!("SKIP: {MODEL} is absent");
        return;
    }
    let prompt = "bench_qwen35 witness prompt #4270: name three rivers.";
    let config = BenchConfig {
        warmup: 1,
        iterations: 2,
        max_tokens: 4,
        prompt: prompt.to_string(),
        quiet: true,
    };
    let result = run_realizar_benchmark(Path::new(MODEL), &config).expect("bench runs");
    let mapped = realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map");
    let tokens = mapped.model.encode(prompt).expect("tokenizer");
    let generates = realizar::session::entries_for(&tokens)
        .into_iter()
        .filter(|e| e.arch == "qwen35" && e.kind == realizar::session::EntryKind::Generate)
        .count();
    assert_eq!(
        generates, 3,
        "1 warmup + 2 iterations through Session::generate"
    );
    assert_eq!(result.iteration_times.len(), 2);
    assert!(result.total_tokens > 0, "{result:?}");
    assert!(result.time_to_first_token <= result.median_time.max(result.mean_time));
}
