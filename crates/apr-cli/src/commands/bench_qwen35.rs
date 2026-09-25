/// #4270: bench a Qwen3.5 hybrid GGUF by timing the one engine (`Qwen35Session`,
/// the same session `apr run`/`chat`/`serve` drive) — never a bench-local loop.
/// The session picks CUDA unless `use_cuda` is false, exactly as `apr run` does,
/// and its route notices (including any CPU fallback) are printed.
#[cfg(feature = "inference")]
fn run_qwen35_session_benchmark(
    mapped: &realizar::gguf::MappedGGUFModel,
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    config: &BenchConfig,
    use_cuda: bool,
    start: Instant,
    tracer: &TracerImpl,
) -> Result<BenchResult> {
    use realizar::gguf::qwen35_session::Qwen35Session;

    let mut session = Qwen35Session::load(mapped, !use_cuda)
        .map_err(|e| CliError::ValidationFailed(format!("Qwen3.5 hybrid: {e}")))?;
    for notice in session.notices() {
        bench_log(config, notice);
    }
    let device = if session.on_gpu() {
        " (qwen35 session, GPU)"
    } else {
        " (qwen35 session, CPU)"
    };
    bench_log_ready(config, start.elapsed(), device);

    bench_log(config, &"Running warmup...".yellow().to_string());
    for i in 0..config.warmup {
        session.forget_prefix();
        let _ = session.generate(prompt_tokens, gen_config, &mut |_| true);
        bench_log_iter(config, i, Duration::ZERO, None);
    }
    bench_log_done(config);

    bench_log(config, &"Running benchmark...".yellow().to_string());
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;
    let budget_us = config.max_tokens as u64 * 100_000;
    for i in 0..config.iterations {
        let (turn, iter_time, ttft) =
            session_timed_turn(&mut session, prompt_tokens, gen_config, tracer, budget_us)?;
        let tokens_generated = turn.tokens.len().saturating_sub(prompt_tokens.len());
        iteration_times.push(iter_time);
        total_tokens += tokens_generated;
        if i == 0 {
            first_token_time = ttft;
        }
        bench_log_iter(config, i, iter_time, Some(tokens_generated));
    }
    bench_log_done(config);
    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}

/// One timed turn: `(turn, total, time to first token)`. The prompt must be
/// prefilled whole — a bench that reused a cached prefix would report decode as
/// if it were prefill + decode.
#[cfg(feature = "inference")]
fn session_timed_turn(
    session: &mut realizar::gguf::qwen35_session::Qwen35Session,
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    tracer: &TracerImpl,
    budget_us: u64,
) -> Result<(realizar::session::Turn, Duration, Duration)> {
    // #4445: every timed turn repeats the same prompt, which the session would
    // otherwise resume from its checkpoint; forget it so the turn prefills whole.
    session.forget_prefix();
    let t0 = Instant::now();
    let mut first: Option<Duration> = None;
    let traced = tracer.trace("bench_qwen35_session_iter", budget_us, || {
        session.generate(prompt_tokens, gen_config, &mut |_| {
            first.get_or_insert_with(|| t0.elapsed());
            true
        })
    });
    let turn = traced
        .result
        .map_err(|e| CliError::ValidationFailed(format!("qwen35 session generate: {e}")))?;
    if turn.reused != 0 {
        return Err(CliError::ValidationFailed(format!(
            "qwen35 bench: {} prompt tokens were served from a cached prefix, not prefilled",
            turn.reused
        )));
    }
    // TTFT and the total are read off the same clock, so ttft <= total holds.
    let total = t0.elapsed();
    Ok((turn, total, first.unwrap_or(total)))
}
