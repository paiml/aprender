/// Detect Ollama model name from GGUF filename (BUG-QA-001 fix)
/// Matches model size to avoid unfair comparison (e.g., 0.5B APR vs 1.5B Ollama)
/// Detect the matching Ollama model tag for fair like-for-like comparison.
///
/// For quantized GGUF: uses the default Ollama tag (Q4_K_M quantized).
/// For F32/F16 (SafeTensors, APR): uses the `-instruct-fp16` Ollama tag
/// so we compare unquantized vs unquantized.
///
/// Detects model size from filename, or falls back to file size heuristic
/// for hash-named pacha-cached files.
fn detect_ollama_model_from_path(path: &Path) -> String {
    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let filename_lower = filename.to_lowercase();

    let size = detect_size_from_filename(&filename_lower)
        .unwrap_or_else(|| estimate_size_from_file(path));

    // Default Ollama tag uses Q4_K_M — fair comparison for quantized GGUF
    format!("qwen2.5-coder:{size}")
}

/// Measure Ollama throughput for comparison (GGUF only)
/// BUG-QA-002 FIX: Use Ollama's eval_duration instead of wall clock time
/// (wall clock includes HTTP overhead, making Ollama look 10x slower)
///
/// `raw` sends the prompt untemplated (#5254): an imported GGUF can come back
/// with no chat template (`TEMPLATE {{ .Prompt }}`), and a throughput figure
/// must not depend on whether ollama found one.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json! macro internally uses unwrap()
fn measure_ollama_throughput(model: &str, raw: bool, config: &QaConfig) -> Result<OllamaRates> {
    // Use curl to send a request to Ollama
    let prompt = "Write a hello world program in Python:";

    // Match parity gate: use 128 tokens minimum to amortize prefill overhead
    let parity_max_tokens = config.max_tokens.max(128);
    let request_body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "raw": raw,
        "options": {
            "num_predict": parity_max_tokens,
            "temperature": 0.0
        }
    });

    // [eval_count, eval_duration, prompt_eval_count, prompt_eval_duration]
    let mut sums = [0u64; 4];

    for _ in 0..config.iterations.min(3) {
        let output = std::process::Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "http://localhost:11434/api/generate",
                "-H",
                "Content-Type: application/json",
                "-d",
                &request_body.to_string(),
            ])
            .output();

        if let Ok(output) = output {
            if let Ok(response) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                add_ollama_counters(&response, &mut sums);
            }
        }
    }

    Ok(OllamaRates {
        decode_tps: rate(sums[0], sums[1]).unwrap_or(0.0),
        prefill_tps: rate(sums[2], sums[3]),
        // Only for an import (#5254); the size-matched tag path asks nothing new.
        resident_mb: if raw { ollama_resident_mb(model) } else { None },
    })
}

/// What ollama reported for the parity prompt, summed over the timed requests.
/// Every figure is ollama's own (`eval_*`, `prompt_eval_*`), never a wall clock.
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct OllamaRates {
    /// Decode tok/s: `eval_count / eval_duration`
    decode_tps: f64,
    /// Prefill tok/s: `prompt_eval_count / prompt_eval_duration`, when reported
    prefill_tps: Option<f64>,
    /// What `/api/ps` says the loaded model occupies (`size`), MiB, read while
    /// it is still loaded. ollama's own figure, not a process RSS.
    resident_mb: Option<f64>,
}

/// #5254: an ollama model imported from THIS GGUF for one qa run, removed on
/// drop. The name carries the pid so two runs on one host never `ollama rm`
/// each other's import.
#[cfg(feature = "inference")]
struct OllamaImport {
    name: String,
}

#[cfg(feature = "inference")]
impl OllamaImport {
    fn create(path: &Path) -> std::result::Result<Self, String> {
        sweep_stale_imports();
        let name = format!("apr-qa-{}", std::process::id());
        let abs = std::fs::canonicalize(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let dir = std::env::temp_dir().join(&name);
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let modelfile = dir.join("Modelfile");
        let out = std::fs::write(&modelfile, format!("FROM {}\n", abs.display()))
            .map_err(|e| format!("cannot write {}: {e}", modelfile.display()))
            .map(|()| {
                std::process::Command::new("ollama")
                    .args(["create", &name, "-f"])
                    .arg(&modelfile)
                    .output()
            });
        let _ = std::fs::remove_dir_all(&dir);
        match out? {
            Ok(o) if o.status.success() => Ok(Self { name }),
            Ok(o) => Err(format!(
                "ollama create failed ({}): {}",
                o.status,
                String::from_utf8_lossy(&o.stderr).lines().last().unwrap_or("")
            )),
            Err(e) => Err(format!("cannot run ollama: {e}")),
        }
    }
}

#[cfg(feature = "inference")]
impl Drop for OllamaImport {
    fn drop(&mut self) {
        unload_and_remove(&self.name);
    }
}

/// Unload first (`keep_alive: 0`): `ollama rm` alone leaves the runner resident
/// for its keep-alive, holding memory while apr's side is timed on a
/// unified-memory device.
#[cfg(feature = "inference")]
fn unload_and_remove(name: &str) {
    let body = format!(r#"{{"model":"{name}","keep_alive":0}}"#);
    let _ = std::process::Command::new("curl")
        .args(["-s", "-X", "POST", "http://localhost:11434/api/generate", "-d", &body])
        .output();
    let _ = std::process::Command::new("ollama").args(["rm", name]).output();
}

/// A run killed before `Drop` leaves its `apr-qa-<pid>` import (a full copy of
/// the GGUF) in ollama's store. Remove every one whose process is gone.
#[cfg(feature = "inference")]
fn sweep_stale_imports() {
    let Ok(out) = std::process::Command::new("ollama").arg("list").output() else {
        return;
    };
    for name in stale_import_names(&String::from_utf8_lossy(&out.stdout), |pid| {
        Path::new(&format!("/proc/{pid}")).exists()
    }) {
        unload_and_remove(&name);
    }
}

/// `apr-qa-<pid>` names in `ollama list` output whose pid is not alive.
#[cfg(feature = "inference")]
fn stale_import_names(list: &str, alive: impl Fn(u32) -> bool) -> Vec<String> {
    list.lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter_map(|tag| {
            let name = tag.strip_suffix(":latest").unwrap_or(tag);
            let pid = name.strip_prefix("apr-qa-")?.parse::<u32>().ok()?;
            (!alive(pid)).then(|| name.to_string())
        })
        .collect()
}

/// Sum ollama's own counters out of one `/api/generate` response.
#[cfg(feature = "inference")]
fn add_ollama_counters(response: &serde_json::Value, sums: &mut [u64; 4]) {
    let field = |k: &str| response.get(k).and_then(serde_json::Value::as_u64);
    if let (Some(n), Some(ns)) = (field("eval_count"), field("eval_duration")) {
        sums[0] += n;
        sums[1] += ns;
    }
    if let (Some(n), Some(ns)) = (field("prompt_eval_count"), field("prompt_eval_duration")) {
        sums[2] += n;
        sums[3] += ns;
    }
}

/// Tokens over nanoseconds as tok/s; `None` when either side is zero.
#[cfg(feature = "inference")]
fn rate(tokens: u64, ns: u64) -> Option<f64> {
    (tokens > 0 && ns > 0).then(|| tokens as f64 / (ns as f64 / 1_000_000_000.0))
}

/// `size` of `model` in an `/api/ps` body, in MiB. A tag without `:` is
/// listed by ollama as `<name>:latest`.
#[cfg(feature = "inference")]
fn resident_mb_in_ps(ps: &serde_json::Value, model: &str) -> Option<f64> {
    let latest = format!("{model}:latest");
    ps.get("models")?
        .as_array()?
        .iter()
        .find(|m| {
            let name = m.get("name").and_then(serde_json::Value::as_str);
            name == Some(model) || name == Some(latest.as_str())
        })?
        .get("size")?
        .as_u64()
        .map(|b| b as f64 / (1024.0 * 1024.0))
}

/// Ask the running ollama what the model occupies right after the timed runs.
#[cfg(feature = "inference")]
fn ollama_resident_mb(model: &str) -> Option<f64> {
    let out = std::process::Command::new("curl")
        .args(["-s", "http://localhost:11434/api/ps"])
        .output()
        .ok()?;
    let ps = serde_json::from_slice::<serde_json::Value>(&out.stdout).ok()?;
    resident_mb_in_ps(&ps, model)
}

/// Print a gate result to the terminal
fn print_gate_result(result: &GateResult) {
    let badge = if result.skipped {
        output::badge_skip("SKIP")
    } else if result.passed {
        output::badge_pass("PASS")
    } else {
        output::badge_fail("FAIL")
    };

    let name = gate_display_name(&result.name);

    println!(
        "  {} {} {}",
        badge,
        name.white().bold(),
        result.message.dimmed()
    );

    if !result.skipped {
        println!(
            "       {}",
            output::duration_fmt(result.duration_ms).dimmed()
        );
    }
    println!();
}
