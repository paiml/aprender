# Review brief — PR #4428 (replaces #4311), REPLAY DELTA only

Ticket intent: #4311 was an emergency fold of 9 PRs into one batch. Operator ruling: no main-merge into PRs, so #4311 is
rebuilt as a fresh branch from current main with the fold merges cherry-picked (-m 1). The folded PRs carry their own
reviews. Review ONLY what the replay itself authored:

1. PORT: main deleted crates/apr-cli/src/commands/gguf_generate_result.rs (0.69.3 merge-back moved print_roofline_profile
   into run_trace_print.rs with the OLD throughput-only classification). #4229/#4211's change — backend-aware
   classify_roofline(tok_per_sec, used_gpu) + "(wall clock, load included)" on the Throughput line — is ported there.
   Then split into helpers because the complexity ratchet flagged cognitive 29 > 25. Must be behaviour-identical to
   #4229's original function (reproduced below as ORIGINAL). Tests: run_tests_benchmark_profile_4211.rs (included in run_07.rs).
2. CONFLICT: scripts/check_clippy_feature_matrix.sh — main had the pre-#4147 version; resolved by taking #4147's side of
   every hunk (adds the no-default axis + its self-test plant).
3. ROADMAP: docs/roadmaps/roadmap.yaml = origin/main's file plus exactly the 7 entries whose fragments this branch adds
   (GH-4149, GH-4175, PMAT-4041, PMAT-4152, PMAT-4193, PMAT-4259, PMAT-4273), spliced verbatim from the aggregator's
   output. check_roadmap_ids_unique / sorted / fragment_required / diff_additive all pass locally.
4. DROPPED picks (already on main): #4046, #4224 (superseded by 5fa13fe8c v0.69.3 merge-back), 68c842319 (empty),
   d5c5b1d35's release_section book/contract + tests_engine_identity/dense_session hunks (main's version kept: main already
   has BorrowedCudaForward in ADAPTERS).

Return JSON: {"verdict":"PASS|FAIL","summary":"...","findings":[{"file","line","claim","grounding":"cited|measured|asserted"}]}.
Default to FAIL if the port is not behaviour-identical, or a resolution drops content.

## ORIGINAL (#4229, from gguf_generate_result.rs @ 08001d2c2)
```rust
/// Roofline classification of one run: `(compute %, memory %, bottleneck, recommendation)`.
///
/// Based on Ivanov et al. (2021): M=1 decode is memory-bandwidth bound.
///
/// #4211: the backend that RAN is an input. The tok/s thresholds alone once
/// told a CUDA run (`GPU used: yes`) that it was CPU-bound and to "Enable GPU
/// with --gpu". On a GPU run the throughput is wall clock including load and
/// one-off CUDA setup, so a low number is not evidence of a CPU bottleneck;
/// the GPU arm never names the CPU and never recommends `--gpu`. A CPU run
/// never claims tensor cores. `None` (backend not reported) keeps the
/// throughput-only tiers, but its top tier names no backend: only a run that
/// reported `GPU used: yes` is told its tensor cores are engaged.
fn classify_roofline(
    tok_per_sec: f64,
    used_gpu: Option<bool>,
) -> (u8, u8, &'static str, &'static str) {
    if used_gpu == Some(true) {
        return if tok_per_sec > 50.0 {
            (
                65,
                35,
                "Compute (GPU tensor cores engaged)",
                "Efficient — GPU-accelerated path active",
            )
        } else {
            (
                20,
                80,
                "Memory bandwidth (GPU VRAM); wall-clock tok/s includes load and one-off CUDA setup",
                "Measure decode-only throughput with `apr bench` before tuning",
            )
        };
    }
    if tok_per_sec > 50.0 {
        return if used_gpu == Some(false) {
            (
                40,
                60,
                "Mixed (CPU SIMD path, memory bandwidth limited)",
                "Efficient for CPU — enable GPU with --gpu for more",
            )
        } else {
            (
                65,
                35,
                "Compute (high throughput; backend not reported)",
                "Efficient at this throughput",
            )
        };
    }
    if tok_per_sec > 20.0 {
        (
            40,
            60,
            "Mixed (memory bandwidth limited)",
            "Try quantized model (Q4K) for less data movement",
        )
    } else if tok_per_sec > 5.0 {
        (
            20,
            80,
            "Memory bandwidth (DRAM → cache)",
            "Enable GPU with --gpu, or use smaller quantization",
        )
    } else {
        (
            10,
            90,
            "Memory bandwidth (CPU, no SIMD saturation)",
            "Model too large for CPU — use GPU or smaller model",
        )
    }
}

/// Print roofline profiling analysis (PMAT-480).
///
/// Estimates compute vs memory boundedness from throughput. For real
/// per-brick µs timing, use `apr profile <model> --granular` which
/// integrates with trueno's BrickProfiler.
fn print_roofline_profile(result: &RunResult, max_tokens: usize) {
    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
    let total_ms = result.duration_secs * 1000.0;
    let tok_per_sec = if result.duration_secs > 0.0 {
        tokens_generated as f64 / result.duration_secs
    } else {
        0.0
    };

    let (compute_pct, memory_pct, bottleneck, recommendation) =
        classify_roofline(tok_per_sec, result.used_gpu);

    eprintln!();
    eprintln!("{}", "=== Roofline Profile (PMAT-480) ===".cyan().bold());
```

## DIFF vs origin/main: run_trace_print.rs + check_clippy_feature_matrix.sh + run_07.rs
```diff
diff --git a/crates/apr-cli/src/commands/run_07.rs b/crates/apr-cli/src/commands/run_07.rs
index 658d6c501..32e727160 100644
--- a/crates/apr-cli/src/commands/run_07.rs
+++ b/crates/apr-cli/src/commands/run_07.rs
@@ -13,4 +13,5 @@ include!("run_tests_stream_output.rs");
 include!("run_tests_accel_reconcile.rs");
 include!("run_tests_usage_3718.rs");
 include!("run_tests_top_k_default_3754.rs");
+include!("run_tests_benchmark_profile_4211.rs");
 }
diff --git a/crates/apr-cli/src/commands/run_trace_print.rs b/crates/apr-cli/src/commands/run_trace_print.rs
index 43f2b70d2..5c83bc042 100644
--- a/crates/apr-cli/src/commands/run_trace_print.rs
+++ b/crates/apr-cli/src/commands/run_trace_print.rs
@@ -78,7 +78,11 @@ fn render_layer_trace(result: &RunResult, max_tokens: usize) -> String {
 
     let mut out = String::new();
     let _ = writeln!(out);
-    let _ = writeln!(out, "{}", "=== Layer Trace (APR-TRACE-001) ===".cyan().bold());
+    let _ = writeln!(
+        out,
+        "{}",
+        "=== Layer Trace (APR-TRACE-001) ===".cyan().bold()
+    );
     let _ = writeln!(out);
     let _ = writeln!(
         out,
@@ -182,31 +186,70 @@ fn print_payload_trace(result: &RunResult, max_tokens: usize) {
     eprintln!();
 }
 
-/// Print roofline profiling analysis (PMAT-480).
+/// Roofline classification of one run: `(compute %, memory %, bottleneck, recommendation)`.
 ///
-/// Estimates compute vs memory boundedness from throughput. For real
-/// per-brick µs timing, use `apr profile <model> --granular` which
-/// integrates with trueno's BrickProfiler.
-fn print_roofline_profile(result: &RunResult, max_tokens: usize) {
-    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
-    let total_ms = result.duration_secs * 1000.0;
-    let tok_per_sec = if result.duration_secs > 0.0 {
-        tokens_generated as f64 / result.duration_secs
-    } else {
-        0.0
-    };
+/// Based on Ivanov et al. (2021): M=1 decode is memory-bandwidth bound.
+///
+/// #4211: the backend that RAN is an input. The tok/s thresholds alone once
+/// told a CUDA run (`GPU used: yes`) that it was CPU-bound and to "Enable GPU
+/// with --gpu". On a GPU run the throughput is wall clock including load and
+/// one-off CUDA setup, so a low number is not evidence of a CPU bottleneck;
+/// the GPU arm never names the CPU and never recommends `--gpu`. A CPU run
+/// never claims tensor cores. `None` (backend not reported) keeps the
+/// throughput-only tiers, but its top tier names no backend: only a run that
+/// reported `GPU used: yes` is told its tensor cores are engaged.
+fn classify_roofline(
+    tok_per_sec: f64,
+    used_gpu: Option<bool>,
+) -> (u8, u8, &'static str, &'static str) {
+    match used_gpu {
+        Some(true) => classify_roofline_gpu(tok_per_sec),
+        _ if tok_per_sec > 50.0 => classify_roofline_fast_non_gpu(used_gpu),
+        _ => classify_roofline_tiers(tok_per_sec),
+    }
+}
 
-    // Roofline classification based on Ivanov et al. (2021):
-    // M=1 decode is memory-bandwidth bound. High tok/s implies GPU
-    // compute is engaged (batched prefill or tensor cores).
-    let (compute_pct, memory_pct, bottleneck, recommendation) = if tok_per_sec > 50.0 {
+/// GPU arm (#4211): never names the CPU, never recommends `--gpu`.
+fn classify_roofline_gpu(tok_per_sec: f64) -> (u8, u8, &'static str, &'static str) {
+    if tok_per_sec > 50.0 {
         (
             65,
             35,
             "Compute (GPU tensor cores engaged)",
             "Efficient — GPU-accelerated path active",
         )
-    } else if tok_per_sec > 20.0 {
+    } else {
+        (
+            20,
+            80,
+            "Memory bandwidth (GPU VRAM); wall-clock tok/s includes load and one-off CUDA setup",
+            "Measure decode-only throughput with `apr bench` before tuning",
+        )
+    }
+}
+
+/// Above 50 tok/s without a GPU report: a CPU run never claims tensor cores.
+fn classify_roofline_fast_non_gpu(used_gpu: Option<bool>) -> (u8, u8, &'static str, &'static str) {
+    if used_gpu == Some(false) {
+        (
+            40,
+            60,
+            "Mixed (CPU SIMD path, memory bandwidth limited)",
+            "Efficient for CPU — enable GPU with --gpu for more",
+        )
+    } else {
+        (
+            65,
+            35,
+            "Compute (high throughput; backend not reported)",
+            "Efficient at this throughput",
+        )
+    }
+}
+
+/// The throughput-only tiers at or below 50 tok/s.
+fn classify_roofline_tiers(tok_per_sec: f64) -> (u8, u8, &'static str, &'static str) {
+    if tok_per_sec > 20.0 {
         (
             40,
             60,
@@ -227,15 +270,41 @@ fn print_roofline_profile(result: &RunResult, max_tokens: usize) {
             "Memory bandwidth (CPU, no SIMD saturation)",
             "Model too large for CPU — use GPU or smaller model",
         )
+    }
+}
+
+/// Print roofline profiling analysis (PMAT-480).
+///
+/// Estimates compute vs memory boundedness from throughput. For real
+/// per-brick µs timing, use `apr profile <model> --granular` which
+/// integrates with trueno's BrickProfiler.
+fn print_roofline_profile(result: &RunResult, max_tokens: usize) {
+    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
+    let total_ms = result.duration_secs * 1000.0;
+    let tok_per_sec = if result.duration_secs > 0.0 {
+        tokens_generated as f64 / result.duration_secs
+    } else {
+        0.0
     };
 
+    let (compute_pct, memory_pct, bottleneck, recommendation) =
+        classify_roofline(tok_per_sec, result.used_gpu);
+
     eprintln!();
     eprintln!("{}", "=== Roofline Profile (PMAT-480) ===".cyan().bold());
     eprintln!();
-    eprintln!("  Throughput:     {tok_per_sec:.1} tok/s");
+    eprintln!("  Throughput:     {tok_per_sec:.1} tok/s (wall clock, load included)");
     eprintln!("  Latency:        {total_ms:.1} ms ({tokens_generated} tokens)");
-    eprintln!("  Per-token:      {:.2} ms", total_ms / tokens_generated.max(1) as f64);
-    eprintln!("  GPU used:       {}", result.used_gpu.map_or("unknown", |g| if g { "yes" } else { "no" }));
+    eprintln!(
+        "  Per-token:      {:.2} ms",
+        total_ms / tokens_generated.max(1) as f64
+    );
+    eprintln!(
+        "  GPU used:       {}",
+        result
+            .used_gpu
+            .map_or("unknown", |g| if g { "yes" } else { "no" })
+    );
     eprintln!();
     eprintln!("  {}", "Roofline Classification".bold());
     eprintln!("  Compute bound:  {compute_pct}%");
diff --git a/scripts/check_clippy_feature_matrix.sh b/scripts/check_clippy_feature_matrix.sh
index 5b0520f27..550e433c5 100755
--- a/scripts/check_clippy_feature_matrix.sh
+++ b/scripts/check_clippy_feature_matrix.sh
@@ -15,18 +15,22 @@
 #   full        --features full     pulls in cuda-batch, training-gpu, code, xet and trueno-explain
 #   dev         --features dev
 #   dhat-heap   --features dhat-heap
+#   no-default  --no-default-features   the minimal crates.io build apr-cli's manifest promises (#4041: SUPPORTED;
+#               it had 8 compile errors because nothing in-tree builds it — the facade pins default-features)
 #
 # EXCLUDED, by name and ticket, never by omission:
 #   wgpu         --features wgpu does not COMPILE (5 errors) — #4056 (P1)
-#   no-default   --no-default-features does not COMPILE (8 errors) — #4041 decides whether it is a supported
-#                configuration; it joins this list when it is.
 #
 # THE UNIVERSE IS DERIVED (#3837's "enumerate the other feature axes"). apr-cli's feature table is read from
 # `cargo metadata`, and every feature must be linted by some axis — directly, or because an axis's feature set
 # (plus `default` unless the axis disables it) implies it — or be EXCLUDED with a ticket. A new feature that no axis
 # reaches is a refusal naming it, before any clippy runs. --self-test plants one to prove it.
 #
-# --self-test plants `x as u32` on a u32 inside cuda-only code and requires: cuda axis RED, default axis GREEN.
+# --self-test plants `x as u32` on a u32 inside cuda-only code AND a type error inside code compiled only WITHOUT
+# `inference` (a lint would not do: apr-cli's lib.rs allows clippy::all and more),
+# and requires: cuda and full RED (the cuda plant), no-default RED (the no-inference plant), default, dev and
+# dhat-heap GREEN. The second plant is what proves the no-default axis really builds without defaults: an axis
+# whose flags were dropped would lint the default build and stay green.
 # The restore is trapped BEFORE the plant and uses `git checkout --`, so an interrupted run leaves no planted file.
 #
 # Exit: 0 every axis clean · 1 an axis has findings · 2 could not check.
@@ -35,7 +39,7 @@ set -uo pipefail
 ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
 cd "$ROOT" || exit 2
 PROG=check_clippy_feature_matrix
-AXES=(default cuda full dev dhat-heap)
+AXES=(default cuda full dev dhat-heap no-default)
 flags_for() { # prints nothing; fills the global array F for axis $1
   case "$1" in
     default)   F=() ;;
@@ -43,12 +47,13 @@ flags_for() { # prints nothing; fills the global array F for axis $1
     full)      F=(--features full) ;;
     dev)       F=(--features dev) ;;
     dhat-heap) F=(--features dhat-heap) ;;
+    no-default) F=(--no-default-features) ;;
     *) echo "$PROG: unknown axis $1" >&2; return 2 ;;
   esac
 }
 # Features no axis may claim, each with its ticket (the no-default MODE is excluded above, not a feature).
 EXCLUDED_FEATURES="wgpu=#4056"
-EXCLUDED="wgpu (#4056: does not compile), no-default (#4041: does not compile)"
+EXCLUDED="wgpu (#4056: does not compile)"
 
 # Every apr-cli feature must be reached by an axis or excluded with a ticket: derived from cargo metadata.
 coverage() { # -> 0 every feature accounted for; 1 names each unreached feature; 2 metadata unreadable
@@ -110,9 +115,10 @@ check() { # -> 0 all clean, 1 findings; prints one line per axis
 
 TMP=$(mktemp -d) || exit 2
 PLANT_FILE=crates/aprender-train/src/autograd/cuda_forward/matmul_f16.rs
+PLANT_ND_FILE=crates/apr-cli/src/commands/explain.rs   # #4041: any apr-cli file; the plant carries its own cfg
 MANIFEST=crates/apr-cli/Cargo.toml
 cleanup() {
-  [ -n "${PLANTED:-}" ] && git checkout -- "$PLANT_FILE" 2>/dev/null
+  [ -n "${PLANTED:-}" ] && git checkout -- "$PLANT_FILE" "$PLANT_ND_FILE" 2>/dev/null
   [ -n "${PLANTED_FEATURE:-}" ] && git checkout -- "$MANIFEST" 2>/dev/null
   rm -rf "$TMP"
 }
@@ -132,18 +138,24 @@ if [ "${1:-}" = "--self-test" ]; then
   fi
   echo "  ok    an unreached feature is refused by name, before any clippy"
   # Plant 2: a cuda-only finding reddens ONLY the cuda-reaching axes.
-  git diff --quiet -- "$PLANT_FILE" || { echo "$PROG: $PLANT_FILE has local edits; the self-test will not plant over them" >&2; exit 2; }
+  git diff --quiet -- "$PLANT_FILE" "$PLANT_ND_FILE" || { echo "$PROG: a plant file has local edits; the self-test will not plant over them" >&2; exit 2; }
   grep -q '#!\[cfg(feature = "cuda")\]\|^#\[cfg(feature = "cuda")\]' "$PLANT_FILE" || { echo "$PROG: $PLANT_FILE is no longer cuda-gated; move the plant" >&2; exit 2; }
   PLANTED=1
   printf '\n#[cfg(feature = "cuda")]\n#[allow(dead_code)]\nfn planted_3837_self_test(x: u32) -> u32 {\n    x as u32\n}\n' >> "$PLANT_FILE"
+  # #4041's plant is a TYPE ERROR, not a lint: apr-cli's lib.rs opens with #![allow(clippy::all, clippy::pedantic,
+  # clippy::disallowed_methods)] and more, so a lint planted in apr-cli stays green on every axis (measured: a cast,
+  # an Option::unwrap, an unused variable). The property this plant proves is that the no-default axis really
+  # compiles with `inference` off, and only a build without it compiles this function.
+  printf '\n#[cfg(not(feature = "inference"))]\n#[allow(dead_code)]\nfn planted_4041_self_test() -> u32 {\n    "compiled only without inference"\n}\n' >> "$PLANT_ND_FILE"
   out=$(check); rc=$?
   printf '%s\n' "$out"
-  git checkout -- "$PLANT_FILE"; PLANTED=
+  git checkout -- "$PLANT_FILE" "$PLANT_ND_FILE"; PLANTED=
   if grep -q '^  FAIL  cuda' <<< "$out" && grep -q '^  FAIL  full' <<< "$out" && grep -q '^  ok    default' <<< "$out" \
-     && grep -q '^  ok    dev' <<< "$out" && grep -q '^  ok    dhat-heap' <<< "$out"; then
-    echo "$PROG --self-test: PASS — the planted cuda-only finding turned exactly the cuda-reaching axes (cuda, full) red"; exit 0
+     && grep -q '^  ok    dev' <<< "$out" && grep -q '^  ok    dhat-heap' <<< "$out" \
+     && grep -q '^  FAIL  no-default' <<< "$out"; then
+    echo "$PROG --self-test: PASS — the cuda plant turned exactly cuda and full red, and the no-inference plant turned exactly no-default red"; exit 0
   fi
-  echo "$PROG --self-test: FAIL (rc $rc) — want cuda and full RED, default/dev/dhat-heap GREEN"; exit 1
+  echo "$PROG --self-test: FAIL (rc $rc) — want cuda, full and no-default RED; default, dev and dhat-heap GREEN"; exit 1
 fi
 
 echo "$PROG: axes ${AXES[*]}; excluded: $EXCLUDED"
```

## TEST FILE
```rust
    // #4211: `apr run --benchmark --json` stdout must be ONE JSON document, and
    // `--profile` must not call a GPU run CPU-bound or tell it to use --gpu.

    fn bench_result_4211(tok_per_sec: Option<f64>, used_gpu: Option<bool>) -> RunResult {
        RunResult {
            text: "hello".to_string(),
            duration_secs: 8.892,
            cached: true,
            tokens_generated: Some(56),
            tok_per_sec,
            used_gpu,
            gpu_attempted: used_gpu,
            generated_tokens: None,
            token_texts: None,
            usage: Default::default(),
        }
    }

    #[test]
    fn benchmark_json_stdout_is_exactly_one_json_document_4211() {
        let r = bench_result_4211(Some(6.3), Some(true));
        let (stdout, stderr) = render_benchmark_results(&r, "m.gguf", "json", 32);
        // The whole of stdout parses as one document: a human preamble, or a
        // second document after it, makes this fail (the `jq .` of the ticket).
        let v: serde_json::Value = serde_json::from_str(stdout.trim())
            .unwrap_or_else(|e| panic!("stdout is not one JSON document ({e}): {stdout:?}"));
        assert!(v.is_object(), "stdout JSON is not an object: {v}");
        assert_eq!(stdout.trim().lines().count(), 1, "stdout: {stdout:?}");
        // The human block still reaches the user, on stderr.
        assert!(stderr.contains("Benchmark Results"), "stderr: {stderr:?}");
        assert!(!stdout.contains("Benchmark Results"), "stdout: {stdout:?}");
    }

    #[test]
    fn benchmark_reports_one_throughput_with_its_basis_4211() {
        // Engine figure present: stdout reports it — the number stderr's
        // "Generated N tokens in X ms (Y tok/s)" line prints — not the wall-clock
        // 56 / 8.892 = 6.3. 9.7 is distinct from 6.3 so the two cannot alias.
        // The backend marked where generation began: setup is excluded.
        let mut r = bench_result_4211(Some(9.7), Some(true));
        r.usage.generation_ms = Some(5773);
        r.usage.setup_ms = Some(3119);
        let (stdout, stderr) = render_benchmark_results(&r, "m.gguf", "json", 32);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
        assert_eq!(v["tok_s"].as_f64(), Some(9.7), "{v}");
        assert_eq!(v["tok_s_basis"], "generation", "{v}");
        assert_eq!(v["generation_ms"].as_u64(), Some(5773), "{v}");
        assert_eq!(v["latency_basis"], "wall", "{v}");
        assert!(stderr.contains("setup excluded"), "stderr: {stderr:?}");

        // The path did not mark it (wgpu, for one): the engine figure still
        // includes weight upload + F2, and must not be called setup-excluded.
        let r = bench_result_4211(Some(9.7), Some(true));
        let (stdout, stderr) = render_benchmark_results(&r, "m.gguf", "json", 32);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
        assert_eq!(v["tok_s"].as_f64(), Some(9.7), "{v}");
        assert_eq!(v["tok_s_basis"], "inference_incl_setup", "{v}");
        assert!(v["generation_ms"].is_null(), "{v}");
        assert!(!stderr.contains("excluded"), "stderr: {stderr:?}");

        // No engine figure: wall clock, and labelled as such.
        let r = bench_result_4211(None, Some(false));
        let (stdout, _) = render_benchmark_results(&r, "m.gguf", "json", 32);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
        assert_eq!(v["tok_s"].as_f64(), Some(6.3), "{v}"); // 56 / 8.892
        assert_eq!(v["tok_s_basis"], "wall", "{v}");
    }

    #[test]
    fn benchmark_json_and_human_line_agree_at_a_rounding_tie_4211() {
        // 6.25 is exact in binary64 (25 tokens / 4.0 s). `.round()` breaks the tie
        // away from zero and `{:.1}` does not, so rounding twice printed two
        // throughputs for one run. Whichever digit wins, both streams show it.
        let mut r = bench_result_4211(Some(6.25), Some(true));
        r.usage.generation_ms = Some(4000);
        let (stdout, stderr) = render_benchmark_results(&r, "m.gguf", "json", 32);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
        let human: f64 = stderr
            .lines()
            .find_map(|l| l.strip_prefix("tok/s: "))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("no tok/s line on stderr: {stderr:?}"));
        assert_eq!(v["tok_s"].as_f64(), Some(human), "json {v} vs stderr {stderr:?}");
    }

    #[test]
    fn benchmark_human_format_keeps_the_block_on_stdout_4211() {
        let r = bench_result_4211(Some(6.3), None);
        let (stdout, stderr) = render_benchmark_results(&r, "m.gguf", "text", 32);
        assert!(stdout.contains("Benchmark Results"), "stdout: {stdout:?}");
        assert!(stdout.contains("tok/s: 6.3"), "stdout: {stdout:?}");
        assert!(stderr.is_empty(), "stderr: {stderr:?}");
    }

    #[test]
    fn roofline_on_a_gpu_run_never_says_cpu_or_use_gpu_4211() {
        // Every tier the throughput-only table had, including the two the
        // ticket quotes (6.x tok/s and <5 tok/s on CUDA).
        for tps in [0.0, 1.0, 4.1, 5.0, 6.3, 12.0, 20.0, 35.0, 50.0, 93.9, 250.0] {
            let (c, m, bottleneck, rec) = classify_roofline(tps, Some(true));
            assert_eq!(u32::from(c) + u32::from(m), 100, "tps {tps}");
            assert!(!bottleneck.contains("CPU"), "tps {tps}: {bottleneck}");
            assert!(!bottleneck.contains("DRAM"), "tps {tps}: {bottleneck}");
            assert!(!rec.contains("--gpu"), "tps {tps}: {rec}");
            assert!(!rec.contains("CPU"), "tps {tps}: {rec}");
        }
    }

    #[test]
    fn roofline_on_a_cpu_run_never_claims_tensor_cores_4211() {
        for tps in [0.0, 4.1, 6.3, 35.0, 50.1, 93.9, 250.0] {
            let (_, _, bottleneck, rec) = classify_roofline(tps, Some(false));
            assert!(!bottleneck.contains("GPU"), "tps {tps}: {bottleneck}");
            assert!(!rec.contains("GPU-accelerated"), "tps {tps}: {rec}");
        }
        // The CPU-bound readings the ticket saw on CUDA still apply to a CPU run.
        assert!(classify_roofline(6.3, Some(false)).3.contains("--gpu"));
        assert!(classify_roofline(4.1, Some(false)).2.contains("CPU"));
    }

    #[test]
    fn roofline_with_unreported_backend_names_no_backend_4211() {
        for tps in [0.0, 4.1, 6.3, 35.0, 50.1, 93.9, 250.0] {
            let (_, _, bottleneck, rec) = classify_roofline(tps, None);
            assert!(!bottleneck.contains("GPU"), "tps {tps}: {bottleneck}");
            assert!(!bottleneck.contains("tensor cores"), "tps {tps}: {bottleneck}");
            assert!(!rec.contains("GPU-accelerated"), "tps {tps}: {rec}");
        }
    }
```

## ROADMAP diff stat vs origin/main
 docs/roadmaps/roadmap.yaml | 129 +++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 129 insertions(+)
+- id: GH-4149
+- id: GH-4175
+- id: PMAT-4041
+- id: PMAT-4152
+- id: PMAT-4193
+- id: PMAT-4259
+- id: PMAT-4273
