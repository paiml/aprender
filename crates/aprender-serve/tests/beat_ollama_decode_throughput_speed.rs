//! BEAT-OLLAMA-DECODE-THROUGHPUT — Pillar-4 SPEED beat (PMAT-755, audit gap #4).
//! **ENFORCED manual/GPU gate (NOT a CPU-CI gate).**
//!
//! `#[ignore]`d: it needs an NVIDIA GPU, an `apr` binary built `--features cuda`,
//! a resident `ollama` daemon + the Q4_K_M model, and the matching GGUF on disk.
//! NONE of those exist on the CPU-only self-hosted CI runners, so this ENFORCED
//! gate is MANUAL/operator-gated (lambda-vector RTX 4090 or gx10 GB10), exactly
//! like the cuda-oxide throughput beat. The CPU unit tests below DO run in CI.
//!
//! ## VERDICT 2026-07-31: the 1.371x BEAT CLAIM IS WITHDRAWN. This is now a
//! ## NO-COLLAPSE PARITY FLOOR, not a beat. apr does not currently win here.
//!
//! The old header asserted "median ratio 1.371x, worst single run 1.230x, every
//! single run clears 1.10x, bootstrapped ~0% false-FAIL". That claim is not
//! reproducible on this host and has been removed rather than re-explained.
//!
//! FOUR INDEPENDENT MEASUREMENTS, one host (lambda RTX 4090 sm_89):
//!   date        apr median   ollama median   ratio    source
//!   2026-06-15    412.3         300.7        1.371x   promotion claim (#2067)
//!   2026-07-29    332.7         299.9        1.109x   cuda-nightly (PASSED)
//!   2026-07-31    342.4         328.6        1.042x   cuda-nightly (FAILED)
//!   2026-07-31    318.2         313.5        1.015x   idle box, this harness
//!
//! Read the OLLAMA column first: 300.7 / 299.9 / 328.6 / 313.5 over six weeks.
//! The incumbent is reproducible, so the drift is not the measuring rig. apr's
//! own column moved 412 -> ~303-342. The 2026-07-29 "PASS" at 1.109x was already
//! this regression squeaking past a threshold with 0.8% headroom; nobody looked
//! because it was green.
//!
//! WHAT CHANGED: #2323 (2026-07-27) made `auto_q4k` return Mwv on every device.
//! The previous default on sm_89 was HwDp4a, whose INT8 Q8_1 activation quant is
//! numerically degraded (F2 first-token cosine 0.9186 < the 0.95 floor). The
//! 412.3 figure was measured under that old default.
//!
//! DO NOT read that as "#2323 cost us 23%" — that is NOT what was measured. Re-run
//! today with the surviving opt-in, `HW_DP4A_Q4K=1`, and the differential is
//! **20.3 tok/s**, not 412: HwDp4a is F2-REJECTED, wgpu then fails its own 0.99
//! gate, and the run finishes on CPU SIMD. That is precisely the ~20 tok/s
//! collapse #2323 was written to fix. So the honest statement is narrow: the
//! 1.371x number was taken under a kernel default that no longer exists and can
//! no longer be reproduced here. A correct 318 tok/s is worth far more than a
//! degraded kernel the F2 gate refuses to let serve a token.
//!
//! WHY THE FLOOR IS 0.90 AND NOT 1.10. Keeping 1.10 asserts a win apr does not
//! currently have; it would fail ~every other night and teach people to ignore
//! this gate. Lowering it to ~1.00 would quietly redefine "beat" as "tie". So the
//! assertion is renamed to what it can actually prove: apr must not COLLAPSE.
//! 0.90 sits 12% under the worst observed median (1.015) so it does not flake,
//! and still catches the failure class that matters — the CPU-fallback collapse
//! is ratio ~0.065, a 14x violation.
//!
//! RECOVERING THE BEAT is tracked separately: either restore DP4A-class decode
//! speed with activation quant that clears F2, or make Mwv faster. Until apr
//! measures >= 1.10x on this host again, Pillar-4 must NOT claim a GPU decode
//! win over ollama on sm_89.
//!
//! ## DEPENDENCY: the median premise still needs #2049 on main
//! The ~1-in-6 catastrophic decode stall is fixed by #2049 (FP8-warmup OOB read).
//! If #2049 is reverted/absent the stall returns and any median gate here flakes.
//!
//! ## What it measures (be honest about scope)
//! STEADY-STATE GPU DECODE throughput only — the marginal token-generation rate
//! with model load + FP8-weight-cache build + CUDA-graph capture + prefill
//! amortized out. apr's one-shot CLI has a large (~3.4-3.9s) fixed per-invocation
//! startup cost that ollama's resident daemon avoids; short-prompt one-shot
//! WALL-CLOCK still favors ollama and is a SEPARATE, conceded comparison (NOT
//! measured here). The decode rate is the apples-to-apples kernel-throughput claim.
//!
//! ## Method (differential — cancels apr's fixed overhead)
//! Time `apr run --gpu --benchmark` at two forced token counts (128 and 384) on
//! the SAME long prompt; the marginal decode rate is
//! `(384-128) / (ms@384 - ms@128) * 1000` tok/s. Compare to ollama's `--verbose`
//! "eval rate" (already decode-only — it excludes prompt-eval). apr side uses the
//! MEDIAN-OF-7 clean trials (the #2049 fix removed the stall, so the median is
//! stable); ollama side uses its median (it is tight). Same session, same GPU,
//! same GGUF weights.
//!
//! ## Run recipe (NVIDIA host with apr --features cuda + ollama + model)
//! ```text
//! APR_BIN=/mnt/nvme-raid0/targets/aprender/release/apr \
//! APR_GGUF=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//! OLLAMA_MODEL=qwen2.5-coder:1.5b-instruct-q4_K_M \
//! cargo test -p aprender-serve --test beat_ollama_decode_throughput_speed \
//!   -- --ignored --nocapture
//! ```
//!
//! Contract: contracts/beat-ollama-decode-throughput-speed-v1.yaml.

#![cfg(test)]

use std::path::Path;
use std::process::Command;

/// NO-COLLAPSE FLOOR (mirrors the contract's `beat_threshold`). apr's MEDIAN-OF-7
/// decode must stay within this factor of ollama's median.
///
/// This is NOT a beat threshold. It was 1.10 while the harness believed apr ran
/// at 1.371x; the three post-#2323 measurements are 1.109 / 1.042 / 1.015, so 1.10
/// now fails about every other night for a reason the gate cannot fix. 0.90 sits
/// 12% below the worst observed median — it will not flake — while still catching
/// the collapse that actually costs users: an F2-rejected GPU path falling to CPU
/// SIMD measures ratio ~0.065, violating this floor by 14x. See the header.
const ENFORCED_THRESHOLD: f64 = 0.90;

/// Forced token counts for the differential apr measurement (amortizes fixed cost).
const N_LOW: u32 = 128;
const N_HIGH: u32 = 384;

/// apr trials for the ENFORCED median: median-of-7 gives extra non-flakiness margin
/// over the validated median-of-5 now that the ~1-in-6 stall is fixed by #2049. We
/// attempt a few extra so the median has 7 clean samples even if a trial early-EOSes.
const APR_MEDIAN_N: usize = 7;
const APR_ATTEMPTS: usize = 9;

/// ollama trials (its distribution is tight, so a small median is stable).
const OLLAMA_TRIALS: usize = 5;

/// A long, open-ended prompt that does NOT terminate early, so apr generates the
/// full forced token budget at both N_LOW and N_HIGH.
const PROMPT: &str = "Write a detailed 1000-word essay about the history of computing, \
covering Babbage, Turing, von Neumann, transistors, integrated circuits, microprocessors, \
the internet, and modern AI. Be thorough and do not stop early.";

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Median of a slice of f64 (sorts a copy). Panics on empty input.
fn median(xs: &[f64]) -> f64 {
    assert!(!xs.is_empty(), "median of empty slice");
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Run `apr run --gpu --benchmark` once and return `(tokens_generated, decode_ms)`
/// parsed from the `Generated <N> tokens in <ms>ms` line. `decode_ms` here is the
/// raw wall time for the generation (still includes fixed prefill+startup — the
/// caller differences two token counts to cancel it).
fn apr_run(apr_bin: &str, gguf: &Path, max_tokens: u32) -> Option<(u32, f64)> {
    let out = Command::new(apr_bin)
        .args([
            "run",
            &gguf.display().to_string(),
            "--prompt",
            PROMPT,
            "--max-tokens",
            &max_tokens.to_string(),
            "--gpu",
            "--benchmark",
        ])
        .output()
        .expect("spawn apr (is APR_BIN built --features cuda?)");
    // apr prints diagnostics to stderr and the Generated line may land on either;
    // scan both.
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_generated(&text)
}

/// Parse `Generated <tokens> tokens in <ms>ms` → (tokens, ms).
fn parse_generated(text: &str) -> Option<(u32, f64)> {
    let line = text
        .lines()
        .find(|l| l.contains("Generated") && l.contains("tokens in"))?;
    // ... "Generated 384 tokens in 4632.3ms (82.9 tok/s)"
    let after_gen = line.split("Generated").nth(1)?.trim();
    let tokens: u32 = after_gen.split_whitespace().next()?.parse().ok()?;
    let after_in = line.split("in").nth(1)?.trim(); // "4632.3ms (82.9 tok/s)"
    let ms: f64 = after_in
        .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
        .split_whitespace()
        .next()?
        .trim_end_matches("ms")
        .parse()
        .ok()?;
    Some((tokens, ms))
}

/// One clean apr steady-state decode measurement (marginal tok/s), or None if a
/// run hit early EOS (didn't reach the forced budget) and must be discarded.
fn apr_decode_tps_once(apr_bin: &str, gguf: &Path) -> Option<f64> {
    let (lo_tok, lo_ms) = apr_run(apr_bin, gguf, N_LOW)?;
    let (hi_tok, hi_ms) = apr_run(apr_bin, gguf, N_HIGH)?;
    // Both runs must have generated the full forced budget, else the differential
    // is meaningless (the model stopped early).
    if lo_tok != N_LOW || hi_tok != N_HIGH {
        return None;
    }
    let dms = hi_ms - lo_ms;
    if dms <= 0.0 {
        return None; // scheduling hiccup (e.g. JIT stall on the LOW run)
    }
    let marginal_tokens = f64::from(N_HIGH - N_LOW);
    Some(marginal_tokens / (dms / 1000.0))
}

/// One ollama warm decode measurement: `ollama run <model> <prompt> --verbose`,
/// parse the "eval rate: <X> tokens/s" line (decode-only; excludes prompt eval).
fn ollama_eval_tps_once(model: &str) -> Option<f64> {
    let out = Command::new("ollama")
        .args(["run", model, PROMPT, "--verbose"])
        .output()
        .expect("spawn ollama (is it installed and is the model pulled?)");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    text.lines()
        .filter(|l| l.contains("eval rate") && !l.contains("prompt eval"))
        .find_map(|l| {
            l.split(':')
                .nth(1)?
                .split_whitespace()
                .next()?
                .parse::<f64>()
                .ok()
        })
}

#[test]
#[ignore = "ENFORCED manual/GPU gate: needs NVIDIA GPU + apr --features cuda (incl. #2049 FP8 \
            stall fix) + ollama + Q4_K_M model (no NVIDIA CI runner; status=enforced, see contract)"]
fn beat_ollama_decode_throughput_speed() {
    // APR_BIN is REQUIRED — there is deliberately no default path.
    //
    // This used to default to /mnt/nvme-raid0/targets/aprender/release/apr. That
    // directory is ORPHANED: nothing in the repo writes it (`git grep target-dir`
    // finds no producer), so it is hand-maintained convention that goes stale by
    // construction. On 2026-08-01 it held 0.60.0 while HEAD was 0.62.0 — six days
    // and two minor versions behind — and a beat measuring a stale binary reports
    // a number about code nobody is shipping. cuda-nightly.yml already overrides
    // APR_BIN for exactly this reason; the default only ever served whoever forgot.
    //
    // There is also no correct path to substitute: `.cargo/config.toml` redirects
    // cargo's target-dir and is gitignored, so the main checkout and a worktree
    // build to different places. Callers must resolve it — `scripts/apr_bin.sh`
    // asks cargo and proves the binary's embedded SHA matches HEAD.
    let apr_bin = std::env::var("APR_BIN").unwrap_or_else(|_| {
        panic!(
            "APR_BIN must be set to the apr binary under test. There is no default: \
             any hardcoded path is stale in some checkout and right in another. \
             Resolve it with `. scripts/apr_bin.sh` (exports $APR, asserts SHA == HEAD), \
             then re-run with APR_BIN=\"$APR\"."
        )
    });
    let gguf = env_or(
        "APR_GGUF",
        &format!(
            "{}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
            std::env::var("HOME").unwrap_or_default()
        ),
    );
    let ollama_model = env_or("OLLAMA_MODEL", "qwen2.5-coder:1.5b-instruct-q4_K_M");
    let gguf_path = Path::new(&gguf);

    assert!(
        gguf_path.exists(),
        "APR_GGUF not found: {gguf} — point APR_GGUF at the Q4_K_M GGUF (same weights ollama serves)"
    );

    // --- apr: collect clean differential decode trials. With the #2049 FP8 stall
    // fix, the ~1-in-6 catastrophic stall is gone (re-measured 0/8), so the MEDIAN
    // is the right estimator. We attempt a few extra to backfill the rare early-EOS
    // so the gate's median has APR_MEDIAN_N clean samples. ---
    let mut apr_tps = Vec::new();
    for _ in 0..APR_ATTEMPTS {
        if let Some(t) = apr_decode_tps_once(&apr_bin, gguf_path) {
            apr_tps.push(t);
        }
        if apr_tps.len() >= APR_MEDIAN_N {
            break;
        }
    }
    assert!(
        apr_tps.len() >= APR_MEDIAN_N,
        "could not collect {APR_MEDIAN_N} clean apr decode trials for the median-of-{APR_MEDIAN_N} \
         gate (got {} in {APR_ATTEMPTS} attempts; if many are early-EOS/stalls the #2049 FP8 fix \
         may be missing from this binary — see contract DEPENDENCY)",
        apr_tps.len()
    );
    // Take exactly APR_MEDIAN_N samples for the enforced median (median-of-7).
    let apr_med = median(&apr_tps[..APR_MEDIAN_N]);
    let apr_best = apr_tps.iter().copied().fold(f64::MIN, f64::max);

    // --- ollama: median over a few warm eval-rate measurements (tight distribution) ---
    let mut ollama_tps = Vec::new();
    for _ in 0..OLLAMA_TRIALS {
        let t = ollama_eval_tps_once(&ollama_model)
            .expect("failed to parse ollama eval rate (is the model pulled and the daemon up?)");
        ollama_tps.push(t);
    }
    let ollama_med = median(&ollama_tps);

    let ratio_med = apr_med / ollama_med;
    let ratio_best = apr_best / ollama_med;

    eprintln!(
        "BEAT-OLLAMA-DECODE-THROUGHPUT [ENFORCED median-of-{APR_MEDIAN_N} >= {ENFORCED_THRESHOLD:.2}x]: \
         apr_median7={apr_med:.1} apr_best={apr_best:.1} ollama_median={ollama_med:.1} tok/s | \
         ratio_median={ratio_med:.3}x ratio_best={ratio_best:.3}x (steady-state GPU decode) | \
         apr trials={apr_tps:?} ollama trials={ollama_tps:?}\n\
         NOTE: status=enforced — this gate's non-flaky median premise DEPENDS on #2049 (FP8 \
         warmup OOB stall fix) being on the decode path (contract \
         beat-ollama-decode-throughput-speed-v1.yaml)."
    );

    // NO-COLLAPSE assertion. Deliberately NOT "apr beats ollama" - see the header:
    // that claim was withdrawn on 2026-07-31 because it is not reproducible. What
    // this still proves is that apr is serving decode on the GPU at all.
    //
    // The old message offered two diagnoses ("apr's advantage regressed, OR #2049
    // is missing") and BOTH were wrong when it finally fired on 2026-07-31 - the
    // cause was #2323 changing the default Q4K kernel. A failure message that
    // confidently names the wrong cause is worse than one that names none, so this
    // one reports the measurement and points at the checks that discriminate.
    assert!(
        ratio_med >= ENFORCED_THRESHOLD,
        "DECODE-COLLAPSE beat-ollama-decode-throughput: apr median-of-{APR_MEDIAN_N} decode \
         {apr_med:.1} tok/s is below {ENFORCED_THRESHOLD:.2}x ollama's median {ollama_med:.1} tok/s \
         (ratio_median {ratio_med:.3}). This floor is NOT a beat threshold - at this depth apr is \
         very likely not decoding on the GPU at all. Check, in order: (1) is the F2 first-token \
         gate REJECTING the CUDA path (look for 'GPU diverges from CPU' / cosine < 0.95)? A \
         rejected CUDA path falls to wgpu, then to CPU SIMD, and measures ~20 tok/s - ratio ~0.065. \
         (2) is APR_BIN built --features cuda? (3) is HW_DP4A_Q4K set, re-selecting the degraded \
         kernel #2323 removed as the default? (contract beat-ollama-decode-throughput-speed-v1.yaml)"
    );
}

// --- Pure-CPU unit tests for the parsing helpers (these DO run in normal CI) ---

#[test]
fn parse_generated_extracts_tokens_and_ms() {
    let s = "noise\nGenerated 384 tokens in 4632.3ms (82.9 tok/s)\nmore noise";
    assert_eq!(parse_generated(s), Some((384, 4632.3)));
}

#[test]
fn parse_generated_handles_low_count() {
    let s = "Generated 128 tokens in 4119.1ms (31.1 tok/s)";
    assert_eq!(parse_generated(s), Some((128, 4119.1)));
}

#[test]
fn parse_generated_none_when_absent() {
    assert_eq!(parse_generated("no generation line here"), None);
}

#[test]
fn median_odd_and_even() {
    assert!((median(&[366.3, 423.2, 437.8]) - 423.2).abs() < 1e-9);
    assert!((median(&[300.0, 310.0, 320.0, 330.0]) - 315.0).abs() < 1e-9);
}

#[test]
fn median_of_seven_picks_middle() {
    // The enforced gate uses median-of-7; verify the estimator is the 4th-smallest.
    let xs = [458.0, 411.2, 430.4, 421.3, 369.9, 413.4, 384.3];
    assert!(
        (median(&xs) - 413.4).abs() < 1e-9,
        "median-of-7 is the 4th-smallest"
    );
}

#[test]
fn enforced_threshold_is_a_real_beat_with_margin() {
    // status=enforced: the gate asserts apr median-of-7 >= ollama median x 1.10.
    // 1.10 is a real beat (> 1.0) and sits well under the re-measured 1.37x median
    // and even the worst single run's 1.23x, so the gate has a wide non-flaky margin.
    // `black_box` defeats const-folding so clippy doesn't flag a constant assertion.
    let thresh = std::hint::black_box(ENFORCED_THRESHOLD);
    let measured_median_ratio = std::hint::black_box(1.371_f64);
    let worst_single_run_ratio = std::hint::black_box(1.230_f64);
    assert!(
        thresh > 1.0,
        "enforced threshold must be a real beat (> 1.0x)"
    );
    assert!(
        thresh < worst_single_run_ratio,
        "enforced 1.10x threshold must sit under the worst re-measured single run (1.23x) \
         so even worst-case single samples clear it"
    );
    assert!(
        worst_single_run_ratio < measured_median_ratio,
        "sanity: worst single run < median"
    );
}
