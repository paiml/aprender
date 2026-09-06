//! BEAT-OLLAMA-DECODE-THROUGHPUT — Pillar-4 SPEED beat (PMAT-755, audit gap #4).
//! **ENFORCED GPU gate, run nightly on gx10.**
//!
//! `#[ignore]`d: it needs an NVIDIA GPU, an `apr` binary built `--features cuda`,
//! a resident `ollama` daemon + the Q4_K_M model, and the matching GGUF on disk.
//! None of those exist on the CPU-only clean-room runners, so it runs in
//! `.github/workflows/cuda-nightly.yml` on **gx10** (GB10, sm_121), which is the
//! fleet's only sanctioned GPU runner. ollama and the model are DECLARED there —
//! paiml/infra `machines/gx10/forjar.yaml`, tag `beat-incumbent` — rather than
//! hand-installed, because a comparative beat whose incumbent is undeclared is
//! one apt-get away from silently not running.
//!
//! It previously ran on lambda-vector (RTX 4090, sm_89). That leg is retired:
//! lambda-labs is the workstation and must never be a CI host (paiml/infra#359).
//!
//! ## 2026-09-01: THE FLOOR DID NOT MOVE WITH THE HOST (aprender#2835)
//!
//! Retiring the `ada-4090` leg also deleted the predicate that scoped this step to it
//! (`&& matrix.name == 'ada-4090'`). **Removing a matrix leg re-targets every step
//! whose predicate named it.** From 2026-08-30 this gate asserted an sm_89 floor on
//! sm_121 and went red four nights running; before that it reports `skipped` on both
//! legs, so those four are the first executions it has ever had on this silicon.
//!
//! Every number below this line is an sm_89 number. That is now enforced rather than
//! merely documented: `SILICON_FLOORS` is keyed by compute capability, sm_121 has no
//! entry, and an uncalibrated silicon produces `UNCALIBRATED-SILICON` — a refusal to
//! assert a threshold derived on other hardware. It is NOT a pass (that is the
//! `ada-4090 only` skip that hid this gate for months) and NOT a claim that apr
//! regressed.
//!
//! The failure text was wrong in the same way. At ratio 0.619 it led with "very likely
//! not decoding on the GPU at all", whose own cited signature is ~0.065 — a diagnosis
//! its own arithmetic excluded. This file already contained the rule ("a failure
//! message that confidently names the wrong cause is worse than one that names none");
//! it had not been applied to the ladder itself. The diagnosis is now conditional on
//! the measured ratio.
//!
//! Whether GB10 is honestly ~0.62x or carries a real sm_121 decode deficit is OPEN
//! (#2835; #2800 argues the #2786 GB10 shortfall is a real deficit). Those imply
//! opposite fixes, and this gate must not pick one by accident — which is exactly what
//! inheriting 0.90 did.
//!
//! The unit tests below now DO run in CI: this target was added to `ci.yml`'s beat
//! chain. The header's own admission that they did not was still true today.
//!
//! TWO CLAIMS THAT USED TO SIT HERE WERE FALSE, and both are recorded rather than
//! quietly deleted:
//!
//!   - "The CPU unit tests below DO run in CI." They did not. This is an
//!     integration test TARGET, so `cargo test --lib` never reaches it, and the
//!     only workflow that named it ran `-- --ignored` (the beat) — the unit tests
//!     below were compiled and skipped. One of them had been RED for a month:
//!     `enforced_threshold_is_a_real_beat_with_margin` asserted `thresh > 1.0`
//!     against the 1.371x provenance this very header withdrew, while the
//!     constant it read had been 0.90 since. Nobody saw it because nothing ran
//!     it. It is now `enforced_threshold_is_a_no_collapse_floor_not_a_beat` and
//!     asserts what the constant means.
//!   - "no NVIDIA CI runner" (in the `#[ignore]` reason). There is one: gx10.
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

/// A no-collapse floor is DERIVED ON ONE SILICON AND DOES NOT TRANSFER TO ANOTHER.
///
/// This table exists because the untransferable thing was transferred. #2740 retired
/// the `ada-4090` matrix leg — correctly; lambda-vector is the workstation and must
/// never be a CI host — and in the same commit the beat step lost the predicate that
/// had scoped it to that leg:
///
/// ```text
/// -  - name: Pillar-4 ... beat (ada-4090 only)
/// -    if: steps.decide.outputs.proceed == 'true' && matrix.name == 'ada-4090'
/// +  - name: Pillar-4 ... beat
/// +    if: steps.decide.outputs.proceed == 'true'
/// ```
///
/// **Removing a matrix leg re-targets every step whose predicate named it.** The
/// assertion did not move hosts; the host moved out from under the assertion, and a
/// single global constant had no way to notice. Every number in this file's header is
/// an sm_89 number — the four measurements, the 1.015 worst median, the ~300 tok/s
/// incumbent — and since 2026-08-30 the gate has been asserting them against sm_121.
///
/// It had never run there before: on 2026-08-29 and earlier the step reports
/// `skipped` on both matrix legs. So the first four executions in the file's history
/// are the first four measurements of this silicon, and they are NOT a calibration —
/// §8's rule is that a threshold comes from samples, never from invention, and four
/// nights is not a sample set. They are recorded in `GB10_OBSERVED` below as data.
struct SiliconFloor {
    /// Compute capability exactly as `nvidia-smi --query-gpu=compute_cap` prints it.
    compute_cap: &'static str,
    /// Human name, for the failure message.
    arch: &'static str,
    /// apr's median-of-N decode must stay within this factor of ollama's median.
    floor: f64,
    /// What the number was derived FROM. A floor with no derivation is an invention.
    derived_from: &'static str,
}

/// One entry per silicon this gate has been calibrated on. **Absence is meaningful**
/// and is handled explicitly — see `UNCALIBRATED` in the assertion below. Adding a
/// silicon here requires the derivation, not just the number.
const SILICON_FLOORS: &[SiliconFloor] = &[SiliconFloor {
    compute_cap: "8.9",
    arch: "sm_89 (RTX 4090)",
    floor: 0.90,
    derived_from: "four measurements 2026-06-15..2026-07-31 on lambda-vector; worst \
                   observed median 1.015, and 0.90 sits 12% under it so it does not \
                   flake, while still catching the CPU-SIMD collapse at ratio ~0.065",
}];

/// The four GB10 executions, recorded as DATA and deliberately not turned into a
/// floor. ollama is stable within 0.8% across them, so this is a reproducible
/// measurement of apr on this silicon and not a noisy rig:
///
/// ```text
///   date        apr median-of-7   ollama median   ratio
///   2026-08-30      105.6             182.3       0.579
///   2026-08-30      117.9             182.3       0.647
///   2026-08-31      116.0             181.8       0.638
///   2026-09-01      112.0             180.9       0.619
/// ```
///
/// Whether that is an honest GB10 number or a real sm_121 decode deficit is OPEN
/// (aprender#2835, and #2800 argues the GB10 shortfall on #2786 is a real deficit).
/// The two answers imply opposite fixes — recalibrate, or fix the kernel — and this
/// gate must not pick one by accident, which is exactly what inheriting 0.90 did.
const GB10_OBSERVED: &[f64] = &[0.579, 0.647, 0.638, 0.619];

/// Retained as the sm_89 floor's spelling for the contract mirror and the unit test
/// below. Reading it directly in an assertion is what this fix removes.
const ENFORCED_THRESHOLD: f64 = 0.90;

/// Below this ratio the CPU-SIMD fallback hypothesis is live (that collapse measures
/// ~0.065). Above it, offering that hypothesis is naming a cause the measurement
/// already excludes — which the previous failure message did at 0.619.
const CPU_FALLBACK_RATIO_CEILING: f64 = 0.20;

/// Look up the floor for a compute capability. `None` means UNCALIBRATED, which is a
/// distinct outcome from "passes" and from "fails".
fn floor_for(compute_cap: &str) -> Option<&'static SiliconFloor> {
    SILICON_FLOORS
        .iter()
        .find(|f| f.compute_cap == compute_cap.trim())
}

/// Ask the driver which silicon this actually is. Never inferred from the runner
/// label: a label is a claim about provisioning, and this gate has already been
/// wrong once about which host it was running on.
fn detect_compute_cap() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let first = s.lines().next()?.trim().to_string();
    if first.is_empty() {
        None
    } else {
        Some(first)
    }
}

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
            stall fix) + ollama + Q4_K_M model; runs nightly on gx10 (GB10) via cuda-nightly.yml; status=enforced, see contract"]
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
    // THREE OUTCOMES, NOT TWO. A floor belongs to the silicon it was derived on, so
    // before comparing anything we ask which silicon this is. "No entry" is its own
    // verdict: not a pass (that is the `ada-4090 only` skip that hid this gate for
    // months) and not a failure against a number nobody derived here (that is what
    // has been happening since 2026-08-30).
    let cap = detect_compute_cap();
    let cap_str = cap
        .clone()
        .unwrap_or_else(|| "<nvidia-smi did not answer>".to_string());

    let Some(known) = cap.as_deref().and_then(floor_for) else {
        panic!(
            "UNCALIBRATED-SILICON beat-ollama-decode-throughput: this host reports compute_cap \
             {cap_str}, which has NO calibrated no-collapse floor in SILICON_FLOORS. Measured \
             here: apr median-of-{APR_MEDIAN_N} {apr_med:.1} tok/s vs ollama median \
             {ollama_med:.1} tok/s (ratio_median {ratio_med:.3}).\n\n\
             This is NOT a statement that apr regressed. It is a refusal to assert a threshold \
             derived on other hardware. Every number in this file's header - the four \
             measurements, the 1.015 worst median, the ~300 tok/s incumbent - is sm_89, and \
             #2740 re-targeted this step to a new silicon by deleting the matrix leg its \
             predicate named. Removing a matrix leg re-targets every step whose predicate \
             named it.\n\n\
             To resolve, do ONE of: (a) add a SILICON_FLOORS entry for {cap_str} WITH its \
             derivation - samples, not a number picked to make this green; or (b) restore a \
             host predicate so this step runs only where it is calibrated; or (c) if the \
             deficit is real, fix it and say so - #2800 argues the GB10 shortfall on #2786 is \
             a real deficit, and GB10_OBSERVED in this file records {n_obs} nights at \
             {obs_lo:.3}-{obs_hi:.3} with ollama stable within 0.8%. (a) and (c) are opposite \
             conclusions; this gate must not pick one by accident. See aprender#2835. \
             (contract beat-ollama-decode-throughput-speed-v1.yaml)",
            n_obs = GB10_OBSERVED.len(),
            obs_lo = GB10_OBSERVED.iter().copied().fold(f64::INFINITY, f64::min),
            obs_hi = GB10_OBSERVED
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        );
    };

    // The diagnosis is now CONDITIONAL on the measurement, because the previous one was
    // not. It said "at this depth apr is very likely not decoding on the GPU at all" and
    // offered the CPU-SIMD fallback first - while its own numbers read 112 tok/s against
    // a fallback that measures ~20. A failure message that names a cause its own
    // arithmetic excludes is worse than one that names none; that lesson is three
    // paragraphs up in this file and was not applied to the ladder itself.
    let diagnosis = if ratio_med < CPU_FALLBACK_RATIO_CEILING {
        "At this depth apr is very likely not decoding on the GPU at all. Check, in order: \
         (1) is the F2 first-token gate REJECTING the CUDA path (look for 'GPU diverges from \
         CPU' / cosine < 0.95)? A rejected CUDA path falls to wgpu, then to CPU SIMD, and \
         measures ~20 tok/s - ratio ~0.065. (2) is APR_BIN built --features cuda? (3) is \
         HW_DP4A_Q4K set, re-selecting the degraded kernel #2323 removed as the default?"
    } else {
        "This ratio is well ABOVE the CPU-SIMD fallback band (~0.065), so apr IS decoding on \
         the GPU and the CPU-fallback hypotheses do not apply - do not spend time on them. \
         This is a real throughput deficit against the incumbent on calibrated silicon: \
         compare the kernel actually selected (apr trace / APR_BIN --features cuda) against \
         the one this floor was derived under."
    };

    assert!(
        ratio_med >= known.floor,
        "DECODE-COLLAPSE beat-ollama-decode-throughput on {arch} (compute_cap {cap_str}): apr \
         median-of-{APR_MEDIAN_N} decode {apr_med:.1} tok/s is below {floor:.2}x ollama's median \
         {ollama_med:.1} tok/s (ratio_median {ratio_med:.3}). This floor is NOT a beat threshold; \
         it was derived as: {derived}. {diagnosis} \
         (contract beat-ollama-decode-throughput-speed-v1.yaml)",
        arch = known.arch,
        floor = known.floor,
        derived = known.derived_from,
    );
}

// --- Pure-CPU unit tests for the parsing helpers (these DO run in normal CI) ---

#[test]
fn sm_89_is_calibrated_and_carries_its_derivation() {
    let f = floor_for("8.9").expect("sm_89 must stay calibrated; it is the only derived floor");
    assert!(
        (f.floor - ENFORCED_THRESHOLD).abs() < f64::EPSILON,
        "the sm_89 entry and ENFORCED_THRESHOLD must not drift apart"
    );
    assert!(
        !f.derived_from.is_empty(),
        "a floor with no derivation is an invention, not a calibration"
    );
}

#[test]
fn every_floor_states_what_it_was_derived_from() {
    // The rule that makes this table different from a constant. A number may be added
    // here only with the samples behind it; that is what stops the next silicon
    // inheriting a figure nobody measured on it.
    for f in SILICON_FLOORS {
        assert!(
            !f.derived_from.is_empty(),
            "SILICON_FLOORS entry {} has no derivation",
            f.arch
        );
        assert!(
            f.floor > 0.0 && f.floor < 2.0,
            "SILICON_FLOORS entry {} has an implausible floor {}",
            f.arch,
            f.floor
        );
    }
}

#[test]
fn gb10_is_deliberately_uncalibrated() {
    // GB10 reports compute_cap 12.1. If someone adds an entry for it, this test must be
    // the thing that makes them justify it — deleting this test is the visible act.
    assert!(
        floor_for("12.1").is_none(),
        "GB10/sm_121 has no derived floor. Four nights is data, not a calibration (S8: a \
         threshold comes from samples, never from invention). If you are adding one, bring \
         the derivation and update GB10_OBSERVED and aprender#2835."
    );
    assert!(
        floor_for("<nvidia-smi did not answer>").is_none(),
        "an unanswered probe must never resolve to a floor"
    );
}

#[test]
fn inheriting_the_sm_89_floor_would_have_failed_every_gb10_night() {
    // This is the regression the calibration table exists to prevent, stated as an
    // assertion rather than as a comment. Every recorded GB10 night violates the sm_89
    // floor — which is precisely why asserting it there produced four red nights that
    // read as an apr regression rather than as a re-targeted gate.
    assert!(
        !GB10_OBSERVED.is_empty(),
        "the observations must not be emptied"
    );
    for r in GB10_OBSERVED {
        assert!(
            *r < ENFORCED_THRESHOLD,
            "GB10 observation {r} is at or above the sm_89 floor; if that is now true the \
             premise of aprender#2835 has changed and this file needs re-reading"
        );
    }
}

#[test]
fn the_cpu_fallback_diagnosis_does_not_fire_on_the_observed_gb10_band() {
    // The message defect: at ratio 0.619 the old text led with "very likely not decoding
    // on the GPU at all", whose own cited signature is ~0.065. The ceiling must sit
    // ABOVE the collapse band and BELOW every observed GB10 ratio, or the diagnosis is
    // wrong in one direction or the other.
    const DOCUMENTED_CPU_COLLAPSE_RATIO: f64 = 0.065;
    // A const block, so this half is a COMPILE failure rather than a test failure: the
    // ceiling dropping below the collapse it exists to catch should not be something you
    // can build and ship while the suite happens not to run.
    const {
        assert!(
            CPU_FALLBACK_RATIO_CEILING > DOCUMENTED_CPU_COLLAPSE_RATIO,
            "the ceiling must still admit the collapse it was written to catch"
        );
    }
    for r in GB10_OBSERVED {
        assert!(
            *r > CPU_FALLBACK_RATIO_CEILING,
            "observed GB10 ratio {r} would trigger the CPU-fallback diagnosis, which its own \
             magnitude excludes"
        );
    }
}

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
fn enforced_threshold_is_a_no_collapse_floor_not_a_beat() {
    // WAS `enforced_threshold_is_a_real_beat_with_margin`, and it asserted
    // `thresh > 1.0` against provenance (1.371x median, 1.230x worst run) that
    // this file's own header WITHDREW on 2026-07-31. ENFORCED_THRESHOLD has been
    // 0.90 since, so the assertion was false — and nobody saw it, because no
    // workflow executed this target: `cargo test --lib` does not reach an
    // integration test, and the only site that named it (cuda-nightly's ada-4090
    // leg) ran the ignored beat, not these unit tests.
    //
    // Measured on 2026-08-29, running the target directly:
    //     test enforced_threshold_is_a_real_beat_with_margin ... FAILED
    //     panicked at 'enforced threshold must be a real beat (> 1.0x)'
    //
    // The header even claimed "The CPU unit tests below DO run in CI." They did
    // not. So this now asserts what the constant actually means.
    //
    // `black_box` defeats const-folding so clippy does not flag a constant
    // assertion.
    let thresh = std::hint::black_box(ENFORCED_THRESHOLD);
    // The failure this floor exists to catch: decode falling back to CPU SIMD,
    // measured at ratio ~0.065 (a 14x violation of this floor).
    let cpu_fallback_collapse_ratio = std::hint::black_box(0.065_f64);
    // The worst honestly re-measured GPU run on sm_89 (2026-07-31, idle box).
    let worst_measured_gpu_ratio = std::hint::black_box(1.015_f64);

    assert!(
        thresh < worst_measured_gpu_ratio,
        "the floor must sit UNDER the worst measured GPU run, or it fails on a \
         healthy night and teaches everyone to ignore it"
    );
    assert!(
        thresh > cpu_fallback_collapse_ratio,
        "the floor must sit ABOVE the CPU-fallback collapse, or it cannot catch \
         the one failure it exists for"
    );
    assert!(
        thresh <= 1.0,
        "this is a NO-COLLAPSE FLOOR, not a beat: a threshold above 1.0 asserts a \
         win apr does not currently have, and the 1.371x claim it came from was \
         withdrawn (see this file's header)"
    );
}
