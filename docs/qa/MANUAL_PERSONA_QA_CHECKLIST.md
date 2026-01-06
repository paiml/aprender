# 🎭 Manual Persona QA Checklist: "The Gauntlet"

**Version:** 1.0.0
**Target Repos:** `aprender` (Training/Format) & `realizar` (Inference/Runtime)
**Objective:** Falsify "Production Ready" claims through rigorous, persona-driven manual testing.
**Philosophy:** "If it feels janky, it is broken." - The Gamer's QA Motto

---

## 🕹️ Persona 1: "The Speedrunner" (Performance & Latency)
**Role:** High-frequency trader / Competitive gamer demanding <10ms latency.
**Focus:** `realizar` Inference Engine & `apr serve`

### 🧪 Workflows

1.  **The "Cold Start" Dash**
    *   **Action:** Run `apr serve --model tinyllama.gguf` on a cold boot (clear cache/RAM).
    *   **Expectation:** Server ready in <2s. First token <500ms.
    *   **Falsify If:** Server hangs on "Loading weights...", TTFT > 1s, or fan spins up before prompt.

2.  **The "Button Masher" (Concurrency)**
    *   **Action:** Spawn 50 parallel requests using `ab` or `k6` to `/chat/completions`.
    *   **Expectation:** Zero dropped requests. P99 latency < 200ms.
    *   **Falsify If:** 500 errors, connection resets, or "Out of Memory" panic.

3.  **The "Alt-Tab" (Context Switching)**
    *   **Action:** Load Model A (TinyLlama), run inference. Immediately load Model B (Phi-2). Switch back to A.
    *   **Expectation:** Zero-copy mmap swapping. Instant transition.
    *   **Falsify If:** Full reload delay (>5s) or RAM usage doubles (memory leak).

4.  **The "Frame Perfect" (Token Streaming)**
    *   **Action:** Watch the stdout stream visually.
    *   **Expectation:** Smooth, consistent token pacing (like a typewriter).
    *   **Falsify If:** "Stuttering" (bursts of tokens then silence), erratic delays, or hanging at end-of-text.

5.  **The "Hardware Hacker" (Backend Check)**
    *   **Action:** Run on CPU-only machine, then GPU machine.
    *   **Expectation:** CLI explicitly logs `Backend: CPU (AVX2)` or `Backend: CUDA`.
    *   **Falsify If:** Silent fallback to CPU on a GPU node, or crashing on non-AVX hardware.

---

## 🔬 Persona 2: "The Skeptic Scientist" (Correctness & Reproducibility)
**Role:** ML Researcher verification model integrity.
**Focus:** `aprender` Training & `apr convert`

### 🧪 Workflows

6.  **The "Determinist"**
    *   **Action:** Run `apr train` or `apr chat` with `--seed 42` and `temperature 0` twice.
    *   **Expectation:** Bit-exact identical output/weights.
    *   **Falsify If:** Even a single float/token differs.

7.  **The "Chaos Monkey" (Corrupt Inputs)**
    *   **Action:** Feed a `tokenizer_config.json` with malformed JSON or a 1GB text file as a prompt.
    *   **Expectation:** Graceful `AprenderError::Io` or `Validation` error.
    *   **Falsify If:** Rust panic (`unwrap` on `None`), segfault, or infinite loop.

8.  **The "Template Detective"**
    *   **Action:** Load a model with a *weird* chat template (e.g., custom Jinja with logic).
    *   **Expectation:** `apr inspect` shows the correct template source. Chat follows the logic exactly.
    *   **Falsify If:** Falls back to "Raw" format without warning, or misinterprets `{{ system }}` tags.

9.  **The "Unit Test Purist"**
    *   **Action:** Run the `scripts/verify-chat-models.sh` script.
    *   **Expectation:** All green ticks.
    *   **Falsify If:** Script requires manual intervention (installing `jq`, `curl` failures) or reports "Skip" on standard models.

10. **The "Precision Snob" (Quantization)**
    *   **Action:** Convert `fp32` -> `q4_0` -> `apr chat`.
    *   **Expectation:** Coherent English output.
    *   **Falsify If:** Gibberish output ("The the the..."), NaN values, or sudden context loss.

---

## 🛠️ Persona 3: "The DevOps Janitor" (Deployment & Ops)
**Role:** SRE deploying to Kubernetes/Edge.
**Focus:** `apr-cli` & `Dockerfile`

### 🧪 Workflows

11. **The "Zero-Config" Install**
    *   **Action:** `cargo install apr-cli` -> `apr run hf://...` on a fresh Alpine container.
    *   **Expectation:** Just works. Auto-downloads dependencies.
    *   **Falsify If:** "Library not found (libssl/libtorch)", "GLIBC version too old", or manual path setup needed.

12. **The "Log Diver"**
    *   **Action:** Start server, send bad request, kill process.
    *   **Expectation:** Structured JSON logs. Request IDs. Clean shutdown signal.
    *   **Falsify If:** Multiline stack traces in stdout, unhandled `SIGTERM`, or logs missing timestamps.

13. **The "Network Partition"**
    *   **Action:** Start download, pull network cable (or `iptables` drop). Reconnect.
    *   **Expectation:** Auto-resume or clean error with retry hint.
    *   **Falsify If:** Corrupt file left on disk that crashes next run.

14. **The "Disk Filler"**
    *   **Action:** Run with `APR_CACHE_DIR` on a full partition.
    *   **Expectation:** "No space left on device" error message.
    *   **Falsify If:** Silent corruption or partial writes treated as success.

15. **The "Permission Denied"**
    *   **Action:** Run `apr` as `nobody` user trying to write to `/root`.
    *   **Expectation:** "Permission denied" error.
    *   **Falsify If:** Panic or infinite retry loop.

---

## 🎨 Persona 4: "The Artist" (UX & Aesthetics)
**Role:** Creative coder / Frontend dev.
**Focus:** TUI & CLI Output

### 🧪 Workflows

16. **The "Pixel Peeper" (TUI Glitches)**
    *   **Action:** Resize terminal window rapidly while `apr chat` is streaming.
    *   **Expectation:** Text reflows correctly. No tearing. Input bar stays at bottom.
    *   **Falsify If:** Text overlaps, cursor jumps to wrong line, or raw ANSI codes leak.

17. **The "HelpSeeker"**
    *   **Action:** Run `apr --help` and `apr chat --help`.
    *   **Expectation:** Beautifully formatted, grouped args, examples provided.
    *   **Falsify If:** Missing docs for flags, untyped "string" args where enums exist, or default clap formatting.

18. **The "Progress Watcher"**
    *   **Action:** Download a 70B model.
    *   **Expectation:** Smooth progress bar (PB/s, ETA).
    *   **Falsify If:** Bar jumps from 0% to 100%, spamming new lines, or no update for >10s.

19. **The "Color Blind"**
    *   **Action:** Run with `NO_COLOR=1`.
    *   **Expectation:** Plain text output, distinguishable structure.
    *   **Falsify If:** Information is conveyed *only* by color (e.g., Red=Error, Green=Success with no text prefix).

20. **The "Unicode World"**
    *   **Action:** Chat in Japanese/Emoji. "こんにちは 👋"
    *   **Expectation:** Correct alignment, no width issues.
    *   **Falsify If:** Cursors misaligned (width=2 chars vs width=1), broken encoding.

---

## 🚦 Testing Protocol

1.  **Assign Personas:** Rotate team members daily.
2.  **Log "Fails":** Any hiccup, weird log, or feeling of "jank" is a bug ticket.
3.  **Video Proof:** Record "The Speedrunner" and "The Pixel Peeper" sessions for UI review.
4.  **Verdict:**
    *   🟢 **Gold Master:** All 20 checks pass with zero friction.
    *   🟡 **Beta:** Functional but "janky" (UI glitches, minor lags).
    *   🔴 **Alpha:** Crashes, panics, or manual intervention needed.
