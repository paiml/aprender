# Appendix F: Advanced Popperian Falsification Protocols

> Archived from qwen2.5-coder-showcase-demo.md (lines 8065-8136)

## Appendix F: The Popperian Enhancement - Advanced Falsification Protocols

> "In so far as a scientific statement speaks about reality, it must be falsifiable; and in so far as it is not falsifiable, it does not speak about reality." — Karl Popper

This section elevates our testing methodology from "Verification" (showing it works) to "Falsification" (trying to prove it fails).

### F.1 Bold Conjectures (Theories to Refute)

We posit the following Bold Conjectures. A single counter-example refutes the entire conjecture.

| ID | Conjecture (Hypothesis) | Refutation Condition (Falsifier) | Risk |
|----|-------------------------|----------------------------------|------|
| **C-001** | **The Isomorphism Conjecture:** APR F32 is mathematically identical to SafeTensors F32. | Any single tensor $t$ where $|APR(t) - ST(t)| > \epsilon$ (where $\epsilon = 1e^{-6}$). | **Catastrophic** (Format invalid) |
| **C-002** | **The Determinism Conjecture:** Given fixed seed $S$ and temperature $T=0$, `apr run` produces identical token sequence $K$ on any hardware. | $Output(CPU) \neq Output(GPU)$ or $Output(Run_1) \neq Output(Run_2)$. | **Critical** (Inference untrustworthy) |
| **C-003** | **The Containment Conjecture:** An `.apr` file is fully self-contained and requires no external network or file access. | Any `File::open()` or `http::get()` outside the `.apr` bundle during inference. | **Major** (Design violation) |
| **C-004** | **The Zero-Panic Conjecture:** No input sequence, however malformed, can cause the runtime to panic. | Any panic (SIGABRT, `unwrap()` failure). | **Safety** (DoS vulnerability) |
| **C-005** | **The Linear Scaling Conjecture:** Inference latency $L$ scales linearly with token count $N$ ($O(N)$) for prefill, not quadratically ($O(N^2)$). | $L(2N) > 2.5 \times L(N)$. | **Performance** (KV cache failure) |

### F.2 Active Refutation Protocols (The "Torture" Tests)

We do not just run "happy path" tests. We actively attack the system.

#### R-001: The "Empty Space" Attack (Refuting C-004)
**Hypothesis:** The tokenizer handles whitespace-only prompts correctly.
**Attack:**
```bash
apr run model.apr "   " --max-tokens 10
```
**Falsification:** Panic, infinite loop, or garbage output.
**Current Status:** ✅ CORROBORATED (Returns empty/EOS).

#### R-002: The "Babel" Attack (Refuting C-001)
**Hypothesis:** Tokenizer merges are language-agnostic.
**Attack:**
```bash
apr run model.apr "こんにちは世界" (Japanese)
apr run model.apr "مرحبا بالعالم" (Arabic)
apr run model.apr "👋🌍" (Emoji)
```
**Falsification:** Garbage tokens or replacement characters ``.
**Current Status:** ⚠️ SUSPECT (Needs verification).

#### R-003: The "Amnesia" Attack (Refuting C-005)
**Hypothesis:** KV Cache correctly handles context shifts.
**Attack:**
1. Feed 4096 tokens.
2. Feed 1 token "Therefore,".
3. Check latency.
**Falsification:** If Token 4097 takes > 100ms (re-processing previous 4096), KV cache is broken.
**Current Status:** ✅ CORROBORATED (O(1) generation step verified).

#### R-004: The "Air Gap" Attack (Refuting C-003)
**Hypothesis:** System works without internet.
**Attack:**
```bash
unshare -n apr run model.apr "Test"  # Run in network namespace with no interfaces
```
**Falsification:** Connection error or hang.
**Current Status:** ✅ CORROBORATED (Embedded tokenizer used).

### F.3 The "Stop the Line" Criteria

If any of the following occur, the release is IMMEDIATELY rejected (Status: 🛑).

1.  **Regression of > 10%** in throughput on reference hardware.
2.  **Any Panic** in the Falsification Suite.
3.  **Non-Deterministic Output** at Temp=0.
4.  **License Violation** (e.g., accidental inclusion of non-Apache2 code).


---

