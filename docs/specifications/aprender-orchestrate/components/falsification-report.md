# Cross-Specification Falsification Report

> Methodology: Popperian falsification (arXiv:2502.09858)
> Scope: All specs added 2026-04-02 (apr-code, multi-provider-api, agent-and-playbook extensions, presentar-probar-integration)
> Status: **12 contradictions found, 8 unfalsifiable claims, 6 missing failure modes, 4 circular dependencies**

---

## 1. CONTRADICTIONS BETWEEN SPECS

### C-001: Sovereign tier vs auto-compact summarization

**Specs in conflict:** agent-and-playbook.md §7.2 vs apr-code.md §5.4

Auto-compact at 80% context "summarizes history into condensed form." Summarization requires an LLM call. Under Sovereign tier with an 8K context model (the minimum per apr-code §5.2), the model hits 80% at ~6.4K tokens — roughly 15 tool calls. The agent then needs to summarize using the same small model that is already at capacity.

**Problem:** Summarization quality from a 3B model at near-capacity is likely garbage. The spec assumes compaction is free and high-quality; it's neither.

**Fix required:** Spec must define compaction strategy for small local models. Options: (a) extractive compaction only (no LLM call, just drop old tool results), (b) require 32K+ for agentic use, (c) admit Sovereign tier with 8K models cannot sustain multi-turn conversations.

---

### C-002: "Single binary" vs batuta + presentar-terminal + trueno-rag + pmat + renacer

**Specs in conflict:** apr-code.md §1 ("Single binary, no npm, no Python, no Docker") vs apr-code.md §2 (9 crate dependencies)

The claim is "one Rust binary." But `pmat` is a separate binary (`pmat query`). `renacer` is a separate binary. The spec lists `pmat_query` as a builtin tool (§4.1) but pmat is invoked via shell. Same for renacer. These are external binaries, not linked libraries.

**Problem:** "Single binary" is false if the agent shells out to `pmat`, `renacer`, `cargo`, `git`, etc. The binary is single but the system is not self-contained.

**Fix required:** Either (a) embed pmat/renacer as library dependencies (not CLI), (b) redefine the claim as "single primary binary with optional stack tools," or (c) accept graceful degradation when stack tools are missing.

---

### C-003: Pricing table in two locations

**Specs in conflict:** multi-provider-api.md §6.3 (`~/.config/batuta/pricing.toml`) vs apr-code.md §10.1 (`~/.apr/config.toml`)

Config lives under `~/.config/batuta/` in the multi-provider spec but `~/.apr/` in the apr-code spec. Which is authoritative? The agent runtime is batuta but the user-facing tool is `apr`.

**Problem:** Two config hierarchies = confused users, split-brain settings.

**Fix required:** Pick one. Recommendation: `~/.apr/` is the user-facing config root (matches `apr` CLI), with `batuta` reading from it. Or use XDG (`~/.config/apr/`).

---

### C-004: Session storage in two locations

**Specs in conflict:** agent-and-playbook.md §11.1 (`.batuta/sessions/`) vs apr-code.md §6.1 (`~/.apr/sessions/`)

Same problem as C-003 but for session data.

**Fix required:** One canonical location. `~/.apr/sessions/` since `apr code` is the user-facing command.

---

### ~~C-005: Default privacy tier contradiction~~ — RESOLVED

**Resolution (PMAT-111):** `apr code` is now Sovereign-only by design. No tier selection, no remote providers. The contradiction was eliminated by removing multi-provider support from apr code entirely. The multi-provider-api.md spec still applies to `batuta agent` (general-purpose agent runtime) but NOT to `apr code`.

---

### C-006: Failover cascade skips retryable errors vs always continuing

**Specs in conflict:** multi-provider-api.md §8 RoutingDriver code vs §5.3 failover triggers

The RoutingDriver code shows `Err(e) if e.is_retryable() => continue` — this continues to the NEXT provider on retryable errors. But §5.1 says we should RETRY the same provider with backoff first. The code skips retry and goes straight to failover.

**Problem:** The backoff spec (§5.1) is never invoked in the routing code (§8). They're architecturally disconnected.

**Fix required:** The RoutingDriver must retry with backoff BEFORE cascading. The code should be: retry with backoff on same provider → if retries exhausted → cascade to next.

---

### ~~C-007: Context window exceeded as failover trigger is dangerous~~ — RESOLVED

**Resolution (PMAT-111):** For `apr code`, there is no failover — only compaction. When context exceeds the local model's window, extractive compaction drops old tool results (agent-and-playbook.md §7.2). The sovereignty guarantee is maintained. The multi-provider failover in the general agent runtime still has the guard from the earlier fix (C-007 in multi-provider-api.md).

---

### C-008: 60fps frame budget contradicts streaming token rendering

**Specs in conflict:** presentar-probar-integration.md §3.3 ("Zero heap allocations in steady state") vs §6.2 (BrickHouse 16ms budget)

Streaming tokens arrive at ~50ms intervals (§3.1). Each token appends to `VecDeque<String>` which allocates. The "zero heap allocation" claim applies to the renderer, but token accumulation IS heap allocation. Also, markdown state machine processing on each token is not O(1).

**Problem:** "Zero heap alloc in steady state" is misleading when every token is a heap allocation.

**Fix required:** Clarify that zero-alloc applies to the render path only, not the data path. The token VecDeque should use a pre-allocated ring buffer with fixed capacity.

---

### C-009: Pixel coverage >= 80% threshold is arbitrary and untested

**Specs in conflict:** presentar-probar-integration.md §5.1 (80% threshold) vs agent-ux-v1.yaml (same)

Why 80%? No justification. The 20% uncovered area could be the exact region where bugs hide. Probar's own thresholds define STANDARD=80% and HIGH=90%, but the spec doesn't explain why STANDARD is sufficient.

**Problem:** Unfalsifiable threshold. We can't prove 80% is enough without knowing what the 20% uncovered area contains.

**Fix required:** Either justify 80% empirically (e.g., "80% covers all interactive regions; remaining 20% is borders/padding") or increase to 90% with explicit exclusion list for non-interactive regions.

---

### C-010: Playbook mutation testing claims 100% kill rate

**Specs in conflict:** presentar-probar-integration.md §4.3 ("Expected: 100% mutation kill rate") vs reality

100% mutation kill rate is aspirational, not a spec. M5 (assertion weakening) mutations that weaken a strong assertion to a slightly-less-strong assertion are often not killed by tests that only check the weakened version. Claiming 100% as the target means any single surviving mutant is a spec violation.

**Problem:** Setting 100% as the pass criteria means the spec can never be met in practice. This makes the QA gate unfalsifiable (it always "fails" trivially).

**Fix required:** Set realistic target (e.g., >=90% mutation kill rate). Document which mutation types are expected to survive and why.

---

### C-011: Format translation "≈" (approximately equal) is unfalsifiable

**Specs in conflict:** multi-provider-api.md §14.1 and provider-routing-v1.yaml

The format translation equation uses `≈` (approximately equal). But the invariant says "Content text identical after round-trip." These contradict — is it exact or approximate? For tool arguments, OpenAI serializes as JSON string while Anthropic uses JSON object. `{"a": 1}` vs `"{\"a\": 1}"` — are these "approximately equal"?

**Problem:** `≈` is not a testable predicate. The falsification test (FALSIFY-MPA-005) asserts "content equality" but the equation says "approximately."

**Fix required:** Define equality precisely. Content text: byte-identical. Tool arguments: JSON semantically equivalent (parse both, deep compare). Role mapping: exact. Drop `≈`, use `==` with defined comparison semantics per field.

---

### C-012: Hook ordering vs parallel tool execution

**Specs in conflict:** agent-and-playbook.md §9.1 (hooks run per tool call) vs §8.1 (parallel tool execution)

If 5 tools execute in parallel and each has pre/post hooks, do the hooks interleave? Can a pre-hook on tool A observe the post-hook result of tool B (which started simultaneously)? The spec says hooks are "per tool call" but doesn't define cross-tool hook ordering in parallel context.

**Problem:** Hook ordering is well-defined for sequential execution but undefined for parallel. An audit log hook might record events in non-deterministic order.

**Fix required:** Define that hooks are per-tool-call scoped. Pre-hooks run before the specific tool. Post-hooks run after. Cross-tool ordering is non-deterministic but each tool's hook chain is atomic. Audit log must include timestamps, not rely on ordering.

---

## 2. UNFALSIFIABLE CLAIMS

### U-001: "Stack-native tools are preferred because they return richer metadata"

apr-code.md §4.2. This is a design preference, not a testable claim. What is "richer"? How much richer justifies the complexity? FALSIFY-AC-008 tests relevance but not "richness."

**Fix:** Replace with measurable claim: "pmat query returns TDG grade + complexity for every result; grep returns none."

### U-002: "Toyota Production System principles applied"

Multiple specs. Labeling design patterns with TPS names (Jidoka, Poka-Yoke, etc.) is marketing, not engineering. The principles are metaphors, not testable properties. "Jidoka: stop-on-error" is just error handling. Calling it Jidoka doesn't make it more correct.

**Fix:** Keep the table for design philosophy documentation but remove from falsification sections. Only testable properties matter.

### U-003: "Deep integration with all 20+ PAIML crates"

apr-code.md §1. "Deep" is subjective. Phase 1 only uses 3 crates (batuta, presentar-terminal, realizar). Full integration is Phase 3+.

**Fix:** Replace with "integrates with N crates at launch, M total planned."

### U-004: "Best quality" for Claude Sonnet 4

apr-code.md §5.3. This is a marketing claim, not a spec property. Quality depends on the task.

**Fix:** Remove quality column or replace with measurable: "Highest SWE-bench score among supported models."

### U-005: "Deterministic IDs" for tool call translation

multi-provider-api.md §3.3. "Generate deterministic IDs" — deterministic from what input? If from content hash, two identical tool calls get the same ID (collision). If from sequence number, the translation is stateful and non-pure.

**Fix:** Specify the ID generation scheme: `blake3(provider_id || message_index || tool_index)`.

### U-006: "Tests ARE the interface"

presentar-probar-integration.md §6. This is probar's philosophy, not a falsifiable property of our spec. The claim is that Brick assertions enforce correctness, but the assertions themselves could be wrong (a Brick that asserts `ContrastRatio(4.5)` but measures contrast incorrectly).

**Fix:** Acknowledge that Brick assertions are only as good as their implementation. Add: "Brick verification itself is tested by probar property tests."

### U-007: "Crash recovery is lossless"

agent-and-playbook.md §16.1. JSONL append is not lossless if the OS doesn't flush to disk. A kill -9 between write() and fsync() loses the last message. The spec says "truncate to last complete" but doesn't specify fsync policy.

**Fix:** Specify: each message append followed by fsync(). Accept that the last message MAY be lost (at-most-once delivery). "Lossless" means "no corruption" not "no data loss."

### U-008: "Zero-cost when no streaming is active"

apr-code.md §14.2. The TUI renders a status bar and session info even when idle. "Zero-cost" is unmeasurable. Does a 0.1ms idle render count as zero?

**Fix:** Replace with measurable: "<0.5ms render time when no streaming active."

---

## 3. MISSING FAILURE MODES

### F-001: What happens when Landlock is unavailable?

Kernel <5.13 doesn't support Landlock. The spec (agent-and-playbook.md §10.1) lists "Linux 5.13+" but doesn't define behavior on older kernels. Does the agent refuse to start? Run unsandboxed? Warn and continue?

**Fix:** Define: kernel <5.13 → warn + run with application-level-only sandbox (capability + hooks). Log that kernel sandbox is inactive.

### F-002: What happens when no model is available?

apr-code.md §5.1 lists 4 search locations but doesn't define behavior when ALL are empty and `--offline` is set. Exit code 5 is defined but the user experience is not.

**Fix:** Define: print clear error message with instructions to download a model (`apr pull qwen3:1.7b-q4k`). Suggest smallest viable model. **DONE** — implemented in `print_no_model_error()` (PMAT-113).

### F-003: What happens when APR.md and CLAUDE.md conflict?

apr-code.md §3.4 lists discovery order (APR.md preferred over CLAUDE.md) but doesn't define behavior when both exist and contradict. E.g., APR.md says `Tier: Sovereign`, CLAUDE.md says nothing about tier.

**Fix:** Define: APR.md takes precedence. CLAUDE.md instructions merged only if they don't conflict. Log which file contributed which instructions.

### F-004: What happens when context compaction drops relevant context?

agent-and-playbook.md §7.3. The spec says micro-compact "strips tool result bodies." But a tool result might contain critical information the agent needs for its next decision. The spec has no mechanism for the agent to realize it lost context.

**Fix:** After compaction, re-inject a "compaction summary" that lists what was dropped: "Tool results from turns 1-15 were compacted. If you need details, re-run the tool." This lets the agent self-recover.

### F-005: What happens on provider mid-stream failure?

multi-provider-api.md §4. The SSE parser handles `Error` events but doesn't define behavior when the HTTP connection drops mid-stream (no `[DONE]`, no error, just TCP RST). Partial tool use JSON is left incomplete.

**Fix:** Define: on connection drop, parser emits `Error { message: "stream interrupted" }`. Agent retries the turn from scratch (not from partial). Partial tool use JSON is discarded, not attempted to parse.

### F-006: What happens when BrickHouse budget is exceeded?

presentar-probar-integration.md §6.2 says "Jidoka: if any brick exceeds budget, entire frame fails." But "frame fails" means... what? Blank screen? Previous frame frozen? Error message?

**Fix:** Define: frame failure renders the PREVIOUS frame (frozen) + a status bar message "Frame budget exceeded (23ms > 16ms)." The next frame attempts normally. After 10 consecutive failures, degrade to Minimal layout.

---

## 4. CIRCULAR DEPENDENCIES

### D-001: apr-cli depends on batuta, batuta depends on aprender

apr-code.md §8.3: `apr-cli` depends on `batuta`. But batuta's Cargo.toml already depends on `aprender`. This creates: `aprender → (workspace) → apr-cli → batuta → aprender`. Cargo handles workspace cycles, but this means `apr code` can only be built from the aprender workspace, not standalone.

**Fix:** Acknowledge this is an intentional workspace dependency, not a circular crate dependency. The `code` feature flag on apr-cli optionally depends on batuta, which already depends on aprender. This works because Cargo resolves at the crate level, not workspace level. Document: "apr-cli `code` feature requires building from a workspace that includes both aprender and batuta."

### D-002: Probar tests the agent; the agent uses probar's Brick trait

presentar-probar-integration.md: probar tests the agent TUI. But the agent TUI implements probar's Brick trait from presentar-core. So probar tests code that depends on probar's abstractions. This is conceptually circular (probar defines the interface AND verifies it).

**Fix:** This is actually fine — probar defines `Brick` in presentar-core (not in probar itself). The testing layer (jugar-probar) is a dev-dependency. Clarify in the spec that Brick trait lives in presentar-core, not probar.

### D-003: Contract depends on contract

apr-code-v1.yaml `depends_on: [agent-loop-v1, provider-routing-v1, agent-ux-v1]`. agent-ux-v1.yaml `depends_on: [agent-loop-v1]`. This is fine as a DAG but means apr-code-v1 cannot be verified without first verifying all three dependencies. If agent-loop-v1 has a bug, apr-code-v1 inherits it silently.

**Fix:** Make dependency chain explicit in QA gate: "apr-code-v1 QA gate runs ONLY after agent-loop-v1, provider-routing-v1, and agent-ux-v1 QA gates pass."

### D-004: Sovereignty verified by renacer, but renacer requires running the agent

apr-code-v1.yaml: "Zero network syscalls verified by renacer." But renacer traces the running process. To verify sovereignty, you must run the agent under renacer, which itself may make syscalls (ptrace attach, /proc reads). The trace must distinguish renacer's syscalls from the agent's.

**Fix:** Renacer traces the agent as a child process. Filter renacer's own syscalls (ptrace, /proc) from the trace. Assert: child process network syscalls == 0. Document that renacer itself is trusted computing base.

---

## 5. RECOMMENDATIONS

### Priority 1 (Must fix before implementation)

| ID | Issue | Fix |
|----|-------|-----|
| C-001 | Compaction with small models | Define extractive-only compaction for <16K context |
| ~~C-005~~ | ~~Default Sovereign blocks providers~~ | **RESOLVED** (PMAT-111): apr code is Sovereign-only |
| C-006 | Backoff skipped in routing code | Add retry loop before cascade |
| ~~C-007~~ | ~~Context failover violates sovereignty~~ | **RESOLVED** (PMAT-111): apr code compacts, never failovers |
| F-005 | Mid-stream connection drop | Define: discard partial, retry turn |
| C-011 | Translation `≈` is untestable | Define exact comparison semantics per field |

### Priority 2 (Should fix before v0.2)

| ID | Issue | Fix |
|----|-------|-----|
| C-002 | "Single binary" misleading | Redefine as "single primary binary" |
| C-003/004 | Config/session path split | Unify under `~/.apr/` |
| C-010 | 100% mutation kill rate unrealistic | Target >=90% |
| F-001 | Landlock unavailable | Graceful degradation with warning |
| F-004 | Lost context after compaction | Add compaction summary injection |
| F-006 | BrickHouse budget exceeded behavior | Freeze previous frame + degrade |

### Priority 3 (Nice to fix)

| ID | Issue | Fix |
|----|-------|-----|
| C-008 | Zero-alloc claim misleading | Clarify: render path only |
| C-009 | 80% pixel threshold arbitrary | Justify or increase to 90% |
| C-012 | Hook ordering in parallel | Define per-tool-call atomicity |
| U-005 | Deterministic ID scheme | Specify blake3 hash |
