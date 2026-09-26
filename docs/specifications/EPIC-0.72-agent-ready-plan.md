# EPIC 0.72.0 "Agent Ready": plan (paiml/aprender#4000)

**Status:** plan for operator review. Nothing is applied: no child issues, milestone moves, or closes.
**Ticket:** PMAT-4000 · **kind:** docs · **Ratchet:** slice 3 of 5 of DEBT-RATCHET-001 (#3997, PR #4003)

Baselines come from §6's commands, run 2026-09-23 on `origin/main` @ `49fe19c28`. The operator's approval of the
Anthropic adapter is quoted on #4000 as "yes to 1".

## 1. Exit bar, made measurable

| Bar | Unit | Baseline | 0.72 threshold |
|---|---|---|---|
| **E-1** One conformance suite passes on every model × backend 0.71 certifies | conformance cells = (suite case × model × backend) from the 0.71 ladder universe | **no suite exists.** `contracts/apr-serve-openai-compat-v1.yaml` (622 lines, 57 `- id:` entries) describes the OpenAI surface; there is no Anthropic shape at all | 0 RED cells over the 0.71 certified matrix, **and at least one case per row R-1..R-5 per shape** (quorum fix: an empty suite would otherwise pass). The Anthropic cases live in a new `contracts/apr-serve-anthropic-messages-v1.yaml` |
| **E-2** An agent harness completes a tool-calling task end to end | harness run receipts: (Claude Code via `/v1/messages`) and (an OpenAI-client harness, e.g. Alfredo's) | none | 2 green receipts, each with the gateway's binary sha and the transcript hash |

**What the gateway serves today** (route literals in `crates/aprender-serve/src` + `crates/apr-cli/src`):
`/v1/chat/completions` (184 references), `/v1/completions` (39), `/v1/embeddings` (19), `/v1/models` (16), plus
`/v1/chat/completions/stream`, `/v1/batch/completions`, `/v1/generate`, `/v1/tokenize` and others. **`/v1/messages`:
0 server routes.** It appears only in client code (`crates/aprender-orchestrate/src/agent/driver/remote.rs` builds Anthropic
requests), so the codebase already speaks the shape as a client, which the adapter can reuse.

## 2. Rows

| Row | Item | done_when | Baseline (measured) | First-green proof |
|---|---|---|---|---|
| **R-1** | `/v1/messages` Anthropic Messages adapter in `apr serve` (non-streaming + SSE `message_start … message_stop`, `tool_use`/`tool_result` blocks, `stop_reason`) | the conformance suite's Anthropic cases pass against `apr serve` | 0 server routes | the Anthropic SDK's own request fixtures, replayed against `apr serve`: green. The same fixtures against a build with the route removed: RED (404) |
| **R-2** | #3825 tool calling: `tools`/`tool_choice` forwarded, `tool_calls` returned (also over `--ollama-compat`) | the conformance tool-call cases green on the OpenAI and Anthropic shapes **and on `--ollama-compat`** | OPEN | a tool-call case where the model must call `get_weather`: green; the same case with `tools` stripped by a planted mutant: RED |
| **R-3** | Reproducible sampling: #3760 (safetensors/AprTransformer never samples), #3786 (APR Q4K GPU chat seeds from a wall-clock hash), #3754 (`--temperature` alone is greedy) | the suite's seed case: same seed → byte-identical output (×3 runs); different seed at T>0 → different output; **and #3754's case: `--temperature 0.8` given alone (no `--top-k`) must sample, i.e. two seeds differ** | 3/3 OPEN (0.69.1 milestone) | both halves of the seed case on each certified backend. **The different-seed half is the positive control**: without it, a greedy engine passes the same-seed half |
| **R-4** | Honest telemetry: #3718 (`prompt_tokens`/`completion_tokens`), #3981 (`tok_per_sec` includes model load + F2 validation), #3598 row 1 | `usage` present and exact on both shapes; tok/s excludes load time | 3/3 OPEN | a case with a known prompt token count asserts the exact number. A 10× load-time delay (planted) must not move tok/s by more than the noise band |
| **R-5** | Serve surface: #3979 (`.apr` routes, `GET /`, SSE `[DONE]`/`finish_reason`), #3978 (`apr code`), #3987 qwen3moe verbs if still open | the suite's streaming cases (`[DONE]` last, `finish_reason` set) green on `.apr`, GGUF and safetensors routers | 2/2 OPEN (+ #3987) | the streaming case per router; a router with `[DONE]` removed must go RED |
| **R-6** | The conformance suite itself, **in CI** | a CI job (CPU, small model) runs the suite on every PR that touches `aprender-serve`; the certified matrix runs nightly on the GPU hosts | not built | the first PR run is green; a mutant that drops `finish_reason` is RED in CI |
| **R-7** | MCP server (#2794) | `apr mcp` passes the MCP inspector's conformance checks | OPEN (0.70.0 milestone) | inspector run: green; a tool with a malformed schema: RED |
| **R-8** | One gateway per host, pinned by binary sha; CRUX measurement uses a dedicated instance | a `GET /` or `/health` response carries `binary_sha256`; consumers refuse a gateway whose sha differs from the one they expect | not built | a consumer pointed at a gateway with the wrong sha refuses (RED); the right sha passes |
| **R-9** | **Ratchet slice 3 of 5** | the DEBT-RATCHET-001 slice-3 gates | see #4003 | see #4003 |

## 3. Ratchet slice 3 of 5 (from #4003 §3, proposed)

| Pillar | 0.72 floor |
|---|---|
| A: P₀ bp | ≥ 9,146; `P_cuda` ≥ `B_cuda + 1·s_cuda` |
| B-1: E2 call sites | ≥ 307 (total ≥ 510) |
| B-2: contracts with no falsifier | ≤ 7 |
| C: ONT rows bound | ≥ 22 |
| D-1 / D-2 / D-3 | ≤ 170 / ≤ 7 / ≤ 117 |

## 4. Open questions for the quorum to DECIDE

- **Q1 (the epic's open decision): consumer migration timing.** Should the infra quorum lane, arbiter ask/decide, and
  paiml-implement migrate to the gateway **in 0.72**, or does aprender ship the gateway and suite in 0.72 while the
  other repos migrate after? Plan's recommendation: **aprender ships the gateway + suite in 0.72, and exactly ONE
  consumer migrates in 0.72 as the proof of the client contract** (the infra quorum local lane, the smallest). The rest
  migrate in their own repos' cycles after 0.72, each gated by the same suite. Reasons:
  - a gateway with zero consumers at release is untested as a gateway: E-2 needs a real client;
  - migrating every consumer puts three other repos' release trains on aprender's critical path, and 0.72 cannot
    control their queues;
  - one consumer is enough to find a contract break before the others depend on it.
- **Q2. Streaming shape for tool calls.** OpenAI streams `tool_calls` as argument deltas, and Anthropic streams
  `input_json_delta`. Should the adapter translate at the edge from one internal event stream? Recommendation: yes. One
  internal event stream, two edge encoders, and the suite runs both encoders over the same recorded stream.
- **Q3. Which model runs the CI (CPU) suite?** Recommendation: the smallest certified Qwen with tool-call training
  (qwen3-1.7b-q4km, already a ladder rung). A model that never emits tool calls would make R-2's case vacuous.

## 5. Out of scope

New model support (0.74); performance (0.73).

## 6. Commands

```bash
grep -rn '/v1/messages' crates/aprender-serve/src          # no hits (no server route)
grep -rhoE '"/v1/[a-z_/{}:]+"' crates/aprender-serve/src crates/apr-cli/src | sort | uniq -c | sort -rn
grep -cE '^\s*- id:' contracts/apr-serve-openai-compat-v1.yaml     # 57
for i in 3825 3760 3786 3754 3718 3981 3598 3979 3978 3987 2794; do gh issue view $i -R paiml/aprender --json state,milestone; done
```


## Quorum record: decision quorum, 2026-09-23 (aprender-cb)

**Lanes (ADVISORY: single family, all gemini):** gemini-3.1-pro-high, gemini-3.8-flash-high, gemini-3.7-flash-high,
all returning PASS-with-changes. gpt-oss returned 429. 3/3 exited 3 on foreign fleet ref motion, with every clone
byte-identical. Conversations: `6d4c0e07`, `c593218c`, `60bc18f8`.

| Q | Decision (tally) | Applied as |
|---|---|---|
| **Q1: consumer-migration timing (the operator asked the quorum to decide this)** | **DECIDED 3/3:** the gateway and the conformance suite ship in 0.72; **exactly one consumer, the infra quorum local lane, migrates in 0.72** as the client-contract proof; every other consumer (arbiter, paiml-implement, cookbook, external harnesses) migrates after 0.72 in its own repo's cycle, gated by the same suite | E-2's OpenAI-side receipt comes from the migrated quorum lane |
| Q2 | **one internal event stream, two edge encoders** (OpenAI deltas and Anthropic `input_json_delta`), 3/3 | R-1/R-2 |
| Q3 | **qwen3-1.7b-q4km** for the CPU CI suite, 3/3 | R-6 |

**Must-fix items applied:**
- the orchestrate path;
- E-1's minimum case count;
- the Anthropic contract file is named;
- R-2 covers `--ollama-compat`;
- R-3 covers #3754.

**Must-fix items carried to step 2 as child-issue acceptance:**
- exact `done_when` commands;
- measured behaviour baselines for R-2..R-9 (replacing "OPEN");
- negative controls for R-4's exact token count and R-5's `finish_reason`;
- the slice-3 baseline column and commands (#4003 §7).

