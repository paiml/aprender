# Implementation receipt: PMAT-3720 — the apr response contract

Branch `PMAT-3720-response-contract`, stacked on `PMAT-3723-thinking-derived` (#3723/#3755/#3801
supply the reasoning split this contract reports). Quorum base: `f97cf7866`, the tip of that
branch. The schema is the one accepted on #3720 (issuecomment-5767765474), pinned as
`contracts/apr-response-contract-v1.yaml` (`pv validate`: 0 errors; `pv lint`: PASS, pv 0.69.0
built from this tree). The fragment `docs/roadmaps/entries/PMAT-3720.yaml` holds the issue's
done_when verbatim.

## done_when 1 — weights digest

- `apr run --json` / `--stream`: `model_digest` is the sha256 (lower-case hex, what `sha256sum`
  prints) of the file `run_model` resolved. It is hashed on its own thread beside inference, only
  for a machine-readable run. The document also carries `apr_version` and `apr_git_sha`. A path
  that is not one file (a model directory) reports `null`, not a wrong digest.
- `apr serve`: the model is hashed ONCE at startup, from the GGUF's mapped bytes (exactly what was
  loaded) or from the APR/SafeTensors file, and printed as "Model sha256: … (Ns)". A router layer
  puts `x-apr-model-digest` on EVERY response, streaming ones too. Completion bodies carry
  `model_digest`, `apr_version` and `apr_git_sha`. A server given no identity sends none.
- Case rows: the digest equals `sha256sum` of the file on every hardware row, in the run document,
  in the serve body and in the serve header.

## done_when 2 — seed determinism

Three samplers drew from process entropy. Each now draws from an RNG seeded by the request:

| sampler | used by | before |
|---|---|---|
| `OwnedQuantizedModelCuda::next_token` (resident decode loop) | `apr run --gpu`, serve's direct CUDA path, streaming | `sample_topk`: unseeded, and it dropped `top_p` |
| `generate_gpu_resident_logprobs` | serve with `logprobs` | `sample_topk`: unseeded |
| `select_batched_token` (continuous batching) | `apr serve --gpu`'s scheduler | `rand::rng()` |

Completions carry `deterministic`, plus `nondeterminism_reason` when it is false. False, with the
reason stated, for the wgpu `GpuModel` (its `GpuGenerateConfig` carries no seed) and for a
scheduler that admits more than one request at a time. `apr serve --gpu`'s default is
`max_batch=1`.

**Measured, sampling ENGAGED** (the sampled text differs from greedy on every row; `apr run`'s
`--top-k` defaults to 1 = greedy, so the first measurement, which passed only `--temperature`,
was greedy and its "identical" proved nothing). The same model, prompt, seed and sampling were run
twice, greedy and `temperature 0.7 / top_k 40 / top_p 0.9 / seed 42`:

| host / backend | models | apr run | apr serve |
|---|---|---|---|
| lambda RTX 4090, CUDA (`3f0cdf588`) | Qwen3-8B, qwen2.5-coder-1.5b | byte-identical ×2 | byte-identical ×2 |
| lambda Threadripper 7960X, CPU (`e638060e8`) | Qwen3-1.7B, qwen2.5-coder-1.5b | byte-identical ×2 | byte-identical ×2 |
| gx10 GB10, CUDA (`e638060e8`) | Qwen3-8B, qwen2.5-coder-1.5b | **NOT MEASURED** | **NOT MEASURED** |

Before the fix (`075203530`, same flags), the sampled rows DIFFERED between the two runs on both
models, for run and for serve.

**gx10 is unmeasured.** The binary is built there (`e638060e8`,
`/mnt/nvme-raid0/agent-wt/pmat-3720-gx10`) and the run was queued at `gpu-q --prio 1`, but it was
still behind two tickets at the wind-down and never started. It writes to
`/mnt/nvme-raid0/agent-wt/pmat-3723-hw/det-gx10` if it runs; judge it with
`python3 scripts/seed_determinism_cases_json.py "gx10=cuda=<dir>" out.json`. done_when 2 names
gx10, so this row is NOT complete for that host.

## done_when 3 — empty completion = error

Zero tokens generated, or only a think block (whose reasoning is still reported), is
`CliError::EmptyCompletion`, kind `empty_completion`: exit 8 on `apr run`, HTTP 422 on serve.
A whitespace answer the model did generate is an answer. (The first cut used `trim().is_empty()`,
which would also have refused a `max_tokens: 1` request whose first token is a newline. It was
narrowed to #3720's rule.) `think_block_unclosed` is typed (`RealizarError::ThinkBlockUnclosed
{budget}`), exit 8 / 422.

## done_when 4 — refused vs did-not-run

- Every `CliError` variant has `kind()` (snake_case, one per variant), `status()` and `envelope()`
  = `{status, error{kind, message, exit_code}}`. `exit_code` is the process's own and is never
  re-mapped. REFUSED means a capability, limit or policy said no before the work ran (no model,
  bad input, a build or host without the capability, a gate that refused the backend, a mode the
  model cannot honour). FAILED means it ran and errored.
- A `--json` / `--stream` run writes ONE document on EVERY exit. A run that ends before it has a
  result prints the envelope with `model` (and `"event": "final"` on a stream).
- Serve is ADDITIVE: `error` stays the message string every existing client reads, and `status`
  and `error_kind` are added beside it (`thinking_mode_unsupported` / `invalid_request` refused;
  `think_block_unclosed` / `empty_completion` failed). The SSE unclosed-block event carries the
  same fields.

## done_when 5 — SHACL cells

Not in this row: they are #3715's shape to add, against `contracts/apr-response-contract-v1.yaml`.

## Existing tests changed (stated, not hidden)

Four serve tests whose random-weight fixtures generate ZERO tokens now get the named
`empty_completion` 422: `gpu_model` (2), `tests_15` (1), and
`stream_mode_pp27::timings_absent_is_null_not_zero`, whose claim (no zero timings) is still
asserted on that body. Each accepts ONLY that 422, checked on `error_kind`. The 512-draw `top_p`
test now holds one RNG across its draws; a fresh seeded RNG per draw would make all 512 draws
identical and hollow the test.

## Tests

- `run_tests_response_contract.rs`: every variant's kind/status/exit/envelope; ok carries no
  error; empty and reasoning-only answers fail by name with the reasoning kept; generated
  whitespace is ok; a run with no result writes one document; the stream's final event carries
  the envelope; the document carries the digest and the build; the digest of "abc" is its known
  sha256, and a directory has none.
- `response_contract_3720_tests` (openai_handlers): ok bodies with and without an identity; zero
  tokens and reasoning-only fail by name and generated whitespace is 200; unclosed is a named
  failure; an off-only template refuses ON by kind; the digest header on `/health`, and none
  without an identity; deterministic true by default, false with a reason under concurrent
  batching.
- `pmat764_select_batched_token_tests`: the same seed draws the same tokens, another seed differs.
- Suites: serve api + chat_template 2090 pass; apr-cli run + serve 559 pass;
  gguf::cuda::generation 11 pass (`--features cuda`). Complexity ratchet: none new, none grown.

## Found, not fixed here

- `apr run --top-k` defaults to 1 (documented "1 = greedy"), so `--temperature` alone never
  samples. llama.cpp's default is 40. That's a UX decision for the cop, not this row.
- The wgpu `GpuModel` scheduler (`gpu/scheduler`) and `cli/inference.rs` still sample unseeded.
  The wgpu path has no seed in its config, which is why it reports `deterministic: false`.
