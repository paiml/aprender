# PMAT-3760 sampler sweep: every next-token site in aprender-serve and aprender-core generation

Line numbers are at the #3760 branch head. **Draws** means that at `temperature > 0` with `top_k != 1` the site samples from a probability distribution. **Never draws** is the #3760 defect: the site returns the argmax (or some other fixed token) whatever the sampling flags say.

Scope:

- `crates/aprender-serve/src` (library name `realizar`) and `crates/aprender-core/src/nn/generation`.
- Excluded: tests, benches, and names that are not token selection (`metrics` "samples", `observability` sampling rate, SHAP `nsamples`, `warmup` sample prompts).

## A. Route through the shared sampler (`realizar::sampling`)

Each of these makes one seeded draw through `draw_seeded`, and every one of them is on a shipped path.

| site | reached by | before #3760 | now |
|---|---|---|---|
| `sampling.rs:52` `draw`, `:34` `draw_seeded`, `:26` `is_greedy` | — | — | **the shared sampler**: the body moved verbatim from `OwnedQuantizedModel::sample_topk_with_draw` |
| `gguf/inference/generate_quantized.rs:181` `sample_topk_with_draw`, `:213` `sample_topk_seeded` | dense CPU `generate_with_cache` (`apr run` GGUF CPU, `apr run` .apr CPU via `run_apr_quantized_cpu_inference`, serve quantized chat); qwen35 CPU/GPU decode; dense CUDA `next_token` | draws, seeded | draws, seeded, now delegating to `sampling::draw` |
| `apr_transformer/generation.rs:38` `sample_from_logits` | `AprTransformer::generate_with_cache[_streaming]`: `apr chat` on SafeTensors, serve's APR CPU fallback (apr-cli `serve/handlers.rs`, `handler_apr_cpu_completion.rs`), serve `/generate` and batch on an AprTransformer server (`api/batch.rs`), `apr serve` simple and chat (apr-cli `serve/simple.rs`, `serve/chat.rs`) | **never drew**: argmax of the top-k/top-p survivors | draws, seeded from `GenerateConfig.seed` (new field, `DEFAULT_SEED`) |
| `infer/mod_log_transformer_eos.rs:38` `decode_with_transformer` (was `greedy_decode_with_transformer`) | `apr run` on `.safetensors` (CPU, and sharded) | **never drew**: sampling parameters never read | greedy path is byte-identical; sampled path draws, seeded from `InferenceConfig.seed` |
| `gpu/scheduler/ops.rs:168` `sample_topk` (through `model_forward_block_gpu.rs:407` `sample_topk_generate`) | `GpuModel::generate` → `generate_optimized` / `generate_refcell`: serve's GpuModel chat and completions backends | **never drew**: sorted by probability, returned the first | draws, seeded from `GpuGenerateConfig.seed` (new field) |
| `gpu/scheduler/kv_forward_block.rs:388` `sample_topk`, `:303` `sample_token` | `GpuModel` `kv::generate_with_cache` | **never drew**, same shape | draws, seeded |

## B. Greedy-only decoders: sampled requests no longer reach them

These pick the argmax by design and take no sampling parameters. A sampled request used to go to them anyway and silently decode greedily. It now runs on a loop that draws, and prints why.

| site | reached by | now |
|---|---|---|
| `infer/gguf_gpu_generate.rs:80` `try_wgpu_generate` (inline LM-head argmax) | `apr run` GGUF when not `--no-gpu` on a build with realizar's default `gpu` feature and a working wgpu adapter | taken only when `is_greedy`. Otherwise `WGPU_SAMPLING_NOTICE` is printed and the CPU loop runs (`wgpu_can_serve`, `:68`) |
| `infer/gguf_gpu_generate.rs:467` `try_apr_wgpu_inference` (inline argmax) | `apr run` .apr, same condition | same guard |
| `SafeTensorsCudaModel::generate(input, max_tokens, eos_id)` via `try_safetensors_cuda_inference` | `apr run` .safetensors on a `cuda` build | taken only when `is_greedy`. Otherwise `SAFETENSORS_CUDA_SAMPLING_NOTICE` is printed and the CPU loop runs |
| `api/gpu_completions_handler.rs` (GpuModel completions) | serve `/v1/completions` on a GpuModel | was `top_k: 1`, so the request `temperature` did nothing. Now uses the same rule as the CPU completions handlers (`1` at temperature 0, else `40`; #3754's `DEFAULT_TOP_K` replaces the literal when both land) |

## C. Greedy by construction: argmax is correct

These are called only on a greedy decision, either behind the `temperature == 0 || top_k == 1` predicate or through an API that takes no sampling parameters.

| site | why argmax is correct |
|---|---|
| `gguf/ops.rs:329` `argmax`; `gguf/inference/generate_quantized.rs:128` `argmax`; `gguf/inference/fails.rs:88` `argmax` | the greedy branch of the dense and qwen35 loops |
| `apr_transformer/generation.rs:22` `argmax_logits`; `infer/mod_log_transformer_eos.rs:24` `greedy_argmax`; `gpu/scheduler/kv_forward_block.rs:378` `argmax` | the greedy branch of their loops. Each keeps its own tie-breaking so greedy output stays byte-identical |
| `gguf/cuda/uses.rs:219` `forward_gpu_resident_to_token_id`; `cuda/executor/layers/reduces.rs:144/283/418` and `par-062.rs:124/263/401` (`gpu_argmax`, `batched_gpu_argmax`, `forward_graphed_replay_to_token_id`); `par-121.rs:438` `batched_argmax_from_logits`; `batched_forward.rs:257/442/496` | GPU-side argmax fast paths. `gguf/cuda/generate_1.rs:219` `next_token` takes them only when `greedy && !penalty_active`, and otherwise downloads the logits and samples through `sample_topk_seeded` |
| `gpu/scheduler/batch.rs:129` `forward_single_token_greedy`, `:329` `argmax`, `:368` `optimized_lm_head_argmax_transposed`; `gpu/scheduler/wrapper.rs:198` `argmax` | greedy-only APIs (`generate_gpu(prompt, max_tokens)` takes no sampling parameters) |
| `gguf/parity.rs:41/228`; `infer/inference_result.rs:1294` `argmax_u32` | parity and F2-guard comparisons, not generation |
| `gpu/planner.rs:88` `use_greedy_path` | the predicate that routes to the greedy fast path |
| `cli/apr_inference.rs:278` `argmax` | the greedy branch of realizar's own CLI inference (`apr` does not call `realizar::cli`) |

## D. Draw, but not through the shared seeded draw (follow-ups)

These are not the #3760 defect: they do sample. Each still differs from the shared seeded draw. They are listed so the next change finds them, not hidden.

| site | reached by | how it draws | follow-up |
|---|---|---|---|
| `infer/qwen3_moe_generate.rs:52` `sample_from_logits` | qwen3_moe `apr run` / serve | seeded `StdRng` from `config.seed`; the implementation `sample_topk_with_draw` was ported from | route to `sampling::draw` (same algorithm; needs a byte-parity check) |
| `gguf/inference/generate_quantized.rs:197` `sample_topk` | `cli/inference.rs:14` `sample_next_token` (realizar's own CLI, not `apr`) | the shared `draw`, but `rand::rng()`, so not reproducible | plumb a seed or delete if the CLI is dead |
| `gguf/inference/fails.rs:105` `sample_advanced` (and `:97` `sample_topk`) | one internal caller (Candle-parity sampler) | its own filter, `rand::rng()`, unseeded | route to `sampling::draw` plus a seed |
| `cli/apr_inference.rs:288` `sample_with_temperature` | `api/apr_q4k_scheduler.rs:303/333`: serve APR Q4K GPU chat | its own filter; the "RNG" is a hash of `SystemTime::now()`, so the request `seed` cannot reproduce it | **route to `sampling::draw_seeded` with the request seed; highest priority in D (a serve path)** |
| `generate/mod.rs:63/278/314/356/407` (`sample_token`, `sample_top_k`, `sample_top_p`, `sample_greedy`, `sample_from_distribution`) | the `generate` Sampler API for the dense f32 `Model` (registry and demo models) | caller-supplied `rng_value`, its own filter code | route its top-k/top-p to `sampling::draw` |
| `generate/algorithms.rs:23/130/207/297` (min-p, mirostat, tail-free, typical), `dry_penalty.rs:276` (eta), `dynamic_temperature.rs:389` | the advanced sampler chain | different algorithms by design, not top-k/top-p | none: distinct algorithms |
| `speculative.rs:179`, `speculative_config.rs:115-300` | speculative decoding draft/verify | `rand` RNG, unseeded | seed from the request when speculative decoding is served |
| `delayed_eos_model.rs:316/385`, `mock_model.rs:26` | test and mock models | deterministic stand-ins | none |
| aprender-core `nn/generation/nucleus_sampler.rs:106/203/284/323` | aprender (training library) generation | its own filter, unseeded `rand` | layering: aprender-core sits below realizar and cannot call `realizar::sampling`. A shared draw would have to live in aprender-core or a leaf crate |

## Verification of this sweep

**Enumeration.** Sites were found with:

```
git grep -n -E 'fn [a-z_0-9]*(sample|argmax|greedy|to_token_id)[a-z_0-9]*\s*[<(]'
```

over the two trees, excluding tests, plus the inline-argmax decode loops found by reading each `apr run` / serve dispatch chain (the wgpu loops, the SafeTensors CUDA `generate`, and GPU completions' `top_k: 1`). 83 raw matches, of which the non-token ones are excluded above.

**Real models.** CPU, qwen2.5-coder-0.5b-instruct, prompt "Write one sentence about the sea.", 24 tokens. main is `apr 0.69.0 (a9502d992)`; the branch is `(8202b67bb)`.

| model | build | sampled ≠ greedy | same seed ×2 | seed 99 ≠ 1234 | `--top-k 1` = greedy | greedy output |
|---|---|---|---|---|---|---|
| `.safetensors` | main | **NO** | IDENTICAL | **NO** | yes | — |
| `.safetensors` | branch | YES | IDENTICAL | YES | yes | byte-identical to main |
| `.apr` | main | YES | IDENTICAL | YES | yes | — |
| `.apr` | branch | YES | IDENTICAL | YES | yes | byte-identical to main |

`.apr` already sampled on main: it goes through the GGUF sampler.
