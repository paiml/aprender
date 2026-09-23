# #3899 — tinyllama's GPU divergence is a near-tie, not a CUDA defect

**Worker:** aprender worker two. **Branch:** `feat/iq3s-gpu-gemv`.

## Verdict

**Not a CUDA correctness defect. Rule A is not violated.** The row's disposition
is the ill-posed-comparison problem (#3873): a text-equality gate run on a prompt
that puts the model in a degenerate regime.

## 1. The architecture hypothesis, refuted

The hypothesis was that the Q4_K GPU path might be right for qwen geometry and
wrong for `llama`, since tinyllama is the fleet's only llama-arch Q4_K model and
the only one producing GPU gibberish. Four models, one coherent prompt
("The capital of France is"), 192-token budget, greedy, one session:

| model | arch | quant | first divergence | identical |
|---|---|---|---|---|
| tinyllama-1.1B | llama | Q4_K | **143** | no |
| TinyLLama-5M | **llama** | F16 | — | **yes** |
| Qwen3-1.7B | qwen3 | Q4_K | — | **yes** |
| qwen2.5-1.5B | qwen2 | Q4_K | — | **yes** |

A second llama-arch model that does **not** diverge kills the hypothesis as
stated. This was disconfirming evidence for the cop's position, sought
deliberately.

I could not get a second llama **Q4_K** model — tinyllama is the only one on the
box — so "Q4_K × llama specifically" is narrowed rather than eliminated by this
table alone. §3 closes it.

## 2. The divergence is stable and the streams reconverge

```
gpu_run1 == gpu_run2                    (greedy, deterministic)
token 143: CPU picks 2982, GPU picks 26014
  CPU "...one of the most famous LANDMARKS IN PARIS. It was built in the 12th
       century and has been a symbol of Parisian culture and history."
  GPU "...one of the most famous CHURCHES IN THE WORLD. It was built in the
       12th century and has been a symbol of Parisian culture and history."
tokens agreeing AFTER the split: 45/48 (93.8%)
```

Both continuations are fluent and true of the same landmark. A wrong kernel does
not emit a coherent alternative phrasing that rejoins the original within three
tokens; it degrades.

## 3. The number, which is what actually settles it

§1 and §2 are inferences from *semantic plausibility*. "Both words read well" and
"the logits were within fp error" are different claims, and only the second
matters. CPU logits recomputed at exactly the prefix both legs had consumed:

```
gap = logit(2982) - logit(26014) = 14.78509 - 15.02783 = -0.24275
top1 = 15.02783   top10 = 11.76790   spread = 3.25994
gap as fraction of local spread = 7.4%
cpu_tok_is_argmax = FALSE          rank_of_gpu_tok = 1
```

**0.24 logits against a 3.26 spread.** Well inside what fp accumulation order
moves. Nothing shifted a logit by a large absolute amount, so the plausibility of
the two continuations was not luck.

**And the GPU is not the deviant party**: `top1 == logit(26014)`, so the CPU's own
model recomputed at that prefix prefers the GPU's token. The leg that picked the
non-argmax token was the CPU's incremental path.

**Limit, kept out of the headline:** `forward_hidden_states` is a full forward
and `generate_with_cache` is incremental, so that second point is not a
controlled comparison and "the CPU cached path has a bug" does not follow from
one recomputation. The 0.24-of-3.26 does not depend on resolving that.

## 4. A guard I wrote, could not falsify, and deleted

`a_divergence_between_the_legs_is_isolated_and_reconverges` asserted that a
divergence must be isolated and the streams must reconverge (≥75%; measured
93.8%). It replaced an earlier guard of mine whose 64-token budget made it a
*true* assertion narrower than its name — green forever past 143.

**Four honest attempts to red it, all failed:**

| fault | where | result |
|---|---|---|
| `+0.02*(row%7)` | CPU fused Q4_K dot | green — too small to flip an argmax |
| `+2.0*(row%7)` | CPU fused Q4_K dot | green — additive, RMSNorm removes it |
| `+3.0*(i%5)` on logits | `fails.rs::generate_with_cache` | green — **wrong function** |
| `+3.0*(i%5)` on logits | `generate_quantized.rs` (verified live) | green — **shared by both legs**, so it perturbs them identically |
| `*= 1+0.30*(row%7)` | CPU fused Q4_K dot | green — and printed nothing, i.e. **no divergence at all** |

The third is the instructive one: a panic probe proved `fails.rs:193` is NOT the
`generate_with_cache` this path runs — `generate_quantized.rs:237` is. Three
faults had been planted in a function that never executes here. That is the
fourth instance this session of *the scope I was working in is not the scope the
behaviour lives in*, and it was caught by a reachability probe rather than by
reasoning.

The last row is the one that condemns the guard: a 30% per-row multiplicative
distortion of the CPU Q4_K matvec produced **no divergence**, which means that
function is not on the CPU generate path either, despite a panic firing from
somewhere. I do not know what the CPU leg's matmul path actually is, and a guard
whose subject I cannot locate is not a guard.

**Deleted rather than shipped.** An unfalsified green reads as coverage and is
worse than its absence. The findings above are the durable part; the test was
not.

**Process note:** three of my mutation attempts silently planted nothing because
the anchor appeared 2 or 3 times, and `assert s.count(old) == 1` refused each
time. Without it I would have reported "the function is not on the path" as a
confident, wrong conclusion about why the guard passes.

## 5. Not claimed

* Not claimed the CPU incremental path has a defect. §3's limit applies.
* Not claimed tinyllama passes the golden gate. It does not, for #3873's reason.
* Not claimed the reconvergence guard is wrong — only that I could not construct
  a fault that reds it, which is a different and sufficient reason not to ship it.
