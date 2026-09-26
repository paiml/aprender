# EXT-27 (aprender#4409) — M1b artifact quality, first record

KL(BF16 reference ‖ arm) and top-1 agreement per arm, measured with llama.cpp
`d1d3c3396aa13a5f239109a822666c4870490ad5` (`llama-perplexity --kl-divergence`,
`-c 512`, 36 chunks) on `corpus.txt`, 2026-09-26 on lambda-vector (CPU).

| input | sha256 | source |
|---|---|---|
| corpus.txt | `32f70802fd34…` | four book chapters at 8204b06fb (gradient-descent, decision-trees, kmeans-clustering, test-first-philosophy), 64,824 bytes |
| reference | `9e6e2841a75f…` | unsloth/Qwen3.5-4B-GGUF@e87f1764 `Qwen3.5-4B-BF16.gguf` — independent of apr |
| ours | `97784e904050…` | `apr import` of Qwen/Qwen3.5-4B@851bf6e8 safetensors, then `apr quantize -s q4k --format gguf`, apr built at 2718c5214 (batch/0.70.0; carries the #4418 qwen35 GGUF fix — apr 0.69.3's GGUF does not load in llama.cpp) |
| plant | `186256778202…` | ours with `blk.10.ffn_down.weight` re-quantized q4_K → q2_K by `llama-quantize --tensor-type-file logs/plant-types.txt` (every other 2-D tensor pinned to its own type; `plant-quant.log` shows exactly one conversion) |

Arm pins are the EXT-24 pins (`evidence/ext-001/EXT-24/crux-bind-receipt.json`).

| arm | mean KL (nats) | top-1 agreement |
|---|---|---|
| ours | 0.028094 ± 0.000760 | 93.638 ± 0.255 % |
| unsloth-q4_k_m | 0.018684 ± 0.000703 | 95.294 ± 0.221 % |
| bartowski-q4_k_m | 0.016423 ± 0.000517 | 95.251 ± 0.222 % |
| unsloth-ud-q4_k_xl (separate recipe) | 0.013781 ± 0.000482 | 95.991 ± 0.205 % |
| plant (ours + one Q2_K tensor) | 0.031958 ± 0.000823 | 93.290 ± 0.261 % |

- `m1b-first-record.json` — the first record: no baseline, so it sets it (R-15).
- `m1b-plant-q2k.json` — the plant measured against that baseline. Every arm's KL gap
  widens by 0.003864 nats (EPS 1e-4) and every top-1 gap by 0.348 pp (EPS 0.1 pp):
  M1b is RED. `falsify_ext_021_degraded_quant_red` reads both files (FALSIFY-EXT-021);
  resetting the plant's figures to ours' turns that test RED (mutation checked).
- **ollama-qwen3.5-4b is not in the record: unrunnable at the pin.** llama.cpp
  d1d3c3396 refuses the blob (`qwen35.rope.dimension_sections has wrong array length;
  expected 4, got 3`, `logs/kl-ollama-qwen3.5-4b.log`). Measuring it needs Ollama's
  own runtime or a later pin; a re-pin is a new baseline, never a comparison.
- Ours trails every arm in absolute terms. That is recorded, not gated (R-15), and
  filed in `docs/findings/aprender-cb.jsonl`.
- Reproduce: `logs/lane.sh <arm>…` after the reference pass
  `llama-perplexity -m reference-bf16.gguf -f corpus.txt -c 512 -t 16 --kl-divergence-base ref-logits.kld`.
  The 4.6 GB `ref-logits.kld` is not committed.
