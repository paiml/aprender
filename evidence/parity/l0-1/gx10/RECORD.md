# L0-1 RED-first record — gx10 (2026-09-08T12:5xZ), BEFORE any kernel edit

The second required host (card item vi). Same command, same prompt, same judging rule as
`../lambda/RECORD.md`; a DIFFERENT GPU architecture, a different ISA, a different kernel
path (the sm_121 JIT workaround fires here and does not exist on lambda).

- host: `gx10-a5b5`, aarch64, GPU 0 **NVIDIA GB10** (`nvidia-smi -L`; UUID GPU-e9e2a99f…), driver 590.48.01
- binary: `~/.cargo/bin/apr` = `apr 0.65.2 (c04eda87)` — built from `c04eda87d`, the tip of `main` on 2026-09-08 — sha256 prefix `21d182d69505159c`
- mechanism ENGAGED, not intended (verification discipline #2): the stderr of every run carries
  `[GH-480] Patched N backward branch(es) for sm_121 JIT workaround` (an sm_121-only code path)
  and the `Arch:` line the model reports; a CPU-only build has no `apr parity` subcommand at all.
- command: `apr parity <model> --prompt "<the 78-token corpus prompt>" --json`
  (`scripts/check_model_parity.sh`'s `$PROMPT`, byte-identical to lambda's), n=5 per model
  (`n5/runs.log`, `n5/<model>.<i>.json`); run 1 of each is the canonical record beside this file.
- models: `/home/noah/models/qwen2.5-coder-{1.5b,7b}-instruct-q4_k_m.gguf`

## Result (78 positions each, `metrics[].cosine_similarity`)

| model | positions < 0.98 | min cosine (position) | max abs Δlogit at position 0 | argmax mismatches | n=5 stdev of min cosine |
|---|---|---|---|---|---|
| qwen2.5-coder-1.5b-instruct-q4_k_m | **1** | **0.950611 (position 0, token_id 785)** | 12.0087 | 4 | 0.0 |
| qwen2.5-coder-7b-instruct-q4_k_m | 0 | 0.998465 (position 0) | 0.8032 | 1 | 0.0 |

## The cross-host reading — the divergence is NOT architecture-specific

| model | lambda / RTX 4090 / sm_89 / x86_64 | gx10 / GB10 / sm_121 / aarch64 | Δ |
|---|---|---|---|
| 1.5B min cosine | 0.950827 | 0.950611 | 0.000216 |
| 7B min cosine | 0.998607 | 0.998465 | 0.000142 |

Two GPU generations, two ISAs, two host architectures, one of them running a JIT workaround the
other does not have, and the known-bad pair lands within 2.2e-4 of the same number — while the
known-good pair on the same silicon is 0.9985+. Whatever L0-1b's op is, it is reached on **both**
kernel paths and it is selected by the MODEL, not by the device: the 1.5B is
hidden=1536 heads=12 kv_heads=2 GQA=6, the 7B is hidden=3584 heads=28 kv_heads=4 GQA=7.
That refutes, before any lane is dispatched, every hypothesis of the form "an sm_89 kernel is
wrong" or "the sm_121 JIT patch corrupts a branch" — L0-1b's five whys start from the
shape-selected path, not from the device.

Determinism: min cosine is byte-stable across n=5 on both hosts (stdev 0.0 in all four cells), so
a single run is a sufficient witness here and a flake is not an available explanation.

## Threshold

0.98 now sits between TWO measured known-good pairs (7B@lambda 0.998607, 7B@gx10 0.998465; margin
0.0185 under the lower of the two) and TWO measured known-bad pairs (1.5B, 0.9506–0.9508). That is
the second pair `evidence/parity/thresholds.yaml` was waiting on, and the `[U]` on that file is
lifted by this record.

## The records are portable, and the five-run series is one file (2026-09-08)

Every record here was re-taken with the model named RELATIVELY — `cd ~/models && apr parity
./<model>.gguf --prompt "<the 78-token corpus prompt>" --json` — with the same pinned binary
and the same prompt. The `metrics` array and every other key are byte-identical to the
absolute-path run; only the `model` field differs. That keeps `check_hardcoded_paths.sh`'s
shipped-path count flat without deleting a measurement: nothing is redacted, the run was
simply invoked the way it should have been.

`n5/` no longer carries five JSON files per model. All five runs were byte-identical to each
other AND to the canonical record beside this file — `stdev 0` understates it; the whole file
was the same file — so five copies carried nothing the recorded sha256 does not. See
`n5/DETERMINISM.md` for the hashes and `n5/runs.log` for the ten exit codes.
