# C14 live — the release gate over the model manifest (lambda, 2026-09-08)

Binary: `apr 0.65.2 (c08e2682d)` built `--features cuda` from `agent/L0-1b`, sha256 `776cbbdb4306b5d8`.
Host: noah-Lambda-Vector, RTX 4090 (24045 MB), driver 570.207, sm_89.
Command: `bash scripts/check_model_parity.sh --manifest --apr <bin>` — the C14 criterion verbatim.

```
=== C14 model parity on noah-Lambda-Vector (apr 0.65.2 (c08e2682d); thresholds evidence/parity/thresholds.yaml; models /home/noah/models) ===
PASS qwen2.5-coder-1.5b-instruct: 78 positions, min cosine 0.9998 at position 36 >= 0.98 (basis measured n=5 on two known-good pairs (7B@lambda 0.998607, 7B@gx10 0.998465; stdev 0) against two known-bad pairs (1.5B@lambda 0.950827, 1.5B@gx10 0.950611); 0.98 separates them with 0.0185 margin under the lower good floor)
PASS qwen2.5-coder-0.5b-instruct: 78 positions, min cosine 0.9996 at position 0 >= 0.98 (basis measured n=5 on two known-good pairs (7B@lambda 0.998607, 7B@gx10 0.998465; stdev 0) against two known-bad pairs (1.5B@lambda 0.950827, 1.5B@gx10 0.950611); 0.98 separates them with 0.0185 margin under the lower good floor)
PASS qwen2.5-coder-7b-instruct: 78 positions, min cosine 0.9996 at position 22 >= 0.98 (basis measured n=5 on two known-good pairs (7B@lambda 0.998607, 7B@gx10 0.998465; stdev 0) against two known-bad pairs (1.5B@lambda 0.950827, 1.5B@gx10 0.950611); 0.98 separates them with 0.0185 margin under the lower good floor)
UNMEASURED qwen2.5-0.5b-instruct: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen3.5-9b-instruct: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen2-0.5b-instruct: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen2.5-coder-32b: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen3-coder-30b: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen3.5-0.8b: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen2.5-0.5b: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen3.5-9b: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen2-0.5b: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen3-30b: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen3-4b: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
UNMEASURED qwen2-7b: no file under /home/noah/models (reported; the fleet-level rule belongs to the release)
C14: measured=3 rc=0
```

## Reading

- **`qwen2.5-coder-1.5b-instruct` PASS at min cosine 0.9998** — the model #2971 is about, over 78 positions, against a threshold with a measured two-host basis. Pre-fix it read 0.950827 on this host.
- 0.5B and 7B also PASS; three models measured, `rc=0`.
- 14 models UNMEASURED because this host does not hold them. The script reports rather than fails: no single host holds every model the README names, and the fleet-level rule (every cited model measured on >= 1 GPU host) belongs to the release, not to one box.

## The user-visible form

```
=== Model Chat (GGUF Format) ===
  Chat Template: ChatML
Loaded GGUF format in 0.54s (1117.3 MB)
Detected ChatML chat template
[GGUF CUDA: NVIDIA GeForce RTX 4090 (24045 MB VRAM) — pre-cached]
Assistant: 2 + 2 equals 4.
```

No `falling back to CPU`: the request ran on CUDA and answered correctly. That is #2971 from the outside.
