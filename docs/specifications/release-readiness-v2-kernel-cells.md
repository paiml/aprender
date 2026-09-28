# Release readiness v2 — kernel cells, derived model cells (#3715 v2, 0.71)

Status: DESIGN STUB (L1 look-ahead, branch `la-71/3715-kernel-cells`). Owner: kreg (aprender-85).
Depends on: KREG-001 (#4539, #3430). Supersedes the 08:13Z "gate ≤ 2 h" kaizen as the 0.71 target.

Operator, verbatim (2026-09-28 08:15Z): "rebuild the release-readiness shape on KERNEL receipts.
:Kernel × :Backend × :ShapeClass → Receipt(Pass) with sh:minCount 1; model cells DERIVED by SHACL
rule (all kernels of the model Pass on that host's backend) + one e2e smoke per model×host.
Falsifiers: remove one kernel receipt → every model using it goes RED; a model with an unregistered
kernel → RED. Target: gate ≤ 30 min, 0 unvalidated cells."

## 1. Why

v1 (`contracts/release-readiness-v1.yaml`) owes one end-to-end Pass receipt per
model × host × verb {run, chat, serve, code} × thinking {on, off} × context rung. Every cell re-runs
the same handful of kernels through a full model load, so the gate costs hours, and a kernel defect
shows up as dozens of red model cells rather than one red kernel. v2 moves the expensive axis to the
thing that is actually shared, the kernel. A kernel is measured once per backend and shape class, and
the model cells are derived from that measurement.

The invariant is unchanged: **0 cells without a Pass receipt**. What changes is which cells are
measured and which are derived.

## 2. The cells

| Cell | Key | Evidence | Cost |
|------|-----|----------|------|
| `release:KernelParityCell` | kernel_id × backend × shape_class | a `kernel-parity-receipt/v1` (KREG AC-3) whose `host_arch`/`sm` is the host's, `verdict` pass, fresh | seconds each (CPU receipts today: 0.1 s for 11 rows) |
| `release:ModelCell` | model × host | **derived**: one `release:usesKernel` edge per kernel on the model's dispatch path on that host's backend | 0: no run |
| `release:SmokeCell` | model × host | one e2e `apr run` (short prompt, 1 context rung) `rr2-smoke-receipt/v1`, verdict pass, fresh, with a `kernel_path` from KREG | ~1–2 min each |

The class is `release:KernelParityCell`, not `release:KernelCell`. v1's extractor already emits
`release:KernelCell` for its per-(host, kernel, quant) diff cells, and `release-readiness-v1.kernel`
targets that class. The v2 cells go into the same release graph, so with a shared name each
version's shape would grade the other's cells (found wiring KTEST-08, 2026-09-28).

The axes come from existing data, not from a new list:
- `kernel_id`, `backend`, `shape_class` (`m1`/`m_any`) are fields of `crates/aprender-serve/kernel-registry.json`.
  `:Backend` and `:ShapeClass` exist today only as literals (`kreg:backend`, `kreg:shape_class`). v2
  makes them IRIs (`kreg:backend/cuda`, `kreg:shape/m1`) so a host can name the backend it serves.
- Hosts are the `required: true` entries of `contracts/model-capability-ladder-v1.yaml` `ladder.hosts[]`.
  Models are the ladder rungs plus the measured inventory, as in v1.

## 3. "Derived by SHACL rule" without `sh:rule`

The in-tree engine is SHACL **Core** only (`crates/aprender-contracts/src/ontology/shapes.rs:1-19`).
`sh:rule` and `sh:sparql` are refused at parse. v2 does not add inference. It gets the same
semantics from Core:

1. The extractor (`ontology/extract/release_evidence.rs`, which already derives v1 cells) emits
   `ModelCell --release:usesKernel--> KernelParityCell` for every kernel on the model's dispatch path on
   that host's backend. It emits **edges only, never a verdict**. A verdict computed in Rust and then
   checked by a shape would be the check checking itself.
2. The shape does the derivation:
   ```yaml
   release:ModelCell:
     - {path: release:usesKernel, minCount: 1, node: release:KernelParityCellPass}   # sh:node, one level: supported
     - {path: release:unregisteredQtype, maxCount: 0}
     - {path: release:smoke, minCount: 1, node: release:SmokeCellPass}
   release:KernelParityCellPass:   # verdict pass, withinBound, fresh, receipt minCount 1
   ```
   A KernelParityCell with no receipt fails `KernelParityCellPass`, so every ModelCell that points at it fails
   through `sh:node`. That is the first falsifier, enforced by the shape and not by the extractor.

## 4. The model → kernel map (the load-bearing part)

This map comes from two independent sources, and they must agree:

- **Static:** the per-tensor ggml types of the model file, each passed through `admit(backend,
  ggml_type, layout)` (`crates/aprender-serve/src/kernel_registry.rs:280`). A qtype with no admitting
  row becomes a `release:unregisteredQtype` literal, which is the second falsifier. **Gap:** the GGUF
  extractor records only `general.file_type` and the tensor count (`extract/gguf.rs:8,276`). It needs
  a per-tensor qtype set.
- **Observed:** the kernel ids the smoke run actually dispatched (`kernel_path`, OBS-15, declared in
  `contracts/kernel-registry-v1.yaml:86-87`). **Gap:** no emitter writes it into a receipt yet. Rule:
  observed ⊆ static, and a kernel that ran but was not predicted is RED. Without this check the static
  map is a claim nobody tested.

**The smoke receipt** (`rr2-smoke-receipt/v1`, `<v2-dir>/<host>/smoke/*.json`) is `{schema, host,
model_sha256, apr_sha, verdict, kernel_path}`. `kernel_path` is the `apr-kernel-path-v1` object that
OBS-15 defines (aprender#4574, aprender-01); v2 does not define a second one. Only a `source: kreg` path
names registry ids, so a `null` or `source: trace` path sets `release:kernelPathKnown false` on the smoke,
and the model cell is RED: the static map was not checked against a run. `kernel_cells::judge_smoke_receipt`
reads it; `build_cells` judges each receipt against its own model's static map.

`apr parity --per-op` kernel-diff receipts (`release_inputs.rs:289-325`) already record a dispatch
path, but under labels such as `q4k_gemv@q4_k` rather than registry `kernel_id`s. v2 maps those
labels to registry ids once, in the registry itself, as a `labels` field, so no second list exists.

**Per-forward ops** (embedding, RMSNorm/LayerNorm, RoPE, attention, KV write, SwiGLU/GELU, residual add,
argmax) run on f32 activations whatever the file's types, so no `ggml_type` key reaches them. They are the
registry's `ops[]` rows (closed shape `kernel-registry-v1.op`, which refuses a type field on them), on every
map of their backend and host arch for a model with tensor types, narrowed by an optional `archs` list
against `general.architecture` (LayerNorm for `phi2`, say). An unknown architecture takes every op row, so
not knowing it can only add RED cells. They are not `ggml_type: null` rows of `kernels[]`: that would need
`ggml_type` at minCount 0, and SHACL Core cannot then require a type on a typed row. `kernel_cells::op_kernels`
reads them; a registry with no `ops[]`, a type key on an op, `archs` on a typed row, or an id shared across
the two arrays refuses the registry.

**Coverage gap, and the first finding for 0.71:** the registry holds 41 `kernels[]` rows, all
`matvec` or `gemv`, and ten CPU `ops[]` rows (embedding, RMSNorm, LayerNorm, RoPE, attention, KV write, SwiGLU,
GELU, residual add, greedy argmax), none with a receipt. No CUDA op is registered, and neither is top-k/top-p sampling. Under the
unregistered-kernel falsifier, **every model is RED today**. That is correct: it is the "0
unvalidated cells" target stated honestly. Registering the remaining dispatch ops is phase P1.

## 5. Freshness and reuse (from kaizen (1), 08:13Z)

A kernel receipt is fresh when its **`input_set_hash`** (KTEST-001 §5.1) matches the one the gate
recomputes from the release tree: a sha256 over the kernel's `source_file`, its registry row (every
field but `tolerance`), the pinned toolchain, the driver, the device and the oracle. `apr_sha` is not part of the key. A release whose diff
does not touch a kernel reuses its receipt, and the gate reports the reuse percentage. The gate does not recompute the hash itself: aprender-serve's freshness
test writes `kreg-input-sets/v1` (`KREG_INPUT_SETS_OUT`, at `KREG_GIT_SHA`), and
`kernel_cells::parse_input_sets` reads it, refusing a file computed at any other commit. A kernel absent
from that file judges stale. A model cell is
fresh when every KernelParityCell it uses is fresh and its smoke ran at the release commit.

## 6. Falsifiers (each a committed RED case in the shapes-gate case table)

| Id | Plant | Required RED |
|----|-------|--------------|
| RR2-F1 | delete one kernel receipt | exactly the ModelCells whose static set contains that kernel, and no others (the test asserts the set) |
| RR2-F2 | a model tensor with a qtype no row admits | that ModelCell, via `unregisteredQtype` |
| RR2-F3 | the smoke's `kernel_path` contains a kernel outside the static set | that ModelCell |
| RR2-F4 | kernel source changed, receipt input key stale | that KernelParityCell and its dependants |
| RR2-F5 | no smoke receipt for model × host | that ModelCell |
| RR2-F6 | receipt measured on another arch/sm than the host's backend | that KernelParityCell |
| S-SAN | a cuda kernel with no, a dirty (any of memcheck/racecheck/initcheck/synccheck not CLEAN), or a > 7 d compute-sanitizer run (KTEST-05 `receipt.json`) | that KernelParityCell only (`release-readiness-v2.sanitizer`); cpu kernels are not asked |

S-SAN attributes a sanitizer run to the kernels it dispatched, so it needs the run's `kernel_path`
(OBS-15), just as RR2-F3 does. The run is `ktest-05-sanitizer-receipt-v2` (`<v2-dir>/<host>/sanitizer/*.json`):
KTEST-05's v1 receipt plus `host`, a `source: kreg` `kernel_path`, and on each tool row run with
`--kernel-name`, `covers`, the registry ids that filter kept. An unfiltered tool checked every dispatched
kernel. A filtered tool with no `covers` checked none, because its reach is unknown: registry rows carry no
CUDA symbol name to match the regex against. A kernel is clean only when every one of the four tools checked
it in some run and no check of it was dirty, and fresh only when every tool checked it in a run at most 7 d
older than `--gate-utc`. The extractor reads no clock, so with no `--gate-utc` every run is stale. The gx10
run of 2026-09-28 filters racecheck to attention/rope/norm kernels, so the gemv kernels stay RED on S-SAN until
a racecheck run covers them. Until KTEST-05 writes v2 receipts, every cuda kernel cell is RED on S-SAN.

## 7. Budget: ≤ 30 min

Kernel receipts: 41 rows today, and ~100 once every op is registered. Each takes 0.1–5 s per host,
so under 10 min cold and about 1 min with reuse. Smokes: roughly 8 models × 2 required hosts ×
1–2 min, run in parallel across the two hosts, under 16 min. The verb × thinking × context grid
leaves the release gate and becomes nightly full-depth, once per model × context (kaizen (3)).
Long-context risk moves to the attention kernels' shape classes, and that needs a sequence-length
`shape_class`. This is an open question below.

## 8. Phases

- **P1 (0.71):** register every dispatch op (registry rows plus `labels`; the v2 map reads `ops[]`; ten CPU op
  rows cover the CPU decode path; CUDA's and sampling are open); per-tensor qtype in the
  GGUF extractor; `release-readiness-v2.yaml` shapes; extractor edges; the RR2-F1…F6 case table.
- **P2:** CUDA kernel receipts on lambda (sm_89) and gx10 (sm_121); the `kernel_path` emitter in the
  smoke.
- **P3:** `scripts/release/release_readiness.sh` switches to v2; the v1 grid runs nightly; v1 is
  retired after two green trains on v2.

## 9. Open questions (for the cop and operator)

1. Does the long-context rung stay a release cell, or does it become a KernelParityCell shape class
   (`seq_len` bucket) on attention kernels?
2. Is CPU a required backend for release, or does it stay nightly? v1 releases on cuda only; the
   ladder rungs also list cpu.
3. The smoke verb: is `run` alone enough, or should it be `serve` (which exercises batching kernels)?
