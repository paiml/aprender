# Book Execution Validation Harness Spec (BOOK-CLOSEOUT-001 Phase 6)

**Status**: PROPOSED (2026-05-23)
**Owner**: aprender-core
**Parent spec**: `docs/specifications/book-completeness-spec.md` § Phase 6
**Target close**: post-hiatus authoring window (estimate revised below)
**Predecessor**: PR #1902 (Phases 1-5 — structural correctness)

## Problem statement

Phases 1-5 of BOOK-CLOSEOUT-001 enforce that every CLI subcommand has a chapter (`book/src/cli/<cmd>.md`) and every public `aprender-core` module has a chapter (`book/src/lib/<mod>.md`), and that each chapter contains a fenced code block of the right language. **They do not check that the code in those blocks actually works.**

Today, a stub like:

```markdown
## Example
\`\`\`bash
apr run NONEXISTENT-MODEL --invalid-flag
\`\`\`
```

passes `scripts/check_book_example_block.sh` because the fenced block exists. The block is syntactically a `bash` block, regardless of whether `apr` would accept it. The same holds for `rust` blocks in `book/src/lib/*.md`: the regex matches the fence, never the type-checker.

This Phase 6 spec closes the **shape-vs-behavior** gap by adding an execution validation harness.

## Goal

Every fenced code block in `book/src/cli/*.md` and `book/src/lib/*.md` is verified to:

- **rust** blocks: compile (via `cargo check`)
- **bash** blocks: execute without error (exit 0), respecting a per-example **cost annotation** that tells CI how much work the example needs

The harness must be:
1. **Fast in common case**: pure-`cargo check` and trivial `bash` finish in ~5 minutes on `ubuntu-latest`.
2. **Honest about expense**: examples that need a model, a GPU, or a destructive side-effect are explicitly annotated and routed to the right CI job.
3. **Falsifiable**: two new contracts in `apr-book-completeness-v1.yaml` formalize the "every example runs / every block compiles" predictions.

## Inventory (HEAD of `feat/book-completeness-spec`)

| Surface | Count | Source |
|---|---|---|
| CLI chapter stubs | 103 | `git ls-tree feat/book-completeness-spec book/src/cli/ \| wc -l` |
| Lib chapter stubs | 69 | `git ls-tree feat/book-completeness-spec book/src/lib/ \| wc -l` |
| Existing aprender-core examples (golden references) | 150 | `ls crates/aprender-core/examples/ \| wc -l` |
| Existing CLI commands with destructive side-effects | ~10 | `publish, pull, push, finetune, train, pretrain, distill, encrypt, decrypt, merge, rm, convert, export, quantize, prune, compile` |
| Self-hosted GPU runners | available | `qwen-story-daily.yml: runs-on: [self-hosted, gpu]` |

## 1. Per-example cost annotation schema

Every fenced code block in `book/src/cli/*.md` and `book/src/lib/*.md` must be preceded (within 3 lines above the opening fence) by an HTML comment of the form:

```html
<!-- example-cost: <class>[; key=value[; key=value...]] -->
```

### Cost classes (closed set — extending requires a contract amendment)

| Class | Meaning | CI dispatch |
|---|---|---|
| `trivial` | Runs in < 5s. Needs nothing beyond `apr` itself (or stdlib for rust). | `ubuntu-latest`, run inline |
| `model-required` | Needs a GGUF in the model cache. Specify `model=<id>` and optionally `size=<class>` | `ubuntu-latest` with cache prelude; falls back to `--mock` mode if cache miss |
| `gpu` | Needs CUDA / wgpu. May also need a model (combine with `model=`) | `[self-hosted, gpu]` runner; `ubuntu-latest` skips with allowed-skip marker |
| `destructive` | Mutates external state (`apr publish`, `apr rm`, network upload, file deletion outside `/tmp`). | Rewritten to dry-run/mock form by the executor before invocation. |
| `compile-only` | rust block — must `cargo check` but is not expected to be wrapped in `fn main()` (used for trait/import-only snippets). | `ubuntu-latest`, harness adds a `fn main() {}` wrapper. |
| `skip` | Explicitly opt-out (with reason). Counted as failure unless `reason=` is set. | Skipped, but contract requires a `reason=` argument. |

### Argument grammar (lightweight key=value, semicolon-separated)

- `model=<id>` — example needs `<id>` available via `apr pull`. Canonical id is the CLI accept form, e.g. `qwen2.5-coder-1.5b` (no quotes, no spaces).
- `size=tiny|small|medium|large` — coarse hint for cache eviction. `tiny<200MB`, `small<1.5GB`, `medium<5GB`, `large>=5GB`.
- `gpu_arch=<arch>` — for `gpu` class; e.g. `sm_89` or `cuda12`.
- `reason="<text>"` — required for `skip`. Free-form, parsed verbatim.
- `mock_as="<rewrite>"` — for `destructive`; the exact substitution the executor will perform before invocation.

### Examples

```html
<!-- example-cost: trivial -->
<!-- example-cost: model-required; model=qwen2.5-coder-1.5b; size=small -->
<!-- example-cost: gpu; model=qwen2.5-coder-7b; size=medium; gpu_arch=sm_89 -->
<!-- example-cost: destructive; mock_as="apr publish --dry-run" -->
<!-- example-cost: skip; reason="depends on huggingface.co network — covered by integration-tests-network.yml" -->
<!-- example-cost: compile-only -->
```

### Discoverability rule

If an annotation is missing or malformed, the extractor (§ 2) emits the row with `cost=UNANNOTATED` and the gate fails with a list of unannotated files. There is no implicit default — annotations are mandatory. This is the design counterweight to having 172 stubs: silent classification drift is the obvious failure mode if we let `cost` default to `trivial`.

## 2. Extractor script design

### `scripts/extract-book-examples.sh`

Plain bash + python. Single pass over `book/src/{cli,lib}/*.md`. Emits NDJSON (one record per fenced block) to stdout. Exit 0 iff every block in scope has a valid cost annotation.

See the original Plan agent design output (Plan agent ad80c15e8444ae1e1) for the complete 90-line python implementation. Key features:

- Regex match `<!-- example-cost: ... -->` within 3 lines preceding a `^```(\w+)$` fence
- Validate cost class against the closed set
- Emit NDJSON with full positional metadata (path, line_start, line_end)
- Exit non-zero if any block is unannotated

## 3. Per-cost-class executor

`scripts/check_book_examples_executable.sh` consumes NDJSON, dispatches per cost class:

- **trivial**: run inline with hardened PATH, 10s timeout, refuse known model-binding commands
- **model-required**: check `apr list --json` for model; one-shot `apr pull` on miss; 60s timeout
- **gpu**: SKIP on CPU runners; execute on `[self-hosted, gpu]` with 120s timeout
- **destructive**: require `mock_as=` rewrite; execute the mock, not the original
- **skip**: count + require reason
- **compile-only**: rust-only; handled by §4 compile harness

Aggregator job ensures every gpu-class example ran on at least one runner in the CI matrix.

## 4. Rust compilation harness

`scripts/check_book_examples_compile.sh` generates `crates/aprender-core/examples/book_<slug>.rs` for each rust block, then single-shot `cargo check -p aprender-core --examples`. Manifest-tracked cleanup of generated files. `book_` prefix reserved.

Wrapper logic:
- If block declares `fn main(...)`: include verbatim with `#![allow(dead_code, unused_imports, unused_variables)]`
- Else: wrap in `fn main() { <code> }`

Rejected alternatives: `mdbook test` (slow per-block), `rustdoc --test per-block` (orphan-blocks need wrapping anyway).

## 5. Two new falsifiers

Append to `contracts/apr-book-completeness-v1.yaml`:

```yaml
  - id: FALSIFY-BOOK-EXAMPLE-EXECUTES-001
    name: bash_examples_executable
    test_harness: "bash scripts/check_book_examples_executable.sh"
    expected_output: "exit 0; every annotated bash example exits 0 or is explicitly skipped"
    if_fails: "Fix the example, re-annotate cost, or mark skip with reason."

  - id: FALSIFY-BOOK-EXAMPLE-COMPILES-001
    name: rust_examples_compile
    test_harness: "bash scripts/check_book_examples_compile.sh"
    expected_output: "exit 0; cargo check emits no error lines"
    if_fails: "Fix the rust block, mark compile-only, or mark skip."
```

Extend equations block with `example_executable` and `example_compiles` invariants.

## 6. CI integration

Add 4 jobs to `.github/workflows/book.yml`:

- `examples-compile`: ubuntu-latest, `cargo check --examples`
- `examples-execute-cpu`: ubuntu-latest, runs all non-gpu examples
- `examples-execute-gpu`: `[self-hosted, gpu]`, runs gpu-class examples
- `examples-gpu-aggregate`: ubuntu-latest, asserts every gpu example ran on ≥1 runner

All depend on the existing `build` job. The aggregate job prevents "GPU runner silently misconfigured" from passing the gate.

## 7. Migration path

3-tier classification for the 172 existing stubs:

**Tier 1 (auto, ~70%)**: `scripts/classify-book-examples.sh` heuristics:
- `apr --help` / `apr <cmd> --help` → `trivial`
- `apr list/diagnose/gpu` (no model) → `trivial`
- Contains a model literal → `model-required; model=<extracted>`
- Contains `apr publish/push/rm/encrypt/decrypt/finetune/train/pretrain/distill` → `destructive`
- Contains `--gpu` or `--backend cuda` → `gpu`
- Library rust block with only `use` → `compile-only`

**Tier 2 (manual, ~25%)**: ambiguous cases, gpu, destructive — ~50 stubs, ~3 hours focused review.

**Tier 3 (LLM-assisted, ~5%)**: residual ambiguous; operator-driven prompts, NOT inline in CI.

Migration ordering (6 PRs, 6a-6f) so the falsifier never goes "enabled but unsatisfiable":

- 6a: extractor (warning-only mode)
- 6b: auto-classify Tier 1 (~120 stubs)
- 6c: manual Tier 2 (~50 stubs)
- 6d: flip falsifiers to required + extractor strict mode
- 6e: examples-compile + examples-execute-cpu jobs
- 6f: examples-execute-gpu + aggregator

## 8. Cost estimate (revised from parent spec's 8-12 hr)

| Sub-task | Estimate |
|---|---|
| Extractor + NDJSON | 1.5 hr |
| Executable harness (5 cost-class executors) | 2.5 hr |
| Compile harness + manifest lifecycle | 1.5 hr |
| Auto-classifier + annotation tool | 1.5 hr |
| Manual annotation (Tier 2 ~50 stubs) | 3 hr |
| Contract extension (2 falsifiers + equations) | 0.5 hr |
| CI workflow additions (4 jobs incl. aggregator) | 1.5 hr |
| GPU runner pre-staging | 1 hr |
| End-to-end shakedown (fix bugs surfaced by harness) | 2 hr |
| Documentation + PR descriptions (6 PRs) | 1 hr |
| **Subtotal** | **16 hr** |
| Contingency (10%) | 1.5 hr |
| **Total** | **17.5 hr** |

**Revised: 16-18 hr** (vs parent spec's 8-12 hr). Dominant overruns: Tier 2 manual annotation (3 hr) and end-to-end shakedown (2 hr).

## Definition of done

### Per-PR sub-deliverables

- [ ] **6a**: extractor lands; `bash scripts/extract-book-examples.sh > /tmp/blocks.ndjson` produces valid NDJSON; warning-only mode.
- [ ] **6b**: ~120 stubs auto-annotated; bulk diff reviewed.
- [ ] **6c**: ~50 stubs manually annotated; every block has a `cost`.
- [ ] **6d**: contract extended; extractor in strict mode.
- [ ] **6e**: examples-compile + examples-execute-cpu jobs; both pass on main.
- [ ] **6f**: examples-execute-gpu + aggregate jobs; GPU fleet pre-staged with cache.

### Aggregate definition of done

- [ ] `bash scripts/extract-book-examples.sh` exits 0 (every block annotated)
- [ ] `bash scripts/check_book_examples_compile.sh` exits 0
- [ ] `bash scripts/check_book_examples_executable.sh` exits 0 on `ubuntu-latest` (skip-set = gpu-class only)
- [ ] `bash scripts/check_book_examples_executable.sh` exits 0 on `[self-hosted, gpu]` (skip-set = ∅)
- [ ] `pv validate contracts/apr-book-completeness-v1.yaml` exits 0
- [ ] `pmat comply check` reports `is_compliant=true` with 0 Fail-status checks for both new falsifiers
- [ ] `book/src/cli/*.md` and `book/src/lib/*.md`: 0 unannotated blocks

## Out of scope

- Authoring real content in the stubs (separate spec)
- Output assertion (exit-code only — snapshot lib is separate)
- Network-dependent examples (`apr pull hf://...` is `skip; reason="network"`)
- Cross-platform (Linux only)
- Examples outside `cli/` and `lib/` (already covered by mdbook test on methodology chapters)

## Top 5 most consequential design decisions

1. **Mandatory cost annotation with no implicit default** — every block must carry the annotation or the extractor fails. Silent misclassification of 172 stubs is the obvious failure mode if default were `trivial`. Cost: ~3 hours of manual review; benefit: per-chapter auditability.

2. **Generated `examples/book_<slug>.rs` + single `cargo check` over `mdbook test`** — single invocation amortizes dep-compilation 3-5x faster than per-block. Tradeoff: on-disk pollution mitigated by `book_` prefix + manifest cleanup.

3. **Explicit `mock_as=` for destructive examples (vs. flag-transform)** — not every destructive command has `--dry-run`. Trades shape-verification for explicitness; maintainer on the hook for mock honesty.

4. **Split CPU/GPU jobs with aggregator** — prevents "GPU runner silently misconfigured" from passing. Cost: 1 extra job; benefit: no silent skips.

5. **Six-PR phased migration with informational-then-blocking cutover** — gate becomes blocking only at PR 6d, after annotation is universal. Eliminates half-annotated stuck-state. Cost: 6 PRs vs 1; benefit: each independently revertable.

## Related

- `docs/specifications/book-completeness-spec.md` § Phase 6 (parent)
- PR #1902 (structural predecessor)
- `contracts/apr-book-completeness-v1.yaml`
- `.github/workflows/book.yml`
- `.github/workflows/qwen-story-daily.yml` (GPU job precedent)
- `crates/aprender-core/examples/apr_embed.rs` (hand-written example shape)
