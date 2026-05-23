# Book Completeness Spec (BOOK-CLOSEOUT-001)

**Status**: PROPOSED (drafted 2026-05-23)
**Owner**: aprender-core
**Target close**: within hiatus window (~5 hours of focused work, before 2026-05-25)

## Problem statement

The `aprender` book under `book/src/` has two coverage gaps captured by post-publish v0.35.2 dogfood (PR #1901):

1. **168 broken file links** (closed by #1901 — strip-prose-only fix; pages stay missing)
2. **~30 unique chapter files referenced from prose but never authored**, and **~50+ CLI subcommands without dedicated chapters**

The book builds clean today (PR #1901), but the linkcheck zero state is **not enforced**. Any new chapter that references a missing target will silently regress us back to 168 broken links over the hiatus.

This spec defines a 5-phase closeout that uses **provable-contracts + pmat comply** to:

- Generate stub chapters for every public CLI subcommand + core library module
- Bind each chapter to a PCU contract (the pattern already in use for 257 `apr-page-*-v1.yaml` files)
- Enforce zero broken links via CI
- Enforce CLI-subcommand → book-chapter parity via CI

## Inventory (source of truth)

| Surface | Count | Source command |
|---|---|---|
| CLI subcommands | 103 | `apr --help \| awk '/^Commands:/{f=1;next} f && /^  [a-z]/{print $1}'` |
| Existing chapter files | 256 (post #1901) | `find book/src -name '*.md' -not -name SUMMARY.md \| wc -l` |
| SUMMARY.md entries | 250 (post #1901) | `grep -oE '\(\./[^)]+\.md\)' book/src/SUMMARY.md \| wc -l` |
| Existing PCU contracts | 257 | `ls contracts/apr-page-*-v1.yaml \| wc -l` |
| Existing book-ch contracts | 12 | `ls contracts/apr-book-ch*-v1.yaml \| wc -l` |
| Broken file links | 0 (post #1901), was 168 | `mdbook-linkcheck --standalone \| grep -c "File not found:"` |
| aprender-core public modules | TBD | `cargo doc --no-deps && find target/doc/aprender -name 'index.html'` |

## Phase 1 — Linkcheck CI gate (lock in PR #1901's work)

**Goal**: zero new broken links can land on `main`.

**Deliverables**:
- `.github/workflows/book.yml` adds a step:
  ```yaml
  - name: book linkcheck
    run: |
      cargo install mdbook mdbook-linkcheck --locked
      cd book
      mdbook-linkcheck --standalone 2>&1 | tee /tmp/linkcheck.log
      ! grep -q "File not found:" /tmp/linkcheck.log
  ```
- New contract `contracts/apr-book-linkcheck-v1.yaml` with `FALSIFY-BOOK-LINKCHECK-001`: `mdbook-linkcheck reports zero File not found errors on every CI run`.

**Estimate**: 30 min. (Mostly waiting for CI to confirm.)

## Phase 2 — Generate stub chapters for every uncovered CLI subcommand

**Goal**: every `apr <cmd>` has at least a stub chapter; no command is undocumented.

**Approach**: auto-generation script seeds each stub from `apr <cmd> --help` output. Each stub has:
1. PCU header (so contract gate passes): `<!-- PCU: cli-<cmd> | contract: contracts/apr-page-cli-<cmd>-v1.yaml -->`
2. Title (`# apr <cmd>`)
3. One-line description (from `--help` first line)
4. Synopsis (the `Usage:` block from `--help`)
5. Options table (parsed from `--help`)
6. `## Walkthrough` section with a TODO marker (`<!-- TODO: walkthrough -->`)
7. Cross-link to source code + canonical contract

**Deliverable**: `scripts/gen-cli-chapter-stubs.sh` (~80 LOC bash + python) that:
- Reads `apr --help` to get the 103 commands
- For each, checks if a chapter file matching `book/src/cli/<cmd>.md` or `book/src/examples/<cmd>.md` exists
- If not, generates the stub
- Adds to `book/src/SUMMARY.md` under a new `# CLI Reference` section
- Generates matching `contracts/apr-page-cli-<cmd>-v1.yaml` (copy of the existing PCU template)

**Estimate**: 1.5 hr (script + verification + manual review of 5-10 sample stubs).

## Phase 3 — Generate stub chapters for every aprender-core public module

**Goal**: every public Rust module in `aprender-core` has at least a stub chapter (library reference parity with CLI).

**Approach**: parse `cargo doc --no-deps -p aprender-core` JSON output. For each top-level `mod` with `pub` visibility, generate a stub:
1. PCU header
2. Title (`# Module: aprender::<mod>`)
3. One-line description (from the module's doc comment)
4. Public API table (struct / enum / trait / fn names)
5. `## Usage` section with TODO marker
6. Cross-link to the rustdoc page

**Deliverable**: `scripts/gen-lib-chapter-stubs.sh`. Same shape as Phase 2.

**Estimate**: 1.5 hr.

## Phase 4 — Book completeness contract + CI gate

**Goal**: prevent regression. A PR that adds a new CLI subcommand without a matching chapter MUST fail CI.

**Deliverable**: `contracts/apr-book-completeness-v1.yaml`:

```yaml
metadata:
  version: 1.0.0
  kind: BookCompletenessContract
  description: Every public surface (CLI subcommand + library module) has a book chapter.

equations:
  cli_chapter_count:
    formula: count(book_chapters_referencing_cli) == count(apr_subcommands)
    invariants:
      - "no apr <cmd> without a chapter"
      - "no orphan chapter without a CLI command"

  lib_chapter_count:
    formula: count(book_chapters_referencing_lib_module) >= count(aprender_core_pub_modules)

  linkcheck_zero:
    formula: linkcheck.file_not_found_count == 0

falsification_tests:
  - id: FALSIFY-BOOK-CLI-PARITY-001
    prediction: "running scripts/check_book_cli_parity.sh exits 0"
    test_harness: "bash scripts/check_book_cli_parity.sh"
    expected_output: "exit 0"
  - id: FALSIFY-BOOK-LIB-PARITY-001
    prediction: "running scripts/check_book_lib_parity.sh exits 0"
  - id: FALSIFY-BOOK-LINKCHECK-001
    prediction: "mdbook-linkcheck reports zero file-not-found errors"
```

Wired into `pmat comply` via:
```bash
pv lint contracts/apr-book-*.yaml
pmat comply check  # gates is_compliant=true on the new bookcompleteness contract
```

CI step in `book.yml`:
```yaml
- name: book completeness gate
  run: |
    bash scripts/check_book_cli_parity.sh
    bash scripts/check_book_lib_parity.sh
    pv validate contracts/apr-book-completeness-v1.yaml
```

**Estimate**: 1 hr.

## Phase 5 — README contract extension

**Goal**: the README's at-HEAD numbers stay accurate (already enforced for crates/contracts/CLI counts; extend to book).

**Deliverable**: amend `contracts/readme-claims-v1.yaml` to add:

- `FALSIFY-README-005`: README's book chapter count claim matches `find book/src -name '*.md' \| wc -l`
- `FALSIFY-README-006`: README's CLI-coverage claim matches `bash scripts/check_book_cli_parity.sh` result

**Estimate**: 30 min.

## Phase 6 — Execution validation harness (NOT in initial ship; framed honestly)

**Honest status**: Phases 1-5 enforce **structural** correctness only. The example-block gates (`FALSIFY-BOOK-EXAMPLE-001` and `FALSIFY-BOOK-LIB-EXAMPLE-001`) only check that a fenced bash/rust code block EXISTS. They do NOT check that the bash actually runs, that the rust actually compiles, or that the output matches expectations.

A chapter with `apr run nonexistent-model --broken-flag` would pass the current gates.

This is a real limitation. Phase 6 closes it.

**Goal**: every fenced code block in `book/src/cli/*.md` and `book/src/lib/*.md` is verified to compile (rust) or execute without error (bash), against a CI fixture that includes a small model cache.

**Deliverable sketch**:

1. **Per-example cost annotation** (HTML comment in each chapter):
   ```markdown
   <!-- example-cost: trivial -->                    <!-- runs in <1s, no model needed -->
   <!-- example-cost: model-required, model: 1.5b --> <!-- needs 1.5B GGUF in cache -->
   <!-- example-cost: gpu -->                         <!-- needs CUDA -->
   <!-- example-cost: destructive -->                 <!-- mutates state, MOCK in CI -->
   ```

2. **Extractor**: `scripts/extract-book-examples.sh` parses every `.md` under `book/src/{cli,lib}/`, emits a JSONL stream of `{path, language, code, cost, model_needed}`.

3. **Per-cost-class executor**:
   - `trivial`: run inline (`bash -c '<code>'`), assert exit 0
   - `model-required`: spin up Docker container with model fixture, run, assert exit 0
   - `gpu`: skip in CPU-only CI, gate on self-hosted GPU runner
   - `destructive`: rewrite to no-op (`apr publish` → `apr publish --dry-run`), assert exit 0

4. **Rust example compilation**: `cargo +stable check --edition 2021` against a generated `examples/book-<mod>.rs` wrapping each ```rust block in a `fn main() {}`.

5. **New falsifier**:
   ```yaml
   - id: FALSIFY-BOOK-EXAMPLE-EXECUTES-001
     name: bash_examples_run_without_error
     test_harness: bash scripts/check_book_examples_executable.sh
     expected_output: "exit 0; every annotated bash example exits 0"
     if_fails: "Fix the broken example, OR re-annotate its cost-class."

   - id: FALSIFY-BOOK-EXAMPLE-COMPILES-001
     name: rust_examples_compile
     test_harness: bash scripts/check_book_examples_compile.sh
     expected_output: "exit 0; every rust block compiles"
   ```

**Estimate**: 8-12 hr. Out of scope for hiatus-close window. Three pieces are missing today:

1. Per-chapter cost annotation (manual or LLM-assisted classification of 172 chapters)
2. A test harness that respects the annotation (skip-policy + sandbox + Docker fixture)
3. A model cache fixture in CI that lets `apr run qwen2.5-coder-1.5b` succeed without a ~5min per-run download

**Why it's still listed here**: so the gap is documented. Phase 1-5 enforce that every command HAS a chapter with a code-block-shaped runnable example. Phase 6 enforces that the code-block actually works. The two layers compose.

## Total estimate

| Phase | Status | Effort |
|---|---|---|
| 1. Linkcheck CI gate | ✓ shipped | 30 min |
| 2. CLI stub generation (103 stubs) | ✓ shipped | 1.5 hr |
| 3. Lib stub generation (69 stubs) | ✓ shipped | 1.5 hr |
| 4. Completeness contract + CI gate | ✓ shipped | 1 hr |
| 5. README contract extension | ✓ shipped | 30 min |
| 6. Execution validation harness | NOT SHIPPED — gap documented | 8-12 hr (post-hiatus) |
| **Total (shipped before hiatus)** | | **~5 hr** |

Phases 1-5: single sitting, delivered as one bundle PR (#1902). Phase 6 deferred to post-hiatus authoring.

## Honest scope statement (per "shape vs behavior" five-whys)

The shipped phases (1-5) enforce **structural** correctness:
- ✓ Every CLI subcommand has a chapter
- ✓ Every public module has a chapter
- ✓ Each chapter has a fenced code block (bash for CLI, rust for lib)
- ✓ Zero broken file links

They do NOT enforce **behavioral** correctness:
- ✗ The bash code in a chapter actually runs
- ✗ The rust code in a chapter actually compiles
- ✗ The output matches what the prose claims it produces

This is a real shape-vs-behavior gap. Phase 6 (above) sketches the closure plan and explicitly defers it to post-hiatus authoring. Documenting the gap is itself a contract obligation — see PR #1902 description.

## Out of scope

- **Authoring real content for the stub chapters**. Stubs are scaffolding. Content can be filled in post-hiatus without breaking the gate.
- **Cross-repo doc links** (`../../../docs/specifications/*.md`). mdbook-linkcheck can't follow outside the book root; those are inherently external links. Treated as "informational" not enforced.
- **The 184 "Potential incomplete link" warnings** (interval notation `[0, 1]` parsed as link refs). False positives; not addressed.
- **Re-authoring the 30 missing prose chapters** (Linear Regression tutorial, Toyota Way jidoka, etc.). These referenced names — the canonical pages exist under `ml-fundamentals/` — but #1901 stripped the link tags rather than relativize. A separate spec ("BOOK-CHAPTER-REVIVAL-001") covers reviving these via path corrections rather than scaffolding.

## Definition of done

### Structural (Phases 1-5 — SHIPPED in #1902)

- [x] `mdbook-linkcheck --standalone` reports 0 broken file links on every PR
- [x] `bash scripts/check_book_cli_parity.sh` exits 0 (every `apr <cmd>` has a chapter)
- [x] `bash scripts/check_book_lib_parity.sh` exits 0 (every `pub mod aprender::*` has a chapter)
- [x] `bash scripts/check_book_example_block.sh` exits 0 (every CLI chapter has a fenced bash block)
- [x] `bash scripts/check_book_lib_example_block.sh` exits 0 (every lib chapter has a fenced rust block)
- [x] `pv validate contracts/apr-book-completeness-v1.yaml` exits 0
- [x] CI workflow `.github/workflows/book.yml` enforces all gates on PR
- [x] Book builds clean: `cd book && mdbook build` exits 0
- [x] README claims accurate: `bash scripts/check_readme_claims.sh` exits 0 (extended to 6 falsifiers)

### Behavioral (Phase 6 — NOT SHIPPED)

- [ ] `scripts/check_book_examples_executable.sh` exits 0 (every annotated bash example runs without error)
- [ ] `scripts/check_book_examples_compile.sh` exits 0 (every rust example compiles)
- [ ] Per-chapter `<!-- example-cost: ... -->` annotation present on every example
- [ ] CI fixture cache contains `qwen2.5-coder-1.5b` for model-required examples
- [ ] Docker sandbox image `aprender-test` exists for destructive-class examples

## Related

- PR #1901: zero broken links (predecessor — strip-only fix)
- `docs/specifications/apr-book-spec.md`: original book authoring spec
- `contracts/apr-book-ch*-v1.yaml`: chapter-binding contracts (12 existing, extend pattern to all chapters)
- `contracts/apr-page-*-v1.yaml`: PCU contracts (257 existing, extend pattern to all chapters)
- `memory/feedback_falsify_simple_before_deep.md`: methodology — when in doubt, falsify simple explanations first
