# PMAT-3847 / ONT-4c — implementation plan to grill

Row text is `docs/specifications/paiml-ontology.md` §5 ONT-4c in paiml/infra (v4.11); worked contracts
in Appendix B.4 (readme), B.5 (llm-context), B.6 (apr-model), B.8 (csv); extractor contract in §3.7.

## Measured baseline (aprender 8c7822f34, in-tree pv 0.69.0, `pv lint contracts/ --gate shapes --format json`)

    verdict Pass
    by_entity_type  apr-model 0 · code 270 · gguf 8 · lean 410 · parity-receipt 7 · pv-contract 1764
                    (no readme, llm-context or csv key exists at all)
    pc_extract      8 controls, all "fired"

The probe this row must satisfy:

    tracked contracts/readme-root.yaml && tracked contracts/claude-md.yaml &&
    pv lint contracts/ --gate shapes --format json | jq -e '
      .by_entity_type.readme>=1 and .by_entity_type["llm-context"]>=1 and
      .by_entity_type["apr-model"]>=1 and .by_entity_type.csv>=1 and
      .pc_extract.readme=="fired" and .pc_extract["apr-model"]=="fired" and .verdict=="Pass"'
    && present '^---' README.md && merged ONT-4c

## Plan

- **P1 `extract:readme`** — `ontology/extract/readme.rs`: frontmatter (`schema_version`, `kind`,
  `entrypoints`, `verified_commands`, `contract_count`) + `##` headings + fenced commands →
  `readme:schemaVersion/kind/entrypoint/verifiedCommand/section/contractCount`. `resolves: path` → the
  entrypoint is in the tree; `resolves: ci-step` → each verified command appears verbatim in a workflow
  `run:` step. `contracts/readme-root.yaml`; frontmatter added to `README.md`; `readme_gen` emits it from
  census so the README cannot drift from its own contract. Positive control: a README citing a command no
  workflow runs.
- **P2 `extract:llm-context`** — `llm_context.rs`, `llm:section/referencedPath/command/declaredTool/neverRule`;
  `contracts/claude-md.yaml`; frontmatter on `CLAUDE.md`. Control: a nonexistent referenced path.
- **P3 `extract:csv`** — `csv.rs`, `csv:column{name,dtype}/header/rows/sha256`; one `contracts/csv-<name>.yaml`.
  Control: a header/row column-count mismatch.
- **P4 the `.apr` contract + Σ** — `contracts/model-<name>.yaml` over a TRACKED `.apr` (reuses ONT-4c1's
  `apr_model.rs`, which already fires its control); Σ `entity_types` marks readme, llm-context, apr-model, csv
  `implemented: true`; three new entries in `extract_controls()` (a committed test already asserts
  `pc_extract` keys == Σ's implemented types, so this cannot drift).
- **P5** dogfood R-23 (both corpora), receipt, acceptance criteria written from the diff.
- **P6** PR, quorum, arm.

## The five decisions I want grilled — I have NOT taken them

1. **No tracked `.apr` file has a ladder receipt.** All four receipts under `evidence/dogfood/models/*/`
   carry GGUF `rungs[]` (`qwen2-1.5b-q4km`, …); the ten tracked `.apr` files (`crates/apr-format/tests/fixtures/golden_v2.apr`,
   `crates/aprender-serve/models/mnist_784x2.apr`, `crates/aprender-tsp/models/berlin52-aco.apr`, …) appear in
   none of them. B.6's worked contract requires `model:parityReceipt minCount: 1, resolves: receipt`.
   **Proposal: the shipped `model-*.yaml` omits `parityReceipt`** (and `arch`/`quant`/`contextLength`, which an
   `.apr` header does not carry for these files) and keeps `tensorCount`, `sha256`, `format`; the row's RED
   clause "`model:parityReceipt` with no PP-LLAMA receipt for the sha → reject" is satisfied by a FIXTURE
   contract that does claim one, not by the shipped contract. **Is that the row, or is it weakening it?**
2. **Which `.apr`?** `crates/apr-format/tests/fixtures/golden_v2.apr` is the file §3.7 already names as the
   `pc_extract["apr-model"]` control, so the entity and the control would be the same bytes; a product model
   like `crates/aprender-serve/models/mnist_784x2.apr` is a truer "entity under contract" but is not the
   control. Which?
3. **Which CSV?** Nine are tracked. `crates/aprender-core/src/datasets/iris.csv` (a dataset the library ships)
   vs `evidence/task-132-residual-b/nvidia-smi-during-run.csv` (evidence, the shape B.8 was written for, but
   machine-generated and wide). B.8 also requires `csv:producer resolves: ci-step` — **for iris there is no
   producing CI step at all.** Drop `producer` for a shipped dataset, or pick a CSV that has one?
4. **`readme:verifiedCommand resolves: ci-step` against the REAL README.** aprender's README is long and its
   fenced blocks hold many commands no workflow runs verbatim. Proposal: the frontmatter's
   `verified_commands` list is the closed set the shape grades (author-declared, each checked against the
   workflows), NOT every fenced command in the file; the extractor still emits every fenced command as
   `readme:section`-adjacent data. **Does that make the gate vacuous** — an author can simply declare two
   easy commands — and if so what is the non-vacuous rule that still passes on the real README?
5. **`llm:referencedPath resolves: path` against the REAL CLAUDE.md**, same question: every path mentioned in
   prose vs a declared frontmatter list. And `llm:section in [Purpose, Rules, Commands, Layout, Never]` is a
   CLOSED set that aprender's actual CLAUDE.md headings do not match.

For each: answer with the decision you would take, the reason, and — this is what I most want — **the way it
could be wrong**, i.e. what a later reader would find if the decision is the lazy one. Judge whether each
proposal keeps the row's assertion or hollows it out. A decision that makes the probe pass while measuring
nothing is the failure mode this repo is built to refuse.
