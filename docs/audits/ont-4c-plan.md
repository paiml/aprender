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

---

# Round 2 — what the first grill refused, and the one question it did not settle

One countable lane (`gemini-3.1-pro-low`, SUCCESS, witness verified) returned **FAIL with 5 cited findings**,
one per decision. Five further gemini envelopes across three rounds carried FAIL too and were voided by 429s
or by an external ref move; they are uncounted but they converge. The only PASS came from an author-family
claude lane, ineligible on independence grounds. **I accept the FAIL. The plan above is withdrawn.**

The findings, and what I have done with each:

| # | Finding | Action |
|---|---|---|
| D1 | omitting `parityReceipt`/`arch`/`quant`/`contextLength` leaves a contract its target trivially satisfies | **accepted** — see below |
| D2 | `golden_v2.apr` is the `pc_extract` control's own bytes | **accepted** — the entity becomes `crates/aprender-serve/models/mnist_784x2.apr` |
| D3 | dropping `csv:producer` to fit `iris.csv` weakens B.8 | **accepted** — the entity becomes a CSV that HAS a producing step, and the constraint stays |
| D4 | grading an author-declared `verified_commands` list is shrinkable at will | **accepted as a refutation; its proposed fix is unsatisfiable — measured below** |
| D5 | same shape for `referencedPath` | **accepted, and the fix is satisfiable** — measured below |

## D5 is satisfiable, so it is simply done

`CLAUDE.md` mentions **45** distinct backticked slash-bearing paths. **39 exist; 6 do not**
(`aprender-contracts/src/schema/`, `.cargo/config.toml`, `entrenar/cuda`, `examples/qwen_inference.rs`,
`realizar/cuda`, `src/models/`). So grading every path in the PROSE — no frontmatter list — costs six
corrections, and at least two of the six (`entrenar/cuda`, `realizar/cuda`) are not paths at all but
crate/feature pairs, which is itself the kind of thing a path constraint should surface. Adopted as the lane
wrote it.

## D4's fix is unsatisfiable on this corpus, and that is a measured fact about the ROW

The lane's fix was "grade every fenced command present in the actual README file". Measured on this tree:

- README has **18** fenced blocks: 14 `bash`, 1 `yaml`, 1 `toml`, 1 `rust`. (Using the fence INFO STRING, not
  a first-word heuristic, already kills the false positives — a `toml` fence's `aprender = "0.35"` is not a
  command.)
- The 14 `bash` fences hold **43 distinct command lines**.
- **6 of the 43 appear verbatim in a workflow `run:` step. 37 do not.**

The 37 are things like `apr chat qwen2.5-coder-1.5b-instruct-q4k` (an interactive REPL),
`apr pull hf://Qwen/Qwen2.5-Coder-0.5B-Instruct` (a network fetch) and `apr --help`. ONT-4c's RED clause says
a fenced command appearing in no workflow `run:` step is a **reject**. Applied literally to every fenced
command, that clause cannot be satisfied by a product README: the only ways to green it are to delete the
usage examples, or to add 37 workflow steps that run them — CI theatre, a worse defect than the one the
clause is aimed at.

So there are three candidate readings and **I want the quorum to pick one**, not me:

**R-A — two predicates over the complete fenced set, neither author-declared.** Every `bash`-fenced command is
extracted. One that resolves to a workflow `run:` step is emitted as `readme:verifiedCommand`; one that does
not is emitted as `readme:documentedCommand` and resolves instead against the binary's own command surface
(`apr --help` / `pv --help` subcommand list), so a README documenting a verb the binary does not have is a
reject. Nothing is author-declared; the author cannot shrink either set without editing the fences, which is
a visible change to the graded document. **Cost:** it does not implement the row's RED clause literally — a
non-CI command is classified rather than rejected.

**R-B — the row is literal, and the entity is not this README.** Keep `resolves: ci-step` over every fenced
command exactly as written and accept that aprender's root README cannot be the `readme` entity; contract a
document whose fenced commands genuinely are CI steps. **Cost:** §4.1 names `contracts/readme-root.yaml` and
the probe asserts `present '^---' README.md`, so this reading contradicts the row's own artifacts.

**R-C — the row is right and the README is wrong.** Take the 37 as 37 real findings and fix the README: move
interactive and network examples out of `bash` fences into `text` fences, leaving only commands CI runs.
**Cost:** a large README rewrite inside a row about extractors, and a fence language chosen to dodge a gate is
the author-shrinkable move in a different costume — unless the rule is exactly that `text` fences are not
commands, which is defensible but must be said out loud.

Grade R-A, R-B and R-C against one question: **which one leaves a gate that can fail on a future README, and
by what edit?** Name the edit. If your answer is R-A, say whether classifying rather than rejecting is a
softening of ONT-4c's RED clause that needs the SPEC amended (infra, `docs/specifications/paiml-ontology.md`)
rather than absorbed silently here — I would rather amend the row in the open than implement something the
row does not say.

Then re-grade D1 with the entity fixed to `crates/aprender-serve/models/mnist_784x2.apr`: no tracked `.apr`
has any ladder receipt, so `model:parityReceipt minCount: 1, resolves: receipt` cannot pass on any of them
today. Is the honest move to (i) drop `parityReceipt` and carry a different non-trivial constraint, (ii)
produce a real receipt for that file in this PR, or (iii) declare the row `blocked_on:` a receipt the way
R-25 made ONT-4c4 declare its blocker? Name which, and why the other two are worse.
