# PMAT-3777 implementation receipt: #3745 S2, release cells DERIVED from apr's own surface

**Read this before judging scope.** `pmat work status PMAT-3777` prints only the ticket's title. The criteria are
the issue's, https://github.com/paiml/aprender/issues/3777, quoted verbatim from #3745 (issuecomment-5764661626 and
the S2.5 amendment issuecomment-5765674584):

- [ ] S2.1 A pv extractor `cli-surface`, registered implemented in `contracts/ontology.yaml`, with its own pc_extract planted control (the #3706 Σ-tie).
- [ ] S2.2 The cell set is computed: every model-role command × host-inventory models × input shapes × mode args (a deterministic pairwise covering array) × thinking × rung. Commands without a model owe one runs-or-refuses cell. The derived cell count is printed in the receipt.
- [ ] S2.3 `release-readiness-v1` targets the derived set through `Inputs.cells`, and `release_evidence.rs`'s `VERBS` const is **deleted**. Issue mutants 1 and 2 are RED.
- [ ] S2.4 `:CruxCell` plus the per-verb CRUX mapping entry (#3739): a derived verb with no mapping entry is a violation.
- [ ] S2.5 The flag-effect oracle: for every derived **mode** arg, some cell must exist where toggling it changes the observable output: stdout bytes, the JSON fields, or the token ids. A mode arg with no observable effect on any cell is RED unless its typed marker (S1) declares it a no-op for that model class, with the class named. Sampling args additionally carry a8's controls: sampled ≠ greedy for some seed; the same seed ×2 byte-identical; seed A ≠ seed B; top-k 1 / T 0 byte-greedy.

**The design rulings this implements** (the cop, on my measured design note #3745 issuecomment-5765981405):
- **D1**: pairwise over modes ∪ {input shape, thinking, rung} per (command, model, host), ≈35k cells rather than ≈280k for the literal product, plus a per-host projected wall time from measured medians, never guessed.
- **D2**: a typed `generates` field on the surface. fc added it (apr-cli-surface/v1.1, #3745 amendment 3); embed, rerank and eval are non-generating via the typed `EncodeText` marker.
- **D3**: one-factor-at-a-time cells for S2.5 per representative model per architecture, plus `output_sha256`.

The receipt interface is agreed with aprender-62 (the producer; S3 has ONE judge, pv) and with aprender-76 (the CRUX correspondence file and receipts, #3739).

## What the diff does, criterion by criterion
- **S2.1** `ontology/extract/cli_surface.rs`: the Σ entity type `cli-surface` (registered in `contracts/ontology.yaml`). Its pc_extract control is drawn every run: a planted surface must type `run` as a ModelCommand when its arg's ROLE is `model`, and not when it is `other`. #3706's Σ-tie test passes. An unknown schema or role is refused by name.
- **S2.2** `covering.rs` (deterministic IPOG, strength 2; `conflicts_with` pairs never share a row) and `release_cells.rs::derive`:
  - every model-role leaf command × each host's measured models × a covering array over its mode args (own plus globals) ∪ {input shape} ∪ ({thinking, rung} when `generates`);
  - commands without a model owe one ProbeCell per host (runs, or refuses by name);
  - the derived cell count is printed (`release.cells`), and `pv extract --cells-out` writes the list.
- **S2.3** `release_evidence.rs`:
  - the **`VERBS` const is deleted** and the derived set flows through `build(…, &Inputs { surface, … })`;
  - rows key by `cell_id` alone;
  - classes Cell, RefusalCell, ModelCell, ProbeCell and EffectCell, each with its own shape;
  - the release node needs a surface.
  - **Issue mutant 1** (a new flag on a generating command owes cells nobody ran) and **issue mutant 2** (the `--prompt` + `--chat` cell missing) are RED, and each is a CLI case.
- **S2.4** `release_crux.rs`:
  - the correspondence file (`evidence/crux/verb-correspondence.yaml`, agreed with 76) needs an entry per derived verb covering every engine the file declares;
  - every (host, model, mapped verb, thinking, rung) owes a `:CruxCell`;
  - RED, UNJUDGED and absent are violations; ALL_WRONG is named, not a violation, pending the cop.
- **CRUX request modes** (#3739 slice 4, agreed with 76): a correspondence entry may DECLARE `modes:` (the `serve run` entry declares `[nonstream, stream]`). Each CRUX obligation of that verb then owes ≥ 1 `:CruxCell` per declared mode, and a missing one is named `modeMissing`. pv never names the verb that streams.
- **S2.5** `release:ModeEffect`: per (command, mode arg, level), it must be observed in an effect cell whose `output_sha256` differs from its base's, and a flag with no observable effect is RED.
- **S2.5 sampling (a8's controls)** follow the cop's ruling, "SamplingArg: yes … their role becomes `sampling`, derived from construction", which fc landed at 17dfb1291 as role `sampling` + `sampling_kind`.
  - Per generating command with typed sampling args × representative model per arch × host, there are five control cells (`t0`, `topk1`, `seed-a`, `seed-a-again`, `seed-b`; the knob VALUES are the controls' own constants, never apr names), plus a `release:SamplingCheck`.
  - The checks are: sampled ≠ greedy; the same seed ×2 is identical; seed A ≠ seed B; top-k 1 == T 0.
  - A control the command cannot express (no typed knob of that kind) is named `underivable`, never passed.
- **#3748** (62's F2 requirement): a generating Cell needs `f2Measured`, meaning F2 ran fresh or from THIS binary's own receipt (`f2_source`, `f2_receipt_binary_sha` full 40-hex).
- **Cop ruling on CRUX ALL_WRONG:** named and counted per model, never a violation. A `positive_control` row that is ALL_WRONG, or a model with no measured control, DECLINES the gate (exit 2, `ShapesOutcome::HarnessBroken`).

## Measured
| check | result |
|---|---|
| derivation on fc's real surface (7ed033b63, v1.1) with the 16 ladder host-model pairs | 14,984 cells (11,176 matrix, 3,042 effect, 390 base, 376 probe) in 9.2 s, 166 MB; `run` covers all 12 (thinking, rung) pairs per model and host |
| shrink-only ceilings (`evidence/release/surface-ratchet.json`) | unknown_args 259 (all foreign), stdin_undeclared 71, measured on the same surface |
| `cargo test -p aprender-contracts --lib` | 1709 passed |
| `cargo test -p aprender-contracts-cli` (every target) | green; `ont_release_readiness` has 30 cases (the green base is synthesized from pv's own derived list) |
| `cargo clippy -p aprender-contracts -p aprender-contracts-cli --lib --tests -- -D warnings` | clean |
| mutant: the covering array skips its vertical pass | 2 covering cases FAIL |
| mutant: derivation drops the input-shape factor | issue-mutant-2 case FAILS |
| mutant: every flag counts as observed | the flag-effect case FAILS |
| the ont4b shape ratchet | 18 → 25; the seven new shapes are named in the test |

## Not in this diff (so it carries `Refs`)
- Wiring at T-1 / T-4 (aprender-f0), and migrating #3712 B1 to pv as the one judge (aprender-62, S3.4).
- The #3715 proof at `evidence/release/proof-0.69.0/` is #3715's historical result over the retired verb list. S2's universe supersedes it.
