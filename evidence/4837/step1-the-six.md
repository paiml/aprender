# #4837 step 1: the six corpus disagreements, measured live

Receipt: intel, `52f181a86c`, `make oracle` (shacl 0.3.21). The output is `step1-differential.json`.
All 20 W3C cases agree. The single disagreement is the corpus: the oracle reports 736 violations and pv reports
730. All 6 extra results are `closed`, every one is only in the oracle, and none is only in pv.

| # | focus node | contract | rdf:type values on the focus |
|---|------------|----------|------------------------------|
| 1 | csv/csv-train | csv-train.yaml | csv:Dataset |
| 2 | llm/claude-md | claude-md.yaml | llm:LlmContext |
| 3 | model/4a75ee80… | model-setfit-slice.yaml | model:AprModel |
| 4 | model/4a75ee80… | model-setfit-slice.yaml | model:Model |
| 5 | readme/readme-root | readme-root.yaml | readme:Readme |
| 6 | refusal/refusal-receipt-v1.refusals.0 | refusal-receipt-v1.yaml | refusal:Refusal |

## Cause

`check_closed` (crates/aprender-contracts/src/ontology/shapes.rs) always allows `rdf:type` on a closed shape. Its doc
comment says "declared, ignored, or rdf:type". SHACL 1.0 §4.8.1 does not do this: `rdf:type` passes a closed shape
only when `sh:ignoredProperties` lists it. The W3C node/closed-002 case lists it explicitly for that reason. The
Turtle export writes `sh:ignoredProperties` exactly as the YAML gives it, and no closed contract shape lists
`rdf:type`. So the oracle reports one `closed` result per rdf:type value. The model node has two values, which gives
two results and makes 1+1+2+1+1 = 6 exactly.

Every other predicate on the five focus nodes is declared on its shape (checked against contracts/contracts.nt).
The sixth closed shape, parity-receipt-v2's `parity-receipt-complete`, has zero ParityReceipt instances in the
corpus, so it yields no result on either side.

## Options (a ruling, not a choice made here)

- A. **Make the export say what pv checks.** For a closed shape, add `rdf:type` to the exported
  `sh:ignoredProperties`. No pv verdict changes. `contracts/shapes.ttl` is regenerated, the differential goes to 0
  and the ratchet goes from 6 to 0. Recommended: pv's documented reading stays, and the exported shapes stop
  claiming something stricter than pv enforces.
- B. **Make pv follow SHACL.** Drop the implicit `rdf:type` allowance, and add `rdf:type` to `ignoredProperties` in
  the 5 contracts. This is stricter, because a closed shape that omits it now fails. It changes what the gate
  accepts, so it is a sign-off row.
