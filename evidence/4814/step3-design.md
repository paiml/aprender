# #4814 step 3 design: adding the missing SHACL Core components

- **Base:** branch `feat/4814-pv-shacl-core` @ `60b79e4a63` (step 2, fail closed)
- **Inputs:** `baseline.md` in this directory; the W3C suite @ `976ed12ad3`; `crates/aprender-contracts/src/ontology/{shapes,rdf,w3c}.rs`
- **Written:** 2026-10-05 by aprender-wshacl. This is a design only: no code has been written for it.

## Rules every slice follows

1. **One component group per commit.** Each commit vendors the group's W3C cases as `w3c/*.yaml`, adds them to
   `CASES`, and removes them from `NOT_VENDORED`.
2. **`make oracle` green**, and `tests/oracle/differential.json` regenerated in the same commit. The oracle is
   pinned to `shacl =0.3.21`, so the slice's cases are checked against an outside engine and not only against
   pv itself.
3. **One mutant per group**, applied with a script that aborts if the patch did not apply. Each mutant must turn
   at least one of the group's vendored cases RED. The kill is recorded in `evidence/4814/`.
4. **A key is added to `PROPERTY_KEYS` or `NODE_KEYS` in the same commit as its validator arm, and never
   before.** Until that commit, the key stays `Unsupported` (exit 3), so the gate stays fail-closed throughout.
5. **Ratchet:** the vendored case count only grows, and `NOT_VENDORED` only shrinks (slice 0 makes this
   meaningful).
6. **Gate rules:** each slice changes what the gate accepts. Each one is built, gets two reviews, and gets one
   row in the cop's sign-off table. None is merged before Noah signs it off.

## Slice 0: account for all 98 cases (no change to what the gate accepts)

Today `the_table_of_ont_0_is_accounted_for_case_by_case` pins `CASES + NOT_VENDORED == 35`. That leaves 63 suite
cases in neither list, so "NOT_VENDORED only shrinks" proves nothing.

- Add all 63 to `NOT_VENDORED`, each with the reason it is out: the component, target or path kind it needs.
- Change the pin to 98.
- Add a test that reads the case ids from a vendored copy of the 7 manifests' `mf:include` lists, so the 98 is
  derived and not typed in. This copy is a plain list file; there is no Turtle reader on the gate path (R-13).

This slice is a test-only change, and it is the denominator for every ratchet that follows.

## Slices, in order

Each case count below is measured from the suite. "Unlocks" counts the cases this slice alone makes vendorable.
Cases that also need a later slice are counted under the later one.

| # | slice | new keys | cases it unlocks | cost / risk | serves |
|---|---|---|---|---|---|
| 1 | value range: min/maxExclusive, min/maxInclusive | property keys | property/{minExclusive-001,-002, maxExclusive-001, maxInclusive-001}: **4** | low. Reuses `compare_terms` (SPARQL `<`). An incomparable value is a result, never a skip, the same rule as `check_pairs` | — |
| 2 | equals, disjoint | property keys | property/equals-001, property/disjoint-001: **2** | low. Reuses `graph.objects(focus, other)`. Term equality must be exact on (value, datatype) | **#4600** (sh:equals) |
| 3 | hasValue | property key | property/hasValue-001: **1** | low. One literal or IRI, with the same scalar typing as `in_entry` | #3715 (release-readiness shape) |
| 4 | targets: targetNode, targetSubjectsOf, targetObjectsOf, implicit class target | node keys | targets/*: **6** of 7 (multipleTargets needs `in` + targetSubjectsOf). It also retires the `targetNode → rdf:type ex:Focus` rewrite that every vendored case uses today | medium. `validate` changes from one class walk to a union of focus sets. Needs a reverse lookup, `Graph::subjects(pred)` | translation fidelity of every case |
| 5 | constraints on a node shape itself (the focus node is the value) | node keys mirror property keys | node/{minInclusive-001..003, min/maxExclusive-001, maxInclusive-001, equals-001, disjoint-001, hasValue-001, languageIn-001}, plus about 12 of the 16 current `NOT_VENDORED`: **≈20** | medium. Factor `check_value` so it takes a value set and a path that is `Option`; a node shape has no `resultPath` | broadest unlock |
| 6 | inversePath | path form `^p` in a mapping `{inverse: p}` | path/path-inverse-001, path-strange-001/-002, path-complex-002: **4** (personexample also needs it) | low once slice 4's `Graph::subjects` exists | — |
| 7 | sequence, alternative, zeroOrMore, oneOrMore, zeroOrOne paths | path as a YAML structure, never a string | path/*: **7** | medium. Closures need a cycle-safe walk. **The string forms stay Unsupported (F8)**: a path expression is only ever a YAML structure | — |
| 8 | not, and, or, xone | node keys holding shape refs | node/{and,not,or,xone}-*, property/{and,not,or,or-datatypes}-*: **≈11** | high. A shape becomes a value: needs named shapes and a recursion guard (SHACL leaves recursion undefined, so pv refuses a cycle as Malformed) | — |
| 9 | qualifiedValueShape, qualifiedMin/MaxCount, qualifiedValueShapesDisjoint | property keys | property/qualified*: **3**, node/qualified-001 | high. Needs slice 8's shape-as-value | — |
| 10 | languageIn, uniqueLang | property keys | property/languageIn-001, uniqueLang-001/-002, node/languageIn-001: **4** | **needs a `Term` change**: see the finding below | — |
| 11 | misc: sh:Info, deactivated, message | node+property keys | misc/*: **4** | low | — |

`complex/shacl-shacl` (the SHACL-for-SHACL shapes graph) and `validation-reports/shared` stay in `NOT_VENDORED`
for good, with that reason. Both need a shapes graph read as RDF, and the gate path has no Turtle reader (R-13).

**Order rationale:** slices 1 to 3 are cheap property-level components that reuse existing machinery, and slice 2
is a named dependency (#4600). Slice 4 removes the translation rewrite before the case count grows. Slice 5 has
the biggest unlock. Slices 8 and 9 are last among the components because they are the only ones that change
the engine's model (a shape becomes a value).

## Finding during design: language tags are dropped (a fail-closed hole, F9)

`w3c.rs::object_term` reads `"chat"@fr` as `Term::Literal { value: "chat", datatype: rdf:langString }`. The tag
is lost. `rdf::Term` has no field for it.

Consequence today: `"chat"@fr` and `"chat"@en` are the same term. Any `in`, and the future `equals`, `disjoint`
and `hasValue`, would treat them as equal. Nothing in the corpus uses language tags today, so this is latent.
It is still a silent wrong answer, not a refusal.

Proposal, inside slice 10 and not before: `Term::Literal` gains `lang: Option<String>`. Until then, add one line
to step 2's fail-closed set: **a langString literal reaching any comparing component is Unsupported.** That
should be a ruling-free tightening, because it refuses input that no contract has. It still changes what the
gate accepts, so it goes in the same sign-off row as step 2.

## Uses (plan step 5), checked against the slices

- **#4600 sh:equals (checked 17:55Z against the issue body):** the shape is `sh:equals (:inputBytesSent
  :inputBytesConsumed)` on every verb receipt, and "a Pass without these fields is a violation". Slice 2 gives the
  first half. `equals` alone passes when BOTH fields are absent (two empty sets are equal), so the second half is
  `minCount: 1` on each field, which pv already has. Equality is exact on the term: the extractor must write both
  counts with the same datatype (`xsd:integer`), or a correct receipt reads as a violation. Slice 5 is not needed.
- **#3715 release-readiness (checked 17:56Z):** `hasValue` (slice 3) covers `verdict=Pass`, `backend=cuda` and
  `fallback=false` (YAML `false` is `xsd:boolean`, so the extractor must type it the same way). It does not cover the
  rest:
  - "exactly one `:Receipt` per (verb, context rung)" is `qualifiedValueShape` + `qualifiedMinCount 1` /
    `qualifiedMaxCount 1`: **slice 9**, which needs slice 8.
  - "`apr_sha` = release commit" is a value known only at release time: either a generated `hasValue` or an
    `equals` against a release node's property. That is a design question for #3715, not for #4814.
  - "focus = every `:Model` in the inventory of every required `:Host`" is a `targetObjectsOf` target: **slice 4**.
  - `hasValue` has no IRI form yet. If the extractor writes `verdict` as an IRI (`ont:Pass`), slice 3 cannot express
    it and an IRI form must come first.
- **#3611 lessThan:** done on main, with 3 cases vendored. Its only gap is the stale `differential.json`, which
  slice 1's first `make oracle` regeneration closes. Check it, but do not close it (brief).

## Finding during slice 2: subsumption never checks the property pairs (F10)

`lint/subsumption.rs::weakened` covers counts, datatype, class, nodeKind, pattern, in, the lengths, and, from
slices 1 and 2, the range bounds and `equals`/`disjoint`. It has no `lessThan` or `lessThanOrEquals` row (#3611
added the components but not their weakening checks). So a sub-shape that drops a super-shape's `lessThan` is
not reported. Adding the two rows is stricter, so it changes what the gate accepts and needs a sign-off row. It is
recorded here and not folded into slice 2.

## Slice 2 note: language tags (F9) on `equals` / `disjoint`

Slice 2 applies the F9 rule locally: a language-tagged value is a result on both components, never a pass,
because `Term` cannot tell `"a"@en` from `"a"@fr`. The general fail-closed rule (for `in`, `hasValue`, and the
range components, which already refuse it through `compare_terms`) is still part of slice 10.

## Slice 5 design: constraints on a node shape itself (written 18:55Z, before slices 1 to 4 have compiled)

Survey of the 21 `node/*` cases that wait on slice 5 (from the suite @ `976ed12ad3`):

- 19 of the 21 name their focus with `sh:targetNode`, and at least 14 of those 19 target a **literal** (counted from the
  first non-`ex:` `targetNode` line of each file only) (`7`, `3.9`,
  `"Aldi"`, `"…"^^xsd:dateTime`, `"<span>…</span>"^^rdf:HTML`, `"true"^^xsd:boolean`). Two (`in-001`,
  `node-001`) have no target at all, so they need no slice-5 work on targets. They need the implicit class
  target or none, and they stay out.
- So slice 5 needs a **literal `targetNode`** first. Slice 4 reads `targetNode` as an IRI only. A YAML string
  cannot mean both, because in `in` and `hasValue` a YAML string is an `xsd:string` literal. So the literal form
  is a mapping, `{literal: "7", datatype: xsd:integer}`, and a bare string stays an IRI. `Targets::nodes` becomes
  `Vec<Term>`. A literal focus is still named by its N-Triples form, as `targetObjectsOf` names it in slice 4.

Engine:

- `NODE_KEYS` gains the value components that make sense on the focus node: class, datatype, nodeKind,
  `in`, pattern, minLength, maxLength, the four range bounds, hasValue, equals, disjoint, node.
  `languageIn` stays refused (slice 10, F9).
- A node shape's value set is `{focus}`. `check_value` is factored so it takes the value set and a path that is
  an `Option`. A node-level result has no `resultPath` (W3C: none), which `Expected.path: None` already models.
- equals and disjoint on a node shape compare `{focus}` with the focus's values of the named property.
- The gate still refuses every node-level constraint in a contract shape, as it refuses non-class targets since
  slice 4. Its plant creates an IRI node of the target class, so a node-level `datatype` or `pattern` would be
  planted against a value it can never match. The contract gate's accepted set stays unchanged.

### Finding F11: YAML numbers are not Turtle numbers

`in_entry` types a YAML float as `xsd:double`. In Turtle a bare `3.9` is an `xsd:decimal`, and a bare `7` is an
`xsd:integer`, which agrees. The range components compare across numeric types (`compare_terms`), so the
mismatch is harmless there. But `in` and `hasValue` compare terms exactly, so `hasValue: 3.9` never matches a
decimal `3.9` in the data. That is a silent wrong answer, not a refusal.

Proposal: in slice 5, the `{literal, datatype}` mapping is accepted wherever a scalar term is (`in`, `hasValue`,
`targetNode`), so a translation can write the exact type. A bare YAML float in `in` or `hasValue` becomes
`Malformed` ("write `{literal, datatype}`"). That is a tightening: today no contract uses a float there. It
changes what the gate accepts, so it gets its own sign-off row and is not folded silently into slice 5.

Expected unlock: up to 18 of the 21, if the literal `targetNode` and the mapping form land with slice 5. Excluded
are `in-001` and `node-001` (no target) and `nodeKind-001` (blank-node kinds). `languageIn-001` is not among the 21;
it waits on slice 10.
Each count is to be re-measured against the vendored YAML before the commit says so.

## Slice 5 as built (measured, not compiled yet)

- Node-level constraints (SHACL §2.1): a node shape may carry `class, datatype, nodeKind, in, pattern, minLength,
  maxLength, min/maxExclusive, min/maxInclusive, hasValue, equals, disjoint, node`. The value set is `{focus}` and
  the result has no `resultPath`. `languageIn` stays `Unsupported` there too.
- Literal `targetNode`: `{literal: "7", datatype: xsd:integer}`, exactly those two keys and no default datatype. A
  bare scalar stays an IRI. The mapping form is accepted only under `targetNode`. The `in`/`hasValue` mapping and the
  bare-float `Malformed` above are NOT in this slice. They tighten the gate, so they wait for their own sign-off row.
- The gate (`shapes_gate::refuse_node_level`) refuses node-level constraints in a contract shape, nested `node`
  shapes included, with `Unsupported{component: first key}`. No contract's verdict can change, so there is no
  sign-off row for this slice.
- Unlock measured against the vendored YAML: 7 cases, not 18. They are node/class-001, disjoint-001, equals-001,
  hasValue-001, minInclusive-001, node-001 and nodeKind-001. CASES goes from 29 to 36 and NOT_VENDORED loses 7. The
  rest keep written reasons: blank-node focus (R-15), language tags (F9), IRI `in` members, two `sh:class` values,
  `sh:flags`, dateTime time zones, and the Warning-only harness. The 18 estimate counted cases that need those.
- The mutant for the receipt (MUTANT 7) is to skip the `own` block in `validate_focus`. It must turn
  `every_embedded_case_parses_and_passes` RED.

## Slice 5 receipt (intel, 8b6f111b6f, 2026-10-06 04:00Z)

The first run at 8f6acc4ff7 had 2300 passing and 1 failing. That failure was an order-only assert: slice 5 lists literal
focus nodes after the IRIs. 8b6f111b6f sorts the result before comparing, and its rerun is green:
- `cargo test -p aprender-contracts --tests`: 2538 passed, 0 failed.
- pv builds, and the shapes gate exits 0.
- Mutants 1-8 each turn their test RED. Mutant 7 (the node-level block) and mutant 8 (`refuse_node_level`) are new.
- `aprender-contracts-cli --tests`: 39 binaries, 0 failed.
- The pinned oracle (shacl 0.3.21) agrees on 36/36 W3C cases. The only disagreement is the corpus 736 vs 730, which is
  #4837.
- No contract changes its verdict.

## Slice 6 as built: inverse path (written 2026-10-06 08:40Z, before compiling; the receipt follows)

**Finding F13: the table overcounted slice 6, which unlocks 1 case and not 4.** Read from the four case files at
976ed12ad3:
- `path-strange-001` and `-002` give a path node that is both an RDF list and an `sh:inversePath`. W3C grades the
  results by the sequence `( ex:p ex:q )`.
- `path-complex-002` is a sequence of two inverse paths.

All three need slice 7's sequence path. Their `NOT_VENDORED` reasons now say so. Only `path/path-inverse-001` is
inverse-only, so CASES goes from 36 to 37.

- **Form:** `path: {inverse: <predicate>}`, a YAML mapping and never a string (F8). The values are the subjects `s`
  of `s p focus`, and a result's path is `^<p>`. Any other path mapping keeps the parser's pre-slice-6 error word for
  word ("a property has no `path`").
- **Closed shapes:** an inverse property adds no predicate to the allowed set (SHACL §4.8.1, IRI paths only).
- **Export:** Turtle writes `sh:path [ sh:inversePath <p> ]`.
- **Gate:** `refuse_inverse_path` refuses an inverse path in a contract shape, nested `node` shapes included. It
  uses the pre-slice-6 text, so the gate accepts and prints exactly what it did. R-19's subsumption matches
  properties by predicate alone, and `Graph::subjects` is a full scan.
- **The receipt checks:** the shapes-gate JSON must be byte-identical to slice 5's (durations aside). Mutants 9 to 15
  cover each of the parts above and must each turn their test RED.

## Slice 6 receipt (intel, dbbda70b17, 2026-10-06 12:39Z)

The cli step was stopped three times before it finished: twice by intel load alarms (11:15Z and 12:05Z), and once
by my own guard, which re-read a replayed alarm line. The steps before it come from the first run at the same commit.
- `cargo test -p aprender-contracts --tests`: 2543 passed, 0 failed, in 16 binaries (slice 5 had 2538).
- pv builds, and the shapes gate exits 0. Its JSON differs from slice 5's only in `w3c_cases_passed`, 36 to 37.
  Everything else is byte-identical, so the gate accepts and prints what it did.
- Mutants 9-15 each turn their test RED (rc 101, 1 failed).
- `aprender-contracts-cli --tests`: 592 passed, 0 failed, in 39 binaries. pv sha256 `ee9c9261…71216cf`.
- The pinned oracle (shacl 0.3.21) agrees on 37/37 W3C cases, `path-inverse-001` included. The only disagreement
  is still the corpus, 736 vs 730, which is #4837.
- No contract changes its verdict.

## Slice 7 as built: sequence path (written 2026-10-06 14:40Z, before compiling; the receipt follows)

Read from the six slice-7 case files at 976ed12ad3 (fetched from the suite's raw URL; `path-inverse-001` from the
same URL is byte-identical to the vendored copy, so the source is the one slice 6 used):
- `path-sequence-001`, `-002` and `path-sequence-duplicate-001` are plain forward sequences.
- `path-complex-002` is the sequence `( [ sh:inversePath ex:p ] [ sh:inversePath ex:p ] )`, written once through
  a shared blank node and once inline. Both shapes read the same path.
- **Finding F14: `path-strange-001` and `-002` stay out.** Their path node is both an RDF list and an
  `sh:inversePath`, which SHACL §2.3.1 calls ill-formed. W3C grades them as the sequence `( ex:p ex:q )`. The YAML
  form cannot write such a node, and writing it as the sequence would be `path-sequence-001` again under another
  name. Their `NOT_VENDORED` reasons now say this. CASES goes from 37 to 41, and NOT_VENDORED drops by 4.

- **Form:** `path: {sequence: [<step>, <step>, …]}`, at least two steps. Each step is a predicate or
  `{inverse: <predicate>}`. A step that is itself a path expression (`ont:b/ont:c`) is refused as a lone path is.
  A nested sequence, an alternative, a one-step or empty sequence, or a non-string step keeps the parser's
  pre-slice-6 error word for word ("a property has no `path`").
- **Model:** `PropertyShape` keeps `(path, inverse)` as the first step and adds `then: Vec<PathStep>` for the rest.
  `is_predicate()` (forward, one step) is what a closed shape, the gate and R-19 read.
- **Values:** a walk from `{focus}` over a `BTreeSet<Term>`. A forward step maps an IRI node to its objects and a
  literal to nothing; an inverse step maps any node to its subjects. The set is what makes
  `path-sequence-duplicate-001` one value (SHACL §2.3.1: a path's value nodes are a set).
- **Result path:** `^<p>` for one inverse step, as in slice 6; `(<s1> <s2> …)` for a sequence, with full IRIs and `^`
  on an inverse step. The W3C harness reads an expected `(ex:p ^ex:q)` the same way.
- **Closed shapes:** a sequence adds no predicate to the allowed set, not even its first step (SHACL §4.8.1).
- **Export:** Turtle writes `sh:path ( <p1> [ sh:inversePath <p2> ] )`.
- **Gate:** `refuse_inverse_path` now refuses any path that is not one forward predicate, nested `node` shapes
  included, with the same pre-slice-6 text. The gate accepts and prints exactly what it did.
- **The receipt checks:** the shapes-gate JSON must differ from slice 6's only in `w3c_cases_passed`, 37 to 41.
  Slice 6's mutants 9-15 were written against lines this slice rewrote, so mutants 16-28 cover slice 6's parts
  again in the new code, plus slice 7's. Each must turn its test RED.
