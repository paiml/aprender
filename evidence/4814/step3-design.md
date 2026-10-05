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
