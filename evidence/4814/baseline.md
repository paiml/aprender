# #4814 baseline: pv's SHACL Core coverage

- **Tree:** `origin/main` @ `11f844a772` (tree `46ea6c532`)
- **Suite:** W3C `w3c/data-shapes` @ `976ed12ad3` (gh-pages), `data-shapes-test-suite/tests/core/`
- **Spec:** the SHACL Core constraint components of SHACL §4 (W3C Rec, `shacl/index.html` in the same checkout)
- **Measured:** 2026-10-05 by aprender-wshacl

## 1. Constraint components: 14 of 29 implemented (5 of them only in part), 15 missing

How it was measured: `PROPERTY_KEYS` and `NODE_KEYS` in `crates/aprender-contracts/src/ontology/shapes.rs`
(every other key is refused at parse), plus the validator arms, checked against the component list in the spec:
`grep -o 'sh:[A-Za-z]*ConstraintComponent' shacl/index.html | sort -u`. The grep gives 32 names. Four of them
are not Core components: `ConstraintComponent`, `RegexConstraintComponent`, `sourceConstraintComponent` and
`SPARQLConstraintComponent`. The grep misses one Core component, `PropertyConstraintComponent`.

| §4 group | component | pv |
|---|---|---|
| 4.1 value type | class | yes, on property shapes only |
| | datatype | yes, on property shapes only |
| | nodeKind | **partial**: `IRI` and `Literal` only. The four blank-node kinds are refused, and the graph has no blank node (R-15) |
| 4.2 cardinality | minCount, maxCount | yes |
| 4.3 value range | minExclusive, minInclusive, maxExclusive, maxInclusive | **no**, refused |
| 4.4 string | minLength, maxLength | yes, on property shapes only |
| | pattern | **partial**: `sh:flags` is refused |
| | languageIn, uniqueLang | **no**, refused |
| 4.5 property pair | lessThan, lessThanOrEquals | yes (#3611) |
| | equals, disjoint | **no**, refused |
| 4.6 logical | not, and, or, xone | **no**, refused |
| 4.7 shape-based | node | **partial**: one level only, from a property shape |
| | property | **partial**: a top-level node shape's `properties:` only |
| | qualifiedMinCount, qualifiedMaxCount | **no**, refused |
| 4.8 other | closed (+ignoredProperties) | **partial**: `rdf:type` is always admitted on a closed shape |
| | hasValue | **no**, refused |
| | in | yes, on property shapes only |

Implemented: class, datatype, nodeKind, minCount, maxCount, minLength, maxLength, pattern, lessThan,
lessThanOrEquals, node, property, closed and in, which makes 14. Five of them are partial: nodeKind, pattern,
node, property and closed. Missing: 15. Total: 29.

### Outside §4 but part of Core

- **Targets:** `targetClass` only. Missing: `targetNode`, `targetSubjectsOf`, `targetObjectsOf`, and the
  implicit class target.
- **Paths:** a single predicate only. Missing: sequence, alternative, inverse, zeroOrMore, oneOrMore and
  zeroOrOne. Each is refused when it is written as a prefixed path.
- **Constraints on a node shape itself:** none. Constraints attach only to property shapes, and that alone
  keeps 12 of the 16 `NOT_VENDORED` cases out.
- **Severity:** Violation and Warning. `sh:Info` is refused, and so are `sh:deactivated` and `sh:message`.

## 2. W3C Core cases: 19 of 98 vendored and passing

| | count |
|---|---|
| Core suite total (`mf:include` in the 7 per-directory manifests: complex 2, misc 5, node 32, path 13, property 38, targets 7, validation-reports 1) | **98** |
| vendored in `w3c.rs` `CASES` | **19** |
| passing in the gate | **19 of 19** |
| `NOT_VENDORED` (excused, each with a reason) | 16 |
| **in neither list** | **63** |

- **Passing:** `ontology::w3c::tests::every_embedded_case_parses_and_passes` asserts `passed == CASES.len()`.
  The test ran green in CI's `workspace-test` on tree `46ea6c532`, measured on PR head `89f5ea7df`. Merge-group
  run 37331310253 reused that result because its tree was identical ("tier=reuse").
- **Oracle differential is stale:** `tests/oracle/differential.json` (last regenerated 2026-09-19 in `daad9f7c53`)
  has `cases: 17`, and its `w3c` map holds the 16 ONT-0 cases. The 3 property-pair cases from #3611
  (`lessThan-001`, `lessThan-002`, `lessThanOrEquals-001`) are vendored and pass in the gate, but no tracked
  oracle run covers them. So "each checked by `make oracle`" holds for 16 of the 19.
- **The ratchet's denominator is 35, not 98:** `the_table_of_ont_0_is_accounted_for_case_by_case` pins
  `CASES + NOT_VENDORED == 35`. The other 63 suite cases are unaccounted for, so `NOT_VENDORED` cannot "only
  shrink" in any useful sense until every one of the 98 is in one list or the other.

## 3. Fail-closed audit (input to plan step 2)

Unknown keys are refused today: a key outside `NODE_KEYS` or `PROPERTY_KEYS` is `Unsupported`, exit 3. But a
**known key holding a value of the wrong YAML type is dropped without an error** in these places:

| # | key | wrong-type value is… | effect |
|---|---|---|---|
| F1 | `targetClass` | treated as absent, so the shape falls back to the Σ default target | the shape checks a different class than the one it names |
| F2 | `datatype`, `class`, `lessThan`, `lessThanOrEquals` | treated as absent | the constraint is never checked, and the shape still reports conforms |
| F3 | `pattern` | treated as absent | the constraint is never checked |
| F4 | `nodeKind` | treated as absent | the constraint is never checked |
| F5 | `severity` | treated as Violation | malformed input is accepted (the fallback is the safe direction) |
| F6 | `ignoredProperties` entries | dropped one by one | malformed input is accepted (the fallback is the safe direction) |
| F7 | `in` entries (a mapping, list or null) | read as an empty-string term | malformed input is accepted |
| F8 | `path` written in full (`http…`) holding `|`, `^` or whitespace | taken as one predicate | a path expression is read as a predicate |

F1 to F4 are the silent skips that plan step 2 targets. F5 to F8 are malformed inputs that are accepted
without a skip.

## Commands

```text
git -C <wt> rev-parse origin/main                                  # 11f844a772…
git clone --depth 1 -b gh-pages https://github.com/w3c/data-shapes  # 976ed12ad3
for d in complex misc node path property targets validation-reports; do
  grep -o '<[^>]*\.ttl>' $d/manifest.ttl | grep -vc manifest; done  # 2 5 32 13 38 7 1 = 98
grep -o 'sh:[A-Za-z]*ConstraintComponent' shacl/index.html | sort -u
jq '{cases, w3c: (.w3c | length)}' tests/oracle/differential.json  # 17, 16
```
