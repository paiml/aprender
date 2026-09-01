# `schemas/` — vendored JSON Schemas for the PR-review receipt

PRREV-002. `PR-REVIEW-SKILL-002 v2` §6.2:

> Schemas are **vendored** under `schemas/` with a recorded SHA-256. Fetching a
> schema over the network at gate time makes the gate depend on an external
> service — the same defect class as the render path.

| file | what it validates | origin |
|---|---|---|
| `sarif-2.1.0.json` | `evidence/pr-review/<pr>/<sha>/findings.sarif` (§4.2) | vendored **verbatim** from OASIS |
| `in-toto-statement-v1.json` | `evidence/pr-review/<pr>/<sha>/receipt.intoto.jsonl` (§4.1) | **authored here** — upstream ships no JSON Schema |
| `MANIFEST.sha256` | the two files above | `sha256sum` format, checked with `sha256sum -c` |
| `sources.json` | provenance: URIs, upstream commits, source-document hashes, every deliberate divergence | — |

Verified by `scripts/check_vendored_schemas.sh`; that guard is itself
mutation-verified by `scripts/mutate_vendored_schemas_guard.sh`.

## Getting the validator

`check-jsonschema` is a Python tool and is **not** currently installed by any
job in this repo. Installing it is the only step that touches the network:

```bash
uv tool install 'check-jsonschema==0.38.0'    # 15 packages, ~25s
```

(`uv`, not `pip` — repo policy.) After that the gate is offline forever: it is
handed a local `--schemafile` path and never a URI.

`check-jsonschema` has ~26 built-in vendored schemas (`--builtin-schema
vendor.*`); **neither SARIF nor in-toto is among them**, which is why these two
are vendored here rather than referenced.

If `check-jsonschema` is absent, `scripts/check_vendored_schemas.sh` exits **2**
and prints the command above. It does not skip. An unmeasured gate is not a
passing gate; the distinct exit code exists so a broken box is never read as a
broken tree.

## How "offline" is proved, rather than asserted

`scripts/check_vendored_schemas.sh` re-runs the entire accept/reject fixture
table inside a **network namespace with no interfaces**:

```bash
unshare -r -n env XDG_CACHE_HOME="$(mktemp -d)" \
    check-jsonschema --schemafile schemas/<schema>.json <fixture>
```

Two things are neutralised at once: there is no route to any host, and there is
no populated schema cache that a previously downloaded copy could be served
from. Statically, the guard also asserts that every `$ref` in both schemas is a
local JSON Pointer — a `$ref` to a remote host would make the gate need the
network even though the `--schemafile` argument is local. That mutant (`M4`) is
in the mutation set, and it turns the whole table RED offline.

## `sarif-2.1.0.json`

Byte-identical to **both** of:

* `https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json`
* `oasis-tcs/sarif-spec@a560296ca8c921f3bdb8d4a8db57ab83dae968a7`,
  `sarif-2.1/schema/sarif-schema-2.1.0.json`

`sha256 c3b4bb2d6093897483348925aaa73af03b3e3f4bd4ca38cef26dcb4212a2682e`,
112768 bytes, JSON Schema **draft-04**, no modifications. Two independent
sources agreeing byte-for-byte is the provenance claim; neither alone would be.

## `in-toto-statement-v1.json`

**This file is authored in this repo, not vendored.** The in-toto Attestation
Framework publishes the Statement layer as Markdown plus protobuf; a recursive
listing of `in-toto/attestation@main` contains exactly two `.json` files and
both are npm manifests. Recording it as "vendored from upstream" would be a
fabricated provenance claim, so `sources.json` records it as `authored-in-repo`
with the seven normative and reference sources it was derived from, each pinned
by SHA-256.

**Authoring rule.** Where the Markdown specification speaks, it wins. Where the
Markdown is silent, the Go reference implementation fills the gap. Every
application of that rule is listed in `sources.json → decisions`. The
consequential ones:

| field | this schema | why |
|---|---|---|
| `subject` | `minItems: 1` | Markdown silent; `go/v1/statement.go` `ErrSubjectRequired` |
| `subject[].digest` | required, `minProperties: 1` | Markdown "Each element MUST have `digest` set" |
| `predicate` | **not** required | Markdown says *optional*; the Go reference disagrees (`ErrPredicateRequired`). The normative document wins, and the divergence is recorded rather than silently resolved. |
| digest values, named algorithms | lowercase hex, exact length | case from `digest_set.md`; length from `HexLength`/`digestLengths`. Go's `hex.DecodeString` also accepts UPPERCASE — rejected here, deliberately: a receipt is byte-compared and a case-variant digest breaks that. |
| digest values, custom algorithms | any non-empty string | Go reference: "use of custom, unsupported algorithms is allowed" |
| `additionalProperties` everywhere | `true` | `spec/v1/README.md` parsing rules: consumers MUST ignore unrecognized fields, producers MAY add extension fields. `additionalProperties: false` here would reject conformant documents. |

### Residual risk, stated rather than hidden

* TypeURI enforces a lowercase **scheme** (`^[a-z][a-z0-9+.-]*:`) but not
  RFC 3986 §6.2.2.1 lowercase-**authority** normalization. A full RFC 3986
  regex was judged more likely to produce a false RED than to catch a defect.
* `ResourceDescriptor.content` is typed `string` with a `contentEncoding`
  annotation; `contentEncoding` is non-validating in 2020-12, so malformed
  base64 passes.
* Nothing here checks that a subject digest matches the artifact it names.
  That is the guard's job (PRREV-003), not the schema's.

## Two things PRREV-003 / PRREV-005 must handle

Both were found while making §6.2 actually run, and neither is fixable in this
ticket.

1. **`receipt.intoto.jsonl` must hold exactly ONE JSON document.**
   `check-jsonschema` parses a `.jsonl` file as plain JSON, not as JSON Lines.
   One pretty-printed object: exit 0. One compact object: exit 0. *Two*
   records: `JSONDecodeError: Extra data: line 2 column 1` — the gate fails as
   a parse error, which reads like a broken file rather than a wrong shape.

2. **§4 and §4.3 disagree about the receipt's container.** §4 calls
   `receipt.intoto.jsonl` "DSSE-wrapped", but §6.2 validates that same path
   against the *Statement* schema, and a DSSE envelope
   (`{payloadType, payload, signatures[]}`) does not validate as a Statement.
   §4.3 then signs with a **detached `minisign`** signature, which needs no
   envelope at all. The two are mutually exclusive. This vendoring implements
   §6.2 as written — the file is a bare Statement — because that is the command
   the guard runs. If DSSE is the intent, a third schema is needed and §6.2
   must validate the decoded payload, not the envelope.

## Case table

Every regex and every constraint above ships a must-match / must-not-match row
under `tests/fixtures/schemas/`, exercised offline by the guard:

* `intoto/accept/` — 7 rows, including the §4.1 receipt itself, a receipt with
  no `predicate`, extension fields at two levels, all named digest algorithms,
  and both `gitCommit` lengths.
* `intoto/reject/` — 13 rows: legacy `_type`, empty `subject`, subject without
  `digest`, empty digest set, uppercase hex, a truncated SHA-256, a base64
  digest, a short `gitCommit`, missing `predicateType`, uppercase scheme, no
  scheme, `subject` not an array, `predicate` not an object.
* `sarif/accept/` — 3 rows including the §3.0 three-state encoding
  (`executionSuccessful: false` + `error` notification) and the §4.2 properties
  bag.
* `sarif/reject/` — 6 rows: wrong `version`, no `runs`, driver without `name`,
  a `level` outside the enum, an unknown root property, an `invocation` without
  `executionSuccessful`.

The accept rows are not decoration. Without them, a validator broken badly
enough to refuse everything scores a perfect pass — which is why
`run_case_table` fails if either side of the table is empty.
