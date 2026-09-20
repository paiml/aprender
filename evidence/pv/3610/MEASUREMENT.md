# #3610 — a shapes gate that reported Pass over zero focus nodes

## The defect, reproduced before anything was changed

`crates/aprender-contracts/src/lint/shapes_gate.rs`:

```rust
if report.focus_nodes_n == 0 {
    return Some(ShapesOutcome::NoFocus { shapes_n });
}
```

The question is asked **globally**, so it is only ever true for a lone contract. In a **directory**,
the other shapes carry the total above zero and an empty shape is invisible.

Fixture `tests/fixtures/ont/shapes-one-empty` — one shape with a focus node, one whose target class
nothing instantiates — on `origin/main`:

```
verdict: Pass   violations: 0   focus_nodes_n: 1
by_shape:       ['empty-shape=0', 'tool-status=1']
armed_shapes:   ['empty-shape', 'tool-status']      <- the claim
not_armed_shapes: []
```

**`armed_shapes` is the tool's claim about what it measured, and it named a shape that graded
nothing.** `by_shape` had carried `empty-shape=0` all along and nothing read it — the same *computed
but unexposed* class as #3597, one layer down, inside the gate itself.

## After

| case | verdict | exit | `declines` |
|---|---|---|---|
| **armed** shape at zero, in a directory | decline `NoFocus` | **2** | named in stderr |
| **unarmed** shape at zero | `Pass` | 0 | `["empty-shape"]` — reported, not swallowed |
| every shape graded something (`json-ok`) | `Pass` | 0 | `[]` |
| lone contract that cannot extract (`json-missing-ref`) | decline | ≠0 | unchanged |

The decline names the shape:

```
shapes: 1 ARMED shape(s) graded ZERO focus nodes and cannot be counted as clean
        (2 shape(s), 1 focus node(s) in total): empty-shape
shapes: `armed_shapes` is the tool's claim about what it MEASURED; a shape that graded
        nothing has no place in it (#3610)
```

## The refusal carries its evidence: full JSON, then exit 2

`slk-session-gate.sh` and the SLK bridge's self-test **capture stdout and parse it regardless of the
exit code** — pv already exits non-zero on `Fail`, so capture-then-parse is their normal path. A
refusal that printed only a bare `decline:` line would read to them as *no `by_shape`* and score
**UNMEASURED** — indistinguishable, from their side, from a broken pv. **So the fix would have turned
their gate grey on the same day it turned ours red.**

The refusal therefore declines through the **ordinary result path** rather than short-circuiting:

```
$ pv lint tests/fixtures/ont/shapes-one-empty/contracts --gate shapes --format json ; echo $?
{ … "verdict": "Unknown(NoFocus)", "extra": {
      "by_shape":         ["empty-shape=0", "tool-status=1"],
      "declines":         ["empty-shape"],
      "armed_shapes":     ["tool-status"],
      "not_armed_shapes": [] } }
decline: NoFocus          # stderr
2
```

**The refusal is the exit code; the report is the evidence.** A refusal that suppresses its own
evidence is the same defect as a gate that swallows a decline, seen from the other side.

Note `not_armed_shapes: []` — the vacuous shape is in **neither** list. `armed_shapes` is the claim
about what was measured and `not_armed_shapes` means *not armed by policy*; filing a vacuity as a
policy choice is how the defect hid in the first place.

## Why the unarmed row exists

Without it the fix would be *"refuse whenever anything is empty"*, which is a different tool. An
unarmed shape at zero did not affect the verdict, so it is **reported** rather than refused — and
**reported rather than swallowed**, which is the half the original defect got wrong. `declines`
carries those names; a field that could only ever be empty would be decoration, which is the defect
one layer up from this one.

## Both venues, and the mutation

The lone-contract venue already declined correctly, so a fix proved only there proves nothing. The
committed table covers both, and RED-turns on the defect: disabling the new per-shape check fails
`an_armed_shape_that_graded_nothing_refuses_in_a_directory` and `the_three_answers_remain_distinct`,
and restoring it gives 5/5.

## The real corpus is unaffected

`pv lint contracts --gate {sigma,relations,shapes}` on aprender: **Pass, 0 violations**, every armed
shape with focus nodes (`ladder-measured=8`, `ladder-green=7`, `ont-shapes-v1=1734`), `declines: []`.
The fix refuses the vacuous case and nothing else.
