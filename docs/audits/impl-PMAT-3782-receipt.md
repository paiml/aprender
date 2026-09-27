# PMAT-3782 — `apr qa` golden_output scores degenerate output as correct

**Row:** GH #3782, milestone 0.69.1. **Worker:** aprender-d8. **Cop:** aprender-3e.
**Branch:** `PMAT-3782-degenerate-golden` off `origin/release/0.69.1-batch-2` @ `1600d6341`.

## Verdict

Fixed, and **the hole is wider than the issue states**. The issue names the
greeting case's `"!"`. Measured against all three golden cases first, the
**arithmetic** case — the flagship "What is 2+2?" test — has the same hole and is
not mentioned: its entire expected answer is the single character `"4"`, so
`"44444444"` scored correct.

`"4"` cannot be dropped the way `"!"` can. It is the right answer. So the fix
could not be only the pattern edit the issue proposes.

## 1. Measured before touching anything

Probe against the live `verify_output`, `"!"` repeated, pre-fix:

| length | rejected? |
|---|---|
| 1 … 11 | **no** |
| 12 and up | yes — `gibberish_repeated_fragment` |

The boundary is exactly `gibberish_repeated_fragment`'s `bytes.len() >= 12`
guard, which then only examines 4-byte fragments. Everything shorter reached the
answer check untouched.

So the issue's headline example, `!` × 64, **is already rejected by the Rust
gate today**. What the Rust gate misses is *short* degenerate output. (The CRUX
judge scored `!` × 64 correct, which is what the issue observed; CRUX is closed
on its side per #3774.) Recording this because the issue's phrasing — "an apr
build that emits token 0 in a loop passes golden case 2" — is true only for loops
under 12 characters.

Then, per case, with each case's own patterns:

| case | patterns | degenerate output | pre-fix |
|---|---|---|---|
| **arithmetic** | `["4"]` | `"44444444"` | **scored CORRECT** |
| **arithmetic** | `["4"]` | `"44444444444"` (11) | **scored CORRECT** |
| greeting | `[…, "!"]` | `"!!!!!!!!"` | **scored CORRECT** |
| greeting | `[…, "!"]` | `"HHHHHHHH"` | rejected (no pattern match) |
| capital | `["Paris"]` | `"PPPPPPPP"` | rejected (no pattern match) |

The greeting and capital cases reject *their* degenerate forms only because a
run of `H` or `P` contains no pattern. The two holes are exactly the two
**single-character patterns**: `"!"` and `"4"`.

## 2. The fix

Two halves, so neither alone has to carry it:

* **General** — a fourth gibberish signal,
  `gibberish_dominant_character` (`output_verification.rs`): 8+ non-space
  characters with 90%+ of them the same character. It sits in the existing
  `detect_gibberish` chain, which runs as Check 2.5 — **before** the answer
  check — so it covers every case, including cases added later.
  The threshold is **CRUX's own** (#3774), deliberately, so the two judges cannot
  disagree about what "degenerate" means on the same completion.
* **Case-local** — `"!"` dropped from the greeting patterns. A bare exclamation
  mark was never evidence of a greeting.

## 3. Mutants

| mutant | result |
|---|---|
| **A** — remove `gibberish_dominant_character` | **RED, 2 of 3**: both arithmetic rows re-open |
| **B** — A, plus restore `"!"` | **RED, 3 of 3**: exactly the pre-fix state |
| **C** — restore `"!"`, guard intact | **GREEN** — the hole does NOT re-open |

Mutant A's message is the useful one, because it names the case the issue does
not:

```
#3782 REGRESSION: 2 of 3 golden cases accept a degenerate completion, so a model
emitting a dead-logit loop passes apr qa's golden_output:
  - arithmetic: "44444444" scored CORRECT. the expected answer IS a single
    character, so any run of it matches — the flagship golden case, and not
    mentioned in #3782
  - arithmetic (11, just under the old 12-byte floor): "44444444444" scored CORRECT.
```

**Mutant C is a deviation from `done_when` 3, which expects "the mutant that
restores `"!"` turns it green".** It does not, and that is the point: with the
general guard in place, restoring the bad pattern is no longer sufficient to
re-open the hole. The issue's expectation assumed the fix would be the pattern
edit alone. Defense in depth is the better outcome, but it is a deviation and is
recorded as one rather than quietly satisfied.

## 4. Over-correction guarded too

A guard that rejects everything is not a guard, and the arithmetic case answers
with **one character** — the case most at risk from a careless length rule.
`real_answers_still_pass` holds `"4"`, `"2 + 2 = 4"`, `"The answer is 4."`,
`"Hello! How are you doing today?"`, `"The capital of France is Paris."` and
`"Hello!!!!!!!!"` (90% is a floor, not a ceiling — heavy but legitimate
punctuation must pass).

## 5. done_when

| # | requirement | status |
|---|---|---|
| 1 | greeting case no longer satisfiable by degenerate output | **MET** — `"!"` dropped |
| 2 | `golden_output` rejects degenerate completions on every case | **MET** — signal runs before the answer check |
| 3 | test feeds `!!!!!!!!` and a token-0 loop → FAIL; mutant turns it green | **MET with a deviation** — see Mutant C above |
| 4 | `scripts/crux_inference_prompts.json` keeps matching | **NOT ACTIONABLE HERE** — see below |

**done_when 4:** `scripts/crux_inference_prompts.json` **does not exist in this
repository**, and there is no drift check against `golden_test_cases` anywhere in
the tree (`grep -rn "crux_inference_prompts"` → no hits in `crates/` or
`scripts/`; the only crux scripts present are `crux_bulk_pmat_work.sh`,
`crux_missing_stories.py`, `crux_scaffold_contracts.py`). The file presumably
lives in the CRUX repo. **If that file mirrors the greeting patterns, it still
carries `"!"` and needs the same edit there** — flagged rather than silently
skipped.

## 6. Checks

`cargo fmt --all --check` rc=0 · `clippy -p apr-cli --lib -D warnings` rc=0 ·
`cargo test -p apr-cli --lib` **7325 passed, 0 failed**.

## 7. Not claimed

* Not run end-to-end through `apr qa` against a real degenerate model — the
  evidence is at the `verify_output` boundary, which is where the decision is
  made and where the issue locates the defect.
* The 90%/8-character threshold is CRUX's, adopted for agreement between the two
  judges, not independently derived. A completion that is 89% one character
  still passes; that is a deliberate floor, not a claim that it is the right one.
