---
status: issues_found
phase: 01-differentiable-minilm-conformance
depth: standard
reviewed: 2026-08-08T00:00:00Z
files_reviewed: 41
critical: 1
warning: 6
info: 6
---

# Phase 01 — Code Review: differentiable MiniLM conformance

**Depth:** standard (per-file, with cross-file tracing of every backward against its forward)
**Base:** `e6dce92a011f1bfa81a4e3c54a3cb8a1e410062b` .. `HEAD` (61 commits)
**Excluded per scope:** `crates/aprender-core/src/generated_contracts.rs`, `contracts/setfit-encoder-conformance-v1.yaml` (grepped for evidence only, not reviewed)
**Deferred items respected:** nothing already logged as D1–D14, D20–D23, D30–D33, D40–D43, D50–D54 is re-reported.

## Summary

The differentiable primitives are, as far as I can verify by hand, mathematically
correct. Every backward I checked reproduces the analytic derivative of its own
forward, including the two piecewise clamp derivatives that were the phase's
declared risk, and forward/backward agree on which branch was taken because both
read the same stored raw norm. The fail-closed input validation is genuinely
fail-closed: no `unwrap()`, no `panic!`, no `assert!` in any production path this
phase added.

The defect is not in the math. It is in the **gate**. One frozen tolerance —
`OPTIMIZER_STEP` — is 1.5× larger than the entire magnitude of the AdamW step it
is supposed to validate, and 175× larger than the decoupled weight-decay term the
harness explicitly claims to detect. Its value was never measured for that family;
`generate_fixtures.py` copies the *gradient* f32/f64 delta into it. The result is
a gate that reports PASS for an optimizer whose learning rate is off by 2×, or
that omits decoupled decay entirely. For a phase whose deliverable is a
trustworthy numerical verdict, that is the highest-cost failure mode there is, so
it is the one Critical finding.

Six Warnings follow, of which two are also gate-trust issues (a duplicated
zero-gradient floor with no agreement test; a tolerance floor formula that drops
the `|x|` factor its own derivation states), one is a new panic path reachable
from public API, one is a mutually-recursive trait default, one is a Makefile
portability break in the new tier3 wiring, and one is an unchecked row width in
the tokenizer boundary.

---

## Critical

### CR-01: `OPTIMIZER_STEP` tolerance exceeds the entire AdamW step it validates — the post-step parity gate cannot falsify a materially wrong optimizer

**Files:**
- `crates/aprender-core/tests/setfit_conformance/tolerances_generated.rs:41` (`OPTIMIZER_STEP = 3.05175781e-5`)
- `crates/aprender-core/tests/setfit_conformance/gradient_gate.rs:244-257` (the assertion), `:200-204` (the claim it disproves)
- `crates/aprender-core/tests/setfit_conformance.rs:52` (the harness's own detection claim)
- Root cause: `scripts/setfit_fixtures/generate_fixtures.py:556-560` and `:92-101`
- Contract obligation: `OBLIG-ENC-04-POST-STEP-PARAMETER-PARITY`

**What is wrong.** AdamW step 1 with the fixture's own hyperparameters
(`lr=2e-5, betas=[0.9,0.999], eps=1e-8, weight_decay=0.01`) has bias-corrected
`m̂ = g` and `v̂ = g²`, so the per-element displacement is
`lr · g/(|g| + 1e-8)`. Computed directly from the committed
`gradients.json` + `optimizer_step.json`:

| quantity | measured |
|---|---|
| max per-element displacement, 35 of 37 tensors | **2.000e-05** (exactly `lr`; every one of those has `max|g| ≥ 6.13e-4 ≫ eps`) |
| max per-element displacement, the 2 key-bias tensors | 1.373e-07 / 1.892e-07 |
| max decoupled-decay term `lr·wd·max|p|` (`max|p| = 0.8738`) | **1.748e-07** |
| `tol::OPTIMIZER_STEP` | **3.052e-05** |

So the tolerance is **1.53× the whole step** and **175× the decay term**.

Three concrete falsifications the gate therefore cannot make:

1. **Halved learning rate.** A Rust `AdamW` using `lr = 1e-5` produces
   `|p_rust − p_torch| = 1e-5 < 3.05e-5` on every tensor → `assert_close` at
   `gradient_gate.rs:251` passes. Clause (f) of `assert_encoder_updates`
   (`setfit_conformance.rs:595`) only requires `d > 0.0`, so it passes too. **The
   error is invisible to both halves of the gate.** Any `lr` in roughly
   `(0, 5.05e-5]` passes.
2. **Weight decay ignored, or coupled (L2) instead of decoupled.** The whole
   effect is ≤1.748e-07, i.e. 175× below the tolerance. This directly contradicts
   `gradient_gate.rs:200-204` (`assert!(wd > 0.0, "...with decay disabled this
   step would not exercise AdamW's DECOUPLED decay at all, which is the half that
   distinguishes it from Adam")`) and `setfit_conformance.rs:52`
   ("**Detected here:** ... a wrong AdamW hyperparameter or decay coupling"). The
   `wd > 0.0` assertion establishes that the *fixture* used decay; it establishes
   nothing about whether the Rust optimizer applied it.
3. **No step at all** passes `assert_close`; it is caught only by clause (f)'s
   `d > 0.0`, which a single-ulp movement satisfies.

**Why the number is wrong.** `generate_fixtures.py:556-560`:

```python
tolerances["optimizer_step"] = {
    "max_abs_f32_f64_delta": grad_delta,      # <-- the GRADIENT delta, not a post-step delta
    "floor": family_floor("optimizer_step"),
    "recommended_tolerance": max(10 * grad_delta, family_floor("optimizer_step")),
}
```

`grad_delta` is the f32/f64 delta of the *gradients*, computed at `:507-509`. No
f64 optimizer step is ever run, so no f32/f64 delta was ever measured for
post-step parameters. And because
`FAMILY_REDUCTION_WIDTH["optimizer_step"] == FAMILY_REDUCTION_WIDTH["gradients"] == 1024`,
the floor is identical too — the committed
`tolerances_measured.json` entries for `gradients` and `optimizer_step` are
byte-identical, which is the tell.

**Fix.** Measure the family it gates, and scale the floor to the *update*, not to
the parameter:

```python
# in generate_fixtures.py, after the float32 step:
o64 = build_slice_model(torch.float64)
opt64 = torch.optim.AdamW([p for n, p in o64.named_parameters()
                           if not n.startswith("pooler.")], **ADAMW_f64)
opt64.zero_grad(set_to_none=True)
pair_loss(o64, ca, cb, remap, torch.float64).backward()
opt64.step()
step_delta = max(max_abs_delta(p32, p64)
                 for (_, p32), (_, p64) in zip(omodel.named_parameters(),
                                               o64.named_parameters()))
# floor scaled to the UPDATE magnitude (lr), not to |p|:
step_floor = FLOOR_K * math.sqrt(FAMILY_REDUCTION_WIDTH["optimizer_step"]) \
             * EPS_F32 * ADAMW["lr"]
tolerances["optimizer_step"] = {
    "max_abs_f32_f64_delta": step_delta,
    "floor": step_floor,
    "recommended_tolerance": max(10 * step_delta, step_floor),
}
# and assert the gate can see what it claims to see:
assert tolerances["optimizer_step"]["recommended_tolerance"] < ADAMW["lr"] * ADAMW["weight_decay"] * max_abs_p / 10, \
    "the post-step tolerance cannot distinguish decoupled decay from no decay"
```

That last assertion is the important half — the same shape as the existing
`activation` guard at `generate_fixtures.py:320-324`, which *does* check that the
tolerance sits far below the effect it must separate. `optimizer_step` has no
such guard, which is exactly why it went unnoticed.

Then edit `OBLIG-ENC-04-POST-STEP-PARAMETER-PARITY.tolerance` in
`contracts/setfit-encoder-conformance-v1.yaml` and regenerate
`tolerances_generated.rs` (`pv diff` will flag the semver bump, as designed).

Independently, harden clause (f) in `setfit_conformance.rs:589-600` from
"moved at all" to "moved by the expected amount": for a non-exempt tensor with
`max|g| ≫ eps` the expected per-element displacement on step 1 is `lr` to within
the decay term, which is a strong, cheap, reference-free check.

**Until this is fixed, `gradient_gate_controlled_adamw_step_matches_the_frozen_reference`
should not be cited as evidence that the AdamW path is correct.**

---

## Warnings

### WR-01: the seeded attention dropout adds a panic path the unseeded path does not have

**Files:** `crates/aprender-core/src/nn/transformer/positional_encoding.rs:537-545`,
`crates/aprender-core/src/nn/transformer/mod.rs:209-212`,
`crates/aprender-core/src/nn/dropout/mod.rs:77-88`

`apply_dropout_seeded` routes `Some(seed)` through `Dropout::with_seed(p, seed)`,
which `assert!((0.0..1.0).contains(&p))`. `MultiHeadAttention::with_dropout` does
not validate `dropout_p` — it is a bare field assignment:

```rust
pub fn with_dropout(mut self, dropout_p: f32) -> Self {
    self.dropout_p = dropout_p;
    self
}
```

So, from entirely public API:

```rust
let mha = MultiHeadAttention::new(64, 8)
    .with_dropout(1.0)                       // accepted silently
    .with_attention_dropout_seed(7);         // opt-in "additive" seed
let _ = mha.forward(&x);                     // PANIC inside forward, training mode
```

The identical call **without** `.with_attention_dropout_seed` does not panic:
`nn::functional::dropout` (`functional.rs:333-358`) has no range check and simply
computes `scale = 1.0/(1.0 - p) = inf`. The seeded path is therefore not
behaviour-preserving, contradicting its own docs at `mod.rs:214-217` ("Opt-in and
additive: a `MultiHeadAttention` built without this call keeps the ambient-RNG
behaviour it had before, **at every dropout setting**") and
`positional_encoding.rs:526-527` ("`None` delegates to `apply_dropout` verbatim,
so every existing caller is byte-for-byte unchanged").

A panic deep inside `forward` is also the wrong place to fail: the neighbouring
`TransformerEncoderLayer::with_dropout` (`mod.rs:513`) already validates at
*construction* via `Dropout::new`, which is the right shape.

**Fix** — validate where the value enters, so both paths agree:

```rust
/// # Panics
/// Panics if `dropout_p` is not in `[0, 1)`.
#[must_use]
pub fn with_dropout(mut self, dropout_p: f32) -> Self {
    assert!(
        (0.0..1.0).contains(&dropout_p),
        "MultiHeadAttention dropout probability must be in [0, 1), got {dropout_p}",
    );
    self.dropout_p = dropout_p;
    self
}
```

(The encoder itself is unaffected — `DROPOUT_P = 0.1` — so this is about the
public surface, not the SetFit path.)

### WR-02: `Module::set_training` and the documented `train → set_training` pattern are mutually recursive

**Files:** `crates/aprender-core/src/nn/module.rs:105-111`,
`crates/aprender-core/src/setfit/encoder.rs:819-833`

The new default is:

```rust
fn set_training(&mut self, training: bool) {
    if training { self.train(); } else { self.eval(); }
}
```

while `BertSentenceEncoder` implements the opposite direction and documents it as
the safe convention:

```rust
/// Delegates to [`Module::set_training`].
/// ... Both spellings therefore route through the one channel.
fn train(&mut self) { self.set_training(true); }
fn eval(&mut self)  { self.set_training(false); }
```

That is safe **only** because `BertSentenceEncoder` also overrides
`set_training` (`encoder.rs:802`). I confirmed it is the only implementor in the
crate that delegates `train` → `set_training`, so there is no live recursion
today. But the next module that copies the pattern — which is precisely what
D33's recorded fix direction recommends crate-wide ("`train`/`eval` delegate to
`set_training`") — and forgets to override `set_training` gets unbounded mutual
recursion and a stack overflow on `model.eval()`. The trait's own docs invite it
(`module.rs:101`: "composites override this method to recurse") without warning
that overriding `train` alone is the trap.

**Fix** — break the cycle at the trait, and pick one direction:

```rust
/// Set training mode recursively. THIS is the propagation channel; leaves
/// override it, composites override it and forward to children.
fn set_training(&mut self, training: bool) {
    // No delegation to train()/eval(): those may be overridden to call THIS,
    // and a default that calls them would then recurse forever.
    let _ = training;
}
fn train(&mut self) { self.set_training(true); }
fn eval(&mut self)  { self.set_training(false); }
```

If that reshuffle is too large for this phase, at minimum add to the
`set_training` doc: "an implementor that overrides `train`/`eval` to call
`set_training` **must** also override `set_training`, or the two defaults
recurse."

### WR-03: the ENC-04 zero-gradient floor lives in two places with no agreement test

**Files:** `crates/aprender-core/tests/setfit_conformance/tolerances_generated.rs:37`,
`crates/aprender-core/tests/fixtures/setfit/gradients.json` (`zero_grad_floor`),
`crates/aprender-core/tests/setfit_conformance/gradient_gate.rs:125,157,170,281`

`assert_encoder_updates` is called with `tol::ZERO_GRAD_FLOOR` (from the
contract) at `gradient_gate.rs:125` and `:281`, `frozen_gate.rs:189` and
`detach_negative.rs:73`, while
`gradient_gate_the_exemption_is_two_sided_against_real_measurements` uses
`g.zero_grad_floor` (from `gradients.json`) at `:157` and `:170`. They agree
today (both `6.1291398e-05`), and I verified it.

But there is a test asserting **contract ↔ generated** agreement
(`conformance_tolerances_agree_with_the_contract`, `setfit_conformance.rs:981`)
and **none** asserting **contract ↔ fixture**. The fixture value is derived from
the observed gradient distribution (`generate_fixtures.py:470-476`, a
largest-log-gap split), so it moves whenever the corpus, the slice, or the loss
moves. A regeneration that shifts it leaves the exemption test and the ENC-04
gate silently applying two different floors — the exact D-14 "these numbers exist
in ONE place" property this phase is built on.

**Fix** — one line in the harness, next to the existing agreement test:

```rust
#[test]
fn conformance_zero_grad_floor_agrees_between_contract_and_fixture() {
    let g: GradientsFixture = read_fixture("gradients.json");
    assert_eq!(
        g.zero_grad_floor.to_bits(),
        tol::ZERO_GRAD_FLOOR.to_bits(),
        "gradients.json.zero_grad_floor ({:e}) and OBLIG-ENC-04-GRADIENT-AND-STEP-GATE \
         ({:e}) disagree; a fixture regeneration moved the floor without a contract edit",
        g.zero_grad_floor, tol::ZERO_GRAD_FLOOR
    );
}
```

### WR-04: the tolerance floor formula drops the `|x|` factor its own derivation states, so every floor is absolute

**File:** `scripts/setfit_fixtures/generate_fixtures.py:73-105`

The comment derives the reduction-order error as
`sqrt(W) * eps_f32 * |x|` and then implements

```python
floor(W) = FLOOR_K * math.sqrt(W) * EPS_F32     # |x| folded into K = 8
```

Folding `|x|` into a constant is correct only for families whose values are
`O(1)`. It is not correct for `gradients`, whose per-tensor `max|g|` in the
committed fixture spans `6.9e-11` to `1.41` — five orders of magnitude. Measured
ratio of `tol::GRADIENTS = 3.05e-5` to each tensor's `max|g|`:

| tensor | max\|g\| | tol / max\|g\| |
|---|---|---|
| `encoder.layer.1.attention.self.query.weight` | 6.129e-04 | **4.98e-02 (5.0 %)** |
| `encoder.layer.1.attention.self.query.bias` | 7.034e-04 | 4.34e-02 |
| `encoder.layer.0.attention.self.key.weight` | 1.488e-03 | 2.05e-02 |
| `embeddings.token_type_embeddings.weight` | 1.411e+00 | 2.16e-05 |

So `OBLIG-ENC-04-NAMED-GRADIENT-PARITY` accepts a **5 %-wrong gradient** on the
layer-1 attention projections while demanding 2e-5 relative on the embedding
tables. The measured f32/f64 delta for the family is `3.63e-07` — the floor is
85× larger and is what actually binds, so the looseness is entirely the floor's
doing, not a real numerical need.

(Two tensors — the analytically-zero key biases — have `tol/max|g| ≈ 4e5`, i.e.
their parity assertion is fully vacuous. That is separately covered by clause (e)
of `assert_encoder_updates`, so it is not an additional hole.)

**Fix** — make the floor scale with the reference magnitude, keeping a small
absolute term so it can never reach zero (the T-1-07 hazard the current comment
correctly warns about):

```python
def family_floor(family: str, ref_scale: float = 1.0) -> float:
    walk = FLOOR_K * math.sqrt(FAMILY_REDUCTION_WIDTH[family]) * EPS_F32
    return max(walk * ref_scale, 1e-9)          # 1e-9 keeps it strictly positive
```

and pass `ref_scale = max|reference|` per family (for `gradients`, per tensor, or
at minimum the family median rather than an implied 1.0). This tightens the gate
without touching any implementation.

### WR-05: `setfit-feature-matrix` hardcodes `target/`, which this repo documents as relocatable

**File:** `Makefile` (new `setfit-feature-matrix` target)

```make
	@cargo tree -p aprender-core --no-default-features -e normal \
		> target/setfit-feature-matrix-tree.txt 2>&1 || \
		{ echo "FAIL: cargo tree failed; ..."; cat target/setfit-feature-matrix-tree.txt; exit 1; }
```

The redirection is opened by the shell *before* `cargo tree` runs. `CLAUDE.md`
and `scripts/apr_bin.sh` both state, at length, that `.cargo/config.toml` in this
repository redirects cargo's target-dir and is gitignored — and `CARGO_TARGET_DIR`
is the standard env override besides. Under either, `target/` may not exist, the
redirection fails with "No such file or directory", `cargo tree` never runs, the
`||` branch fires, the `cat` also fails, and `exit 1` takes down `make tier3`
with a message blaming `cargo tree` for something that has nothing to do with the
D-06 property.

This is the same class the surrounding comment is careful about (never read a
status through a pipe) — the guard is careful about *which* status it reads and
not careful about *where* it writes.

**Fix:**

```make
TARGET_DIR := $(shell cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.target_directory // "target"')
...
	@mkdir -p "$(TARGET_DIR)"
	@cargo tree -p aprender-core --no-default-features -e normal \
		> "$(TARGET_DIR)/setfit-feature-matrix-tree.txt" 2>&1 || \
		{ ...; }
```

(`apr_bin_target_dir` in `scripts/apr_bin.sh` already resolves this exact value
the same way — reuse the idea rather than re-deriving it.)

### WR-06: `encode_batch` checks the row width of `input_ids` only

**File:** `crates/aprender-core/src/setfit/tokenizer.rs:310-320` vs `:374-392`

The file argues the check is load-bearing:

```rust
// BatchLongest padding must make every row the same width. Checked
// rather than assumed: an unequal row would silently corrupt the
// row-major flattening below.
for (i, e) in encodings.iter().enumerate() {
    if e.get_ids().len() != seq { ... }
}
```

and then flattens two more arrays without the same check:

```rust
input_ids.extend_from_slice(e.get_ids());            // width proven == seq
token_type_ids.extend_from_slice(e.get_type_ids());  // width NOT checked
for (pos, m) in e.get_attention_mask().iter().enumerate() { ... }   // width NOT checked
```

`BertSentenceEncoder::validate` (`encoder.rs:507-519`) checks only the **total**
length of each array against `batch * seq`, so compensating differences across
rows (one row +1, another −1) pass validation and produce a batch whose token-type
ids and mask are shifted relative to the input ids by an arbitrary amount. That is
silent numerical corruption in the one place the phase declares to be the trust
boundary.

The `tokenizers` crate does guarantee equal widths today, so this is
defence-in-depth — but the file's own reasoning for checking `get_ids()` applies
verbatim to the other two.

**Fix** — extend the existing loop:

```rust
for (i, e) in encodings.iter().enumerate() {
    for (field, len) in [
        ("input_ids", e.get_ids().len()),
        ("token_type_ids", e.get_type_ids().len()),
        ("attention_mask", e.get_attention_mask().len()),
    ] {
        if len != seq {
            return Err(SetFitError::BatchInvalid {
                reason: format!(
                    "row {i} {field} has length {len} but row 0 has {seq}; padding did not apply"
                ),
            });
        }
    }
}
```

---

## Info

### IN-01: `cosine_similarity_rows` can return `|s| > 1`, contradicting its own rustdoc

**File:** `crates/aprender-core/src/autograd/ops/similarity.rs:27-28`, `:126-134`

The doc states the invariant as a proof:

> The invariant `|out| <= 1` survives both branches, because
> `|<a,b>| <= n_a * n_b <= max(n_a, eps) * max(n_b, eps)`.

That holds in exact arithmetic. The implementation narrows the norms to f32 while
keeping the dot product in f64:

```rust
let na = sa.sqrt() as f32;                 // <-- rounded
let nb = sb.sqrt() as f32;
let da = if na > eps { na } else { eps };
out[row] = (dot / (f64::from(da) * f64::from(db))) as f32;   // dot still f64
```

When `na` rounds *down*, `da·db < n_a·n_b`, and the quotient exceeds 1. Measured:
over 20 000 random f32 rows of width 2..64 with `a == b`, **2 523 (12.6 %)**
returned `> 1.0`, worst `1.0000001192` (1 ulp above 1).

No live failure: `contract_inv_cosine_similarity_rows!` expands to a no-op, the
contract prose hedges correctly ("Values stay within [-1, 1] **up to
floating-point rounding**"), and the only consumer (`mse_loss` against binary
labels) absorbs a 1e-7 excursion. But a future `acos`-based angular objective
would produce `NaN`, and any hardening of that invariant into a real assertion
would fire on legitimate input.

**Fix** — bring the rustdoc in line with the contract (the cheap, correct
option), or eliminate the narrowing by storing f64 norms in
`CosineSimilarityBackward` so forward and backward still take the identical
branch decision. Do **not** clamp `out` to `[-1, 1]`: that would make the stored
`similarity` inconsistent with the derivative computed from `norms_a`/`norms_b`.

### IN-02: `contract_pre_setfit_encoder_forward!` runs before validation, unlike every other op in the phase

**File:** `crates/aprender-core/src/setfit/encoder.rs:377-379`

```rust
pub fn forward_tokens(&self, batch: &SentenceBatch) -> Result<Tensor, SetFitError> {
    contract_pre_setfit_encoder_forward!(batch.input_ids());   // debug_assert!(len > 0)
    let (_, mut layer_outputs) = self.forward_layers(batch)?;  // validation happens HERE
```

`embedding.rs:109-113`, `normalize.rs:90-94` and `similarity.rs:104-106` all place
the precondition macro *after* the guards, and all three document the reason
verbatim: "at entry a `debug_assert!` would turn a fail-closed typed error into a
debug panic on exactly the hostile inputs this op exists to reject." This call
site does the thing those three comments forbid — an empty `input_ids` is a
debug-build panic where `validate` promises `SetFitError::BatchInvalid`.

Not reachable out-of-crate (`SentenceBatch` fields are `pub(crate)` and
`encode_batch` rejects empty inputs and zero-length rows), so this is consistency
rather than exposure. **Fix:** move the macro call into `forward_layers`
immediately after `self.validate(batch)?`.

### IN-03: seeded attention dropout is reproducible only single-threaded

**File:** `crates/aprender-core/src/nn/transformer/mod.rs` (`attention_dropout_calls`, `forward_qkv`)

```rust
let call = self.attention_dropout_calls.fetch_add(1, Ordering::Relaxed);
```

`Module: Send + Sync` and `forward_self` takes `&self`, so two threads sharing one
`MultiHeadAttention` receive call indices in a nondeterministic order and
therefore nondeterministic masks per forward. The docs
(`with_attention_dropout_seed`, and `encoder.rs:103-123`'s site-seed rationale)
present the streams as reproducible without stating the restriction.

Harmless for this phase (every fixture gate runs in eval mode where dropout is
inert, and training is single-threaded), but the claim will be read as
unconditional. **Fix:** document "reproducible across a single-threaded sequence
of forward passes", or derive the call index from a caller-supplied step counter
instead of an interior-mutable atomic.

### IN-04: `broadcast_mask_to`'s release-mode fallback can underflow

**File:** `crates/aprender-core/src/nn/transformer/positional_encoding.rs` (`broadcast_mask_to`, non-broadcastable `else` arm)

```rust
} else {
    debug_assert!(false, "add_mask: mask dim {d} (extent {extent}) is not broadcastable ...");
    idx[t_d].min(extent - 1)
};
```

If a mask dimension has extent `0`, `extent - 1` wraps to `usize::MAX` in a
release build (overflow checks are off by default), `min` then returns `idx[t_d]`,
and `mask_data[m_off]` indexes out of bounds — a panic, in a path the doc
describes as "deterministic and panic-free". Not reachable from
`additive_attention_mask` (which rejects zero dimensions), and `add_mask` is
`pub(super)` with a single caller, so this is latent.

**Fix:** `idx[t_d].min(extent.saturating_sub(1))`.

### IN-05: `assert_close` silently accepts a non-finite *expected* value

**File:** `crates/aprender-core/tests/setfit_conformance.rs:398-414`

```rust
assert!(a.is_finite(), "{what}: element {i} is {a}, not finite");   // actual only
let d = (a - e).abs();
if d > worst { worst = d; worst_at = i; }
```

If a fixture value `e` were `NaN`, `d` is `NaN`, `NaN > worst` is `false`, the
element never becomes `worst`, and the assertion passes for that element. Not
reachable today — `serde_json` rejects bare `NaN`/`Infinity`, and
`jsonfmt._fmt_scalar` (`scripts/setfit_fixtures/jsonfmt.py:30-31`) explicitly
raises on non-finite values — so this is a latent hole in the comparator every
gate in the harness routes through.

**Fix:** `assert!(e.is_finite(), "{what}: FIXTURE element {i} is {e}, not finite");`
alongside the existing check.

### IN-06: the 4.6 MB SetFit fixture corpus now ships inside the published crate

**File:** `crates/aprender-core/Cargo.toml:36-41`

Changing `"tokenizer.json"` → `"/tokenizer.json"` is correct for the stated
CB-510 reason. The side effect is that `crates/aprender-core/tests/` is not
excluded, so the whole corpus — including `gradients.json` (1.8 MB) and
`optimizer_step.json` (1.6 MB) — now goes to crates.io. Measured:

- setfit fixtures: 4.6 MB raw, **1.89 MiB** gzipped
- `src` + `tests` + `contracts` + manifest: **~5.10 MiB** gzipped, against the
  crates.io 10 MiB limit

So it fits with ~50 % headroom today and is not a publish blocker. Worth a
deliberate decision rather than an accident, because a consumer of the published
crate cannot run these gates anyway: `conformance_tolerances_agree_with_the_contract`
(`setfit_conformance.rs:982-1000`) already documents that the workspace-root
`contracts/` directory is absent from the package and skips accordingly.

**Fix (optional):** add `"/crates/aprender-core/tests/fixtures/setfit/"`-equivalent
(`"/tests/fixtures/setfit/"`) to `exclude`, and note in the harness that the
conformance suite is workspace-checkout-only — which it effectively already is.

---

## Reviewed and found sound

These are the highest-risk areas I traced and could not break. Listing them is
part of the verdict.

**Backward vs forward, derived by hand for each:**
- `CosineSimilarityBackward` (`grad_fn.rs:562-621`) — I re-derived the quotient
  rule for both branches and both operands. Above the clamp,
  `∂s/∂a_i = (b_i/d_b − s·a_i/n_a)/n_a` falls out exactly (substituting
  `dot = s·n_a·d_b`); below it, with `d_a ≡ eps` constant, the projection term
  genuinely vanishes and `b_i/(eps·d_b)` is right. The `b` side is the exact
  mirror. The two branch decisions are independent, as documented, and the
  clamped branch never divides by `n_a`, so a zero row cannot produce `NaN`.
- `L2NormalizeRowsBackward` (`:472-511`) — `(I − y yᵀ)/n` reduces to
  `(g_i − y_i·⟨g,y⟩)/n`, which is what the code computes; the clamped branch is
  the plain `I/eps` and correctly has no projection term.
- `MseBackward` (`:638-661`) — `2(p_i − t_i)/n · g` with the `n == 0` guard.
- `MaskedMeanPoolBackward` (`:352-419`) — routes `g[b,h]/n_b` to valid positions
  and leaves padding at the initialised `0.0`; the divisor is recomputed **per
  row** (not hoisted); the destination offset `row*S*H + s*H + j` is correct
  row-major (LAYOUT-001).
- `GeluExactBackward` (`:747-782`) — `Φ(x) + x·φ(x)` with `Φ` evaluated as
  `0.5·erfc(−x/√2)`, which is the numerically right way to get the negative tail.
- `EmbeddingBackward` (`:1459-1483`) — scatter-**add**, so a repeated token
  accumulates rather than overwrites; `embedding_gather` correctly reuses it with
  the flattened `B*S` id list rather than writing a second gather backward.

**Forward/backward branch agreement.** Both piecewise ops store the *raw*
(pre-clamp, f32-narrowed) norm and re-take the *identical* `n > eps` comparison in
backward. Because the same f32-rounded value is used on both sides, forward and
backward can never disagree about which function was evaluated — including at the
exact `n == eps` boundary, which both assign to the constant branch. This is the
defect class the code documents at length, and it is genuinely closed.

**Cody CALERF transcription** (`crates/aprender-common/src/math.rs:143-244`). I
checked all three branches against the published algorithm structure, not just
the coefficient values: the `|x| ≤ 0.46875` direct form uses `A[4]`/`A[0..3]` with
3 iterations and the `A[3]`/`B[3]` tail; the `0.46875 < y ≤ 4` middle branch uses
`C[8]`/`C[0..6]`/`D[0..6]` with 7 iterations and the `C[7]`/`D[7]` tail; the
`y > 4` asymptotic branch uses `P[5]`/`P[0..3]`/`Q[0..3]` with 4 iterations, the
`P[4]`/`Q[4]` tail, and `(SQRPI − r)/y`. The `AINT(y*16)/16` split-exponential is
applied to both erfc branches. `erfc_precise(−x) = 2 − erfc_pos(|x|)` is correct,
and the `NaN` short-circuits are right. The independent-oracle tests (Maclaurin
series + Laplace continued fraction) really are independent of the implementation
under test, which is the property that matters.

**Graph-edge recording.** Every new op sets `requires_grad_(true)`, registers all
operands with `graph.register_tensor`, and records the correct operand-id list; I
checked the recorded id vectors against the backward return arity in each case
(`cosine_similarity_rows` records two ids and returns two gradients, in order).
`additive_attention_mask` correctly records **no** edge and the connectivity
obligation is carried by `add_mask`, which now materialises the broadcast mask as
a constant and applies it through the autograd-aware `Tensor::add` — the
`.zip()`-truncation defect it replaced is genuinely gone, and the fast path
(`scores.shape() == mask.shape()`) also goes through `Tensor::add`, so neither
path can sever.

**Fail-closed validation.** `embedding_gather`, `masked_mean_pool`,
`additive_attention_mask`, `l2_normalize_rows`, `cosine_similarity_rows` and
`mse_loss` each validate rank, zero dimensions, `checked_mul` **before**
allocating, slice lengths, OOV ids, non-binary mask values, all-padding rows,
non-finite inputs, and the epsilon domain. `!(eps.is_finite() && eps > 0.0)`
correctly rejects `NaN` (the naive `eps <= 0.0` would not). No `unwrap()`, no
`panic!`, no `assert!`, no `todo!`, no debug prints in any production line this
phase added — verified by scanning the added lines of the whole diff.

**`VocabRemap::from_json_bytes`** (`import.rs:109-190`). The range check over
`orig_to_slice` completes and returns before the mutual-inverse loop indexes
`slice_to_orig[*slice_row as usize]`, and `slice_to_orig.len() == slice_vocab` is
checked first, so that index cannot go out of bounds. Both directions are
verified, so two canonical ids cannot silently collide onto one slice row.

**Python-reference parity of the pooling/normalise/cosine chain.** torch's
sentence-transformers pooling divides by `clamp(mask.sum, 1e-9)` while Rust
divides by the integer valid count — identical for every non-degenerate row, and
Rust rejects the degenerate row with a typed error rather than emitting the
clamped value. `F.normalize(..., eps=1e-12)` is exactly `x / max(‖x‖, eps)`,
matching `l2_normalize_rows`. `F.cosine_similarity` clamps per factor, matching
the implementation (D12 was closed the right way round — the prose was fixed, not
the code).

**Fixture-generation integrity.** `slice_model.py`'s head-boundary argument is
proven, not asserted by comment: with source `12 × 32`, rows `[0:64)` really are
complete heads 0 and 1, and `assert_head_boundaries` recomputes the offsets.
Upstream digests are verified fail-closed in *both* `slice_model.py` and
`generate_fixtures.py`, so the generator cannot be re-run alone against a
re-pointed artifact. `resolve_apr_bin()` shells `bash scripts/apr_bin.sh` and
reads stdout — which is the documented executable entry point (diagnostics go to
stderr), so the CLAUDE.md "pin the binary" rule is genuinely honoured rather than
gestured at.

**The negative controls are real.** `detach_negative.rs` drives the *same*
`assert_encoder_updates` helper the positive suites use (and enforces that with a
source scan whose needle is assembled at runtime so it cannot trip itself),
requires the rebuilt leaf to receive gradient so the failure cannot be "nothing is
differentiable anywhere", and requires the failure message to name specific
parameters. `frozen_gate.rs` proves frozen tensors are bit-identical **and** that
something moved, so the proof cannot hold vacuously.
`conformance_activation_gate_rejects_the_tanh_approximation` proves the activation
gate can turn red and that the exact-vs-tanh separation (4.735e-4) is 106× the
frozen tolerance.

**Contract ↔ generated tolerance link.** `conformance_tolerances_agree_with_the_contract`
parses the contract with the in-tree schema crate — correctly aliased as
`setfit-contract-schema` to dodge the crates.io `provable_contracts` shadow —
compares by `to_bits()`, and pins both the contract sha256 and its
`metadata.version`. Its single skip (packaged-crate context) is documented and
still asserts the digest is well-formed. `contract_path()` walks up from
`CARGO_MANIFEST_DIR`; I confirmed there is no crate-local
`contracts/setfit-encoder-conformance-v1.yaml` that could shadow the workspace
one.

**`full_weight_parity.rs`** asserts rather than skips when the 86.7 MB checkout is
absent, and verifies the APR digest against the manifest, so a D-10 run can cite
which bytes it tested.

---

_Reviewed: 2026-08-08_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
