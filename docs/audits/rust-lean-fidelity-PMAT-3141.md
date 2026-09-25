# Rust→Lean fidelity: extraction measured, generator diff measured (#3141)

Measured 2026-09-25 on lambda at `paiml/aprender` `5fa13fe8c` (origin/main).
#3141 asked for a measurement plan: extract or hand-write forever, decided on a
number. It also proposed a cheaper gate: diff `pv lean` output against the
committed Lean. Both are measured here.

## Verdict

1. **Extraction (Aeneas) is blocked on floats, not on effort.** The current
   Aeneas nightly translates no `f64` arithmetic. Even `a + b` comes out as
   `sorry`. Every kernel with a `status: proved` theorem is float code, so
   "extract the Rust, prove over the extraction" is not available today for
   any of them. The integer positive control extracts cleanly with 0 `sorry`,
   so this is a float limitation, not a setup failure.
2. **The proposed generator-diff gate would be red forever.** The committed Lean
   is hand-written; `lean_gen` emits scaffolds. Only 6 of 380 paths coincide,
   and those 6 differ in content. A byte-diff gate encodes nothing but that
   fact.
3. **The only YAML→Lean link checked today is equation-level, so the 90
   obligation-level `proved` citations earn nothing.** All 90 name a real
   declaration. The ONT-2a scan counts equation citations only, and 27
   contracts read `self-declared` (70 of 181 Lean claims grounded). Two
   name-scan defects add false negatives on top. See §3.

Recommendation: keep hand-written `Defs/`, and record the Rust connection as
ABSENT (PVL-001 §8) with this receipt as its reason. Re-measure when Aeneas
translates `f64` binops (re-run §1; it takes seconds). Treat `lean_gen` as a
scaffolder, not a source of truth, so no generator-diff gate.

## 1. Aeneas extraction

Tools: charon `nightly-2026.09.24` and aeneas `nightly-2026.09.25-fd27c97`,
linux-x86_64 release assets (tarball sha256 prefixes `0956ba1c96e70882` and
`5812085aed288c8f`). Rust `nightly-2026-09-17`
(`charon toolchain-version`), installed in an isolated `RUSTUP_HOME` with
`rustc-dev`.

```bash
charon rustc --preset=aeneas -- --crate-type=lib <file>.rs
aeneas -backend lean <file>.llbc -dest lean
```

| input | charon | aeneas | result |
|---|---|---|---|
| `aprender::nn::functional::softmax_1d_f64`, body verbatim | rc 0, 1 s | rc 1, 0.5 s | `Improperly typed constant value` at `vec![0.0f64; n]`; whole body → `sorry` |
| same algorithm, floats + slice indexing only (`max_of`, `sum_exp_shifted`, `add(a,b) = a + b` on `f64`) | rc 0 | rc 1 | `Improperly typed constant value` + `Invalid inputs for binop` ×2; **all 3 bodies `sorry`**, including `add` |
| positive control: `max_of` and `add` on `u32` | rc 0 | rc 0 | 0 `sorry`; `def add (a b : Std.U32) : Result Std.U32 := do a + b` |

The extracted `f64` type is Aeneas's `F64`. Even with float support, a theorem
over `ℝ` (all of `Defs/`) would need a bridge from `F64` rounding to `ℝ` before
it could be restated. That gap sits on top of the one measured here.

Why softmax: `contracts/binding.yaml` binds `softmax-kernel-v1` to
`aprender::nn::functional::softmax`. The f32 entry point `softmax_1d`
delegates to trueno SIMD, so the self-contained `f64` path is the fair test.

## 2. `pv lean` output vs the committed tree

```bash
for f in $(grep -l -E '^\s+lean:' contracts/*.yaml); do pv lean "$f" --output-dir $O; done   # 38 contracts
diff -rq $O/ProvableContracts crates/aprender-contracts-staging/lean/ProvableContracts
```

| | files |
|---|---|
| generated | 230 |
| committed (`Defs/` 30, `Theorems/` 131, plus `Basic.lean` etc.) | 162 |
| same path in both | **6** (`Defs/{Softmax,LayerNorm,RMSNorm,BatchNorm,GPU,Tensor}.lean`) — all 6 differ |
| generated only | 224 |
| committed only | 156 |

The generated theorem files are `sorry` stubs, and module names differ
(`Defs/Alibi.lean` committed, `Defs/ALiBi.lean` generated). Generated
`Defs/Softmax.lean` starts `-- DO NOT EDIT — regenerate with pv lean`. The
committed one is a hand-written `noncomputable def softmax {n} (x : RVec n)`
with references.

## 3. Name-level grounding: obligation citations are never consulted

Every `lean: {status: proved, theorem: X}` on a proof obligation in
`contracts/*.yaml` was resolved against the `theorem`/`lemma` declarations in
the committed tree. There are 90 such citations, and **all 90 name a
declaration that exists**.

The ONT-2a grounding scan (`count_lean_theorems_for_contract`,
`proof_status.rs`) reads none of them. It counts one per **equation**
`lean_theorem` that resolves, then grants L4 only when
`grounded + l4_not_applicable >= obligations`. A contract with more
obligations than cited equations therefore cannot be grounded, whatever its
Lean says. `pv proof-status contracts/` at this commit:
`181 lean claimed (70 grounded)` and **27 contracts self-declared**.

Three of the 27, with the equation-level arithmetic:

| contract | obligations | cited equations | N/A | best case `grounded + N/A` |
|---|---|---|---|---|
| `apr-gguf-export-symmetry-v1` | 6 | 1 | 2 | 3 < 6 |
| `tensor-transpose-roundtrip-v1` | 3 | 1 | 0 | 1 < 3 |
| `transpose-kernel-v1` | 5 | 1 | 1 | 2 < 5 |

Even that one equation citation fails to resolve in two of them, because of
name-scan defects:

- `insert_domain_theorems` skips any file where `content.contains("sorry")`.
  `Theorems/TensorTranspose/Roundtrip.lean` has no `sorry` tactic; its line 17
  is a doc comment saying it "compiles sorry-free".
- `Theorems.GgufExportSymmetry.Roundtrip` (domain.file) is never registered:
  the scan inserts `Theorems.GgufExportSymmetry` and `Theorems.Roundtrip`, but
  not the dotted pair.

Everything here fails closed: real proofs are denied credit, and no false
credit is granted. Whether obligation citations should ground L4 is an ONT-2a
design decision. The equation-only rule is deliberate (see the comment on
`is_lean_proved_with_grounding`), so this receipt reports it and does not change
it.
