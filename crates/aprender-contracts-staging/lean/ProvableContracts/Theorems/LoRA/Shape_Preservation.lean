/-!
# Merge Shape Preservation — a simulation of the Rust merge pipeline

Contract: `lora-algebra-v1`, equation `shape_preservation` ("Merge never changes tensor shapes").

This module is a *simulation* (ONT-3b, `kind: simulation`) of the shape behaviour of
`entrenar::merge` (crates/aprender-train/src/merge/mod.rs). It does not model the f32 values,
only what decides a shape: which names are present and each tensor's length.

* `Model` = `HashMap<String, Tensor>` (mod.rs:33). A `Tensor` wraps an `Array1<f32>`
  (autograd/tensor.rs:53), so its whole shape is `len()` (tensor.rs:136). A model is modelled as
  its list of `(name, len)` entries; `NoDupKeys` is the HashMap's unique-key invariant.
* `addLen` is ndarray 0.16's `&Array1 + &Array1` on lengths: equal lengths, or one side of
  length 1 co-broadcasts; anything else panics, modelled as `none`.
* `deltaOne` transcribes `compute_deltas` (mod.rs:52-75) for one model, `mergeWithBase`
  transcribes `merge_with_base` (mod.rs:78-89), `validateModels` transcribes `validate_models`
  (mod.rs:92-104) with `validate_model_keys` (mod.rs:107-116) and `validate_model_shapes`
  (mod.rs:119-133).

Rust iterates a HashMap in an unspecified order, so when several entries are bad the Rust may
name a different one. The theorems below are about success and about shapes, which do not
depend on that order; the witness test (merge/shape_witness_tests.rs) uses one bad entry per case.
Core Lean only (no Mathlib).
-/

namespace ProvableContracts.LoRA.ShapePreservation

/-- `HashMap<String, Tensor>` as the shape sees it: name ↦ `len()`. -/
abbrev Model := List (String × Nat)

/-- `HashMap::get` on the name. -/
def lk : Model → String → Option Nat
  | [], _ => none
  | (k, v) :: t, n => if k = n then some v else lk t n

/-- A HashMap holds each key once. -/
def NoDupKeys : Model → Prop
  | [] => True
  | (k, _) :: t => lk t k = none ∧ NoDupKeys t

/-- `MergeError` (mod.rs:37-49), the arms the shape pipeline can return. -/
inductive MergeError
  | incompatible (name : String)
  | shapeMismatch (name : String)
  | insufficient
  deriving DecidableEq, Repr

/-- ndarray 0.16 `&a + &b` on `Array1` lengths: co-broadcast, else panic (`none`). -/
def addLen (a b : Nat) : Option Nat :=
  if a = b then some a
  else if b = 1 then some a
  else if a = 1 then some b
  else none

/-- `compute_deltas` for one model (mod.rs:55-73): a name missing from the base is
    `IncompatibleArchitectures`, a length that differs is `ShapeMismatch`, otherwise the delta
    `tensor.data() - base_tensor.data()` of two equal lengths has that length. -/
def deltaOne (base : Model) : Model → Except MergeError Model
  | [] => .ok []
  | (n, l) :: t =>
    match lk base n with
    | none => .error (.incompatible n)
    | some bl =>
      if l ≠ bl then .error (.shapeMismatch n)
      else match deltaOne base t with
        | .ok d => .ok ((n, l) :: d)
        | .error e => .error e

/-- `compute_deltas` (mod.rs:52): every model, collected into a `Result<Vec<_>>`. -/
def computeDeltas (models : List Model) (base : Model) : Except MergeError (List Model) :=
  models.mapM (deltaOne base)

/-- `merge_with_base` (mod.rs:78-89): iterate the BASE; a name the delta has is added
    (`base + delta`, may panic), a name it lacks keeps the base tensor. `none` = panic. -/
def mergeWithBase (base delta : Model) : Option Model :=
  match base with
  | [] => some []
  | (n, bl) :: t =>
    let here : Option Nat :=
      match lk delta n with
      | some dl => addLen bl dl
      | none => some bl
    match here, mergeWithBase t delta with
    | some l, some rest => some ((n, l) :: rest)
    | _, _ => none

/-- `validate_model_keys` (mod.rs:107-116): every reference name is in the model. -/
def keysOk (reference model : Model) : Except MergeError Unit :=
  match reference with
  | [] => .ok ()
  | (n, _) :: t =>
    match lk model n with
    | none => .error (.incompatible n)
    | some _ => keysOk t model

/-- `validate_model_shapes` (mod.rs:119-133): every reference length equals the model's. -/
def shapesOk (reference model : Model) : Except MergeError Unit :=
  match reference with
  | [] => .ok ()
  | (n, rl) :: t =>
    match lk model n with
    | some ml => if rl ≠ ml then .error (.shapeMismatch n) else shapesOk t model
    -- unreachable after `keysOk`: the Rust indexes `model[name]` and would panic
    | none => .error (.incompatible n)

/-- `validate_models` (mod.rs:92-104): none is `InsufficientModels`; each later model is
    checked against the first, keys before shapes. -/
def validateModels : List Model → Except MergeError Unit
  | [] => .error .insufficient
  | reference :: rest =>
    rest.foldlM (fun _ m => do keysOk reference m; shapesOk reference m) ()

/-! ## Lemmas -/

theorem lk_mem : ∀ (m : Model) (n : String) (v : Nat), lk m n = some v → (n, v) ∈ m
  | [], _, _, h => by simp [lk] at h
  | (k, w) :: t, n, v, h => by
    unfold lk at h
    by_cases hk : k = n
    · simp [hk] at h; subst hk; subst h; exact List.Mem.head _
    · simp [hk] at h; exact List.Mem.tail _ (lk_mem t n v h)

theorem lk_of_mem : ∀ (m : Model) (n : String) (v : Nat),
    NoDupKeys m → (n, v) ∈ m → lk m n = some v
  | [], _, _, _, h => by cases h
  | (k, w) :: t, n, v, hnd, h => by
    obtain ⟨hk, hnd⟩ := hnd
    unfold lk
    cases h with
    | head => simp
    | tail _ h =>
      by_cases e : k = n
      · subst e
        have := lk_of_mem t k v hnd h
        rw [hk] at this; cases this
      · simp [e]; exact lk_of_mem t n v hnd h

/-! ## Theorems -/

/-- Shape preservation of `merge_with_base`: when every delta tensor the base also names has the
    base's length, the merge does not panic and returns exactly the base's names and lengths. -/
theorem merge_preserves_shape : ∀ (base delta : Model),
    (∀ n bl, (n, bl) ∈ base → ∀ dl, lk delta n = some dl → dl = bl) →
    mergeWithBase base delta = some base
  | [], _, _ => rfl
  | (n, bl) :: t, delta, h => by
    have ih := merge_preserves_shape t delta (fun n' bl' hm => h n' bl' (List.Mem.tail _ hm))
    have hhere : (match lk delta n with | some dl => addLen bl dl | none => some bl) = some bl := by
      cases hd : lk delta n with
      | none => rfl
      | some dl =>
        have := h n bl (List.Mem.head _) dl hd
        subst this; simp [addLen]
    simp only [mergeWithBase, hhere, ih]

/-- `compute_deltas` succeeds only on a model whose every tensor the base names with the same
    length, and the delta it returns has the model's names and lengths. -/
theorem deltas_match_base : ∀ (base m d : Model),
    deltaOne base m = .ok d → d = m ∧ ∀ n l, (n, l) ∈ m → lk base n = some l
  | _, [], d, h => by
    simp [deltaOne] at h
    cases h; exact ⟨rfl, fun _ _ hm => by cases hm⟩
  | base, (n, l) :: t, d, h => by
    unfold deltaOne at h
    cases hb : lk base n with
    | none => simp [hb] at h
    | some bl =>
      simp only [hb] at h
      by_cases hl : l ≠ bl
      · simp [hl] at h
      · simp only [hl, if_false] at h
        have hl' : l = bl := Classical.not_not.mp hl
        cases ht : deltaOne base t with
        | error e => simp [ht] at h
        | ok d' =>
          simp only [ht, Except.ok.injEq] at h
          obtain ⟨hd', hall⟩ := deltas_match_base base t d' ht
          refine ⟨by rw [← h, hd'], ?_⟩
          intro n' l' hm
          cases hm with
          | head => rw [hb, hl']
          | tail _ hm => exact hall n' l' hm

/-- The pipeline `dare_merge` runs (dare.rs:74-92): a delta that `compute_deltas` accepted, merged
    back into a base with unique names, yields the base's shape exactly. -/
theorem shape_preservation (base m d : Model) (hnd : NoDupKeys base)
    (hd : deltaOne base m = .ok d) : mergeWithBase base d = some base := by
  obtain ⟨hdm, hall⟩ := deltas_match_base base m d hd
  apply merge_preserves_shape
  intro n bl hm dl hdl
  subst hdm
  have h1 := hall n dl (lk_mem _ n dl hdl)
  have h2 := lk_of_mem base n bl hnd hm
  rw [h1] at h2; cases h2; rfl

/-- The check is not decorative: a model tensor whose length differs from the base's is refused
    by `compute_deltas` as `ShapeMismatch` on that name. -/
theorem mismatch_refused (base : Model) (n : String) (l bl : Nat) (t : Model)
    (hb : lk base n = some bl) (hl : l ≠ bl) :
    deltaOne base ((n, l) :: t) = .error (.shapeMismatch n) := by
  simp [deltaOne, hb, hl]

/-- The precondition of `merge_preserves_shape` is needed: an unchecked delta whose length is
    neither the base's nor 1 makes `merge_with_base` panic. -/
theorem unchecked_delta_panics (n : String) (bl dl : Nat) (h1 : dl ≠ bl) (h2 : dl ≠ 1) (h3 : bl ≠ 1) :
    mergeWithBase [(n, bl)] [(n, dl)] = none := by
  have : addLen bl dl = none := by
    simp [addLen, Ne.symm h1, h2, h3]
  simp [mergeWithBase, lk, this]

/-! ## Golden table for the Rust witness (merge/shape_witness_tests.rs)

Exhaustive over one parameter `"w"` with lengths `0..=3` and "absent". `#eval` prints the rows
the Rust test hard-codes; re-run `lake env lean` on this file to regenerate them. -/

def lens : List Nat := [0, 1, 2, 3]

/-- `merge_with_base` rows: base length, delta length (`none` = the delta lacks `"w"`),
    merged length (`none` = panic). -/
def mergeRows : List (Nat × Option Nat × Option Nat) :=
  lens.flatMap fun bl => (none :: lens.map some).map fun dl =>
    (bl, dl, (mergeWithBase [("w", bl)] (match dl with | some d => [("w", d)] | none => [])).bind lk'
      )
where lk' (m : Model) : Option Nat := lk m "w"

/-- `compute_deltas` rows: base length (`none` = the base lacks `"w"`), model length, outcome
    (`0` ok with the model's length, `1` IncompatibleArchitectures, `2` ShapeMismatch). -/
def deltaRows : List (Option Nat × Nat × Nat × Nat) :=
  (none :: lens.map some).flatMap fun bl => lens.map fun l =>
    let base : Model := match bl with | some b => [("w", b)] | none => []
    match deltaOne base [("w", l)] with
    | .ok d => (bl, l, 0, (lk d "w").getD 0)
    | .error (.incompatible _) => (bl, l, 1, 0)
    | .error (.shapeMismatch _) => (bl, l, 2, 0)
    | .error .insufficient => (bl, l, 3, 0)

/-- `validate_models` rows over `[reference, second]` with reference `[("w", 2)]`:
    second's `"w"` length (`none` = absent), outcome code as above; plus the empty list. -/
def validateRows : List (Option Nat × Nat) :=
  (none :: lens.map some).map fun l =>
    let second : Model := match l with | some x => [("w", x)] | none => []
    (l, match validateModels [[("w", 2)], second] with
      | .ok () => 0
      | .error (.incompatible _) => 1
      | .error (.shapeMismatch _) => 2
      | .error .insufficient => 3)

#eval mergeRows
#eval deltaRows
#eval validateRows
#eval (match validateModels [] with | .error .insufficient => 3 | _ => 0 : Nat)

#check @shape_preservation
#check @merge_preserves_shape
#check @deltas_match_base

end ProvableContracts.LoRA.ShapePreservation
