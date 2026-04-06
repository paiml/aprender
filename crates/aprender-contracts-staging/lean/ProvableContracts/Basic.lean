import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# ProvableContracts — Shared Definitions

Common notation and utility definitions shared across all kernel
theorem modules.

This module provides the foundation types and notation used by
the theorem-proving layer (Phase 7) of the provable-contracts
pipeline.
-/

open Finset

namespace ProvableContracts

/-- A finite vector of reals, indexed by `Fin n`. -/
abbrev RVec (n : ℕ) := Fin n → ℝ

/-- Sum of all elements in a real vector. -/
noncomputable def RVec.sum {n : ℕ} (v : RVec n) : ℝ :=
  ∑ i : Fin n, v i

end ProvableContracts
