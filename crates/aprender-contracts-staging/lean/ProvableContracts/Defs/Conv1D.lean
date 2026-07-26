import Mathlib.Data.Real.Basic
import Mathlib.Data.List.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import ProvableContracts.Basic

/-!
# Conv1d Kernel Definitions

Mathematical model of single-channel 1-D convolution (cross-correlation),
matching `contracts/conv1d-kernel-v1.yaml`.

A discrete signal is modelled as a total function `x : ℕ → ℝ` (an infinite
tape), and a kernel as a finite `List ℝ` of taps `w = [w₀, …, w_{K-1}]`.
The valid-mode output at position `n` is

  `y[n] = Σ_{k=0}^{K-1} w[k] · x[n + k]`.

This is the standard `conv1d` equation from the contract (single input /
output channel, unit stride, zero bias; the general multi-channel form is a
finite sum of these, so the analytic properties lift verbatim).

For the *shape* obligation we additionally give a concrete `List`-valued
valid convolution `conv1d_valid` whose output length is proved to equal the
contract's `L_out` formula.

## References

- LeCun et al. (1998) Gradient-Based Learning Applied to Document Recognition
- Oppenheim & Schafer (2010) Discrete-Time Signal Processing
-/

namespace ProvableContracts.Conv1D

open Finset

/-- A discrete signal: an infinite real tape indexed by `ℕ`. -/
abbrev Signal := ℕ → ℝ

/-- Single-channel valid 1-D convolution (cross-correlation) at position `n`:
`y[n] = Σ_{k<K} w[k] · x[n + k]`, with `w` the finite list of kernel taps. -/
noncomputable def conv (w : List ℝ) (x : Signal) (n : ℕ) : ℝ :=
  ∑ k ∈ Finset.range w.length, w.getD k 0 * x (n + k)

/-- Scalar linear combination of two signals: `(a • x + b • z)[i] = a·x[i] + b·z[i]`. -/
def lin (a b : ℝ) (x z : Signal) : Signal :=
  fun i => a * x i + b * z i

/-- Left shift of a signal by `s`: `(shift x s)[n] = x[n + s]`. -/
def shift (x : Signal) (s : ℕ) : Signal :=
  fun n => x (n + s)

/-- The all-zero signal. -/
def zeroSignal : Signal := fun _ => 0

/-- Contract output-length formula:
`L_out = ⌊(L + 2·pad − K) / stride⌋ + 1`. -/
def outLen (L K pad stride : ℕ) : ℕ :=
  (L + 2 * pad - K) / stride + 1

/-- Concrete `List`-valued valid convolution of a finite input list `x` with
kernel `w` (unit stride, no padding). Output element `i` is the dot product of
`w` with the length-`K` window `x[i .. i+K)`. -/
noncomputable def conv1d_valid (w x : List ℝ) : List ℝ :=
  (List.range (x.length - w.length + 1)).map
    (fun i => ((w.zip (x.drop i)).map (fun p => p.1 * p.2)).sum)

end ProvableContracts.Conv1D
