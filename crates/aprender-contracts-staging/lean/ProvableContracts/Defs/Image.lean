import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import ProvableContracts.Basic

/-!
# Image Processing Definitions

Definitions for image processing kernel contracts.
-/

namespace ProvableContracts.Image

open Finset

/-- A grayscale image as a vector of reals in [0,1]. -/
abbrev GrayImage (n : ℕ) := Fin n → ℝ

/-- RGB to grayscale: Y = 0.299R + 0.587G + 0.114B. -/
noncomputable def rgb_to_gray (r g b : ℝ) : ℝ :=
  0.299 * r + 0.587 * g + 0.114 * b

/-- Histogram: count elements in each of `bins` buckets. -/
noncomputable def histogram_sum {n : ℕ} (img : GrayImage n) : ℝ :=
  ∑ i : Fin n, img i

/-- 2D convolution output size. -/
def conv2d_output_size (input_size kernel_size : ℕ) : ℕ :=
  input_size - kernel_size + 1

end ProvableContracts.Image
