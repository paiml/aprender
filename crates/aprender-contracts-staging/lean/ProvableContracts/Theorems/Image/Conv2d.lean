import ProvableContracts.Defs.Image

/-!
# Conv2D — Output Size

Proves the fundamental output dimension formula:
  output_size = input_size - kernel_size + 1
-/

namespace ProvableContracts.Image

-- Status: proved
/-- Convolution output size is positive when kernel fits. -/
theorem conv2d_output_positive {input_size kernel_size : ℕ}
    (h : kernel_size ≤ input_size) (hk : 0 < kernel_size) :
    0 < conv2d_output_size input_size kernel_size := by
  unfold conv2d_output_size
  omega

#check @conv2d_output_positive

end ProvableContracts.Image
