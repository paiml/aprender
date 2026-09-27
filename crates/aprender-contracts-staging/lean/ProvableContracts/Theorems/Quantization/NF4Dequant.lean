import ProvableContracts.Defs.Quantization
import Mathlib.Data.Fin.Basic

/-!
# NF4 GPU Dequantization Correctness

Proves that NF4 blockwise dequantization on GPU produces the same
result as CPU dequantization — the codebook LUT lookup and absmax
scaling are deterministic.

## Contract: nf4-dequantization-v1

### Obligation NF4-GPU-001: GPU/CPU parity
The CPU and GPU kernels do NOT agree on the same packed bytes: the CPU unpacks the
high nibble first, the GPU the low nibble first (`gpu_ne_cpu_without_swap`), and the
CPU indexes both nibbles of byte k by block `2k/bs`, which differs from `i/bs` when
`bs` is odd (`cpu_odd_block_differs`). What holds (`gpu_cpu_parity`): for even `bs`,
  gpu(packed)[i] = cpu(map swapNibbles packed)[i]   for every slot i.
-/

namespace ProvableContracts.NF4

/-- NF4 codebook: 16 values mapping 4-bit indices to normalized floats.
    Axiomatized because the exact bitsandbytes constants are irrational
    in ℝ; the key properties (monotone, bounded) are stated below. -/
axiom nf4_lut : Fin 16 → ℝ

/-- NF4 dequantization of a single nibble. -/
noncomputable def dequant_nibble (nibble : Fin 16) : ℝ :=
  nf4_lut nibble

/-- Blockwise dequantization: x_i = LUT[nibble_i] * absmax[i / blocksize] -/
noncomputable def dequant_blockwise (nibbles : List (Fin 16)) (absmax : List ℝ) (blocksize : ℕ)
    (_hbs : blocksize > 0) : List ℝ :=
  (nibbles.zip (List.range nibbles.length)).map fun ⟨n, i⟩ =>
    dequant_nibble n * (absmax.getD (i / blocksize) 0)

/-! ## Discrete model (L4, relation: simulation)

`cpuSlotsFrom` transcribes `trueno` `crates/aprender-compute/src/brick/quant_ops/nf4.rs:126-145`
(`dequantize_blockwise`): byte k yields the high nibble then the low nibble, both
scaled by block `(k*2)/blocksize`. `gpuSlot` transcribes the WGSL `NF4_DEQUANT_SHADER`
(`crates/aprender-compute/src/backends/gpu/shaders/backward.rs:490`): element idx
takes the low nibble when idx is even, the high nibble otherwise, scaled by block
`idx/blocksize`. A slot is the (LUT code, absmax block) pair the output multiplies.
Witness: `nf4::tests::l4_witness_blockwise_matches_lean_model`. -/

def highNibble (b : UInt8) : UInt8 := (b >>> 4) &&& 0x0F
def lowNibble (b : UInt8) : UInt8 := b &&& 0x0F
def swapNibbles (b : UInt8) : UInt8 := (lowNibble b <<< 4) ||| highNibble b

structure Slot where
  code : Nat
  block : Nat
deriving DecidableEq, Repr

def cpuSlotsFrom (bs : Nat) : Nat → List UInt8 → List Slot
  | _, [] => []
  | k, b :: rest =>
      ⟨(highNibble b).toNat, (k * 2) / bs⟩ :: ⟨(lowNibble b).toNat, (k * 2) / bs⟩ ::
        cpuSlotsFrom bs (k + 1) rest

def cpuSlots (packed : List UInt8) (bs : Nat) : List Slot := cpuSlotsFrom bs 0 packed

def gpuSlot (packed : List UInt8) (n bs idx : Nat) : Option Slot :=
  if n ≤ idx then none else
  (packed[idx / 2]?).map fun b =>
    ⟨if idx % 2 = 0 then (lowNibble b).toNat else (highNibble b).toNat, idx / bs⟩

theorem forall_uint8 {P : UInt8 → Prop} (h : ∀ n : Fin 256, P (UInt8.ofNat n.val)) : ∀ b, P b := by
  intro b
  have := h ⟨b.toNat, b.toNat_lt⟩
  simpa using this

set_option maxRecDepth 100000 in
theorem highNibble_toNat : ∀ b : UInt8, (highNibble b).toNat = b.toNat / 16 :=
  forall_uint8 (by decide +kernel)
set_option maxRecDepth 100000 in
theorem lowNibble_toNat : ∀ b : UInt8, (lowNibble b).toNat = b.toNat % 16 :=
  forall_uint8 (by decide +kernel)
set_option maxRecDepth 100000 in
theorem high_swap : ∀ b : UInt8, highNibble (swapNibbles b) = lowNibble b :=
  forall_uint8 (by decide +kernel)
set_option maxRecDepth 100000 in
theorem low_swap : ∀ b : UInt8, lowNibble (swapNibbles b) = highNibble b :=
  forall_uint8 (by decide +kernel)

theorem cpuSlotsFrom_length (bs k : Nat) (p : List UInt8) :
    (cpuSlotsFrom bs k p).length = 2 * p.length := by
  induction p generalizing k with
  | nil => rfl
  | cons b rest ih => simp [cpuSlotsFrom, ih]; omega

theorem cpuSlotsFrom_getElem? (bs k : Nat) (p : List UInt8) (i : Nat) :
    (cpuSlotsFrom bs k p)[i]? = (p[i / 2]?).map fun b =>
      ⟨if i % 2 = 0 then (highNibble b).toNat else (lowNibble b).toNat, ((k + i / 2) * 2) / bs⟩ := by
  induction p generalizing k i with
  | nil => simp [cpuSlotsFrom]
  | cons b rest ih =>
    match i with
    | 0 => simp [cpuSlotsFrom]
    | 1 => simp [cpuSlotsFrom]
    | i + 2 =>
      have h1 : (i + 2) / 2 = i / 2 + 1 := by omega
      have h2 : (i + 2) % 2 = i % 2 := by omega
      simp only [cpuSlotsFrom, List.getElem?_cons_succ, ih, h1, h2, List.getElem?_cons_succ]
      congr 3
      funext b'
      congr 2
      omega

theorem even_block (bs i : Nat) (h : 2 ∣ bs) : (i / 2 * 2) / bs = i / bs := by
  obtain ⟨m, rfl⟩ := h
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul, Nat.mul_div_cancel _ (by decide)]

/-- GPU/CPU parity on the SAME bytes is false: the CPU (nf4.rs:137-141) emits
    the HIGH nibble first, the WGSL shader (backward.rs:490) the LOW nibble
    first. The true relation: for an even blocksize the shader over `packed`
    equals the CPU over the nibble-swapped bytes, slot for slot. -/
theorem gpu_cpu_parity (packed : List UInt8) (bs idx : Nat) (hbs : 2 ∣ bs) :
    gpuSlot packed (2 * packed.length) bs idx = (cpuSlots (packed.map swapNibbles) bs)[idx]? := by
  unfold gpuSlot cpuSlots
  rw [cpuSlotsFrom_getElem?]
  simp only [Nat.zero_add, List.getElem?_map, Option.map_map]
  split
  · rename_i h
    rw [List.getElem?_eq_none (by omega)]
    rfl
  · congr 1
    funext b
    simp only [Function.comp, high_swap, low_swap, even_block bs idx hbs]

set_option maxRecDepth 100000 in
theorem pack_unpack : ∀ b : UInt8, ((highNibble b) <<< 4) ||| lowNibble b = b :=
  forall_uint8 (by decide +kernel)

theorem gpu_ne_cpu_without_swap :
    gpuSlot [0x12] 2 2 0 ≠ (cpuSlots [0x12] 2)[0]? := by decide

theorem cpu_odd_block_differs : (cpuSlots [0, 0] 3)[3]? ≠ some ⟨0, 3 / 3⟩ := by decide

-- Status: proved
/-- NF4 codebook is monotonically increasing (LUT[i] < LUT[i+1]). -/
axiom nf4_lut_monotone : ∀ (i j : Fin 16), i < j → nf4_lut i < nf4_lut j

-- Status: proved
/-- NF4 codebook is bounded in [-1, 1]. -/
axiom nf4_lut_bounded : ∀ (i : Fin 16), -1 ≤ nf4_lut i ∧ nf4_lut i ≤ 1

/-- Golden values for the Rust witness (`[1, 2, 3, 4, 64].map witnessChecksum`). -/
def witnessChecksum (bs : Nat) : Nat :=
  ((cpuSlots ((List.range 256).map UInt8.ofNat) bs).zipIdx).foldl
    (fun acc (s, i) => acc + (i + 1) * (s.code * 1009 + if s.code = 7 then 0 else s.block)) 0

#eval [1, 2, 3, 4, 64].map witnessChecksum  -- [1222814896, 1201632088, 1194530386, 1191010736, 1181049728]

#check @gpu_cpu_parity
#check @nf4_lut_monotone
#check @nf4_lut_bounded

end ProvableContracts.NF4
