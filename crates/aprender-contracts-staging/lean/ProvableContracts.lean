-- Root module: imports all definitions and theorems
import ProvableContracts.Basic
import ProvableContracts.Defs.Softmax
import ProvableContracts.Defs.RMSNorm
import ProvableContracts.Defs.Sigmoid
import ProvableContracts.Defs.CrossEntropy
import ProvableContracts.Defs.LayerNorm
import ProvableContracts.Defs.Transpose
import ProvableContracts.Theorems.Softmax.NonNegativity
import ProvableContracts.Theorems.Softmax.PartitionOfUnity
import ProvableContracts.Theorems.Softmax.Monotonicity
import ProvableContracts.Theorems.Softmax.Bounded
import ProvableContracts.Theorems.Softmax.ShiftInvariance
import ProvableContracts.Theorems.RMSNorm.DenominatorPositive
import ProvableContracts.Theorems.RMSNorm.ScaleInvariance
import ProvableContracts.Theorems.Sigmoid.SigmoidBounded
import ProvableContracts.Theorems.Sigmoid.SigmoidSymmetry
import ProvableContracts.Theorems.Sigmoid.SiluZero
import ProvableContracts.Defs.Gelu
import ProvableContracts.Theorems.Gelu.GeluZero
import ProvableContracts.Theorems.CrossEntropy.LogSoftmaxBound
import ProvableContracts.Theorems.CrossEntropy.NonNegativity
import ProvableContracts.Theorems.LayerNorm.DenominatorPositive
import ProvableContracts.Theorems.LayerNorm.ShiftInvariance
import ProvableContracts.Theorems.Transpose.Involution
-- Linear algebra definitions
import ProvableContracts.Defs.GEMV
import ProvableContracts.Defs.MatMul
import ProvableContracts.Defs.Cholesky
import ProvableContracts.Defs.LU
import ProvableContracts.Defs.QR
import ProvableContracts.Defs.SVD
import ProvableContracts.Defs.BLAS
import ProvableContracts.Defs.Sparse
-- Linear algebra theorems
import ProvableContracts.Theorems.GEMV.Correctness
import ProvableContracts.Theorems.MatMul.Associativity
import ProvableContracts.Theorems.MatMul.Identity
import ProvableContracts.Theorems.Cholesky.SPD
import ProvableContracts.Theorems.LU.Existence
import ProvableContracts.Theorems.QR.Orthogonality
import ProvableContracts.Theorems.SVD.NonNegative
import ProvableContracts.Theorems.BLAS.SyrkSymmetric
import ProvableContracts.Theorems.Sparse.SpMVLinear
-- Elementwise operations
import ProvableContracts.Defs.Elementwise
import ProvableContracts.Theorems.Elementwise.ReLUNonNeg
import ProvableContracts.Theorems.Elementwise.AddCommutative
import ProvableContracts.Theorems.Elementwise.MulScalarAssoc
-- Quantization
import ProvableContracts.Defs.Quantization
import ProvableContracts.Theorems.Quantization.RoundtripBound
-- AdamW optimizer
import ProvableContracts.Defs.AdamW
import ProvableContracts.Theorems.AdamW.WeightDecay
-- Discrete Fourier Transform
import ProvableContracts.Defs.FFT
import ProvableContracts.Theorems.FFT.Parseval
