# PMAT-114: APR CUDA F32 Output Quality Fix

## Status: IN PROGRESS

## Problem
APR CUDA models (from SafeTensors import) produce garbage output ("helf helf helfakak") compared to GGUF's coherent output ("I'm sorry, but I'm not sure what").

## Root Cause Analysis

### Hypothesis 1: Transpose Mismatch
The F32 weights may be transposed incorrectly for the GPU GEMM path.

### Hypothesis 2: Forward Path Lookup
The `gemm_cached_gpu()` may not be finding the cached weights correctly.

### Hypothesis 3: RoPE/Position Issue
RoPE encoding may not be applied correctly to the F32 path.

## Investigation Steps
1. Verify transpose dimensions match what forward path expects
2. Check `has_cached_weight()` returns true for cached F32 weights
3. Compare numerical output between GGUF and APR at each layer

## Fix
TBD - pending investigation
