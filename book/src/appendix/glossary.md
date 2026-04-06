# Glossary

## A

**AVX (Advanced Vector Extensions)**: 256-bit SIMD instruction set for x86_64 CPUs (Sandy Bridge+, 2011+).

**AVX2**: Enhanced version of AVX with FMA (Haswell+, 2013+).

**AVX-512**: 512-bit SIMD instruction set (Zen 4, Sapphire Rapids+, 2022+).

## B

**Backend**: Implementation executing vector operations (Scalar, SSE2, AVX2, GPU).

**Backend Equivalence**: All backends produce identical results.

## C

**CPU Feature Detection**: Runtime SIMD detection using `is_x86_feature_detected!()`.

**Criterion.rs**: Statistical benchmarking framework for Rust.

## E

**Element-wise Operation**: Operation on each element independently (add, mul).

**EXTREME TDD**: Test methodology with >90% coverage, mutation testing.

## F

**FMA (Fused Multiply-Add)**: Instruction computing `a * b + c`.

## G

**GPU (Graphics Processing Unit)**: Massively parallel compute processor.

## N

**NEON**: 128-bit SIMD for ARM64 CPUs.

## S

**SIMD (Single Instruction Multiple Data)**: Parallel execution on multiple elements.

**SSE2**: 128-bit SIMD baseline for x86_64.

## W

**WASM (WebAssembly)**: Portable bytecode for browsers.

**wgpu**: Rust library for GPU compute.
