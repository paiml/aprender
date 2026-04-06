# The "95% Velocity" Mandate

**Objective**: Achieve **95.0%+ Code Coverage** for `trueno` and `trueno-gpu` *immediately* and *correctly*.

**Constraints (Non-Negotiable):**
1.  **Hardware Reality**: This environment has an **NVIDIA GeForce RTX 4090** and an **AMD Threadripper 7960X**. You *will* use them. Do not mock what you can execute.
2.  **No Gaming**: You are forbidden from using `#[cfg(not(cuda))]` or `#[ignore]` to hide complex GPU logic from the coverage report. If it's hard to test, write a better test.
3.  **Speed**: The coverage run must finish in **< 10 minutes**. Optimize the `Makefile` to target *only* the relevant crates (`-p trueno -p trueno-gpu`) and avoid rebuilding dependencies unnecessarily.
4.  **Testing Triad**:
    *   **Unit**: For pure logic (e.g., `emit.rs`).
    *   **Property**: For math/tiling (e.g., `matrix/ops/ml_ops.rs`, `attention/flash.rs`). Use `proptest`.
    *   **Fuzz**: For parsers and memory managers (e.g., `driver/memory.rs`). Steal patterns from `paiml-mcp-agent-toolkit`.

**Execution Protocol:**
1.  **Fix the Makefile**: Ensure `make coverage` runs:
    ```bash
    RUSTFLAGS="-C target-cpu=native" cargo llvm-cov \
      --package trueno --package trueno-gpu \
      --lib \
      --no-default-features \
      --features "parallel,cuda,wgpu,ml-tuner,execution-graph,hardware-detect,tracing" \
      --html
    ```
2.  **Close the Gaps**:
    *   `driver/memory.rs` (29%): Add Fuzz tests for allocation/deallocation sequences.
    *   `attention.rs` (72%): Add Property tests for sequence length edge cases (0, 1, MAX).
3.  **Release**: Upon hitting 95.0%, immediately execute:
    ```bash
    ../batuta publish --all --yes
    ```
4.  **Track**: Log every action with `pmat work`.

**Go. Speed and Truth are your only metrics.**
