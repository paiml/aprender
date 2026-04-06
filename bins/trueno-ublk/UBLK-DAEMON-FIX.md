# trueno-ublk Specification: Zero-Dependency Daemon

**Version:** 2.2.0
**Date:** 2026-01-04
**Status:** Rewrite (Sprint 51) - QA Mandated

## 1. Abstract

The `trueno-ublk` daemon is being rewritten as a zero-dependency, pure-Rust implementation of the `ublk` kernel protocol. To ensure absolute reliability and high performance, this implementation follows an **Extreme TDD** approach with a mandate of **>95% code coverage** and **100% probador-tested TUI**. All progress is monitored via the **PMAT** automated QA framework.

## 2. Design Philosophy: The Toyota Way

### 2.1 Eliminate Waste (Muda)
*   **Overprocessing:** Direct kernel interaction avoids the 500+ lines of wrapper overhead from `libublk`.
*   **Waiting:** Single-threaded `io_uring` loop eliminates executor deadlocks and non-deterministic waiting.

### 2.2 Built-in Quality (Jidoka)
*   **Extreme TDD:** Every feature, from `ioctl` encoding to data-plane processing, must be backed by unit and property-based tests before implementation.
*   **Probador Tested TUI:** The TUI is not just for observation; its accuracy and responsiveness are verified through automated `jugar-probar` test suites.

## 3. Implementation Plan

### 3.1 Modules
*   **`ublk/sys.rs`**: Safe wrappers around `ublk_cmd.h` structs.
*   **`ublk/ctrl.rs`**: Handles the "Control Plane" (add/start/stop) via `ioctl`.
*   **`ublk/io.rs`**: Handles the "Data Plane" via `io_uring` (fetch/commit).
*   **`ublk/daemon.rs`**: The main event loop bridging `io_uring` and `PageStore`.

### 3.2 QA Gate (PMAT)
1.  **Unit Tests:** Must cover all kernel struct offsets and ioctl constants.
2.  **Integration Tests:** Must demonstrate 1GB `dd` roundtrip with checksum verification.
3.  **Coverage:** `cargo llvm-cov` must report >95% for all files in `bins/trueno-ublk/src/ublk/`.
4.  **TUI:** All `ratatui` widgets must have corresponding `jugar-probar` expectations.

## 4. Renacer Verification Matrix: 100-Point Poppian Checklist

### Section A: Protocol Correctness (Coverage > 95%)
1.  [ ] Verify `UblkCtrlCmd` struct layout matches C `sizeof`.
2.  [ ] Verify `UblkIoDesc` struct layout matches C `sizeof`.
3.  [ ] Verify `UBLK_CTRL_ADD_DEV` ioctl number matches kernel header.
4.  [ ] Verify `UBLK_IO_FETCH_REQ` opcode matches kernel.
5.  [ ] Verify `UBLK_IO_COMMIT_AND_FETCH_REQ` logic (chaining).
6.  [ ] Check alignment of mmap buffer (must be page-aligned).
7.  [ ] Verify tag usage: Tags 0..QD-1 are unique and reused correctly.
8.  [ ] Falsify: Send `nr_sectors=0` (expect kernel error/daemon handling).
9.  [ ] Falsify: Send `start_sector` out of bounds.
10. [ ] Verify `UBLK_F_USER_COPY` flag usage.

### Section B: Data Integrity (Coverage > 95%)
11. [ ] Roundtrip: Write 0xDEADBEEF, Read 0xDEADBEEF.
12. [ ] Zero-page logic: Write zeros, verify no allocation in `PageStore`.
13. [ ] Partial pages: Write 512 bytes (sector) -> Read 4KB (page).
14. [ ] Boundary check: Write to last sector.
15. [ ] Offset check: Write to sector 1, verify sector 0 is unchanged.
16. [ ] Data pollution: Fill device with random data, verify checksums.
17. [ ] Concurrent writes to adjacent sectors (race condition check).
18. [ ] Write while reading (atomic guarantees?).
19. [ ] Discard: Verify `BLKDISCARD` clears data in `PageStore`.
20. [ ] Persistence: Kill daemon, restart, is data gone? (RAM-based).

### Section E: TUI & Observability (100% Probador Tested)
51. [ ] **Probador:** Dashboard renders throughput sparkline correctly.
52. [ ] **Probador:** Compression ratio widget updates in real-time.
53. [ ] **Probador:** Entropy distribution bar chart matches data input.
54. [ ] **Probador:** Keyboard shortcuts (e.g., 'q' to quit) function as expected.
55. [ ] **Probador:** Window resizing does not cause panic or layout corruption.

## 5. Timeline & Milestones

*   **T-0h:** Spec validated (v2.2.0).
*   **T+2h:** Protocol unit tests passed (>95% coverage on `sys.rs`).
*   **T+4h:** IO Loop integration tests passed (>95% coverage on `io.rs`).
*   **T+6h:** TUI Probador suite passed (100% component testing).
*   **T+8h:** PMAT final compliance check and shipment.
