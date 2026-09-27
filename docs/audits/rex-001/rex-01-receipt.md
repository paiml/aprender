# REX-01 receipt — executor issues (PMAT-4356, epic #4354)

Filed; no measurement. prereg_sha `ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0`.

## Phase 0 — does an existing declared label already admit these jobs? No.

paiml/infra origin/main `3b1b19e8`, `machines/*/forjar.yaml`:

- **apr binary pin:** present on gx10, lambda-labs and yoga (`apr-pin` + fleet-bins, infra#1027/#1055).
  Absent on intel; mini builds from the tag (infra#1035).
- **`apr serve` unit, Qwen3.5 GGUF weights by sha256, review-lane label:** absent on every host.
  The lambda-labs `ollama-model-qwen35-*` resources are ollama packages, not the GGUF apr loads.
- gx10 runner labels: `gpu,gx10,cuda,blackwell,gb10`, opt-in `gx10`. None of them admits a serve-backed review job.

So every cell stays `NotRun{NoDeclaredExecutor}` until infra#1088 lands (R-4).

## Filed

| Purpose | URL |
|---|---|
| forjar declarations per host (gx10 P1, intel, lambda CPU-only, mini): apr tag, weights path+sha256, serve unit, cgroup, health, exec_sha256, train-active yield, intel concurrency group + non-clean-room label; checkable acceptance | https://github.com/paiml/infra/issues/1088 |
| 4th shadow lane (`quorum.lane_models`, width stays 3, `Unknown{LaneUnavailable}`, review-ledger-v1 rows) + ARB-APR-001 gx10 revision, extending the existing lane issue rather than duplicating it | https://github.com/paiml/paiml-implement/issues/411#issuecomment-5829799553 |
| ARB-APR-001 row 3 cross-reference | https://github.com/paiml/aprender/issues/3574#issuecomment-5829799814 |

## Weights (lambda-vector copies, `sha256sum`, 2026-09-25)

| File | Bytes | sha256 |
|---|---|---|
| Qwen3.5-4B-Q4_K_M.gguf | 2740937888 | 00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4 |
| Qwen3.5-9B-Q4_K_M.gguf | 5680522464 | 03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8 |
| Qwen3.5-27B-Q4_K_M.gguf | 16740812704 | 84b5f7f112156d63836a01a69dc3f11a6ba63b10a23b8ca7a7efaf52d5a2d806 |
