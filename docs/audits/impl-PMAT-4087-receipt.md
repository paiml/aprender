# Receipt — PMAT-4087: the `tensor_contract` qa gate is cached, and cannot mask a checker change

Branch `feat/4087-qa-tensor-contract-cache` off `main@49fe19c28`. Author aprender-dd (claude-opus-5-5).
Code: `crates/apr-cli/src/commands/qa_contract_cache.rs`, its case table `qa_contract_cache_tests.rs`,
`GateCache` / `GateResult.cache` in `qa.rs`, and the wrapper in `qa_gguf.rs` (`run_tensor_contract_gate` →
`run_tensor_contract_uncached`).

## The key covers the checker, by construction
- **Key** = (sha256 of the model file, sha256 of the running `apr` executable).
  - The executable stands in for "the checker's source". Any change the checker could observe (its code, the
    rules it embeds, its helpers, the compiler) is a different binary, so it is a miss.
  - A hand-kept list of checker source files would go stale the first time the checker called a new helper, and
    a stale key is a stale green.
  - `rosetta::validate*` reads no file but the model (checked).
  - The ticket's third key part, a "contract YAML hash", is therefore inside the executable hash: no YAML is read
    at run time.
- **Multi-file models are bypassed:** run, never stored, and marked `cache.source = "bypassed"` with the reason.
  This covers `*.index.json` sharded safetensors, split `-NNNNN-of-NNNNN.gguf` files, and a path that is not a
  regular file (GH-346 resolves those to an index). One file's hash does not cover the other parts.
- **A key that cannot be computed** (unreadable exe or model) is `bypassed`, never a hit.
- **Storage:** an entry stores {schema, key, result, integrity = sha256 of those}, written atomically.
- **On read, the entry is refused** on:
  - a parse failure,
  - a schema mismatch,
  - a key mismatch (an entry under the wrong name),
  - a result for another gate,
  - an integrity mismatch.
  The gate then runs and rewrites the entry: `cache.source = "refused"` with the reason. An entry is never trusted
  on sight.
- **Only a COMPLETED validation is stored.** An I/O error (`Failed to validate`) may be transient, so it is never
  cached.
- **Provenance and timing:** the row carries `cache: {source: run|cache|refused|bypassed, model_sha256,
  checker_sha256, reason}`. A hit reports the time actually spent (hash plus lookup), never the stored run's.
- **Switches:** default on. `APR_QA_NO_CONTRACT_CACHE=1` turns it off; `APR_QA_CACHE_DIR` moves it (default
  `$XDG_CACHE_HOME/apr/qa-cache`, else `~/.cache/apr/qa-cache`). Tests always pass an explicit temp dir.

## Falsifiers (the ticket's three must-REDs)
1. **Checker source mutated → miss → the gate runs and goes RED** (end to end, real builds on lambda):
   - `apr-A` (sha `f1ceb6cc…`) stored a PASS for Qwen3.5-4B-Q4_K_M.
   - One comparison was flipped in `rosetta::collect_validation_failures` (`nan_count > 0` → `== 0`) and the binary
     rebuilt as `apr-B` (sha `88cfd14e…`); the source was restored under a trap.
   - `apr-B` on the same model and the same cache dir: `source=run`, **FAIL, 426 contract violations**. The
     stored PASS was not used.
   - `apr-B` again: `source=cache`, FAIL (its own entry). `apr-A` again: `source=cache`, PASS (its own entry).
   - Unit level: `a_checker_change_is_a_miss`.
2. **One model byte changed → miss:** `a_one_byte_model_change_is_a_miss`.
3. **The key-dropping mutants, each killing its named test:**
   - M1, the checker hash dropped from the entry name → `a_checker_change_is_a_miss` RED.
   - M2, the integrity check skipped → `an_edited_entry_is_refused_the_gate_runs_and_the_entry_is_rewritten` RED.
   - M3, the model hash dropped from the key → `a_one_byte_model_change_is_a_miss` and
     `a_miss_runs_and_stores_then_the_same_key_is_a_hit_that_does_not_run` RED.

   Each was restored with `git checkout --` under a trap.

The case table has 11 tests: miss→hit, checker change, model byte, edited / truncated / wrong-name / other-gate
entries refused, errors not stored, multi-file bypass, split-GGUF name table, and cache off.

## Saving, measured (lambda, apr-A, the gate alone, every other gate skipped)
- `evidence/pmat-4087/lambda-3model-nocache-miss-hit.tsv`, for Qwen3.5 4B / 9B / 27B:
  - uncached (warm) 19.7 / 43.7 / 146.8 s;
  - miss 20.1 / 42.0 / 125.8 s;
  - hit 1.8 / 3.6 / 10.0 s.
- `evidence/pmat-4087/lambda-inventory-miss-hit.tsv`: all 21 single-file GGUFs in `~/models`, which is what
  `model_ladder.sh`'s inventory sweep runs `apr qa` on.
  - All 21 were `run` then `cache`.
  - **Gate total: 10.5 min uncached, 1.8 min cached, 8.7 min saved per ladder re-run with the same binary** (83%).
- **When it pays:** `model_ladder.sh` runs `apr qa` ONCE per rung and per inventory file, not once per backend.
  So the saving is per RE-RUN (a retry, a rerun after a fix elsewhere, the nightly on an unchanged build), not
  within one run.
- **Where a hit is slower:** a model whose gate FAILS FAST, e.g. Qwen3-0.6B-BF16 (0.7 s → 1.7 s), because hashing
  the whole file costs more than the fast failure. The net is in the total above.
- **Failing verdicts are cached like passing ones.** 8 of the 21 models fail the contract (IQ types, BF16, the
  35B IQ4_XS); their hits replay FAIL.

## Lint, honestly
- apr-cli's `lib.rs` allows `clippy::all`, `pedantic`, `disallowed_methods`, `unused_*` and `dead_code`
  crate-wide (#4148, measured by aprender-75), so "clippy clean" was vacuous there.
- Both new files carry a scoped `#![warn(clippy::all, clippy::pedantic, unused_*, dead_code)]`. It found 2 real
  issues, both fixed: a case-sensitive `.gguf` check, and `format!` inside an iterator.
- `clippy -p apr-cli --lib --tests -D warnings`: 0 findings in these files.
- `cargo test -p apr-cli --lib -- commands::qa`: 268 passed.
- `cargo fmt --check` clean.

## Not done
- A hit still hashes the whole model (1–1.5 GB/s). A stat-keyed sha memo could drop that, but only with an
  integrity story of its own. Out of scope.
- #4148, the crate-wide allow.
