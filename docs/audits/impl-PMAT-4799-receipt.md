# PMAT-4799 (sub-ticket of PMAT-4741) receipt: llama_fit_verdict.py ported to `aprender-ci-tools llama-fit-verdict`

Code commit b80872d1ad on batch/0702/py-port-llama-fit-verdict (base 5509934e88), then merge
c85b8e658f of py-port-2/ci-tools @ d676d83079 (all conflicts additive). Judged diff: d676d83079...HEAD.
The .py is kept (N-1): `scripts/model_ladder.sh` calls it on the certification path and
`scripts/check_model_ladder.sh` mutates copies of it. No caller is switched in this diff.

Measured on the x86 build host (CARGO_TARGET_DIR private):
- cargo test -p aprender-ci-tools: 44 passed, 0 failed (33 before the merge brought in dag-status/tarball-build-errors)
- cargo clippy -p aprender-ci-tools --all-targets -- -D warnings: clean
- scripts/tests/ci_tools_py_parity_test.sh under CPython 3.13.1: 201/201 identical (declared 201) on the merge
  (177/177 before it);
  86 llama-fit-verdict cases compare stdout AND the exact exit code: every verdict, pin prefixes,
  Python int() forms (signs, underscores, Unicode digits, 4300-digit limit, \x1c-\x1f whitespace),
  free_mib isdigit edge cases, json.dumps ensure_ascii escapes, 21 GGUF fixtures (every skip path,
  v1/v2/v3, truncation, bad UTF-8 key, seek overflow, a directory, 996 vs 997 nested arrays),
  argv count errors, `-h`/`--help`/`--version` as arguments
- scripts/tests/ci_tools_dag_status_parity_test.sh: 25/25 on the merge
- the section refuses an interpreter other than CPython 3.12/3.13 (checked: 3.10 and 3.14 both
  print "FAIL: not measured")
- planted mutants against the parity harness, each RED (176/177, measured before the merge): MAX_ARRAY_DEPTH 996 -> 995;
  `\x1c-\x1f` dropped from the whitespace class; last Kawi digit dropped from the decimal table
- cargo mutants on crates/aprender-ci-tools/src/llama_fit_verdict.rs: 147 tested, 139 caught,
  0 missed, 8 unviable (a first run missed 33; the unit tests added since kill all of them)

Interpreter pin: the original's answers on Unicode digits and nesting depth depend on the
interpreter. CPython 3.12.3 and 3.13.1 agree (Unicode 15.0/15.1 decimal tables identical;
996 nested arrays parse, 997 do not); 3.10 and 3.14 differ. The port follows 3.12/3.13, which
the ladder's hosts run. README section "llama-fit-verdict follows one interpreter".

## Planted contrary question (reviewers MUST answer, with file:line)
Claim: "After this diff, a model whose GGUF trained context is 2048, with llama-fit-params
printing `-c 4095 -ngl -1`, gets verdict `does-not-fit`, because the port compares the fitted
ctx against a fixed floor of 4096."
Is this claim TRUE or FALSE? Cite the line in the port that computes the floor, the line that
compares against it, and the matching lines of scripts/lib/llama_fit_verdict.py.

Pre-push (x86 build host, b80872d1ad and again on merge c85b8e658f): cargo fmt --all -- --check rc 0; cargo test -p
aprender-contracts --lib 2289 passed, 0 failed; cargo deny check advisories ok.

Quorum (PMAT-4799, judged head c85b8e658f, base d676d83079): AGREED 3/3 PASS. claude-sonnet-5-5
(x2), claude-haiku-4-5, measured == declared; degraded: same-family (policy: agy reserved for
tier-1 diffs). All three lanes answered the planted claim FALSE (floor = min(trained, 4096) = 2048,
4095 is not below it, verdict `fits`). Citations: lane 2 cited llama_fit_verdict.rs:370 (the floor
match), correct; lane 3 cited :338, which is `Record::set`, not the floor, so its citation is wrong
even though its answer is right; lane 1 cited no line. Artifact sha256 849779c55a28d70b… kept out
of tree (local paths).
