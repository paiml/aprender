# Review receipt: the 0.70.1 preflight follow-up for the dogfood's bashrs row (C280.4, Q3)

- Reviewed patch: `git diff 7d2dc28185 -- scripts/check_publish_preflight.sh`, 15 lines, sha256 `5b6aad368f7ce8b53102d685b537ecde1878db8b96ef70fa26ff12f7a98ebe5e`.
- Preflight script reviewed and committed: sha256 `99eb2c6f9e268f321519dffb7db0848db791aee3c90d4abf565a49f8cb31ba2d`. It replaces `423a6fad5ab4a9600f9e09790b1e25d7a912599d4763c42703b8284c33230979`, named in `review-preflight-c280-q1.md`; the two differ only in the two comment lines below.
- The review ran at a4cae95838. The commit that carries this file adds only this file to it, under `evidence/`, so the reviewed script is the same blob.
- Why: the release dogfood at 7d2dc28185 (lambda, `receipt-20261003T134915Z`, 13:49Z to 18:23Z) was NO-GO on one row, verbatim: `[FAIL] bashrs                     1 SEC/DET/IDEM error(s) over 495 file(s): SEC010 — real findings, not #226 false positives`. The error is on the `cp` in the R7 self-test fixture (line 821): its skip comment named SEC014, and bashrs reports SEC010 on that line. bashrs 7.4.1 and 7.4.2 both report it.
- Why the first review missed it: `.bashrsignore` line 14 ignores SEC010 repo-wide, and the first round's lint ran `bashrs lint` without `--no-ignore`, so it printed 0 errors. The gate (the bashrs row of `scripts/dogfood.sh`, and its PR copy `scripts/check_bashrs_gate.sh`) runs with `--no-ignore`. This round ran the gate's own script.
- What it does: the comment and the `disable-next-line` directive above that `cp` name both rules, `SEC010,SEC014`. Comments only; no executable line changes. Naming SEC010 alone would expose a SEC014 warning on the same line (checked: 325 findings instead of 324).
- Author: claude-opus-5-5 (cop aprender-27). Lanes, none the author's model: claude-sonnet-5-5 (plan mode), agy gemini-3.1-pro-high, agy gpt-oss-120b-medium. Packet: the patch, the machine evidence below, excerpts at the fix commit (`check_publish_preflight.sh` 84-88, 544-552 and 805-835; `dogfood.sh` 1033-1092; `check_bashrs_gate.sh` 74-135), the rulings.
- Round 1, 2026-10-03: verify 18:23:38Z to 18:24:03Z, lanes to 18:25:53Z. The gpt-oss-120b-medium answer is short (344 bytes) but states its reasons; Sonnet and Gemini give numbered, cited findings.

## Verdicts

| lane | verdict |
|---|---|
| claude-sonnet-5-5 | VERDICT: PASS |
| gemini-3.1-pro-high | VERDICT: PASS |
| gpt-oss-120b-medium | VERDICT: PASS |

## Machine evidence (verify-f2.sh summary, verbatim)

```
START 2026-10-03T18:23:38Z P HEAD a4cae95838f4a337a9a8527a06740815614bcdc5 bashrs bashrs 7.4.2
PATCH F..F2  1 file changed, 2 insertions(+), 2 deletions(-)
SCOPE crates/src/Cargo.*: 0 bytes of diff; outside evidence/: contracts/model-capability-ladder-v1.yaml scripts/check_publish_preflight.sh 
ROW-AT-F   [FAIL] bashrs                     1 SEC/DET/IDEM error(s) over 495 file(s): SEC010 — real findings, not #226 false positives
GATE-AT-F2 rc=0 bashrs-gate: PASS 495 file(s) linted, receipt matched, 0 SEC/DET/IDEM errors (3 SC10xx suppressed - bashrs#226; 19 other)
LINT-DIFF rc=0 F: 325 findings, F2: 324 findings only at F : [('SEC010', 'error', 821)] only at F2: [] errors at F : [('SEC010', 'error', 821)] errors at F2: [] 
BASHRSIGNORE 14:SEC010 
SELFTEST rc=0 --- 85/85 rows --- broke=0
DONE 2026-10-03T18:24:03Z
```

## Lane output: sonnet

```
VERDICT: PASS

1. **Scope.** F..F2 is two comment lines, 819–820, in `scripts/check_publish_preflight.sh`. I confirmed this with `git diff --stat` (1 file, 2 insertions, 2 deletions). `crates/`, `src/`, `Cargo.toml` and `Cargo.lock` differ from cc4463f by 0 bytes. Outside `evidence/`, F2 vs cc4463f touches only `contracts/model-capability-ladder-v1.yaml` and this script, so C280.5 holds.

2. **No behaviour change.** Both changed lines start with `#`, so bash never executes them. Three things could have read the script's own text, and none is affected:
   - `--help` prints `sed -n '2,48p' "$0"`, and line 819 is far outside that range.
   - Line 614 uses `$0` only to locate the repo root.
   - `git grep` for a checksum or SHA pinned to this script found only an old audit receipt (PMAT-1096, for a different commit).

   The only way these lines matter is through bashrs's `disable-next-line` parsing, which is the point of the patch.

3. **Honesty of the skip.** The `cp` at line 821 has no path-traversal risk.
   - Both sources are `"$SCRIPT_DIR/lib/<fixed name>"`, where `SCRIPT_DIR` comes from `BASH_SOURCE` (line 86). The file names are literals.
   - The destination `$d/scripts/lib/` is under the self-test's `mktemp -d` dir. Line 549 refuses any `$tmp` outside `/tmp`, `/var/folders` or `/mnt`.
   - The command uses `--`, so a leading dash cannot be read as an option.
   - SEC014 may still apply, since it is the generic rule for unquoted-looking `cp` arguments or sources. Skipping it is honest because the F skip already named it with this same reason, and all variables are quoted.

4. **Nothing else hidden.** The directive covers only line 821, since `disable-next-line` is single-line. LINT-DIFF shows F at 325 findings and F2 at 324, with the only difference being `('SEC010','error',821)` gone and nothing new at F2. That matches what the patch is meant to do.

5. **Gate fidelity.** `scripts/check_bashrs_gate.sh` matches the dogfood bashrs row where it matters.
   - It runs `bashrs lint --no-ignore --level error --format json` over the same `git ls-files` surface.
   - It adds the clean sentinel and checks the `Linted N+1 file(s)` receipt.
   - It gates on SEC/DET/IDEM codes and tolerates only SC1020, SC1035 and SC1140.
   - It also runs a positive control, which the dogfood row lacks, so it is at least as strict.
   - The 495-file count matches the dogfood's own count at F, and both used bashrs 7.4.2.
   - **Caveat (not a defect).** The PR-level script is a proxy. The release receipt is the dogfood run itself at F2. Q3 says the whole path re-runs at F2 under "fix", and that rerun is the authoritative proof. Do not tag on the proxy result alone.

6. **Self-test.** 85/85 at F2 with 0 BROKE lines. I did not re-run it; this relies on the supplied evidence. No row was added or removed.

7. **Nothing weaker than C280.** The `.bashrsignore` SEC010 drop is repo-wide, but the dogfood and gate both use `--no-ignore`, so it does not weaken the gate. The patch adds one narrowly scoped, justified suppression and relaxes no other check. No finding blocks a push.

**Recommendation.** Before the push, re-run the real `scripts/dogfood.sh` bashrs row at F2 (not just the proxy). The re-run already planned under Q3 "fix" covers this.
rc=0
```

## Lane output: gemini

```
VERDICT: PASS

1. **Scope:** The diff F..F2 alters exactly two bash comment lines (819-820) in `scripts/check_publish_preflight.sh`. The evidence (`diffscope.out`) confirms that `crates/`, `src/`, and `Cargo.*` files remain identical to `cc4463f`, strictly satisfying ruling C280.5.
2. **Behaviour change:** None. The changes are strictly bash comments used as linter directives. No executable code is altered or affected.
3. **Honesty of the skip:** The `cp` command at line 821 is demonstrably free of path-traversal risk. The source files are literal paths anchored to the deterministic `$SCRIPT_DIR`, and the destination `$d` is securely generated via `mktemp -d`. Because no external user input dictates the paths, SEC010 is a false positive and suppressing it is honest and safe.
4. **Nothing else hidden:** The `LINT-DIFF` evidence verifies that the new skip directive suppresses exactly one finding (`SEC010` at line 821), with no other findings inadvertently hidden or introduced.
5. **Gate equivalence:** The PR gate `scripts/check_bashrs_gate.sh` at F2 correctly mirrors the release dogfood. Its Python classification logic uses the exact same `('SEC', 'DET', 'IDEM')` prefixes for gating faults, meaning a PASS here guarantees a PASS in the dogfood step.
6. **Self-test:** The `scripts/check_publish_preflight.sh` self-test confirms 85/85 rows pass successfully at F2. 
7. **Overall:** The fix safely and correctly resolves the release gate failure without introducing any new risks or requiring waivers.
```

## Lane output: gpt-oss-120b-medium

```
**VERDICT: PASS**  

No defects were found in the two‑line change to `scripts/check_publish_preflight.sh`. The modification only updates bashrs skip comments, does not alter script behavior, correctly reflects that the `cp` operation poses no path‑traversal risk, and results in exactly one fewer lint finding with no new issues introduced.
```
