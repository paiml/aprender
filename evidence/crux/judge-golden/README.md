# CRUX judge golden fixtures

Frozen inputs, and the Python CRUX judge's output on them, so that a port of the judge
(`scripts/lib/crux_inference_judge.py collect`) can be proven against it byte for byte. Under the
no-Python build rule the judge cannot gain a new automated caller. A Rust port is the route, and
these files are its acceptance test.

Nothing here is wired: no workflow, script or guard reads this directory yet.

- `cases.tsv`: one row per case (manifest, prompts, meta, certification; `-` = none).
- `inputs/syn/`: the synthetic cases' inputs, all written by hand.
- `inputs/real/`: relabelled copies of committed 0.69.1 CRUX evidence (see "Relabelling" below):
  `p3962/` (a producer run), `f9-greedy/` (greedy rows) and `cert/` (the prompt certification and its
  two manifests). The 1022 raw output files the real rows name are packed, see "Unpacking" below.
- `inputs/p3962/`, `inputs/f9-greedy/`, `inputs/cert/`: the manifest and metas those cases read. The
  producer manifest drops its `code` rows and points its row paths into `inputs/real/p3962/`. No run
  committed a meta, so each meta is synthetic and carries a `fixture` field saying so.
- `inputs/prompts/`: frozen copies of `scripts/crux_inference_prompts.json` and
  `scripts/crux_inference_prompts.v2.json`, relabelled.
- `SHA256SUMS`: every input a case reads (1073 files), as the judge sees them after unpacking.
  Unpacking ends with `sha256sum -c` over this list.
- `golden/`: the judge's output per case, with the exact command, interpreter and judge commit.
  See `golden/README.md`.

Row paths in a manifest are resolved against the working directory, so run every case from the repo root.

## Unpacking

The judge reads every row's raw output file, and the real cases name 1022 of them. To keep the
directory small they are committed as four JSON Lines bundles under `inputs/real/bundles/`, one per
source (`p3962`, `cert-gpu-sm89`, `cert-arm64-gpu`, `f9-greedy`). Each line is
`{"path": <the row path>, "text": <the file's bytes>}`, sorted by path. Every raw file is valid UTF-8
with no NUL byte, so the text is exact.

**Unpack before running any case.** A missing stdout or stderr file reads as empty, so a case run
over the bundles without unpacking yields a different receipt, not an error. Unpack into a copy of
the tree, not the checkout, and run the cases from that copy. With bash and jq, from the copy's root:

```bash
set -euo pipefail
G=evidence/crux/judge-golden
for b in "$G"/inputs/real/bundles/*.jsonl; do
  jq -j '.path, "\u0000", .text, "\u0000"' "$b" | while IFS= read -r -d '' p && IFS= read -r -d '' t; do
    case "$p" in "$G"/inputs/real/*/*) ;; *) printf 'refused path: %s\n' "$p" >&2; exit 1 ;; esac
    case "$p" in *..* | *//*) printf 'refused path: %s\n' "$p" >&2; exit 1 ;; esac
    awk -v p="$p" '$2 == p { f = 1 } END { exit !f }' "$G/SHA256SUMS" || { printf 'not in SHA256SUMS: %s\n' "$p" >&2; exit 1; }
    if [ -e "$p" ]; then printf 'exists, not overwritten: %s\n' "$p" >&2; exit 1; fi
    mkdir -p "${p%/*}"
    printf '%s' "$t" > "$p"
  done
done
sha256sum -c --quiet "$G/SHA256SUMS"
```

It exits 0 only when all 1073 inputs are present with the listed hashes and every bundled file is one
of them. It fails on a changed byte, a dropped bundle line, a path outside `inputs/real/`, a path
`SHA256SUMS` does not list, and a second unpack over the same tree.

## Cases

Six are synthetic. Every manifest, prompt set, meta and raw output file under `inputs/syn/` was
written by hand (bash and jq) to reach one judge path. None of it is a measurement, and the verdicts
pin the judge's behaviour, not apr's. The synthetic model hash is
`sha256("crux-judge-golden synthetic model")`, and every meta carries a `fixture` field saying so.

| case | inputs | what it pins |
|---|---|---|
| `syn-pass` | apr, llama.cpp, ollama and hf all answer the positive control | the one PASS path (rc 0) |
| `syn-nocontrol` | `syn-pass` with a prompt set that declares no positive control | DECLINE |
| `syn-negative-green` | `syn-pass` with a negative control whose planted wrong apr answer is not judged RED | DECLINE |
| `syn-v2-nocert` | `syn-pass` with a v2 prompt set and no `--certification` | DECLINE (#3962 J2) |
| `syn-branches` | 35 rows, one per parser and classifier branch | refused, timed out, nonzero exit, no JSON, unparseable JSON, no text field, degenerate output, backend fell back, no Assistant turn, echo not found, no timing line, empty ollama stdout, hf with no reported device, device cpu, protocol fault, JSON not in the contract's shape, and a missing stdout file (read as empty) |
| `syn-unknown-engine` | one row from an engine the judge does not know | RED |

Five run over real engine output. Their verdicts are what the judge said about that output; a port
must say the same, not a better verdict.

| case | inputs | what it pins |
|---|---|---|
| `p3962` | the #3962 producer run (apr, llama.cpp and ollama; `run` and `serve`, one `tok` row, one `serve_routes` row) with its own prompt set | 271 cells over real stdout, stderr and serve transcripts; the uncertified path |
| `p3962-cert` | `p3962` with the certification | the certification refused: the producer's prompt set is not the certified one, so its sha256 does not match |
| `f9-greedy` | the F9 greedy rows (apr and llama.cpp) with the v1 prompt set | a receipt with greedy rows and no judged cell (DECLINE) |
| `cert-gpu-sm89` | the certification run's manifest from the `gpu-sm89` host (hf and vllm `chat`) | the certified path over 44 cells |
| `cert-arm64-gpu` | the same from the `arm64-gpu` host (hf and vllm `chat`, llama.cpp `serve`) | the certified path over 238 cells |

## Relabelling

The real inputs are copies, not the committed 0.69.1 files, so that no file here carries a machine
name or a local path. The copies were made by text rules, in bash and sed, with no judge run:

- The two hosts are named by capability: `gpu-sm89` (an sm_89 GPU host) and `arm64-gpu` (an
  arm64 GPU host). This applies to file names, row fields, prompt notes and the synthetic metas' `host`.
- Evidence paths point into `inputs/real/`. An absolute model directory became `<models-dir>/`, an
  absolute work directory became `<work-dir>/`, and the loopback address became `localhost`.
- Two prompt-set notes were reworded to cite the operator, and the v2 notes cite the fixture copy of
  the certification.
- Only the manifests, the prompt sets, the certification and the synthetic metas were rewritten.
  The raw output files the rows name are byte-identical copies.
- The fixture copy of the certification was re-hashed: `prompts_sha256` is the sha256 of the frozen
  v2 copy and `manifests` holds the two relabelled manifests' hashes. `inventory_sha256` is the
  original (the judge does not read it), and a `fixture` field says all this. The certification that
  gates read, `evidence/crux/0.69.1/prompt-certification.json`, is untouched.

The judge was run once over the original inputs and once over these copies, from the same commit.
The first run's real outputs, relabelled by the same rules, equal the second run's except where a
line carries a hash of a relabelled file, or where the judge cuts a quoted value at a fixed length
and the relabelling moved the cut (see `golden/README.md`).

## Not covered yet

- **`code_tests` cells.** They run model-written code in a sandbox (`unshare -rn`, rlimits, a fresh
  tmpdir, `python3 -I`), so their result depends on the host. The producer's `code` rows are dropped.
- Logit-cosine parity. `tok` and `serve_routes` have one real row each, both in `p3962`.
