# CRUX judge golden fixtures

Frozen inputs, and the Python CRUX judge's output on them, so that a port of the judge
(`scripts/lib/crux_inference_judge.py collect`) can be proven against it byte for byte. Under the
no-Python build rule the judge cannot gain a new automated caller. A Rust port is the route, and
these files are its acceptance test.

Nothing here is wired: no workflow, script or guard reads this directory yet.

- `cases.tsv`: one row per case (manifest, prompts, meta, certification; `-` = none).
- `inputs/syn/`: every input file the cases read.
- `SHA256SUMS`: those inputs (40 files). Check with `sha256sum -c evidence/crux/judge-golden/SHA256SUMS`
  from the repo root.
- `golden/`: the judge's output per case, with the exact command, interpreter and judge commit.
  See `golden/README.md`.

Row paths in a manifest are resolved against the working directory, so run every case from the repo root.

## Cases

All six are synthetic. Every manifest, prompt set, meta and raw output file under `inputs/syn/` was
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

## Not covered yet

- **Real engine output.** Cases over the committed 0.69.1 CRUX evidence (a producer run, the greedy
  manifest, the prompt-certification manifests) were judged in the same run but are not kept here:
  their inputs and outputs carry machine-specific labels. They return as relabelled inputs and a new
  run of the judge, after the port.
- **The certified path.** No case here passes a `--certification` receipt.
- **`code_tests` cells.** They run model-written code in a sandbox (`unshare -rn`, rlimits, a fresh
  tmpdir, `python3 -I`), so their result depends on the host.
- Logit-cosine parity, `tok` rows over real tokenizer output, and `serve_routes` borrowing.
