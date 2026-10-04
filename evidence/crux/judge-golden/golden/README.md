# Golden outputs: produced by crux_inference_judge.py @316dee2cd4

Every file under this directory except this README was written by ONE manual run of the Python CRUX
judge, approved as external validation. Nothing in CI or in any script calls the judge to produce or
check these files. They are data: the expected output a port of the judge must reproduce.

| field | value |
|---|---|
| judge | `scripts/lib/crux_inference_judge.py`, subcommand `collect` |
| judge commit | `316dee2cd4b3b3990ad47cc1a7a661c979a4e1d3` |
| `scripts/lib/crux_inference_judge.py` blob | `409d13d039cf5fd88cf79c49b3a4aaaf9d141b6c` |
| `scripts/lib/crux_oracles.py` blob | `343579e3f0b7d3ba00a795c827da5f9971791abe` |
| `scripts/lib/crux_serve_routes.py` blob | `86bdf8abaf252551ac9d6338cbe19c6c4c377346` |
| `scripts/lib/crux_prompt_certify.py` blob | `28bc71bc45f13835a34b3cea30bcdc37326ec776` |
| interpreter | Python 3.13.1 |
| run | 2026-10-04T17:02:43Z to 17:02:49Z, once, under `nice -n 10` |
| inputs | `../SHA256SUMS`; check with `sha256sum -c evidence/crux/judge-golden/SHA256SUMS` from the repo root |
| outputs | `SHA256SUMS` in this directory |

## The exact command

Run once from the repo root, at the judge commit above, with `G=evidence/crux/judge-golden` and
`J=scripts/lib/crux_inference_judge.py`:

```bash
tail -n +2 $G/cases.tsv | while IFS=$'\t' read -r c m p x k; do o=$G/golden/$c; mkdir -p "$o"; set -- collect --manifest "$m" --prompts "$p" --meta "$x" --out-json "$o/receipt.json" --out-md "$o/receipt.md"; [ "$k" = - ] || set -- "$@" --certification "$k"; nice -n 10 python3 "$J" "$@" > "$o/stdout.md" 2> "$o/stderr.txt" < /dev/null; printf '%s\n' "$?" > "$o/rc"; done
```

The run read an 11-case table. The five cases over real engine output were removed afterwards (see
`../README.md`). The six kept cases and their inputs are byte-identical to what the run read and
wrote, so the command over this `cases.tsv` reproduces exactly this directory.

Each case directory holds `receipt.json`, `receipt.md` and `rc` (the judge's exit code). In every
case `stdout.md` was byte-identical to `receipt.md` and `stderr.txt` was empty, so those two files
were not kept. A port must print the markdown receipt on stdout and nothing on stderr.

## Comparing

`judged_at` (JSON) and the `judged <time>` field on line 3 of `receipt.md` are the wall-clock minute
of the run. Mask them on both sides; every other byte must match.

## Results

| case | rc | verdict | cells |
|---|---|---|---|
| `syn-branches` | 1 | RED | 23 |
| `syn-negative-green` | 2 | DECLINE | 1 |
| `syn-nocontrol` | 2 | DECLINE | 1 |
| `syn-pass` | 0 | PASS | 1 |
| `syn-unknown-engine` | 1 | RED | 1 |
| `syn-v2-nocert` | 2 | DECLINE | 1 |
