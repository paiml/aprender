# ONT-10 binary-slice contracts — staged until ONT-4g lands

These are per-binary `pattern` contracts (ONT-001 v4.17 ONT-4g, #4476; ONT-10 #4079). They live here, not in
`contracts/`, because `pv lint contracts/` rejects them until ONT-4g lands two things, and a red lint would turn
the whole `batch/ont-10` carrier red:

1. `entity.type \`binary\` is not in Σ entity_types` — Σ gains `binary` in ONT-4g.
2. `depends_on binary-surface-v1 … a dangling relation` — ONT-4g writes `contracts/binary-surface-v1.yaml`.

Each passes `pv validate` on its own today. When ONT-4g lands: `git mv docs/ont-10/binary-contracts/*.yaml contracts/`
and re-run `pv lint contracts/ --gate shapes`. Each header records what was measured and what is RED today.

| slice | file | binary (package) | commands | RED today |
|---|---|---|---|---|
| S4 (orchestrate 1/2) | binary-aprender-orchestrate-cli-v1.yaml | aprender-orchestrate | 86 CLI leaves; MCP tools/list = 4 | `--version` says "batuta"; 4 MCP tools unledgered, 6 `mcp:*` ledger rows are not tools/list tools |
| S3 (orchestrate 2/2) | binary-aprender-orchestrate-http-v1.yaml | aprender-orchestrate | 92 HTTP routes (banco), set-equal to the ledger | extract gap: dogfood_surfaces.sh reads only aprender-serve routes, so bin:route is empty |
| S9 (apr(apr-cli) 1/2) | binary-apr-cli-apr-v1.yaml | apr (apr-cli) | 264 (262 leaves + debug + sim help); identity + all CLI — S8 takes the 47 HTTP + mcp rows (split by kind: pv refuses qualifiedValueShape) | 15 ledger rows name feature-gated commands (mono, rag eval, data x doctest/hub); 12 pv/capability commands unledgered |
| S10 | binary-alimentar-v1.yaml | alimentar (aprender-data) | 31 | 4 ledger rows name feature-gated commands the default build lacks |
| S16 (1/2) | binary-apr-qa-v1.yaml | apr-qa (aprender-qa-cli) | 15 | none beyond G0.1 |
| S16 (2/2) | binary-aprender-train-lora-v1.yaml | aprender-train-lora | 4 | `--version` says "entrenar-lora" |
| S15 (1/2) | binary-aprender-profile-v1.yaml | aprender-profile | 1 leaf + 46 options; ledger is option rows | `--version` says "renacer" |
| S15 (2/2) | binary-aprender-zram-generator-v1.yaml | aprender-zram-generator | 0; 3 generator positionals | `--version` says "trueno-zram-generator" |
| S2 | binary-aprender-test-cli-v1.yaml | aprender-test-cli | 36 (34 leaves + optional-subcommand groups `comply`, `serve`) | `--version` says "probador"; ledger row `llm experiment` names a group that requires a subcommand |
| S8 (apr(apr-cli) 2/2) | binary-apr-cli-2of2-v1.yaml | apr (apr-cli) | 41 HTTP routes (union over `apr serve` routers, default build) + 9 MCP tools; S9 holds identity + CLI | 2 ledger rows name cuda-only routes (POST /v1/logprobs, /v1/perplexity); extractor drops METHOD and misses apr-cli serve/ routes |
| S14 (1/2) | binary-aprender-train-shell-v1.yaml | aprender-train-shell | REPL: 10 commands, flags -c/-s, 0 subcommands | `--version` says "entrenar-shell"; `-c help` omits `clear` |
| S14 (2/2) | binary-presentar-v1.yaml | presentar (aprender-present-cli) | 7 | none beyond G0.1 |
| S20 | binary-apr-corpus-ingest-v1.yaml | apr-corpus-ingest (apr-cli) | 2 | none beyond G0.1 (pretokenize-bin-v1 cites a nonexistent `run`) |

G0.1 (git sha in `--version`) is a warning on all four and RED on every current build. The S3 half of
aprender-orchestrate (the 92 HTTP routes) is aprender-1c's.

S8 is the HTTP + MCP half of apr (apr-cli); S9 holds identity and every CLI command (the split is by kind, see
S9's header). `apr` is the only binary in S2/S8/S14/S20 whose `--version` carries the sha, so G0.1 is RED on the
other four.

| slice | file | binary (package) | commands | RED today |
|---|---|---|---|---|
| S7 (apr(aprender) 1/2) | binary-aprender-apr-v1.yaml | apr (aprender, the `cargo install aprender` facade) | 264, set-identical to S9; identity + all CLI — S6 takes HTTP + mcp rows | the same 27 paths as S9 (15 feature-gated ledger rows, 12 pv/capability commands unledgered): one ledger fix clears both nodes |
| S13 (1/2) | binary-simular-v1.yaml | simular (aprender-simulate) | 9, incl. user-defined `help`/`version` | 3 ledger rows (GET /, /health, /ws) are library routes behind feature `web` that no simular command serves |
| S13 (2/2) | binary-ptop-v1.yaml | ptop (aprender-present-terminal) | 0; 10 long options | none beyond G0.1. Needs `--features ptop` to exist at all, and `bin:option`, which binary-surface-v1 does not declare yet (S15 uses it too) |

S1 (pv) is not staged here: it is the ONT-4g exemplar, `contracts/bin-aprender-contracts-cli--pv-v1.yaml`. G0.1 is
green on S7 (`apr 0.69.3 (b6cf6d2ede)`) and RED on simular and ptop.
