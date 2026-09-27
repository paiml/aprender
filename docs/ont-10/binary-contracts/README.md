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
| S4 (orchestrate 1/2) | MOVED → contracts/bin-aprender-orchestrate--aprender-orchestrate-v1.yaml (S3 routes folded into the same shape) | aprender-orchestrate | 86 CLI leaves; MCP tools/list = 4 | `--version` says "batuta"; 4 MCP tools unledgered, 6 `mcp:*` ledger rows are not tools/list tools |
| S3 (orchestrate 2/2) | MOVED → contracts/bin-aprender-orchestrate--aprender-orchestrate-v1.yaml (bin:route in the S4 shape) | aprender-orchestrate | 92 HTTP routes (banco), set-equal to the ledger | extract gap: dogfood_surfaces.sh reads only aprender-serve routes, so bin:route is empty |
| S9 (apr(apr-cli) 1/2) | MOVED → contracts/bin-apr-cli--apr-v1.yaml | apr (apr-cli) | 264 (262 leaves + debug + sim help); identity + all CLI — S8 takes the 47 HTTP + mcp rows (split by kind: pv refuses qualifiedValueShape) | 15 ledger rows name feature-gated commands (mono, rag eval, data x doctest/hub); 12 pv/capability commands unledgered: FIXED 2026-09-27 (ledger rows) |
| S10 | MOVED → contracts/bin-aprender-data--aprender-data-v1.yaml | alimentar (aprender-data) | 31 | 4 ledger rows name feature-gated commands the default build lacks |
| S16 (1/2) | MOVED → contracts/bin-aprender-qa-cli--apr-qa-v1.yaml | apr-qa (aprender-qa-cli) | 15 | none beyond G0.1 |
| S16 (2/2) | MOVED → contracts/bin-aprender-train-lora--aprender-train-lora-v1.yaml | aprender-train-lora | 4 | none beyond G0.1 (`--version` said "entrenar-lora" — FIXED) |
| S15 (1/2) | MOVED → contracts/bin-aprender-profile--aprender-profile-v1.yaml | aprender-profile | 1 leaf + 46 options; ledger is option rows | none beyond G0.1 (`--version` said "renacer" — FIXED) |
| S15 (2/2) | MOVED → contracts/bin-aprender-zram-generator--aprender-zram-generator-v1.yaml | aprender-zram-generator | 0; 3 generator positionals | none beyond G0.1 (`--version` said "trueno-zram-generator" — FIXED) |
| S2 | MOVED → contracts/bin-aprender-test-cli--aprender-test-cli-v1.yaml | aprender-test-cli | 37 (34 leaves + optional-subcommand groups `comply`, `serve`, `llm experiment`) | FIXED @f6547b0e1d: `--version` names aprender-test-cli; bare `llm experiment` = status. G0.1 (sha) clears at rc.1 |
| S8 (apr(apr-cli) 2/2) | MOVED → contracts/bin-apr-cli--apr-http-mcp-v1.yaml | apr (apr-cli) | 41 HTTP routes (union over `apr serve` routers, default build) + 9 MCP tools; S9 holds identity + CLI | 2 ledger rows name cuda-only routes (POST /v1/logprobs, /v1/perplexity) — declared in the ledger (verified_hardware = nvidia-cuda), clears when extract:binary honours it; extractor drops METHOD and misses apr-cli serve/ routes |
| S14 (1/2) | MOVED → contracts/bin-aprender-train-shell--aprender-train-shell-v1.yaml | aprender-train-shell | REPL: 10 commands, flags -c/-s, 0 subcommands | none beyond G0.1 (`--version` said "entrenar-shell", `-c help` omitted `clear` — both FIXED) |
| S14 (2/2) | MOVED → contracts/bin-aprender-present-cli--aprender-present-v1.yaml | presentar (aprender-present-cli) | 7 | none beyond G0.1 |
| S20 | MOVED → contracts/bin-apr-cli--aprender-corpus-ingest-v1.yaml | apr-corpus-ingest (apr-cli) | 2 | none beyond G0.1 (pretokenize-bin-v1 cites a nonexistent `run`) |

G0.1 (git sha in `--version`) is a warning on all four and RED on every current build. The S3 half of
aprender-orchestrate (the 92 HTTP routes) is aprender-1c's.

S8 is the HTTP + MCP half of apr (apr-cli); S9 holds identity and every CLI command (the split is by kind, see
S9's header). `apr` is the only binary in S2/S8/S14/S20 whose `--version` carries the sha, so G0.1 is RED on the
other four.

| slice | file | binary (package) | commands | RED today |
|---|---|---|---|---|
| S7 (apr(aprender) 1/2) | MOVED → contracts/bin-aprender--apr-v1.yaml | apr (aprender, the `cargo install aprender` facade) | 264, set-identical to S9; identity + all CLI — S6 takes HTTP + mcp rows | the same 27 paths as S9 (15 feature-gated ledger rows, 12 pv/capability commands unledgered, FIXED 2026-09-27 by ledger rows): one ledger fix clears both nodes |
| S6 (apr(aprender) 2/2) | MOVED → contracts/bin-aprender--apr-http-mcp-v1.yaml | apr (aprender facade) | 41 HTTP routes + 9 MCP tools, equal to S8; live GGUF 404 index 37 ⊂ 41 (other 4 = other formats/gpu_batch); S7 holds identity + CLI | the same 2 cuda-only ledger rows as S8 (POST /v1/logprobs, /v1/perplexity) |
| S13 (1/2) | MOVED → contracts/bin-aprender-simulate--aprender-simulate-v1.yaml | simular (aprender-simulate) | 9, incl. user-defined `help`/`version` | G0.1 FIXED 2026-09-27 (`simular 0.69.3 (<sha9>)`). RULED 2026-09-27 (cop): the 3 library routes behind feature `web` (GET /, /health, /ws), which no simular command serves, stay in the ledger as `unserved: GET …` rows. That is the existing unjoinable kind: counted in bin:auditRow, joined to nothing, so there are 0 orphans. A web serve command is post-0.70, on its own ticket |
| S13 (2/2) | MOVED → contracts/bin-aprender-present-terminal--aprender-ptop-v1.yaml | ptop (aprender-present-terminal) | 0; 10 long options | FIXED 2026-09-27: G0.1 (`ptop 0.69.3 (<sha9>)`). Needs `--features ptop` to exist at all. `bin:option` is declared in binary-surface-v1 (08fe9a794a), and evidence/binary/snapshot.jsonl carries exactly the 10 shaped options. That snapshot is from 68d9f0816e, before the G0.1 fix, so its version string still lacks the sha until the next regen |

S1 (pv) is not staged here: it is the ONT-4g exemplar, `contracts/bin-aprender-contracts-cli--pv-v1.yaml`. G0.1 is
green on S7 (`apr 0.69.3 (b6cf6d2ede)`) and on simular and ptop since 2026-09-27 (batch/ont-10).

| slice | file | binary (package) | commands | RED today |
|---|---|---|---|---|
| S18 (1/3) | NEW → contracts/bin-aprender-ptx-debug--aprender-ptx-debug-v1.yaml | aprender-ptx-debug | 3 | none beyond G0.1 (unknown flag exit 1→2 FIXED) |
| S18 (2/3) | NEW → contracts/bin-aprender-explain--aprender-explain-v1.yaml | aprender-explain | 7 | none beyond G0.1 (name + `-K` value_parser FIXED) |
| S18 (3/3) | NEW → contracts/bin-aprender-db--aprender-db-v1.yaml | aprender-db | 0; `--config`; 3 HTTP routes | none beyond G0.1 (name FIXED); needs `--features server` to exist |
| S12 (1/2) | MOVED → contracts/bin-aprender-rag-cli--aprender-rag-v1.yaml | trueno-rag (aprender-rag-cli) | 6 | 7 ledger rows (`eval compare/gate/generate/judge/metrics/retrieve/sample`) name commands behind non-default feature `eval`; G1.3: `query --format/--mode/--fusion/--rerank` are Strings checked after clap (exit 1) |
| S12 (2/2) | MOVED → contracts/bin-aprender-present-terminal--aprender-score-v1.yaml | score (aprender-present-terminal) | 0; 7 long options + [PATH] | none beyond G0.1. Needs `--features score` to exist at all; bare-row join extended to allow positional placeholders (`score [PATH] (…)`) |

G0.1 is RED on both S12 binaries (`trueno-rag 0.69.3`, `score 0.69.3`: no sha).
