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

G0.1 (git sha in `--version`) is a warning on all four and RED on every current build. The S3 half of
aprender-orchestrate (the 92 HTTP routes) is aprender-1c's.
