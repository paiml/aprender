# evidence/crux/freetoken

CRUX Category P evidence: FreeToken (FlashML-org/FreeToken, Apache-2.0), an edge-native MoE serving engine (Python + CUDA, `ft` CLI, arXiv:2608.16157). Read at `ee43b2a11345d056ba349d22ee073c6828b1364b` on 2026-11-11 through the GitHub API, as text only: nothing was installed or executed, and no `ft` binary exists on the host. Story FX-00 (aprender#6649), epic aprender#6648.

- `help.txt`: `docs/cli.md` verbatim, with a provenance header. The CLI surface the stories are extracted from.
- `readme-verbs.txt`: the verbs, each mapped to its CRUX-P story or to the existing row or epic that already owns it.
- `routes.yaml`: the HTTP routes, file:line of each route decorator, and the sha256 of every source file read.

The apr side of each story was checked against aprender `main` at `0a5da37ea` on 2026-11-11 (see each `contracts/crux-P-NN-v1.yaml`). The FX work tickets #6650–#6670 were folded into the checklist #6742 on 2026-11-05 (GHC-001 R1). They were closed not_planned, not shipped, which is why 12 of the 16 stories are still `missing`.
