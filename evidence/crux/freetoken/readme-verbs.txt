# Canonical FreeToken CLI verbs (FlashML-org/FreeToken @ ee43b2a, captured 2026-11-11)
# Source: README.md and docs/cli.md at
#   https://github.com/FlashML-org/FreeToken/tree/ee43b2a11345d056ba349d22ee073c6828b1364b
# Read through the GitHub API, never executed. Used as the competitor verb list for CRUX category P.
# The CRUX-P-NN after a verb names the story extracted from it; "covered" names the
# existing CRUX row or epic that already owns it (epic aprender#6648 table).

# Install
uv pip install "freetoken[accel]"

# The four-command journey (epic aprender#6648)
ft serve --model <path-or-hf-id>          # zero-flag serve (CRUX-P-06)
ft launch claude|codex|opencode --dry-run # third-party agent launcher (CRUX-P-03)
ft shell                                  # attach to a running server (CRUX-P-04)
ft ctl health|stats|cache|requests        # running-server client (CRUX-P-05)

# ft serve flags extracted as stories
ft serve --reasoning-parser auto|off      # CRUX-P-02
ft serve --sampling-defaults model|none   # CRUX-P-07
ft serve --enable-cache-report            # CRUX-P-08
ft serve --hf-overrides '<json>'          # CRUX-P-10
ft serve --moe-cache-size|--moe-cache-auto  # CRUX-P-12
ft serve --moe-strategy hybrid            # CRUX-P-13
ft serve --moe-collect-stats              # CRUX-P-14
ft serve --embed-weights host             # CRUX-P-16
ft ctl cache --moe N --kv N               # live pool resize, POST /v1/cache/rebuild (CRUX-P-15)

# Other verbs
ft bench bw                               # CRUX-P-11
ft checkpoint --model <hf_dir> --out <ftw_dir>   # FTW format; apr's APR v2 fills this role (epic row 24: have)
scripts/checkpoint_diagnose/              # CRUX-P-09
ft --version

# HTTP surface (routes.yaml)
POST /v1/responses                        # CRUX-P-01
POST /v1/messages                         # covered: aprender#5348 R-1 / E-2 (epic row 1)
--api-key / x-api-key                     # covered: FX-14, aprender#5478 (epic row 3)
--tool-call-parser auto                   # covered: aprender#6571 AG-10..13 (epic row 5)
--cache-type radix                        # covered: CRUX C-08, aprender#6570 SP-11 (epic row 19)
