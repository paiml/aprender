F9 (NEW, NOT FIXED HERE) -- the SAME commit published the SAME ratio to README.md, and
B4 still does not see it.

Found while checking, file by file, WHY the other five files of da069a25f do not fire --
a check done because "the other files have no comparative line" was an assertion, not a
measurement. It was wrong:

  README.md: The `apr` CLI achieves **2.93x Ollama** performance on Qwen2.5-Coder-1.5B ...

match_comparative FIRES on that line. match_target does not suppress it. It is dropped by
match_shipped_surface, because a root-level .md is not on B4's inclusion list -- which the
shipped case table already records as a KNOWN GAP:

  NO-MATCH  CHANGELOG.md  "KNOWN GAP, recorded not widened: a root-level markdown file is
                           outside the universe check_no_claim_literals.sh defends, and
                           widening here would put the two definitions out of step
                           silently"

That row was written before anyone knew README.md carried this claim. It does.

NOT WIDENED IN THIS PR, and the reason is the discipline F6 itself follows: a scope change
ships with a precision measurement over the same 300-commit window, and with the sibling
definition in check_no_claim_literals.sh moved in the same commit, or the two go out of
step silently. Below is the measurement, so the next ticket starts with a number rather
than an argument.

=== how many ADDED root-level *.md lines over the last 300 commits of origin/main
    would B4 fire on, if root .md joined the scope? ===
  added root-level *.md lines = 1457   would-fire = 0

=== and the same question on da069a25f itself (the true positive) ===
  +The `apr` CLI achieves **2.93x Ollama** performance on Qwen2.5-Coder-1.5B with GPU acceleration:
  +| Mode | Throughput | vs Ollama | Status |
