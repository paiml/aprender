# aprender-shell QA Red Team Checklist

**Version:** 0.1.0
**Date:** 2025-11-27
**Target:** 100-point comprehensive validation

## Instructions

- [ ] = Not tested
- [x] = Passed
- [!] = Failed (document issue)
- [~] = Partial/needs investigation

Record tester initials and date for each section completed.

---

## 1. Installation & Setup (10 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 1.1 | `cargo install --path crates/aprender-shell` succeeds | [ ] | |
| 1.2 | Binary appears in `~/.cargo/bin/aprender-shell` | [ ] | |
| 1.3 | `aprender-shell --version` returns version string | [ ] | |
| 1.4 | `aprender-shell --help` displays all commands | [ ] | |
| 1.5 | Install on fresh system without Rust toolchain fails gracefully | [ ] | |
| 1.6 | Docker build completes: `make docker-build` | [ ] | |
| 1.7 | Docker tests pass: `make docker-test-all` | [ ] | |
| 1.8 | Binary runs without shared library errors | [ ] | |
| 1.9 | Install twice (upgrade) works without conflict | [ ] | |
| 1.10 | Uninstall removes binary cleanly | [ ] | |

**Tester:** _______ **Date:** _______

---

## 2. Train Command (10 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 2.1 | `train -f ~/.bash_history -o model.apr` creates model | [ ] | |
| 2.2 | Train with ZSH history format (`: timestamp:0;cmd`) works | [ ] | |
| 2.3 | Train with empty history file produces error message | [ ] | |
| 2.4 | Train with nonexistent file fails gracefully | [ ] | |
| 2.5 | Train with 10 commands produces valid model | [ ] | |
| 2.6 | Train with 10,000+ commands completes in <5s | [ ] | |
| 2.7 | Train with corrupted commands filters them out | [ ] | |
| 2.8 | Train with `-n 2` uses bigram model | [ ] | |
| 2.9 | Train with `-n 5` uses 5-gram model | [ ] | |
| 2.10 | Model file is portable (copy to another machine works) | [ ] | |

**Tester:** _______ **Date:** _______

---

## 3. Suggest Command (15 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 3.1 | `suggest "git "` returns git-related completions | [ ] | |
| 3.2 | `suggest "git s"` returns commands starting with "git s" | [ ] | |
| 3.3 | `suggest ""` (empty) handles gracefully | [ ] | |
| 3.4 | `suggest "nonexistent_cmd_xyz"` returns empty or fallback | [ ] | |
| 3.5 | Suggestions complete in <10ms (P99) | [ ] | |
| 3.6 | Suggestions complete in <50ms under memory pressure | [ ] | |
| 3.7 | Output format is `command\tscore` per line | [ ] | |
| 3.8 | Scores are between 0.0 and 1.0 | [ ] | |
| 3.9 | Results are sorted by score descending | [ ] | |
| 3.10 | `-k 5` limits output to 5 suggestions | [ ] | |
| 3.11 | Partial token "car" matches "cargo" commands | [ ] | |
| 3.12 | Unicode input (emoji, CJK) doesn't crash | [ ] | |
| 3.13 | Very long input (>1000 chars) doesn't hang | [ ] | |
| 3.14 | Null bytes in input handled safely | [ ] | |
| 3.15 | Shell metacharacters (`$()`, backticks) not executed | [ ] | |

**Tester:** _______ **Date:** _______

---

## 4. Security (15 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 4.1 | Sensitive commands (passwords) not suggested | [ ] | |
| 4.2 | `export AWS_SECRET_ACCESS_KEY=xxx` filtered from training | [ ] | |
| 4.3 | `mysql -p password` filtered from training | [ ] | |
| 4.4 | Model file doesn't contain plaintext secrets | [ ] | |
| 4.5 | Path traversal in model path (`../../etc/passwd`) rejected | [ ] | |
| 4.6 | Symlink attacks on model file prevented | [ ] | |
| 4.7 | Model file permissions are 600 or 644 | [ ] | |
| 4.8 | No arbitrary code execution via malformed model | [ ] | |
| 4.9 | Command injection via prefix impossible | [ ] | |
| 4.10 | ZSH widget uses quoted substitution (SC2046) | [ ] | |
| 4.11 | Widget timeout prevents hangs (0.1s default) | [ ] | |
| 4.12 | APRENDER_DISABLED=1 fully disables suggestions | [ ] | |
| 4.13 | No network requests without explicit command | [ ] | |
| 4.14 | HF_TOKEN not logged or stored in model | [ ] | |
| 4.15 | Publish without token doesn't leak credentials | [ ] | |

**Tester:** _______ **Date:** _______

---

## 5. Shell Integration (10 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 5.1 | `zsh-widget` output is valid ZSH syntax | [ ] | |
| 5.2 | `fish-widget` output is valid Fish syntax | [ ] | |
| 5.3 | ZSH widget has start/end markers for uninstall | [ ] | |
| 5.4 | Fish widget has start/end markers for uninstall | [ ] | |
| 5.5 | Widget doesn't break existing shell config | [ ] | |
| 5.6 | Widget works with oh-my-zsh | [ ] | |
| 5.7 | Widget works with prezto | [ ] | |
| 5.8 | Widget works with starship prompt | [ ] | |
| 5.9 | `uninstall --zsh --dry-run` shows what would be removed | [ ] | |
| 5.10 | `uninstall --zsh` removes widget block cleanly | [ ] | |

**Tester:** _______ **Date:** _______

---

## 6. Model Operations (10 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 6.1 | `stats -m model.apr` shows n-gram count | [ ] | |
| 6.2 | `stats` shows vocabulary size | [ ] | |
| 6.3 | `stats` shows top commands | [ ] | |
| 6.4 | `inspect -m model.apr` shows model card | [ ] | |
| 6.5 | `inspect --format json` outputs valid JSON | [ ] | |
| 6.6 | `inspect --format huggingface` outputs valid YAML front matter | [ ] | |
| 6.7 | `export model.json -m model.apr` creates JSON export | [ ] | |
| 6.8 | `import model.json -o new.apr` recreates model | [ ] | |
| 6.9 | Export/import roundtrip preserves suggestions | [ ] | |
| 6.10 | Corrupted model file detected and rejected | [ ] | |

**Tester:** _______ **Date:** _______

---

## 7. Advanced Features (10 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 7.1 | `validate -f history` runs train/test split | [ ] | |
| 7.2 | Validate reports Hit@1, Hit@3, Hit@5 metrics | [ ] | |
| 7.3 | `augment -f history -o model.apr` augments data | [ ] | |
| 7.4 | Augment with `--use-code-eda` enables CodeEDA | [ ] | |
| 7.5 | `analyze -f history` extracts command patterns | [ ] | |
| 7.6 | `analyze --top 10` limits output | [ ] | |
| 7.7 | `tune -f history` runs AutoML hyperparameter search | [ ] | |
| 7.8 | `update -m model.apr -f new_history` incremental update | [ ] | |
| 7.9 | `publish -m model.apr -r org/repo` generates README.md | [ ] | |
| 7.10 | Publish with HF_TOKEN uploads to Hugging Face | [ ] | |

**Tester:** _______ **Date:** _______

---

## 8. Error Handling (10 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 8.1 | Missing required argument shows usage | [ ] | |
| 8.2 | Invalid subcommand shows available commands | [ ] | |
| 8.3 | File permission denied shows clear error | [ ] | |
| 8.4 | Disk full during model save handled | [ ] | |
| 8.5 | Out of memory during training handled | [ ] | |
| 8.6 | Interrupted (Ctrl+C) exits cleanly | [ ] | |
| 8.7 | SIGTERM exits cleanly | [ ] | |
| 8.8 | Invalid UTF-8 in history file handled | [ ] | |
| 8.9 | Binary data in history file handled | [ ] | |
| 8.10 | Network timeout during publish shows retry hint | [ ] | |

**Tester:** _______ **Date:** _______

---

## 9. Performance & Stress (5 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 9.1 | Suggest with 100MB model completes in <100ms | [ ] | |
| 9.2 | 1000 concurrent suggest calls don't crash | [ ] | |
| 9.3 | Memory usage stays <64MB during suggest | [ ] | |
| 9.4 | CPU usage <25% during idle widget polling | [ ] | |
| 9.5 | No memory leaks after 10,000 suggestions | [ ] | |

**Tester:** _______ **Date:** _______

---

## 10. Edge Cases & Fuzzing (5 points)

| # | Test Case | Status | Notes |
|---|-----------|--------|-------|
| 10.1 | Fuzz test: random bytes as prefix (1000 iterations) | [ ] | |
| 10.2 | Fuzz test: random bytes as model file (100 iterations) | [ ] | |
| 10.3 | Fuzz test: random bytes as history file (100 iterations) | [ ] | |
| 10.4 | Maximum path length (4096 chars) handled | [ ] | |
| 10.5 | Zero-byte files handled for all inputs | [ ] | |

**Tester:** _______ **Date:** _______

---

## Summary

| Section | Points | Passed | Failed | Pending |
|---------|--------|--------|--------|---------|
| 1. Installation | 10 | | | |
| 2. Train | 10 | | | |
| 3. Suggest | 15 | | | |
| 4. Security | 15 | | | |
| 5. Shell Integration | 10 | | | |
| 6. Model Operations | 10 | | | |
| 7. Advanced Features | 10 | | | |
| 8. Error Handling | 10 | | | |
| 9. Performance | 5 | | | |
| 10. Edge Cases | 5 | | | |
| **TOTAL** | **100** | | | |

---

## Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| QA Lead | | | |
| Dev Lead | | | |
| Security | | | |

---

## Issue Tracking

Document any failures below:

| Test # | Issue Description | Severity | JIRA/GH Issue |
|--------|-------------------|----------|---------------|
| | | | |
| | | | |
| | | | |

---

## Appendix: Test Commands

```bash
# Quick smoke test
aprender-shell --version
aprender-shell train -f ~/.bash_history -o /tmp/test.model
aprender-shell suggest "git " -m /tmp/test.model
aprender-shell inspect -m /tmp/test.model --format json

# Docker full suite
cd crates/aprender-shell
make docker-test-all

# Performance baseline
time aprender-shell suggest "git " -m /tmp/test.model

# Security: verify no secrets in model
strings /tmp/test.model | grep -i password
strings /tmp/test.model | grep -i secret
strings /tmp/test.model | grep -i token

# Fuzz test prefix (requires cargo-fuzz)
echo "random_$(head -c 100 /dev/urandom | base64)" | \
  xargs -I {} aprender-shell suggest "{}" -m /tmp/test.model 2>&1
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-27 | Claude | Initial 100-point checklist |
