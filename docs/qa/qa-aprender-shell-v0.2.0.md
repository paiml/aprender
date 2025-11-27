# QA Checklist: aprender-shell v0.2.0

**Version Under Test:** 0.2.0 (pre-release)
**Current Published:** 0.1.0
**Test Date:** _______________
**Tester:** _______________
**Platform:** _______________

## Installation Verification

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 1 | `cargo install aprender-shell` | Installs without errors | | |
| 2 | `aprender-shell --version` | Outputs `aprender-shell 0.2.0` | | |
| 3 | `aprender-shell --help` | Shows all commands (train, update, suggest, etc.) | | |
| 4 | `which aprender-shell` | Returns path in ~/.cargo/bin/ | | |
| 5 | Binary size check | Binary < 10MB | | |

## Command: train

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 6 | `aprender-shell train` (auto-detect history) | Trains from ~/.zsh_history or ~/.bash_history | | |
| 7 | `aprender-shell train -f /path/to/history` | Trains from specified file | | |
| 8 | `aprender-shell train -o /tmp/test.model` | Creates model at specified path | | |
| 9 | `aprender-shell train -n 2` | Trains bigram model | | |
| 10 | `aprender-shell train -n 3` | Trains trigram model (default) | | |
| 11 | `aprender-shell train -n 4` | Trains 4-gram model | | |
| 12 | `aprender-shell train -n 5` | Trains 5-gram model | | |
| 13 | `aprender-shell train -n 1` | Errors: N-gram must be 2-5 | | |
| 14 | `aprender-shell train -n 6` | Errors: N-gram must be 2-5 | | |
| 15 | `aprender-shell train --memory-limit 100` | Uses paged storage for large histories | | |
| 16 | Train with empty history file | Handles gracefully, creates minimal model | | |
| 17 | Train with 10 commands | Creates functional model | | |
| 18 | Train with 1000 commands | Creates model in < 5 seconds | | |
| 19 | Train with 100,000 commands | Completes without OOM | | |
| 20 | Train with non-existent history file | Errors with clear message | | |
| 21 | Train with unreadable history file | Errors with permission message | | |
| 22 | Train overwrites existing model | New model replaces old | | |

## Command: update

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 23 | `aprender-shell update` (auto-detect) | Updates existing model incrementally | | |
| 24 | `aprender-shell update -m /tmp/test.model` | Updates specified model | | |
| 25 | `aprender-shell update -q` | Silent output (for hooks) | | |
| 26 | Update non-existent model | Errors or creates new model | | |
| 27 | Update with no new commands | No-op, no error | | |
| 28 | Update with 5 new commands | Model updated in < 100ms | | |
| 29 | Update preserves existing knowledge | Old completions still work | | |
| 30 | Multiple updates in sequence | All updates reflected | | |

## Command: suggest

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 31 | `aprender-shell suggest "git "` | Returns git subcommands | | |
| 32 | `aprender-shell suggest "cd "` | Returns directory suggestions | | |
| 33 | `aprender-shell suggest ""` | Returns most common commands | | |
| 34 | `aprender-shell suggest -c 1 "git "` | Returns 1 suggestion | | |
| 35 | `aprender-shell suggest -c 10 "git "` | Returns up to 10 suggestions | | |
| 36 | `aprender-shell suggest -c 100 "git "` | Returns available suggestions (≤100) | | |
| 37 | `aprender-shell suggest "nonexistent_cmd"` | Returns empty or closest match | | |
| 38 | `aprender-shell suggest -m /tmp/test.model "git "` | Uses specified model | | |
| 39 | Suggest with non-existent model | Errors with clear message | | |
| 40 | Suggest latency < 10ms | Fast response for interactive use | | |
| 41 | Suggest with special characters `"ls -la | grep"` | Handles pipes correctly | | |
| 42 | Suggest with quotes `"echo 'hello'"` | Handles quotes correctly | | |
| 43 | Suggest with unicode `"echo 你好"` | Handles unicode | | |
| 44 | Suggest preserves order by frequency | Most frequent first | | |

## Command: stats

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 45 | `aprender-shell stats` | Shows model statistics | | |
| 46 | Stats shows n-gram count | Displays total n-grams | | |
| 47 | Stats shows vocabulary size | Displays unique tokens | | |
| 48 | Stats shows model size | Displays bytes on disk | | |
| 49 | Stats with non-existent model | Errors with clear message | | |
| 50 | Stats with paged model | Works with --memory-limit | | |

## Command: export

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 51 | `aprender-shell export /tmp/shared.model` | Creates exportable model | | |
| 52 | Export creates valid file | File is readable | | |
| 53 | Export preserves model data | Imported model works same | | |
| 54 | Export to existing file | Overwrites or errors | | |
| 55 | Export to unwritable path | Errors with permission message | | |

## Command: import

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 56 | `aprender-shell import /tmp/shared.model` | Imports to default location | | |
| 57 | `aprender-shell import -o /tmp/imported.model /tmp/shared.model` | Imports to specified path | | |
| 58 | Import invalid file | Errors with format message | | |
| 59 | Import non-existent file | Errors with not found message | | |
| 60 | Imported model produces same suggestions | Consistency verified | | |

## Command: validate

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 61 | `aprender-shell validate` | Runs holdout validation | | |
| 62 | `aprender-shell validate -r 0.8` | Uses 80/20 train/test split | | |
| 63 | `aprender-shell validate -r 0.5` | Uses 50/50 split | | |
| 64 | `aprender-shell validate -r 0.0` | Errors: invalid ratio | | |
| 65 | `aprender-shell validate -r 1.0` | Errors: no test data | | |
| 66 | `aprender-shell validate -n 2` | Validates bigram model | | |
| 67 | Validation reports accuracy metric | Shows percentage | | |
| 68 | Validation reports precision/recall | Shows P/R if available | | |
| 69 | Validation with small history | Works with 20+ commands | | |
| 70 | Validation reproducible | Same result on re-run | | |

## Command: augment

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 71 | `aprender-shell augment` | Augments with synthetic data | | |
| 72 | `aprender-shell augment -a 0.5` | 50% augmentation ratio | | |
| 73 | `aprender-shell augment -a 1.0` | 100% augmentation ratio | | |
| 74 | `aprender-shell augment -q 0.7` | Quality threshold 0.7 | | |
| 75 | `aprender-shell augment --monitor-diversity` | Shows diversity metrics | | |
| 76 | `aprender-shell augment --use-code-eda` | Uses code-aware augmentation | | |
| 77 | Augmented model improves suggestions | Better coverage | | |
| 78 | Augmentation doesn't corrupt model | Original commands still work | | |

## Command: analyze

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 79 | `aprender-shell analyze` | Shows command patterns | | |
| 80 | `aprender-shell analyze -t 5` | Shows top 5 per category | | |
| 81 | `aprender-shell analyze -t 20` | Shows top 20 per category | | |
| 82 | Analyze shows command frequency | Most used commands listed | | |
| 83 | Analyze shows command categories | git, docker, etc. grouped | | |
| 84 | Analyze with empty history | Handles gracefully | | |

## Command: tune

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 85 | `aprender-shell tune` | Runs AutoML tuning | | |
| 86 | `aprender-shell tune -t 5` | Runs 5 trials | | |
| 87 | `aprender-shell tune -t 20` | Runs 20 trials | | |
| 88 | Tune reports best hyperparameters | Shows optimal n-gram size | | |
| 89 | Tune reports validation score | Shows accuracy metric | | |
| 90 | Tune completes in reasonable time | < 2 min for 10 trials | | |

## Command: zsh-widget

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 91 | `aprender-shell zsh-widget` | Outputs valid ZSH code | | |
| 92 | Widget code is sourceable | `source <(aprender-shell zsh-widget)` works | | |
| 93 | Widget binds Ctrl+Space | Keybinding set correctly | | |
| 94 | Widget calls suggest | Invokes aprender-shell suggest | | |

## Edge Cases & Error Handling

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| 95 | Corrupt model file | Errors with clear message, doesn't crash | | |
| 96 | Model file from older version | Handles migration or errors cleanly | | |
| 97 | Disk full during train | Errors gracefully | | |
| 98 | Interrupt during train (Ctrl+C) | Clean exit, no corrupt files | | |
| 99 | Very long command (10KB) | Handles without crash | | |
| 100 | Binary data in history | Skips invalid entries gracefully | | |

## Performance Benchmarks

| # | Metric | Target | Actual | Pass/Fail |
|---|--------|--------|--------|-----------|
| P1 | Train 10K commands | < 2s | | |
| P2 | Suggest latency | < 10ms | | |
| P3 | Update latency | < 100ms | | |
| P4 | Memory usage (10K model) | < 50MB | | |
| P5 | Binary size | < 10MB | | |

## Platform Compatibility

| # | Platform | Status | Notes |
|---|----------|--------|-------|
| C1 | Linux x86_64 | | |
| C2 | Linux aarch64 | | |
| C3 | macOS x86_64 | | |
| C4 | macOS aarch64 (M1/M2) | | |
| C5 | Windows x86_64 | | |

## Integration Tests

| # | Test Case | Expected Result | Pass/Fail | Notes |
|---|-----------|-----------------|-----------|-------|
| I1 | Full workflow: train → suggest → update → suggest | All steps work | | |
| I2 | Export → Import → suggest | Imported model works | | |
| I3 | ZSH integration (interactive) | Tab completion works | | |
| I4 | Bash integration (if supported) | Completion works | | |
| I5 | Multiple users sharing model | No conflicts | | |

---

## Summary

| Category | Total | Passed | Failed | Blocked |
|----------|-------|--------|--------|---------|
| Installation | 5 | | | |
| train | 17 | | | |
| update | 8 | | | |
| suggest | 14 | | | |
| stats | 6 | | | |
| export | 5 | | | |
| import | 5 | | | |
| validate | 10 | | | |
| augment | 8 | | | |
| analyze | 6 | | | |
| tune | 6 | | | |
| zsh-widget | 4 | | | |
| Edge Cases | 6 | | | |
| **TOTAL** | **100** | | | |

## Sign-off

- [ ] All critical tests passed
- [ ] No regressions from v0.1.0
- [ ] Performance targets met
- [ ] Ready for release

**QA Lead Signature:** _______________
**Date:** _______________
