# Fixture README (NOT the project README)

Used by `scripts/check_model_parity.sh --self-test`. This file NAMES the three
refused models — so a predicate that merely greps for the name would fire — while
asserting nothing at all about how one backend compares to another. That is the
state the real README.md is in today, and it is why the three refused models are
reported as UNMEASURED-TOOL instead of turning C14 red.

(The words this file must NOT contain anywhere near a model name are the ones the
predicate keys on; describing them here in prose would make the fixture its own
false positive, which is exactly how the first draft of this file failed.)

```bash
apr inspect --json Qwen3-Coder-30B-A3B-Instruct     # arch=qwen3moe, 30 B params
apr tensors --json Qwen3-Coder-30B-A3B-Instruct     # 579 tensors (MoE expert layout)
apr run qwen3.5-0.8b --prompt "hi"
apr inspect qwen3-30b
```
