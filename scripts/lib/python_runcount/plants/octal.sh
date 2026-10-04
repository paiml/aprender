#!/usr/bin/env bash
# python_runcount.sh --self-test plant: a script name strace prints with octal
# escapes (non-ASCII bytes) and a space (P24). The trace and the shim must agree.
set -euo pipefail
python3 "$(printf 'sub/t\303\266\303\266l.py')"
python3 'sub/two words.py'
