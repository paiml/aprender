#!/usr/bin/env bash
# python_runcount.sh --self-test plant: a descendant daemonizes and outlives the lane (D1, D2).
# It blocks on a fifo the self-test opens afterwards, so it never outlives the test.
set -euo pipefail
( setsid bash -c 'read -r _ < "$1"' _ "$PYRUN_TEST_FIFO" < /dev/null > /dev/null 2>&1 & )
python3 sub/tool.py
