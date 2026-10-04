#!/usr/bin/env bash
# python_runcount.sh --self-test plant: a lane that runs no Python (P0).
set -euo pipefail
printf "%s\n" "no python here" > /dev/null
env true
