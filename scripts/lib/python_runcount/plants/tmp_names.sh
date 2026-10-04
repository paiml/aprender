#!/usr/bin/env bash
# python_runcount.sh --self-test plant: temp names other than /tmp/tmp.XXXXXXXXXX (P20).
# The names are fixed here so the row is deterministic; at run time they are
# mktemp -XXXXXX suffixes, $$ pids, and a mktemp dir inside the repo. The last
# file is a real mktemp in the repo: its 10 X's are random every run.
set -euo pipefail
d=$(mktemp -d)
: > "$d/cell.a1B2c3"
python3 "$d/cell.a1B2c3"
: > "$d/job.48213.py"
python3 "$d/job.48213.py"
printf 'python3 -c pass\n' > "$d/lane.Zp4Kq9"
bash "$d/lane.Zp4Kq9"
mkdir -p wk.Q7rT2z
: > wk.Q7rT2z/tool.py
python3 wk.Q7rT2z/tool.py
f=$(mktemp -p . --suffix .py)
python3 "$f"
rm -rf "${d:?}" wk.Q7rT2z "${f:?}"
