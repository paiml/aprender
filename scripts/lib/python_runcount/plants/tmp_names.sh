#!/usr/bin/env bash
# python_runcount.sh --self-test plant: temp names other than /tmp/tmp.XXXXXXXXXX (P20).
# mktemp -XXXXXX suffixes, $$ pids, a mktemp dir inside the repo and a mktemp
# file inside the repo. Each holds different code, so each is its own entry
# point; none is tracked, so each keys by its content, never by its name.
set -euo pipefail
d=$(mktemp -d)
printf '# a\n' > "$d/cell.a1B2c3"
python3 "$d/cell.a1B2c3"
printf '# b\n' > "$d/job.48213.py"
python3 "$d/job.48213.py"
printf 'python3 -c pass\n' > "$d/lane.Zp4Kq9"
bash "$d/lane.Zp4Kq9"
mkdir -p wk.Q7rT2z
printf '# c\n' > wk.Q7rT2z/tool.py
python3 wk.Q7rT2z/tool.py
f=$(mktemp -p . --suffix .py)
printf '# d\n' > "$f"
python3 "$f"
rm -rf "${d:?}" wk.Q7rT2z "${f:?}"
