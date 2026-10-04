#!/usr/bin/env bash
# python_runcount.sh --self-test plant: two temp scripts with the same code under
# different names (N1). Content keys them, so they are one entry point.
set -euo pipefail
d=$(mktemp -d)
printf '# same\n' > "$d/one.py"
printf '# same\n' > "$d/two.py"
python3 "$d/one.py"
python3 "$d/two.py"
rm -rf "${d:?}"
