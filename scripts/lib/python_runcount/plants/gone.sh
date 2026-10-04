#!/usr/bin/env bash
# python_runcount.sh --self-test plant: a temp script run by its own absolute #!
# line (no shim sees it, so it has no call-time hash) and deleted before
# --count can read it (P26). It keys as gone, by its caller.
set -euo pipefail
d=$(mktemp -d)
printf '#!%s/python3\n# gone\n' "${PYRUN_FAKE_BIN:?}" > "$d/job"
chmod 755 "$d/job"
"$d/job"
rm -rf "${d:?}"
