#!/usr/bin/env bash
# lint-self-referential-default.sh
#
# CONTRACT: streaming-safety-v1 FALSIFY-SS-005
# Detects ..Default::default() or ..Self::default() inside impl Default blocks.
# This is infinite recursion at runtime — a silent bug that compiles cleanly.
#
# Root cause: realizr cf10c0f7 (TTFT regression from F→A+, probador 99.0)
# Five-whys: ..Default::default() in impl Default = self-call = stack overflow
#
# Usage: ./scripts/lint-self-referential-default.sh [path]
# Exit 0 = clean, Exit 1 = bug found

set -euo pipefail
DIR="${1:-.}"

BUGS=$(python3 << 'PYEOF'
import re, glob, sys

bugs = 0
for f in glob.glob(f"{sys.argv[1] if len(sys.argv) > 1 else '.'}/src/**/*.rs", recursive=True):
    try:
        lines = open(f).readlines()
    except:
        continue

    in_default_impl = False
    brace_depth = 0
    impl_type = ""

    for i, line in enumerate(lines):
        stripped = line.strip()

        m = re.match(r'impl\s+Default\s+for\s+(\w+)', stripped)
        if m:
            in_default_impl = True
            impl_type = m.group(1)
            brace_depth = 0

        if in_default_impl:
            brace_depth += stripped.count('{') - stripped.count('}')
            if brace_depth <= 0 and '{' not in stripped and '}' in stripped:
                in_default_impl = False

            if '..Default::default()' in stripped or '..Self::default()' in stripped:
                print(f"ERROR: {f}:{i+1}: ..Default::default() inside impl Default for {impl_type}")
                print(f"  This is INFINITE RECURSION (calls itself). Remove the line.")
                print(f"  Contract: streaming-safety-v1 FALSIFY-SS-005")
                bugs += 1

sys.exit(1 if bugs > 0 else 0)
PYEOF
)

echo "$BUGS"
if [ $? -eq 0 ]; then
    echo "✅ No self-referential Default impls found"
fi
