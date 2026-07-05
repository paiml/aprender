#!/bin/bash
pmat work ls --color never | grep -E "\s+planned\s+" | awk '{print $1}' > remaining.txt
total=$(wc -l < remaining.txt)
echo "Generating L1-L5 contracts for $total tickets..."

cat remaining.txt | xargs -n 1 -P 10 pmat work start > /dev/null 2>&1
echo "Done generating contracts."
