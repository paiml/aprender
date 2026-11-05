# REL1 per-host receipt merge (fallback): sha mismatch, assertion FAIL, allowlist miss and missing x86_64 cuda C14 each turn the merge RED; the baseline merges GO
out=$(bash scripts/release/falsify_merge_host_receipts.sh 2>&1); rc=$?; printf "%s\n" "$out"; [ "$rc" -eq 0 ] && printf "%s\n" "$out" | grep -q "falsify_merge_host_receipts: PASS$"
