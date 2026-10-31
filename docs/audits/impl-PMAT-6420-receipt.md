# PMAT-6420 receipt: union of the 4 workspace-test shards equals the pre-split list

- Measured: 2026-10-26T18:04:07Z (UTC)
- Tree: `1455109f94cbc1d4275fc1210359a115453a95d6` (the receipt commit itself adds only this file)
- Command: `bash scripts/check_shard_union.sh` (reads M from ci/sections.yml pins), exit code 0

```
ok    fragments (pull_request): union of 4 shards == pre-split list (134 commands)
ok    fragments (merge_group): union of 4 shards == pre-split list (134 commands)
```

Both events run N=4 (pins, ci.yml matrix and the sections.yml jobs matrix; `check_shard_count_single_source.sh` ok at count 4).
Not in this receipt: the MQ p95 (post-merge, first 20 green merge_group runs).
