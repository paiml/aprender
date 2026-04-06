# PMAT Bug Report

**Repository:** paiml/paiml-mcp-agent-toolkit

## Bug: `pmat work` commands fail to load roadmap after `pmat work init`

### Environment
- PMAT version: v2.198.0
- OS: Linux 4.4.0
- Rust: 1.75+

### Steps to Reproduce

1. Initialize a new git repository
2. Run `pmat work init`
   - Output: "✅ Created roadmap: ./docs/roadmaps/roadmap.yaml"
   - Output: "✅ Installed commit-msg hook"
3. Verify roadmap file exists:
   ```bash
   ls -la docs/roadmaps/roadmap.yaml
   # File exists with proper content
   ```
4. Try to check status:
   ```bash
   pmat work status
   ```

### Expected Behavior
`pmat work status` should load and display the roadmap from `docs/roadmaps/roadmap.yaml`

### Actual Behavior
```
Error: Failed to load roadmap. Run `pmat work init` first.
```

Alternatively, when using other subcommands:
```bash
pmat roadmap status --format table
# Error: Failed to read roadmap from docs/execution/roadmap.md
```

### Issue Analysis
There appears to be a mismatch in expected roadmap locations:
- `pmat work init` creates: `./docs/roadmaps/roadmap.yaml`
- `pmat work status` tries to load from: Unknown (fails to load)
- `pmat roadmap status` tries to load from: `docs/execution/roadmap.md`

### Impact
- Users cannot use `pmat work start <ticket-id>` to begin work
- Users cannot check work status
- Workflow tracking is broken despite successful initialization

### Workaround
None found. Users can manually reference tickets in commit messages (the commit-msg hook works correctly), but cannot use the workflow tracking features.

### Suggested Fix
Standardize on a single roadmap location:
1. Either use `docs/roadmaps/roadmap.yaml` everywhere, OR
2. Update `pmat work init` to create the file where other commands expect it

### Additional Context
The commit-msg hook (installed by `pmat work init`) correctly looks for the roadmap at `docs/roadmaps/roadmap.yaml`:
```bash
ROADMAP="docs/roadmaps/roadmap.yaml"
```

This suggests `docs/roadmaps/roadmap.yaml` should be the canonical location.

### Files Attached
- Roadmap file: `/home/user/Batuta/docs/roadmaps/roadmap.yaml`
- Commit hook: `/home/user/Batuta/.git/hooks/commit-msg`
