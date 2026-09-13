# Playbook Snapshots

Test fixtures for `apr` CLI playbook verification.

| File | Purpose |
|------|---------|
| `test.apr` | Minimal APR v2 model for snapshot testing. **GENERATED, not committed** — `pixel_regression.rs::test_apr_file()` writes it when absent, and `.gitignore` keeps it out. It is 28 MB, and while it *was* committed the generation path never ran, which hid a real race in its atomic publish (#3051, and the pid-only temp name that broke under `cargo test`'s threads). |
| `hex_dump.txt` | Expected hex output for `apr hex` tests |
| `tree_ascii.txt` | Expected ASCII tree output |
| `tree_mermaid.md` | Expected Mermaid diagram output |
| `flow_full.txt` | Expected full dataflow output |
| `flow_cross_attn.txt` | Expected cross-attention flow output |
