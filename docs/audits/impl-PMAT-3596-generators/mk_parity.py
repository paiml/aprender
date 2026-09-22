"""#3596 parity table from parity-*.json (qwen35_prefill_parity) on each host dir."""
import glob, json, os, sys

print("| host | model | attention | positions | sampled (every 200th) | argmax agree | worst 1 − cosine | worst rel L∞ (logits) | per-token path | batched prefill | tool binary |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    for f in sorted(glob.glob(os.path.join(d, "parity-*.json"))):
        j = json.load(open(f))
        mode = "flash" if "flash" in f else "f32"
        model = os.path.basename(j["model"]).replace("-Q4_K_M.gguf", "")
        worst = min(j["rows"], key=lambda r: r["cosine"])
        print(f"| {label} | {model} | {mode} | {j['n_tokens']:,} | {j['sampled_positions']} | {j['argmax_agree']}/{j['sampled_positions']} | "
              f"{1 - j["worst_cosine"]:.1e} (pos {worst["pos"]}) | {j['worst_rel_linf']:.2e} | {j['per_token_ms']/1000:,.0f} s | "
              f"{j['prefill_ms']:,.0f} ms | qwen35_prefill_parity @ d78eab8eb |")
