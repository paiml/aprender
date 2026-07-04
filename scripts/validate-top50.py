import os
import json
import urllib.request
import subprocess
import time
import sys

def run_cmd(cmd, check=True, timeout=None):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}\n{result.stderr}")
        raise Exception(f"Command failed with {result.returncode}")
    return result

def get_top_50_models():
    url = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=50&filter=gguf"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def get_gguf_file(model_id):
    url = f"https://huggingface.co/api/models/{model_id}/tree/main"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            files = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to fetch tree for {model_id}: {e}")
        return None
    
    # prefer a small quant if possible
    ggufs = [f for f in files if f['path'].endswith('.gguf')]
    if not ggufs:
        return None
    for target in ['Q4_K_M', 'Q2_K', 'Q4_0']:
        for g in ggufs:
            if target.lower() in g['path'].lower():
                return g['path']
    return ggufs[0]['path']

def validate_model(model_id, filename):
    url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
    model_dir = os.path.expanduser("~/.apr/models")
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, os.path.basename(filename))
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename} for {model_id}...")
        try:
            run_cmd(f"curl -sL -o '{filepath}' '{url}'")
        except Exception:
            return "fail (download)"
    
    print(f"Testing {model_id}...")
    try:
        # Run chat with timeout
        cmd = f"echo -e 'What is 2+2?\\n/quit' | timeout 60 target/release/apr chat '{filepath}' --max-tokens 16 --temperature 0.1"
        result = run_cmd(cmd, check=False)
        output = result.stdout + result.stderr
        
        if result.returncode == 0 and ("loaded" in output.lower() or "model" in output.lower()) and ("goodbye" in output.lower() or "quit" in output.lower() or "4" in output):
            return "pass"
        else:
            print(f"Failed output: {output[:500]}")
            return "fail"
    except Exception as e:
        print(f"Error testing: {e}")
        return "error"
    finally:
        # Cleanup to save space
        if os.path.exists(filepath):
            os.remove(filepath)

def main():
    print("Building apr...")
    run_cmd("cargo build --release -p apr-cli --features inference")
    
    print("Fetching top 50 GGUF models...")
    models = get_top_50_models()
    
    results = {}
    for i, model in enumerate(models):
        model_id = model['id']
        print(f"[{i+1}/50] Validating {model_id}")
        
        filename = get_gguf_file(model_id)
        if not filename:
            print(f"No GGUF file found for {model_id}")
            results[model_id] = {"status": "skip", "file": None}
            continue
            
        status = validate_model(model_id, filename)
        results[model_id] = {"status": status, "file": filename}
        
    # Write results
    with open("target/top50-results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Create markdown matrix
    with open("docs/top50-chat-matrix.md", "w") as f:
        f.write("# Top 50 Hugging Face GGUF Models Validation\n\n")
        f.write("| Model | File | Status |\n")
        f.write("|---|---|---|\n")
        for m, data in results.items():
            status_icon = "✅" if data['status'] == "pass" else ("❌" if "fail" in data['status'] else "⚠️")
            f.write(f"| {m} | {data['file']} | {status_icon} |\n")
            
    print("Committing and pushing results...")
    
    branch_name = f"auto-validate-top50-{int(time.time())}"
    run_cmd(f"git checkout -b {branch_name}")
    run_cmd("git add target/top50-results.json docs/top50-chat-matrix.md")
    run_cmd("git commit -m 'chore: autonomously validate top 50 HF models'")
    run_cmd(f"git push -u origin {branch_name}")
    
    print("Creating PR...")
    run_cmd(f"gh pr create -B main -H {branch_name} -t 'Validation of top 50 HF models' -b 'Autonomous validation results'")
    run_cmd("gh pr merge --auto --squash")
    print("Done!")

if __name__ == "__main__":
    main()
