import urllib.request
import json
import subprocess
import argparse
import sys

def get_top_50_models():
    url = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=50&filter=text-generation"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                print(f"Failed to fetch models: {response.status}")
                return []
            data = json.loads(response.read().decode())
            return [model['modelId'] for model in data]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def run_cmd(cmd, timeout=120):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def test_model(model_id, mode="oracle"):
    results = {}
    
    # 1. Oracle test
    success, stdout, stderr = run_cmd(["cargo", "run", "--bin", "apr", "--", "oracle", "--json", f"hf://{model_id}"])
    results["oracle"] = {"success": success, "error": stderr if not success else ""}
    
    if mode == "e2e" and success:
        # Pull model (shallow or full)
        pull_succ, _, pull_err = run_cmd(["cargo", "run", "--bin", "apr", "--", "pull", f"hf://{model_id}"])
        results["pull"] = {"success": pull_succ, "error": pull_err}
        
        if pull_succ:
            # Test chat (pipe input since it's interactive)
            chat_res = subprocess.run(["cargo", "run", "--bin", "apr", "--", "chat", f"hf://{model_id}"], input="Hi\n/exit\n", capture_output=True, text=True, timeout=120)
            chat_succ = chat_res.returncode == 0
            results["chat"] = {"success": chat_succ, "error": chat_res.stderr}
            
            # Test code
            code_succ, _, code_err = run_cmd(["cargo", "run", "--bin", "apr", "--", "code", f"hf://{model_id}", "-p", "def hello():", "--max-tokens", "1"])
            results["code"] = {"success": code_succ, "error": code_err}
            
            # Test serve plan
            serve_succ, _, serve_err = run_cmd(["cargo", "run", "--bin", "apr", "--", "serve", "plan", f"hf://{model_id}"])
            results["serve"] = {"success": serve_succ, "error": serve_err}
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Parity Protocol")
    parser.add_argument("--mode", choices=["oracle", "e2e"], default="oracle", help="Test mode: 'oracle' just checks metadata, 'e2e' tests pull, chat, code, serve.")
    parser.add_argument("--limit", type=int, default=50, help="Number of models to test")
    args = parser.parse_args()

    models = get_top_50_models()[:args.limit]
    print(f"Found {len(models)} models. Verifying with mode={args.mode}...")
    
    all_results = []
    
    for i, model in enumerate(models):
        print(f"[{i+1}/{len(models)}] Testing {model}...")
        res = test_model(model, mode=args.mode)
        
        all_results.append({
            'model': model,
            'results': res
        })
        
        oracle_ok = res.get("oracle", {}).get("success", False)
        print(f"  -> Oracle: {'✅' if oracle_ok else '❌'}")
        
        if args.mode == "e2e" and oracle_ok:
            pull_ok = res.get("pull", {}).get("success", False)
            chat_ok = res.get("chat", {}).get("success", False)
            code_ok = res.get("code", {}).get("success", False)
            serve_ok = res.get("serve", {}).get("success", False)
            print(f"  -> Pull: {'✅' if pull_ok else '❌'} | Chat: {'✅' if chat_ok else '❌'} | Code: {'✅' if code_ok else '❌'} | Serve: {'✅' if serve_ok else '❌'}")
            
    with open("parity_protocol_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("Verification complete. Results saved to parity_protocol_results.json")

if __name__ == "__main__":
    main()
