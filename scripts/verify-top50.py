import urllib.request
import json
import subprocess
import os

url = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=50&filter=gguf"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())

print(f"Found {len(data)} models")
for model in data:
    print(model['id'])
