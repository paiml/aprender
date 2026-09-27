cd /home/noah/src/aprender || exit 1
: > /mnt/nvme-raid0/tmp/af-qm08/comments.jsonl
for n in $(python3 -c 'import json;[print(x["number"]) for x in json.load(open("/mnt/nvme-raid0/tmp/af-qm08/merged.json"))]'); do
  timeout 60 gh api "repos/paiml/aprender/issues/$n/comments?per_page=100" --paginate -q ".[] | select(.body|startswith(\"quorum-review\")) | {pr: $n, at: .created_at, head: (.body|capture(\"\\\"head\\\": \\\"(?<h>[0-9a-f]+)\\\"\").h // null), agreed: (.body|test(\"\\\"agreed\\\": true\")), line: (.body|split(\"\\n\")[0])}" >> /mnt/nvme-raid0/tmp/af-qm08/comments.jsonl 2>> /mnt/nvme-raid0/tmp/af-qm08/comments.err
done
echo x=comments=$?
wc -l < /mnt/nvme-raid0/tmp/af-qm08/comments.jsonl
