journalctl --user --since "2026-09-20 07:09:01 UTC" --until "2026-09-27 07:09:01 UTC" -o json --output-fields=__REALTIME_TIMESTAMP,UNIT,USER_UNIT,MESSAGE,JOB_TYPE,JOB_RESULT --no-pager > /mnt/nvme-raid0/tmp/af-qm08/j.jsonl 2> /mnt/nvme-raid0/tmp/af-qm08/j.err
echo x=journal=$?
wc -l < /mnt/nvme-raid0/tmp/af-qm08/j.jsonl
python3 - <<'PY'
import json,re
start={};end={};tick={}
for l in open('/mnt/nvme-raid0/tmp/af-qm08/j.jsonl'):
    try: d=json.loads(l)
    except: continue
    u=d.get('UNIT') or d.get('USER_UNIT'); m=d.get('MESSAGE')
    if not isinstance(u,str) or not isinstance(m,str): continue
    t=int(d['__REALTIME_TIMESTAMP'])/1e6
    if m.startswith('Started ') and 'quorum-review.sh' in m:
        tk=re.search(r'--ticket\s+(\S+)',m)
        start[u]=t; tick[u]=tk.group(1) if tk else None
    elif u in start and (m.startswith(u+': Consumed') or 'Deactivated' in m or 'Failed with result' in m or m.startswith('Consumed')):
        end[u]=max(end.get(u,0),t)
rows=[{'unit':u,'ticket':tick[u],'start':start[u],'end':end.get(u),'min':(end[u]-start[u])/60 if u in end else None} for u in start]
json.dump(rows,open('/mnt/nvme-raid0/tmp/af-qm08/units.json','w'),indent=0)
ok=sorted(r['min'] for r in rows if r['min'] is not None)
print('units',len(rows),'timed',len(ok))
if ok: print('p50',ok[len(ok)//2],'p90',ok[int(len(ok)*.9)],'max',ok[-1])
PY
echo x=parse=$?
