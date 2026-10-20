#!/bin/bash
# Chaotic-pacing Cinema Studio test (10 cuts in 8 s) through the dev server. Usage: cinema_chaotic.sh [server]
S=${1:-http://127.0.0.1:8805}
P=$(curl -s -X POST $S/api/cinema/compose -H 'Content-Type: application/json' -d '{"scene":"a street drummer on a plastic bucket in a crowded night market, neon signs and steam","genre":"action","era":"2020s","light":"practicals","camera":"modern","lens":"anamorphic","aperture":"f4","pacing":"chaotic","seconds":8}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['prompt']); sys.stderr.write('cuts %s\n' % d['cuts'])")
python3 - "$P" "$S" <<'PY'
import json,sys,urllib.request,time
body=json.dumps({"mode":"t2v","prompt":sys.argv[1],"width":864,"height":480,"seconds":8,"seed":5,"label":"Cinema · chaotic · action"}).encode()
req=urllib.request.Request(sys.argv[2]+"/api/video",data=body,headers={"Content-Type":"application/json","X-Backlot-Async":"1"})
job=json.load(urllib.request.urlopen(req))["job"]; print("job",job,flush=True); t0=time.time()
while True:
    time.sleep(10)
    try:
        r=urllib.request.urlopen(f"{sys.argv[2]}/api/jobs/{job}")
        if r.status==200: d=json.load(r); print("OK",d["url"],d.get("took"),"s"); break
    except urllib.error.HTTPError as e: print("FAILED",e.code,e.read().decode()[:300]); sys.exit(1)
    if time.time()-t0>1800: print("timeout"); sys.exit(1)
PY
