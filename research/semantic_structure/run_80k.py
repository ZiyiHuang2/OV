"""Run the fixed seven-arm protocol, one training process per GPU."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path

root=Path(sys.argv[1]);src=Path(__file__).parent
assert json.loads((root/'preflight_checks.json').read_text())['exact_resume_2_plus_1']
def lane(gpu,arms):
    for arm in arms:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[sys.executable,str(src/'study_80k.py'),'--root',str(root),'--arm',arm]
        info={'arm':arm,'gpu':gpu,'command':cmd,'started_unix':time.time()}
        (root/f'launch_{arm}.json').write_text(json.dumps(info,indent=2))
        with (root/f'{arm}.log').open('w') as log:
            code=subprocess.run(cmd,env=env,cwd=src,stdout=log,stderr=subprocess.STDOUT).returncode
        (root/f'exit_{arm}.json').write_text(json.dumps({'arm':arm,'exit_code':code,'finished_unix':time.time()}))
        if code:raise RuntimeError(f'{arm} failed; see {root}/{arm}.log')
        print('COMPLETE',arm,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    jobs=[pool.submit(lane,0,['B','E','H']),pool.submit(lane,1,['C','F']),pool.submit(lane,2,['D','G'])]
    errors=[]
    for job in jobs:
        try:job.result()
        except Exception as e:errors.append(str(e));print('FAIL',str(e),flush=True)
if errors:
    (root/'FAILED.json').write_text(json.dumps(errors,indent=2));sys.exit(1)
subprocess.run([sys.executable,str(src/'summarize_80k.py'),'--root',str(root)],check=True)
(root/'COMPLETE').write_text('All seven fixed-endpoint runs and matched checks completed.\n')
print('ALL_COMPLETE',flush=True)
