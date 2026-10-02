"""Installation test with independent numerical checks; run on Linux/GPU host."""
import argparse, datetime, json, math, os, platform, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--project',required=True,choices=['cuPDLP-C','HDSDP','SOLNP_plus','cuLoRADS','PDHCG'])
p.add_argument('--mode',default='gpu',choices=['cpu','gpu'])
p.add_argument('--case',default='tiny',choices=['tiny','trace'])
p.add_argument('--binary',type=Path)
p.add_argument('--output',type=Path,default=Path('tests/logs/tiny'))
a=p.parse_args()
root=Path.cwd(); output_root=a.output.resolve(); output_root.mkdir(parents=True,exist_ok=True)
# Existing results never satisfy a new invocation. No user directory is deleted.
out=Path(tempfile.mkdtemp(prefix='run-',dir=output_root))
(output_root/'verification.json').write_text(json.dumps({'pass':False,'state':'running','run_directory':str(out)}))
fixture=root/'examples'
env=os.environ.copy()
env.setdefault('CUDA_VISIBLE_DEVICES','0')
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='SEQUENTIAL')
def execute(cmd):
    (out/'command.json').write_text(json.dumps(cmd,indent=2))
    with (out/'solver.log').open('w') as f:
        proc=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
    return proc.returncode,(out/'solver.log').read_text(errors='replace')
def finite(x): return bool(np.isfinite(np.asarray(x,dtype=float)).all())
def emit(x):
    report.update(x)
    report['run_directory']=str(out)
    encoded=json.dumps(report,indent=2,allow_nan=False)
    (out/'verification.json').write_text(encoded)
    (output_root/'verification.json').write_text(encoded)
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
report={'project':a.project,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host':platform.node(),'primal_feasibility_tolerance':1e-6,'pass':False}
try:
    if a.project=='cuPDLP-C':
        binary=a.binary or root/f'build-coinor-{a.mode}/bin/plc'
        rc,log=execute([str(binary),'-fname',str(fixture/'tiny_lp.mps'),'-ifPre','0','-savesol','1',
            '-out',str(out/'summary.json'),'-outSol',str(out/'solution.json'),'-dPrimalTol','1e-6',
            '-dDualTol','1e-8','-dGapTol','1e-8','-nIterLim','10000','-dTimeLim','60'])
        s=json.loads((out/'summary.json').read_text()); sol=json.loads((out/'solution.json').read_text())
        x=np.asarray(sol['col_value']); y=np.asarray(sol['row_dual']); z=np.asarray(sol['col_dual'])
        A=np.array([[1.,2.],[2.,1.]]); b=np.array([5.,4.]); c=np.array([2.,3.])
        pr=float(max(0.,np.max(b-A@x),np.max(-x))); obj=float(c@x)
        dr=float(max(np.max(np.abs(c-A.T@y-z)),np.max(-y),np.max(-z),0.))
        gap=float(abs(obj-b@y))
        ok=rc==0 and finite([*x,*y,*z,obj]) and s['terminationCode']=='OPTIMAL' and pr<=1e-6 and dr<=1e-6 and gap<=1e-4 and abs(obj-8)<=1e-4 and np.max(np.abs(x-[1,2]))<=1e-4
        emit({'mode':a.mode,'solver_exit':rc,'x':x.tolist(),'objective':obj,'primal_residual':pr,'dual_residual':dr,'absolute_gap':gap,'pass':bool(ok)})
    elif a.project=='PDHCG':
        binary=a.binary or root/'build-coinor/pdhcg'
        rc,log=execute([str(binary),str(fixture/'tiny_qp.qps'),str(out),'--eps_opt','1e-8','--eps_feas','1e-6','--time_limit','60','--iter_limit','10000','--eval_freq','20'])
        s=dict(line.split(': ',1) for line in (out/'tiny_qp_summary.txt').read_text().splitlines() if ': ' in line)
        x=np.loadtxt(out/'tiny_qp_primal_solution.txt',ndmin=1)
        obj=float(x[0]**2+2*x[1]**2-2*x[0]-8*x[1]); pr=float(max(abs(x.sum()-3),max(0.,-x.min())))
        ok=rc==0 and finite(x) and s['Termination Reason']=='OPTIMAL' and pr<=1e-6 and abs(obj+9)<=1e-4 and np.max(np.abs(x-[1,2]))<=1e-4 and float(s['Absolute Dual Residual'])<=1e-6
        emit({'solver_exit':rc,'x':x.tolist(),'objective':obj,'primal_residual':pr,'reported_dual_residual':float(s['Absolute Dual Residual']),'pass':bool(ok)})
    elif a.project in ['HDSDP','SOLNP_plus']:
        binary=a.binary or root/('build-coinor/coinor_sdp_test' if a.project=='HDSDP' else 'build-coinor/test_nlp')
        problem=('trace_sdp.dat-s' if a.case=='trace' else 'tiny_sdp.dat-s') if a.project=='HDSDP' else 'problem.txt'
        rc,log=execute([str(binary),str(fixture/problem)])
        records=[line.split('COINOR_RESULT ',1)[1] for line in log.splitlines() if line.startswith('COINOR_RESULT ')]
        r=json.loads(records[-1]); ok=rc==0 and r['pass']
        if a.project=='HDSDP':
            X=np.array(r['X_raw']).reshape(4,4,order='F'); y=np.array(r['dual'])
            C=np.diag([1.,2.,3.,4.]) if a.case=='trace' else .5*(np.eye(4)-np.ones((4,4)))
            obj=float(np.sum(C*X)); pr=float(abs(np.trace(X)-1)) if a.case=='trace' else float(np.max(np.abs(np.diag(X)-1)))
            mineig=float(np.linalg.eigvalsh(X).min()); target=1. if a.case=='trace' else -6.
            slack=C-y[0]*np.eye(4) if a.case=='trace' else C-np.diag(y)
            ds=float(np.linalg.eigvalsh(slack).min()); gap=float(abs(obj-y.sum()))
            ok=ok and finite([*X.ravel(),*y]) and pr<=1e-6 and mineig>=-1e-6 and ds>=-1e-6 and gap<=1e-3 and abs(obj-target)<=1e-3
            r.update(objective=obj,primal_residual=pr,primal_min_eigenvalue=mineig,dual_min_eigenvalue=ds,absolute_gap=gap)
        else:
            x=np.array(r['x']); obj=float((x[0]-1)**2+(x[1]-2)**2); pr=float(abs(x.sum()-3))
            ok=ok and r['status']==1 and finite(x) and pr<=1e-6 and obj<=1e-6 and np.max(np.abs(x-[1,2]))<=1e-3
            r.update(objective=obj,primal_residual=pr)
        r.update(pass_=bool(ok),solver_exit=rc)
        r['pass']=r.pop('pass_'); emit(r)
    else:
        import h5py
        binary=a.binary or root/'runtime/bin/cuLoRADS'
        rc,log=execute([str(binary),'--filePath',str(fixture/'tiny_sdp.dat-s'),'--outputPath',str(out),'--phase2Tol','1e-6','--timeSecLimit','60'])
        with h5py.File(out/'tiny_sdp.out.mat') as m:
            ref=np.asarray(m['primal_sdp']).reshape(-1)[0]
            U=np.array(m[ref]).T; y=np.array(m['dual']).ravel()
        X=U@U.T; C=.5*(np.eye(4)-np.ones((4,4)))
        obj=float(np.sum(C*X)); pr=float(np.max(np.abs(np.diag(X)-1)))
        ds=float(np.linalg.eigvalsh(C-np.diag(y)).min()); gap=float(abs(obj-y.sum()))
        ok=rc==0 and finite([*X.ravel(),*y]) and pr<=1e-6 and ds>=-1e-6 and abs(obj+6)<=1e-3 and gap<=1e-3
        emit({'solver_exit':rc,'X':X.tolist(),'dual':y.tolist(),'objective':obj,'primal_residual':pr,'dual_min_eigenvalue':ds,'absolute_gap':gap,'pass':bool(ok)})
except Exception as e:
    emit({'error':f'{type(e).__name__}: {e}','pass':False})
sys.exit(0 if report['pass'] else 1)
