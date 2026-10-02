#!/usr/bin/env python3
"""Check that stale success files and incorrect fresh solutions are rejected."""
import argparse,json,pathlib,subprocess,sys,tempfile
p=argparse.ArgumentParser();p.add_argument('--project',required=True,choices=['cuPDLP-C','PDHCG']);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent.parent
def results(out,x):
    if a.project=='cuPDLP-C':
        (out/'summary.json').write_text(json.dumps({'terminationCode':'OPTIMAL'}))
        (out/'solution.json').write_text(json.dumps({'col_value':x,'row_dual':[4/3,1/3],'col_dual':[0,0]}))
    else:
        (out/'tiny_qp_summary.txt').write_text('Termination Reason: OPTIMAL\nAbsolute Dual Residual: 0\n')
        (out/'tiny_qp_primal_solution.txt').write_text('\n'.join(map(str,x)))
with tempfile.TemporaryDirectory(prefix='checker-regression-') as tmp:
    tmp=pathlib.Path(tmp);out=tmp/'results';out.mkdir();results(out,[1,2])
    cmd=[sys.executable,str(root/'tests/run.py'),'--project',a.project,'--output',str(out)]
    for label,binary in [('stale_success_with_noop','/usr/bin/true')]:
        run=subprocess.run(cmd+['--binary',binary],cwd=root,capture_output=True,text=True)
        report=json.loads((out/'verification.json').read_text())
        assert run.returncode!=0 and report['pass'] is False,(label,run.stdout)
        print('PASS checker regression:',label)
    fake=tmp/'wrong_solver.py'
    fake.write_text('#!/usr/bin/env python3\nimport sys,json,pathlib\na=sys.argv\n'
        + ("s=pathlib.Path(a[a.index('-out')+1]); x=pathlib.Path(a[a.index('-outSol')+1]); s.write_text(json.dumps({'terminationCode':'OPTIMAL'})); x.write_text(json.dumps({'col_value':[0,0],'row_dual':[0,0],'col_dual':[0,0]}))\n" if a.project=='cuPDLP-C' else
           "o=pathlib.Path(a[2]); (o/'tiny_qp_summary.txt').write_text('Termination Reason: OPTIMAL\\nAbsolute Dual Residual: 0\\n'); (o/'tiny_qp_primal_solution.txt').write_text('0\\n0\\n')\n"))
    fake.chmod(0o755)
    run=subprocess.run(cmd+['--binary',str(fake)],cwd=root,capture_output=True,text=True)
    report=json.loads((out/'verification.json').read_text())
    assert run.returncode!=0 and report['pass'] is False,run.stdout
    print('PASS checker regression: incorrect_fresh_solution')
