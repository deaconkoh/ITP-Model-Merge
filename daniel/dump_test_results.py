import numpy as np, glob, os, csv

rows=[]
for path in glob.glob(os.path.join('test_results','**','*.npy'), recursive=True):
    parts=path.split(os.sep)
    # test_results/<data_source>/<test_data>/<file>.npy
    data_source, test_data, fname = parts[1], parts[2], parts[3]
    core=fname[len('Result_'):-len('.npy')]
    suf='_'+test_data
    if core.endswith(suf): core=core[:-len(suf)]
    prefix, model = core.split('+', 1)
    mode='greedy' if prefix.endswith('G') else 'sampling'
    a=np.load(path)
    n=a.shape[0]
    mk=round(float(a[:,0].mean()),2)
    if a.shape[1]>=4:
        ca=round(float(a[:,1].mean()),2); pr=round(float(a[:,2].mean()),2); tm=round(float(a[:,3].mean()),4)
    else:
        ca=''; pr=''; tm=round(float(a[:,1].mean()),4)
    rows.append([data_source,test_data,mode,model,n,mk,ca,pr,tm])

rows.sort(key=lambda r:(r[1], r[3]))
out=os.path.join('test_results','test_results_summary.csv')
with open(out,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['data_source','test_data','mode','model','n_instances','makespan','carbon','priority_weighted_completion','time'])
    w.writerows(rows)
print('wrote', os.path.abspath(out))
print('rows:', len(rows))
for r in rows[:8]: print('  ', r)
