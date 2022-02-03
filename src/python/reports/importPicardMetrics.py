#!/home/ec2-user/miniconda3/envs/genomics.py3/bin/python

import sys
import glob
import pandas as pd
import h5py

if len(sys.argv) != 2:
    print("Usage: importPicardMetrics.py <prefix>")
    sys.exit(0)

pref = sys.argv[1]
dir_list = glob.glob(pref + '*')
dir_list.sort()
if len(dir_list) == 0:
    print("No files found with given prefix", pref)
    sys.exit(0)


def readMetricsFile(file):
    func = None
    metricsClass = None
    cvg = []
    with open(file) as f:
        for line in f:
            if line.startswith('## htsjdk.samtools.metrics.StringHeader'):
                header = next(f).strip().split(' ')
                func = header[1]
                # print("FUNC:",func)
                prms = [(h.split('=')[0], h.split('=')[1]) for h in header if len(h.split('=')) == 2]
                # print("PRMS:",len(prms))
                next(f)
            if line.startswith('## METRICS CLASS'):
                metricsClass = line.strip().split('\t')[1].split('.')[-1]
                # print("CLASS:",metricsClass)
                cat = next(f).strip().split('\t')
                val = next(f).strip().split('\t')
                metrics = [(c, v) for (c, v) in zip(cat, val)]
                # print("METRICS:",len(metrics))
            if line.startswith('## HISTOGRAM') and metricsClass and metricsClass.endswith('WgsMetrics'):
                cat = next(f).strip().split('\t')
                cvg = [[], []]
                while len(cvg[0]) <= 200:
                    row = next(f).strip().split('\t')
                    if len(row) < 2: break
                    cvg[0].append(int(row[0]))
                    cvg[1].append(int(row[1]))

    if func == None or metricsClass == None:
        return None, [], None, [], None
    return func, prms, metricsClass, metrics, cvg


prmsDf = pd.DataFrame()
metricsDf = pd.DataFrame()
h5outfile = pref + '.h5'

i = 0
j = 0
print('PREF:', pref)
for file in dir_list:
    fn = file[len(pref) + 1:]
    if fn.endswith('tsv') or fn.endswith('h5'):
        continue
    if fn.endswith('.txt'):
        fn = fn[:-4]
    print('FILE:', fn)
    func, prms, metricsClass, metrics, cvg = readMetricsFile(file)
    # print()

    l = len(prms)
    block = pd.DataFrame({'File': [fn] * l,
                          'Function': [func] * l,
                          'Parameter': [p[0] for p in prms],
                          'Value': [p[1] for p in prms],
                          }, index=range(i, i + l))
    i = i + l
    prmsDf = pd.concat([prmsDf, block])

    l = len(metrics)
    block = pd.DataFrame({'File': [fn] * l,
                          'Class': [metricsClass] * l,
                          'Parameter': [m[0] for m in metrics],
                          'Value': [m[1] for m in metrics],
                          }, index=range(j, j + l))
    j = j + l
    metricsDf = pd.concat([metricsDf, block])

    if cvg:
        cvgDf = pd.DataFrame({'Count': cvg[1]}, index=cvg[0])
        cvgDf.to_hdf(h5outfile, key=fn + "_cvg")

if j == 0:
    print("No Picard metrics found for given prefix", pref)
    sys.exit(0)

prmsDf.to_csv(pref + '.params.tsv', sep='\t')
prmsDf.to_hdf(h5outfile, key="params")
metricsDf.to_csv(pref + '.metrics.tsv', sep='\t')
metricsDf.to_hdf(h5outfile, key="metrics")
print('OUT:', h5outfile)
