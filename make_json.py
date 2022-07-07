import uproot
import numpy as np
from tqdm import tqdm

result = {}

import json
with open("fileset.json", 'r') as f:
    fileset = json.load(f)

for file in tqdm(fileset['DoubleMuon']):
    with uproot.open(file+":LuminosityBlocks") as f:
        runs = f['run'].array()
        lumis = f['luminosityBlock'].array()
        uniqueRuns, counts = np.unique(runs, return_counts=True)
    for run, count in zip(uniqueRuns, counts):
        run = int(run)
        if run not in result:
            result[run] = []
        
        mask = runs==run
        maskedLumis = lumis[mask]

        firstLumi=-1
        lastLumi=-1
        for lumi in lumis[mask]:
            if lastLumi==-1:
                lastLumi=lumi
                firstLumi=lumi
            elif lastLumi != lumi-1:
                result[run].append([firstLumi,lumi])
                firstLumi = lumi
                lastLumi = lumi
            else:
                lastLumi = lumi
        result[run].append([firstLumi, lumi])

#print(result)

import json

with open("lumi.json", 'w') as f:
    json.dump(result, f)