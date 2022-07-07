lookup = "root://cmsxrootd.fnal.gov/"
foldersDY = ["/store/group/lpcpfnano/srothman/v2_simon/2017/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_take3/220624_074716/0000",
             "/store/group/lpcpfnano/srothman/v2_simon/2017/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_take3/220624_074716/0001"]
filesDY = []

foldersDMU = ['/store/group/lpcpfnano/srothman/v2_simon/2017/DoubleMuon/DoubleMuon/DoubleMuon_Run2017C_take2/220624_074202/0000']
filesDMU = []

import subprocess
filesDY = []
filesDMU = []

#total sumw DYJetsToLL = 7171476.0

for folder in foldersDY:
  with subprocess.Popen("eos root://cmseos.fnal.gov ls %s"%folder, stdout=subprocess.PIPE, shell=True) as proc:
    ans = proc.communicate()
    for line in ans[0].splitlines():
      if line.decode().endswith(".root"):
        filesDY.append("%s/%s/%s"%(lookup, folder, line.decode()))
  
for folder in foldersDMU:
  with subprocess.Popen("eos root://cmseos.fnal.gov ls %s"%folder, stdout=subprocess.PIPE, shell=True) as proc:
    ans = proc.communicate()
    for line in ans[0].splitlines():
      if line.decode().endswith(".root"):
        filesDMU.append("%s/%s/%s"%(lookup, folder, line.decode()))


fileset = {
  'DYJetsToLL' : filesDY,
  'DoubleMuon' : filesDMU
}

print(fileset)

import json
with open("fileset_full.json", 'w') as f:
  json.dump(fileset, f, indent='\t')