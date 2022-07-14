lookup = "root://cmsxrootd.fnal.gov/"
foldersDY = ["/store/group/lpcpfnano/srothman/v2_simon/2017/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_take3/220624_074716/0000",
             "/store/group/lpcpfnano/srothman/v2_simon/2017/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_take3/220624_074716/0001"]
filesDY = []

foldersDMU = ['/store/group/lpcpfnano/srothman/v2_simon/2017/DoubleMuon/DoubleMuon/DoubleMuon_Run2017C_take2/220624_074202/0000']
filesDMU = []

foldersCHS = ['/store/group/lpcpfnano/srothman/v2_simon/2017/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_CHS/220709_150604/0000']
filesCHS = []

foldersCS = ['/store/group/lpcpfnano/srothman/v2_simon/2017/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_CS/220709_150640/0000']
filesCS = []

import subprocess

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

for folder in foldersCHS:
  with subprocess.Popen("eos root://cmseos.fnal.gov ls %s"%folder, stdout=subprocess.PIPE, shell=True) as proc:
    ans = proc.communicate()
    for line in ans[0].splitlines():
      if line.decode().endswith(".root"):
        filesCHS.append("%s/%s/%s"%(lookup, folder, line.decode()))

for folder in foldersCS:
  with subprocess.Popen("eos root://cmseos.fnal.gov ls %s"%folder, stdout=subprocess.PIPE, shell=True) as proc:
    ans = proc.communicate()
    for line in ans[0].splitlines():
      if line.decode().endswith(".root"):
        filesCS.append("%s/%s/%s"%(lookup, folder, line.decode()))


fileset_full = {
  'DYJetsToLL' : filesDY,
  'DoubleMuon' : filesDMU
}

import json
with open("fileset_full.json", 'w') as f:
  json.dump(fileset_full, f, indent='\t')

fileset_CHS = {
  'DYJetsToLL' : filesCHS
}

with open("fileset_CHS.json", 'w') as f:
  json.dump(fileset_CHS, f, indent='\t')

fileset_CS = {
  'DYJetsToLL' : filesCS
}

with open("fileset_CS.json", 'w') as f:
  json.dump(fileset_CS, f, indent='\t')

fileset_PUPPI = {
  'DYJetsToLL' : filesDY[:100]
}

with open("fileset_PUPPI.json", 'w') as f:
  json.dump(fileset_PUPPI, f, indent='\t')