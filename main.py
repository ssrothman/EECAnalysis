import json
import awkward as ak
import uproot
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from coffea.nanoevents.methods import vector
import eec
import fastjet

#read in json config
with open("config.json", 'r') as f:
  args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))
print("read in config")

readCut = "("
for trigger in args.triggers:
  readCut += " (" + trigger + "==1) |"
readCut = readCut[:-1] + ")"

readCut += " & " + args.presel

#print(readCut)

#read in uproot file
with uproot.open("copy.root") as f:
  data = f['Events'].arrays(cut=readCut, filter_name = args.branchFilter)

print("read in file")

#apply cuts to muons
mask = np.ones(len(data), dtype=bool)
for cut in args.muCuts:
  if cut.minval == cut.maxval:
    mask = np.logical_and(mask, ak.all(data[cut.var] == cut.minval, axis=1))
  else:
    if cut.minval != -999:
      mask = np.logical_and(mask, ak.all(data[cut.var] >= cut.minval, axis=1))
    if cut.maxval != -999:
      mask = np.logical_and(mask, ak.all(data[cut.var] <= cut.maxval, axis=1))

print("%d/%d (%0.2f%%) pass muon cuts"%(sum(mask), len(mask), 100*sum(mask)/len(mask)))
data = data[mask]

#z selection
muons = ak.to_regular(ak.zip({
    "pt" : data[args.muons.pt],
    'eta' : data[args.muons.eta],
    'phi' : data[args.muons.phi],
    'mass' : data[args.muons.mass]
  },
  behavior = vector.behavior,
  with_name='PtEtaPhiMLorentzVector'
))

mu1 = muons[:,0]
mu2 = muons[:,1]

mass = np.abs(mu1 + mu2)

massWindow = np.logical_and(mass<args.Zsel.maxMass, mass>args.Zsel.minMass)
charge = ak.prod(data[args.muons.charge],axis=1) == -1

Zsel = np.logical_and.reduce((massWindow, charge))

print("%d/%d (%0.2f%%) pass Zsel"%(sum(Zsel), len(Zsel), 100*sum(Zsel)/len(Zsel)))
data = data[Zsel]

#compute eecs
jets = ak.zip({
    "pt" : data[args.jets.pt],
    'eta' : data[args.jets.eta],
    'phi' : data[args.jets.phi],
    'mass' : data[args.jets.mass]
  },
  behavior = vector.behavior,
  with_name='PtEtaPhiMLorentzVector'
)

pfcands = ak.zip({
    "pt" : data[args.pfcands.pt],
    'eta' : data[args.pfcands.eta],
    'phi' : data[args.pfcands.phi],
    'mass' : data[args.pfcands.mass],
  },
  behavior = vector.behavior,
  with_name = 'PtEtaPhiMLorentzVector'
)

#jet clustering
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
cluster = fastjet.ClusterSequence(pfcands, jetdef)

parts = cluster.constituents(args.minJetPt)

#eec
eec_ls = eec.EECLongestSide(2, 10, axis_range=(1e-5,1.0))

#need to cast particle arrays into numpy format recognized by eec library
