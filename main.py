import json
import awkward as ak
import uproot
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from coffea.nanoevents.methods import vector

#read in json config
with open("config.json", 'r') as f:
  args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))

#read in uproot file
with uproot.open("copy.root") as f:
  data = f['Events'].arrays(cut=args.readCut, filter_name = args.branchFilter)

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
ID = ak.all(data[args.muons.ID], axis=1)
charge = ak.prod(data[args.muons.charge],axis=1) == -1

Zsel = np.logical_and.reduce((massWindow, ID, charge))

data = data[Zsel]

#jet plots
jetPt = data[args.jets.pt]
nJets = ak.num(jetPt)
nConstituents = data[args.jets.nConstituents]

plt.hist(ak.flatten(jetPt), bins=100, density=True, histtype='step')
plt.title("Jet pT")
plt.xlabel("Jet pT [GeV]")
plt.ylabel("Density")
plt.show()

nJetRange = np.max(nJets) - np.min(nJets)
plt.hist(nJets, bins=nJetRange, density=True, histtype='step')
plt.title("nJets")
plt.xlabel("nJets")
plt.ylabel("Density")
plt.show()

nConstituentRange = np.max(nConstituents) - np.min(nConstituents)
plt.hist(ak.flatten(nConstituents), bins=nConstituentRange, density=True, histtype='step')
plt.title("nConstituents")
plt.xlabel("nConstituents")
plt.ylabel("Density")
plt.show()
