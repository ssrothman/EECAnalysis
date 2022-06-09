import json
import numpy as np
import awkward as ak
import uproot
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from coffea.nanoevents.methods import vector, candidate
from time import time
import fastjet
from scipy.special import comb
import coffea

from EECProcessor import EECProcessor

#read in json config
with open("config.json", 'r') as f:
  args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))
print("read in config")

readCut = "("
for trigger in args.triggers:
  readCut += " (" + trigger + "==1) |"
readCut = readCut[:-1] + ")"

readCut += " & " + args.presel

print(args.file)

#read in uproot file
#with uproot.open("copy.root") as f:
#with uproot.open(args.file+":Events") as f:
  #data = f.arrays(args.branches, cut=readCut, entry_stop=args.entry_stop)

#print("read in file")

from EECProcessor import EECProcessor
from coffea.nanoevents import NanoEventsFactory, BaseSchema

file = uproot.open(args.file)
data = NanoEventsFactory.from_root(
    file,
    entry_stop=10000,
    metadata={"dataset": "DoubleMuon"},
    schemaclass=BaseSchema,
).events()

#apply cuts...
mask = np.ones(len(data), dtype=bool)

for cut in  args.evtCuts:
    if cut.minval == cut.maxval:
        mask = np.logical_and(mask, data[cut.var] == cut.minval)
    else:
        if cut.minval != -999:
            mask = np.logical_and(mask, data[cut.var] >= cut.minval)
        if cut.maxval != -999:
            mask = np.logical_and(mask, data[cut.var] <= cut.maxval)


for cut in  args.muCuts:
    if cut.minval == cut.maxval:
        mask = np.logical_and(mask, ak.all(data[cut.var] == cut.minval, axis=1))
    else:
        if cut.minval != -999:
            mask = np.logical_and(mask, ak.all(data[cut.var] >= cut.minval, axis=1))
        if cut.maxval != -999:
            mask = np.logical_and(mask, ak.all(data[cut.var] <= cut.maxval, axis=1))

print("%d/%d (%0.2f%%) pass initial cuts"%(sum(mask), len(mask), 100*sum(mask)/len(mask)))
data = data[mask]

#z selection
muons = ak.to_regular(ak.zip({
        "pt" : data[ args.muons.pt],
        'eta' : data[ args.muons.eta],
        'phi' : data[ args.muons.phi],
        'mass' : data[ args.muons.mass]
    },
    behavior = vector.behavior,
    with_name='PtEtaPhiMLorentzVector'
))

mu1 = muons[:,0]
mu2 = muons[:,1]

mass = np.abs(mu1 + mu2)

massWindow = np.logical_and(mass< args.Zsel.maxMass, mass> args.Zsel.minMass)
charge = ak.prod(data[ args.muons.charge],axis=1) == -1

Zsel = np.logical_and.reduce((massWindow, charge))

print("%d/%d (%0.2f%%) pass Z selection"%(sum(Zsel), len(Zsel), 100*sum(Zsel)/len(Zsel)))
data = data[Zsel]

pfcands = ak.zip({
        "pt" : data[ args.pfcands.pt] * data[args.pfcands.puppiWeight],
        'eta' : data[ args.pfcands.eta],
        'phi' : data[ args.pfcands.phi],
        'mass' : data[ args.pfcands.mass],
    },
    behavior = candidate.behavior,
    with_name = 'PtEtaPhiMLorentzVector'
)

pfcands = pfcands[pfcands.pt>0]

pfcandmask = pfcands.pt >  args.minPfCandPt
pfcands = pfcands[pfcandmask]

charge= data[ args.pfcands.charge]

#jet clustering
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm,  args.jetSize)
cluster = fastjet.ClusterSequence(pfcands, jetdef)

parts = cluster.constituents( args.minJetPt)

#need to have the 4vector summation because ak.sum() doesn't work right for 4vecs
jet4vec = ak.zip({
        'x' : ak.sum(parts.x, axis=-1),
        'y' : ak.sum(parts.y, axis=-1),
        'z' : ak.sum(parts.z, axis=-1),
        't' : ak.sum(parts.t, axis=-1)
    },
    behavior = vector.behavior,
    with_name = 'LorentzVector'
)

output["sumw"][dataset] += len(data)
output["mass"].fill(
    dataset=dataset,
    mass=mass[Zsel],
    nMyJets=ak.num(jet4vec),
    nCmsJets=ak.num(data['Jet_pt'])
)
output['myJets'].fill(
    dataset=dataset,
    pt=ak.flatten(jet4vec.pt, axis=None),
    eta=ak.flatten(jet4vec.eta, axis=None),
    phi=ak.flatten(jet4vec.phi, axis=None),
    npart=ak.flatten(ak.num(parts, axis=2), axis=None)
)

output['cmsswJets'].fill(
    dataset=dataset,
    pt=ak.flatten(data['Jet_pt'],axis=None),
    eta=ak.flatten(data['Jet_eta'], axis=None),
    phi=ak.flatten(data['Jet_phi'], axis=None),
    npart=ak.flatten(data["Jet_nConstituents"], axis=None)
)

print(ak.type(jet4vec))
print(ak.type(parts))
print(ak.count(jet4vec))
print(ak.count(data['Jet_pt']))
