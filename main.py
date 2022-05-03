import json
import awkward as ak
import uproot
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from coffea.nanoevents.methods import vector, candidate
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

print(args.file)

#read in uproot file
#with uproot.open("copy.root") as f:
with uproot.open(args.file+":Events") as f:
  data = f.arrays(args.branches, cut=readCut, entry_stop=args.entry_stop)

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

pfcands = ak.zip({
    "pt" : data[args.pfcands.pt],
    'eta' : data[args.pfcands.eta],
    'phi' : data[args.pfcands.phi],
    'mass' : data[args.pfcands.mass],
  },
  behavior = candidate.behavior,
  with_name = 'PtEtaPhiMLorentzVector'
)

charge= data[args.pfcands.charge]

#jet clustering
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, args.jetSize)
cluster = fastjet.ClusterSequence(pfcands, jetdef)

parts = cluster.constituents(args.minJetPt)

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

#eec
eec_ls = eec.EECLongestSide(args.eec.N, args.eec.nBins, axis_range=(args.eec.axisMin,args.eec.axisMax))

flatParts= ak.flatten(parts, axis=1) #remove event axis
#stack (pt, eta, phi, charge)
flatParts = ak.concatenate( (flatParts.pt[:,:,None], 
                             flatParts.eta[:,:,None], 
                             flatParts.phi[:,:,None], 0), axis=2) 

eec_ls(flatParts)
midbins = eec_ls.bin_centers()
binedges = eec_ls.bin_edges()
binwidths = np.log(binedges[1:]) - np.log(binedges[:-1])
hist, errs = eec_ls.get_hist_errs(0, False)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\Delta R$')
plt.ylabel("Projected %d-point correlator"%args.eec.N)
plt.errorbar(midbins, hist/binwidths, 
               xerr=(midbins - binedges[:-1], binedges[1:] - midbins),
               yerr=errs/binwidths, 
               fmt='o',lw=1.5,capsize=1.5,capthick=1,markersize=1.5)
plt.axvline(args.jetSize, c='k')
plt.savefig("proof.png", format='png')
plt.show()

with uproot.recreate('out.root') as f:
  f['eec_hist'] = hist,binedges
  f['eec_errs'] = errs,binedges

