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

'''

t0 = time()
#eec, pkomiske
eec_ls = eec.EECLongestSide(args.eec.N, args.eec.nBins, axis_range=(args.eec.axisMin,args.eec.axisMax))

flatParts= ak.flatten(parts, axis=1) #remove event axis
#stack (pt, eta, phi, charge)
flatParts = ak.concatenate( (flatParts.pt[:,:,None], 
                             flatParts.eta[:,:,None], 
                             flatParts.phi[:,:,None], 0), axis=2) 
print("setting up eec inputs took %0.3f seconds"%(time()-t0))
t9 = time()
eec_ls(flatParts)
print("running eec_ls took %0.3f seconds"%(time()-t0))
midbins = eec_ls.bin_centers()
binedges = eec_ls.bin_edges()
binwidths = (binedges[1:]) - (binedges[:-1])
hist, errs = eec_ls.get_hist_errs(0, False)

#eec, ssrothman
from EEC import ProjectedEEC
myEEC = ProjectedEEC(args.eec.N, args.eec.nBins, args.eec.axisMin, args.eec.axisMax)
myEEC(parts, jet4vec)
myMidbins = myEEC.hist.axes[0].centers
myBinwidths = myEEC.hist.axes[0].widths
myHist = myEEC.hist.values()

#eec_cpp, srothman
import eec_cpp
t0 = time()
pts = np.asarray(ak.flatten(parts.pt/jet4vec.pt, axis=None)).astype(np.float32)
etas = np.asarray(ak.flatten(parts.eta, axis=None)).astype(np.float32)
phis = np.asarray(ak.flatten(parts.phi, axis=None)).astype(np.float32)
nJets = np.asarray(ak.flatten(ak.num(parts,axis=-1), axis=None)).astype(np.int32)
jetIdxs = np.cumsum(nJets).astype(np.int32)
nDRs = comb(nJets, 2)
dRIdxs = np.cumsum(nDRs).astype(np.int32)
nDR = int(dRIdxs[-1])
print("seting up inputs for eec_cpp took %0.3f seconds"%(time()-t0))
t0 = time()
jets = np.column_stack((pts, etas, phis))
dRs, wts = eec_cpp.eec(jets, jetIdxs, args.eec.N, nDR, nDR, dRIdxs)
dRs = np.sqrt(dRs)
print("computing dRs with eec_cpp took %0.3f seconds"%(time()-t0))

import boost_histogram as bh
cpphist = bh.Histogram(bh.axis.Regular(args.eec.nBins, start=args.eec.axisMin, stop=args.eec.axisMax, transform=bh.axis.transform.log))
t0 = time()
cpphist.fill(dRs, weight=wts)
print("filling histogram with cpp values took %0.3f seconds"%(time()-t0))
cppMidbins = cpphist.axes[0].centers
cppBinwidths = cpphist.axes[0].widths
cppHist = cpphist.values()

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\Delta R$')
#plt.ylim(1e-1,1e5)
plt.ylabel("Projected %d-point correlator"%args.eec.N)
plt.errorbar(midbins, hist/binwidths, fmt='o',lw=1.5,markersize=5, label='pkomiske')
plt.errorbar(myMidbins*1.1, myHist/myBinwidths, fmt='o',lw=1.5,markersize=5, label='ssrothman')
plt.errorbar(cppMidbins*1.2, cppHist/cppBinwidths, fmt='o',lw=1.5,markersize=5, label='ssrothman, cpp')
plt.legend()
plt.axvline(args.jetSize, c='k')
plt.savefig(args.plotFile, format='png')
plt.show()

with uproot.recreate('out.root') as f:
  f['eec_hist'] = hist,binedges
  f['eec_errs'] = errs,binedges
'''
