eta = [ [ [0, 0.1, 0.4, 1.0, 0.4], [0, 0, 0]], [[0, 0, 0,]]]
phi = [ [[0, 0.2, 0.4, 0.0, -0.5], [1, 0.2, 3]], [[0,0,0]]]
pt = [ [[1, 2, 0.5, 2, 3], [1, 5, 2]], [[1,0,0]]]
M = [[[0, 0, 0, 0, 1], [0,0,0]], [[0,0,0]]]

import awkward as ak
import numpy as np
from EEC import ProjectedEEC
from coffea.nanoevents.methods import vector
from scipy.special import comb

parts = ak.zip({
    "pt" : pt,
    "eta" : eta,
    "phi" : phi,
    "mass" : M
  },
  behavior = vector.behavior,
  with_name = 'PtEtaPhiMLorentzVector'
)

ak.type(parts)

jet4vec = ak.zip({
    'x' : ak.sum(parts.x, axis=-1),
    'y' : ak.sum(parts.y, axis=-1),
    'z' : ak.sum(parts.z, axis=-1),
    't' : ak.sum(parts.t, axis=-1)
  },
  behavior = vector.behavior,
  with_name = 'LorentzVector'
)
print("jetpT",jet4vec.pt)

myEEC = ProjectedEEC(4, 10, 1e-5, 1)
myEEC(parts, jet4vec)
print(myEEC.hist)

print()
print("#########################################################################")
print("###########################CLIB##########################################")
print("#########################################################################")
print()

import eec_cpp
pts = np.asarray(ak.flatten(parts.pt, axis=None)).astype(np.float32)
etas = np.asarray(ak.flatten(parts.eta, axis=None)).astype(np.float32)
phis = np.asarray(ak.flatten(parts.phi, axis=None)).astype(np.float32)
nJets = np.asarray(ak.flatten(ak.num(parts,axis=-1), axis=None)).astype(np.int32)
jetIdxs = np.cumsum(nJets).astype(np.int32)
nDRs = comb(nJets, 2)
dRIdxs = np.cumsum(nDRs).astype(np.int32)
nDR = int(dRIdxs[-1])
jets = np.column_stack((pts, etas, phis))
dRs, wts = eec_cpp.eec(jets, jetIdxs, 3, nDR, nDR, dRIdxs)
dRs = np.sqrt(dRs)
