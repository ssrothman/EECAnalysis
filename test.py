import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector
import uproot
from scipy.special import comb
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import matplotlib.pyplot as plt
import hist

print("about to read")
#t = uproot.open("nano_mc2017_105.root:Events")
t = NanoEventsFactory.from_root("nano_mc2017_105.root", schemaclass=NanoAODSchema).events()
print("read")

parts = ak.zip({
    "pt" : t.PFCands_pt * t.PFCands_puppiWeight,
    "eta" : t.PFCands_eta,
    "phi" : t.PFCands_phi,
    "mass" : t.PFCands_mass
  },
  behavior = vector.behavior,
  with_name = 'PtEtaPhiMLorentzVector'
)
partsIdx = ak.local_index(parts)
print("made parts")

jets = ak.zip({
    "pt" : t.selectedPatJetsAK4PFPuppi_pt,
    "eta" : t.selectedPatJetsAK4PFPuppi_eta,
    "phi" : t.selectedPatJetsAK4PFPuppi_phi,
    "M" : t.selectedPatJetsAK4PFPuppi_mass,
  },
  behavior = vector.behavior,
  with_name = 'PtEtaPhiMLorentzVector'
)
print("made jets")

jetIdx = t.selectedPatJetsAK4PFPuppiPFCands_jetIdx
PFCIdx = t.selectedPatJetsAK4PFPuppiPFCands_pFCandsIdx

import fastjet
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
cluster = fastjet.ClusterSequence(parts, jetdef)
jetParts = cluster.constituents(30)

import eec
eec_ls = eec.EECLongestSide(2, 20, axis_range=(1e-5,1))

flatParts= ak.flatten(jetParts, axis=1) #remove event axis
#stack (pt, eta, phi, charge)
flatParts = ak.concatenate( (flatParts.pt[:,:,None], 
                             flatParts.eta[:,:,None], 
                             flatParts.phi[:,:,None], 0), axis=2) 

eec_ls(flatParts)
midbins = eec_ls.bin_centers()
binedges = eec_ls.bin_edges()
binwidths = (binedges[1:]) - (binedges[:-1])
histo, errs = eec_ls.get_hist_errs(0, False)

myEEC = hist.Hist(hist.axis.Regular(20, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log))
myEEC.fill(ak.flatten(t.EEC2_dRs), weight=ak.flatten(t.EEC2_wts))