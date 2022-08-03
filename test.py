import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector
import uproot
from scipy.special import comb
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import matplotlib.pyplot as plt
import hist
from processing.roccor import kScaleDT
from coffea import lookup_tools

t = NanoEventsFactory.from_root("nano_mc2017_105.root", schemaclass=NanoAODSchema).events()

muons = t.Muon

print(ak.count(muons))

from time import time

tacc=0
for i in range(1000):
  t0 = time()
  x = kScaleDT(muons.charge, muons.pt, muons.eta, muons.phi, 0, 0)
  tacc += time()-t0
  muons['pt'] = muons.pt*x
print("C extension took %0.3f seconds"%(tacc))

rochester_data = lookup_tools.txt_converters.convert_rochester_file(
    "corrections/roccor/RoccoR2017UL.txt", loaduncs=True)
rochester = lookup_tools.rochester_lookup.rochester_lookup(rochester_data)


tacc=0
for i in range(1000):
  t0 = time()
  x = rochester.kScaleDT(muons.charge, muons.pt, muons.eta, muons.phi)
  tacc += time()-t0
  muons['pt'] = muons.pt*x
print("Coffea lib took %0.3f seconds"%(tacc))
