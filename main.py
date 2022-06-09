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

print(args.file)

from EECProcessor import EECProcessor
from coffea.nanoevents import NanoEventsFactory, BaseSchema

file = uproot.open(args.file)
events = NanoEventsFactory.from_root(
    file,
    entry_stop=10000,
    metadata={"dataset": "DoubleMuon"},
    schemaclass=BaseSchema,
).events()
p = EECProcessor()
out = p.process(events)

print(out)

print(out['EEC'].hist)