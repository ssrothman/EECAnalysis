import json
import uproot
import numpy as np
from types import SimpleNamespace
from time import time
from scipy.special import comb
import pickle

from EECProcessor import EECProcessor

#read in json config
with open("config.json", 'r') as f:
  args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))
print("read in config")

print(args.file)

from EECProcessor import EECProcessor
from coffea.nanoevents import NanoEventsFactory, BaseSchema

fileset = {
  'DYJetsToLL' : [
    'root://cmsxrootd.fnal.gov//store/user/srothman/v2_simon/2017/DYJetsToLL/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/DYJetsToLL_M-50_test/220623_072649/0000/nano_mc2017_9.root'
  ]
}


#file = uproot.open(args.file)
#events = NanoEventsFactory.from_root(
#    file,
#    #entry_stop=5000,
#    metadata={"dataset": "DoubleMuon"},
#    schemaclass=BaseSchema,
#).events()
#p = EECProcessor()
#out = p.process(events)

from coffea import processor
out = processor.run_uproot_job(
  fileset,
  treename='Events',
  processor_instance = EECProcessor(),
  executor=processor.futures_executor,
  executor_args={
    'schema' : BaseSchema
  },
  chunksize=100000
)

with open(args.outname, 'wb') as f:
  pickle.dump(out, f)

print(out)