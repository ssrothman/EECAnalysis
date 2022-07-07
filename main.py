import json
import uproot
import numpy as np
from types import SimpleNamespace
from time import time
from scipy.special import comb
import pickle

from processing.EECProcessor import EECProcessor
from processing.LumiProcessor import LumiProcessor
from coffea import nanoevents

#read in json config
with open("config.json", 'r') as f:
  args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))
print("read in config")

from coffea.nanoevents import NanoEventsFactory, BaseSchema

from coffea import processor

#from fileset import fileset

remote = False
if remote:
  from distributed import Client
  from lpcjobqueue import LPCCondorCluster
  import time
  tic = time.time()

  cluster = LPCCondorCluster(ship_env=True,
                            transfer_input_files=['processing', 'corrections'],
                            memory='12GB')
  cluster.adapt(minimum=1, maximum=100)
  client = Client(cluster)

  exe_args = {
    "client" : client,
    "savemetrics" : True,
    "schema" : nanoevents.BaseSchema,
    "align_clusters" : True
  }

  proc = EECProcessor(list(['DoubleMuon','DYJetsToLL']))
  
  print("Waiting for at least one worker...")
  client.wait_for_workers(1)
  hists, metrics = processor.run_uproot_job(
    "fileset_full.json",

    treename="Events",
    processor_instance=proc,
    executor = processor.dask_executor,
    executor_args=exe_args,
    #maxchunks=10,
  )

  elapsed = time.time() - tic
  print(f"Output: {hists}")
  print(f"Metrics: {metrics}")
  print(f"Finished in {elapsed:.1f}s")
  print(f"Events/s: {metrics['entries'] / elapsed:.0f}")
  with open("output/hists.pickle", 'wb') as f:
    pickle.dump(hists, f)
  with open("output/metrics.pickle", 'wb') as f:
    pickle.dump(metrics, f)
else:
  runner = processor.Runner(
    executor = processor.IterativeExecutor(compression=None, workers=4),
    schema=BaseSchema,
    #maxchunks=4,
    chunksize = 10000000
  )

  out = runner(
    "fileset.json",
    treename='Events',
    processor_instance=EECProcessor(list(['DoubleMuon','DYJetsToLL'])),
  )

  with open(args.outname, 'wb') as f:
    pickle.dump(out, f)

  print(out)