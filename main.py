import json
import pickle
from time import time
from types import SimpleNamespace

import numpy as np
import uproot
from coffea import nanoevents
from scipy.special import comb

from processing.EECProcessor import EECProcessor
from processing.LumiProcessor import LumiProcessor

#read in json config
remote = True
configName = "config_PUPPI.json"
fileSet = "fileset_full.json"

configName = "configs/%s"%configName
fileSet = "filesets/%s"%fileSet

with open(configName, 'r') as f:
  args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))
print("read in config")

from coffea import processor
from coffea.nanoevents import BaseSchema, NanoEventsFactory, NanoAODSchema

#from fileset import fileset

if remote:
  import time

  from distributed import Client
  from lpcjobqueue import LPCCondorCluster
  tic = time.time()

  cluster = LPCCondorCluster(ship_env=True,
                            transfer_input_files=['processing', 'corrections'],
                            memory='12GB')
  cluster.adapt(minimum=5, maximum=100)
  client = Client(cluster)

  exe_args = {
    "client" : client,
    "savemetrics" : True,
    "schema" : NanoAODSchema,
    "align_clusters" : True
  }

  proc = EECProcessor(args)
  
  print("Waiting for at least one worker...")
  client.wait_for_workers(1)
  hists, metrics = processor.run_uproot_job(
    fileSet,
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
  with open(args.histname, 'wb') as f:
    pickle.dump(hists, f)
  with open(args.metricname, 'wb') as f:
    pickle.dump(metrics, f)
else:
  runner = processor.Runner(
    executor = processor.IterativeExecutor(compression=None, workers=4),
    schema=NanoAODSchema,
    #maxchunks=4,
    chunksize = 10000000
  )

  out = runner(
    fileSet,
    treename='Events',
    processor_instance=EECProcessor(args),
  )

  print(args.histname)
  with open(args.histname, 'wb') as f:
    pickle.dump(out, f)

  print(out)
