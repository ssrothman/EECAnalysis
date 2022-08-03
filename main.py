import json
import pickle
from time import time
from types import SimpleNamespace

from numpy import True_

from processing.EECProcessor import EECProcessor
from processing.LumiProcessor import LumiProcessor

from datetime import date

#read in json config
configType = "PUPPI"
filesetType = "test"
remote = False
kind='all'

configName = "config_%s.json"%configType
fileSet = "fileset_%s.json"%filesetType

configName = "configs/%s"%configName
fileSet = "filesets/%s"%fileSet

outPrefix = "%s.%s.%s.%s"%(filesetType, date.today().strftime("%m-%d-%Y"), kind, configType)
print(outPrefix)

with open(configName, 'r') as f:
  args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))
print("read in config")

from coffea import processor
from coffea.nanoevents import NanoAODSchema

#from fileset import fileset

if remote:
  import time

  from distributed import Client
  from lpcjobqueue import LPCCondorCluster

  tic = time.time()

  cluster = LPCCondorCluster(ship_env=False,
                            transfer_input_files=['processing', 'corrections'],
                            memory='6GB',
                            shared_temp_directory="/tmp")
  cluster.adapt(minimum=1, maximum=100)
  client = Client(cluster)

  exe_args = {
    "client" : client,
    "savemetrics" : True,
    "schema" : NanoAODSchema,
    "align_clusters" : True
  }

  proc = EECProcessor(args, kind)
  
  print("Waiting for at least one worker...")
  client.wait_for_workers(1)
  hists, metrics = processor.run_uproot_job(
    fileSet,
    treename="Events",
    processor_instance=proc,
    executor = processor.dask_executor,
    executor_args=exe_args,
    chunksize=10000
    #maxchunks=10,
  )

  elapsed = time.time() - tic
  print(f"Output: {hists}")
  print(f"Metrics: {metrics}")
  print(f"Finished in {elapsed:.1f}s")
  print(f"Events/s: {metrics['entries'] / elapsed:.0f}")
  
  with open("output/%s.hist.pickle"%outPrefix, 'wb') as f:
    pickle.dump(hists, f)
  with open("output/%s.metrics.pickle"%outPrefix, 'wb') as f:
    pickle.dump(metrics, f)

else:
  runner = processor.Runner(
    executor = processor.IterativeExecutor(compression=None, workers=6),
    schema=NanoAODSchema,
    #maxchunks=4,
    chunksize = 10000000
  )

  #hp = h.heap()
  #print("top of everything")
  #print(hp.byrcs)
  #print()

  out = runner(
    fileSet,
    treename='Events',
    processor_instance=EECProcessor(args, kind),
  )

  print(args.histname)
  with open("output/%s.hist.pickle"%outPrefix, 'wb') as f:
    pickle.dump(out, f)

  print(out)
