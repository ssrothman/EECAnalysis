import json
import pickle
from time import time
from types import SimpleNamespace

from numpy import True_

from processing.EECProcessor import EECProcessor
from processing.LumiProcessor import LumiProcessor

from datetime import date

'''
Wants:
Single muon dataset
E^3 E correlators and such
Maybe better EEC/jet linking?

'''

#read in json config
configType = "Puppi"
filesetType = "small_v3_EECs_3PU"
remote = False

configName = "config_%s.json"%configType
fileSet = "fileset_%s.json"%filesetType

configName = "configs/%s"%configName
fileSet = "filesets/%s"%fileSet

outname = "%s_%s_%s/"%(filesetType, date.today().strftime("%m-%d-%Y"), configType)
print(outname)

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

  cluster = LPCCondorCluster(ship_env=True,
                            transfer_input_files=['processing', 'corrections'],
                            memory='4GB',
                            shared_temp_directory="/tmp")
  cluster.adapt(minimum=1, maximum=200)
  client = Client(cluster)

  exe_args = {
    "client" : client,
    "savemetrics" : True,
    "schema" : NanoAODSchema,
    "align_clusters" : True,
    "use_dataframes" : False
  }

  proc = EECProcessor(args, outname)
  
  print("Waiting for at least one worker...")
  client.wait_for_workers(1)
  hists, metrics = processor.run_uproot_job(
    fileSet,
    treename="Events",
    processor_instance=proc,
    executor = processor.dask_executor,
    executor_args=exe_args,
    chunksize=50000
    #maxchunks=10,
  )

  elapsed = time.time() - tic
  print(f"Output: {hists}")
  print(f"Metrics: {metrics}")
  print(f"Finished in {elapsed:.1f}s")
  print(f"Events/s: {metrics['entries'] / elapsed:.0f}")
  
  with open("output/%s.hist.pickle"%outname[:-1], 'wb') as f:
    pickle.dump(hists, f)
  with open("output/%s.metrics.pickle"%outname[:-1], 'wb') as f:
    pickle.dump(metrics, f)

else:
  runner = processor.Runner(
    executor = processor.FuturesExecutor(workers=5),
    schema=NanoAODSchema,
    #maxchunks=4,
    chunksize = 500000000
  )

  #hp = h.heap()
  #print("top of everything")
  #print(hp.byrcs)
  #print()

  hists = runner(
    fileSet,
    treename='Events',
    processor_instance=EECProcessor(args, outname),
  )

  print(hists)

  print(args.histname)
  #with open("output/%s.hist.pickle"%outname, 'wb') as f:
  #  pickle.dump(out, f)

import hist 
from datetime import datetime

dRaxis = hist.axis.Regular(args.dRHist.nBins, 
                            args.dRHist.min, 
                            args.dRHist.max, 
                            transform=getattr(hist.axis.transform, args.dRHist.transform))


with open("summary.txt", 'w') as f:
  f.write("EEC Processor run at ")
  f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
  f.write('\n\n')

  f.write("Total Weights:\n")
  for dataset in hists.keys():
    f.write("%s: %0.2f\n"%(dataset, hists[dataset]))
  f.write('\n')

  f.write("dR bin edges: \n")
  for edge in dRaxis.edges:
    f.write("%0.5e\n"%edge)

xrdpath = "root://cmseos.fnal.gov//store/user/srothman/%s"%outname
import subprocess
subprocess.run(['xrdcopy','-f','summary.txt',xrdpath]) 