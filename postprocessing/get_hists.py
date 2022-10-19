import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import hist
import pyarrow.dataset as ds
from tqdm import tqdm
import pickle
import awkward as ak
import os
import fsspec_xrootd

#TODO: handle missing values from matching
#TODO: read everything in

fs = fsspec_xrootd.XRootDFileSystem(hostid="cmseos.fnal.gov")

def check_exists(writepath, overwrite):
    if fs.exists(writepath):
        print("\tTarget file already exists at:", writepath)
        if overwrite:
            print("\t\tOverwriting...")
            return None
        else:
            print("\tUsing pre-existing histogram...")
            with fsspec_xrootd.XRootDFile(fs=fs, path=writepath, mode='rb') as f:
                return pickle.load(f)

def read_summary(folder):
    '''
    Read dR bin edges from summary.txt
    '''
    edges = []
    sumws = {}

    basepath = '/store/user/srothman/%s'%folder

    with fsspec_xrootd.XRootDFile(fs=fs, path="%s/summary.txt"%basepath, mode='rb') as f:
        startedDRs = False
        startedWTs = False
        for line in f.readlines():
            text = line.decode().strip()

            if text == '':
                startedWTs = False 
                startedDRs = False

            if startedWTs:
                splitted = text.split()
                sumws[splitted[0][:-1]] = float(splitted[1])
            elif 'Total Weights' in text:
                startedWTs = True

            if startedDRs:
                edges.append(float(text))
            elif 'dR bin edges' in text:
                startedDRs = True

    return edges, sumws

def get_muons(folder, overwrite=False):
    basepath = '/store/user/srothman/%s'%folder
    writepath = "%s/%shist.pickle"%(basepath, "muon")

    check = check_exists(writepath, overwrite)
    if check is not None:
        return check

    dataPath = "%s/SingleMuon/muon/parquet"%basepath
    mcPath = "%s/DYJetsToLL/muon/parquet"%basepath

    columns = ['pT', 'eta', 'weight', 'phi']

    dataSET = ds.dataset(dataPath, format='parquet', filesystem=fs)
    mcSET = ds.dataset(mcPath, format='parquet', filesystem=fs)

    HIST = hist.Hist(
        hist.axis.Regular(50, 20, 100, 
                            name='pT', label='$p_{T,\mu}$'),
        hist.axis.Regular(50, 0, 3, 
                            name='eta', label='|$\eta_{\mu}$|'),
         hist.axis.Regular(50, -np.pi, np.pi, 
                            name='phi', label='$\phi_{\mu}$'),                   
        hist.axis.StrCategory(['MC', 'Data'], name='label', label='label'),
        storage=hist.storage.Weight()
    )

    print("\tFilling hist from data...")
    for batch in tqdm(dataSET.to_batches(columns = columns)):
        df = batch.to_pandas()
        
        HIST.fill(
            pT = df.pT,
            eta = np.abs(df.eta),
            phi = df.phi,
            weight = df['weight'],
            label='Data'
        )
    
    print("\tFilling hist from mc...")
    for batch in tqdm(mcSET.to_batches(columns = columns)):
        df = batch.to_pandas()
        
        HIST.fill(
            pT = df.pT,
            eta = np.abs(df.eta),
            phi = df.phi,
            weight = df['weight'],
            label='MC'
        )
    
    with fsspec_xrootd.XRootDFile(fs=fs, path=writepath, mode='wb') as f:
        pickle.dump(HIST, f)

    print("\tDone.")
    print()

    return HIST

def get_Zs(folder, overwrite=False):
    basepath = '/store/user/srothman/%s'%folder
    writepath = "%s/%shist.pickle"%(basepath, "Z")

    check = check_exists(writepath, overwrite)
    if check is not None:
        return check

    dataPath = "%s/SingleMuon/dimuon/parquet"%basepath
    mcPath = "%s/DYJetsToLL/dimuon/parquet"%basepath

    columns = ['pT', 'y', 'phi', 'mass', 'weight']

    dataSET = ds.dataset(dataPath, format='parquet', filesystem=fs)
    mcSET = ds.dataset(mcPath, format='parquet', filesystem=fs)

    HIST = hist.Hist(
        hist.axis.Regular(50, 0, 100, 
                            name='pT', label='$p_{T,Z}$'),
        hist.axis.Regular(50, 0, 3, 
                            name='y', label='|$y_{Z}$|'),
        hist.axis.Regular(50, -np.pi, np.pi, 
                            name='phi', label='$\phi_{Z}$'),  
        hist.axis.Regular(50, 80, 100, 
                            name='mass', label='$m_{Z}$'),                     
        hist.axis.StrCategory(['MC', 'Data'], name='label', label='label'),
        storage=hist.storage.Weight()
    )

    print("\tFilling hist from data...")
    for batch in tqdm(dataSET.to_batches(columns = columns)):
        df = batch.to_pandas()
        
        HIST.fill(
            pT = df.pT,
            y = np.abs(df.y),
            phi = df.phi,
            mass = df.mass,
            weight = df['weight'],
            label='Data'
        )
    
    print("\tFilling hist from mc...")
    for batch in tqdm(mcSET.to_batches(columns = columns)):
        df = batch.to_pandas()
        
        HIST.fill(
            pT = df.pT,
            y = np.abs(df.y),
            phi = df.phi,
            mass = df.mass,
            weight = df['weight'],
            label='MC'
        )
    
    with fsspec_xrootd.XRootDFile(fs=fs, path=writepath, mode='wb') as f:
        pickle.dump(HIST, f)

    print("\tDone.")
    print()

    return HIST

def get_jets(folder, overwrite=False):
    raise NotImplementedError("get_jets still in progress")
    basepath = '/store/user/srothman/%s'%folder
    writepath = "%s/%shist.pickle"%(basepath, "jet")

    check = check_exists(writepath, overwrite)
    if check is not None:
        return check

    dataPath = "%s/SingleMuon/jets/parquet"%basepath
    mcPath = "%s/DYJetsToLL/jets/parquet"%basepath

    dataColumns = ['pT', 'eta', 'weight', 'phi', 'bTag', 'cTag', 'nConstituents']
    mcColumns = dataColumns[:] + ['genPT', 'genPhi', 'genEta', 'genFlav']

    dataSET = ds.dataset(dataPath, format='parquet', filesystem=fs)
    mcSET = ds.dataset(mcPath, format='parquet', filesystem=fs)

    HIST = hist.Hist(
        hist.axis.Regular(50, 0, 500, 
                            name='pT', label='$p_{T,jet}$'),
        hist.axis.Regular(50, 0, 3, 
                            name='eta', label='|$\eta_{jet}$|'),
         hist.axis.Regular(50, -np.pi, np.pi, 
                            name='phi', label='$\phi_{jet}$'),                   
        hist.axis.StrCategory(['MC', 'Data'], name='label', label='label'),
        storage=hist.storage.Weight()
    )

    print("\tFilling hist from data...")
    for batch in tqdm(dataSET.to_batches(columns = dataColumns)):
        df = batch.to_pandas()
        
        HIST.fill(
            pT = df.pT,
            eta = np.abs(df.eta),
            phi = df.phi,
            weight = df['weight'],
            label='Data'
        )
    
    print("\tFilling hist from mc...")
    for batch in tqdm(mcSET.to_batches(columns = mcColumns)):
        df = batch.to_pandas()
        
        HIST.fill(
            pT = df.pT,
            eta = np.abs(df.eta),
            phi = df.phi,
            weight = df['weight'],
            label='MC'
        )
    
    with fsspec_xrootd.XRootDFile(fs=fs, path=writepath, mode='wb') as f:
        pickle.dump(HIST, f)

    print("\tDone.")
    print()

    return HIST

def get_EEC(name, folder, overwrite=False):
    '''
    Name should be one of EEC<N> with N=2,3,4,5,6
                          or EECnonIRC<N> with N=12,22,13
    '''
    edges, _ = read_summary(folder)

    basepath = '/store/user/srothman/%s'%folder

    print("doing EEC with name", name)

    writepath = "%s/%shist.pickle"%(basepath, name)

    check = check_exists(writepath)
    if check is not None:
        return check

    dataPath = "%s/SingleMuon/jets/parquet"%basepath
    mcPath = "%s/DYJetsToLL/jets/parquet"%basepath

    dataColumns = ['pT', 'eta', 'weight']
    mcColumns = dataColumns[:]
    for i in range(50):
        dataColumns.append("%swt%d"%(name,i))
        mcColumns.append("%swt%d"%(name,i))
        mcColumns.append("gen%swt%d"%(name,i))

    dataSET = ds.dataset(dataPath, format='parquet', filesystem=fs)
    mcSET = ds.dataset(mcPath, format='parquet', filesystem=fs)

    HIST = hist.Hist(
        hist.axis.Variable(edges, name='dR', label='$\Delta R$'),
        hist.axis.Regular(10, 0, 500, 
                            name='pT', label='$p_{T,Jet}$'),
        hist.axis.Regular(10, 0, 3, 
                            name='eta', label='$\eta_{Jet}$'),
        hist.axis.StrCategory(['Reco', 'Gen', 'Data'], name='label', label='label'),
        storage=hist.storage.Weight()
    )

    dRs = HIST.axes['dR'].centers

    print("\tFilling hist from data...")
    for batch in tqdm(dataSET.to_batches(columns = dataColumns)):
        df = batch.to_pandas()
        
        for i in range(len(dRs)):
            HIST.fill(
                dR = dRs[i],
                pT = df.pT,
                eta = np.abs(df.eta),
                label = 'Data',
                weight = df['%swt%d'%(name,i)] * df['weight']
            )
    
    print("\tFilling hist from mc...")
    for batch in tqdm(mcSET.to_batches(columns = mcColumns)):
        df = batch.to_pandas()

        mask = (df['%swt0'%(name)] != -999) & (df['gen%swt0'%(name)] != -999)

        for i in range(len(dRs)):
            HIST.fill(
                dR = dRs[i],
                pT = df.pT[mask],
                eta = df.eta[mask],
                label = 'Reco',
                weight = df['%swt%d'%(name,i)][mask] * df['weight'][mask]
            )

            HIST.fill(
                dR = dRs[i],
                pT = df.pT[mask],
                eta = df.eta[mask],
                label = 'Gen',
                weight = df['gen%swt%d'%(name,i)][mask] * df['weight'][mask]
            )

    with fsspec_xrootd.XRootDFile(fs=fs, path=writepath, mode='wb') as f:
        pickle.dump(HIST, f)

    print("\tDone.")
    print()

    return HIST