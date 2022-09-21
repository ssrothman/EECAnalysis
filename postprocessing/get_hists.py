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

folder = 'test_v5_09-21-2022_PUPPI'
basepath = '/store/user/srothman/%s'%folder
histpath = '%s/hists'%basepath

fs = fsspec_xrootd.XRootDFileSystem(hostid="cmseos.fnal.gov")

def read_summary():
    '''
    Read dR bin edges from summary.txt
    '''
    edges = []
    sumws = {}

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
                sumws[splitted[0]] = float(splitted[1])
            elif 'Total Weights' in text:
                startedWTs = True

            if startedDRs:
                edges.append(float(text))
            elif 'dR bin edges' in text:
                startedDRs = True

    return edges, sumws

edges, sumws = read_summary()

def get_EEC(name, edges, overwrite=False):
    '''
    Name should be one of EEC<N> with N=2,3,4,5,6
                          or EECnonIRC<N> with N=12,22,13
    '''
    print("doing EEC with name", name)

    writepath = "%s/%shist.pickle"%(basepath, name)

    if fs.exists(writepath):
        print("\tTarget file already exists at:", writepath)
        if overwrite:
            print("\t\tOverwriting...")
        else:
            print("\tUsing pre-existing histogram...")
            with fsspec_xrootd.XRootDFile(fs=fs, path=writepath, mode='rb') as f:
                return pickle.load(f)



    dataPath = "%s/DoubleMuon/jets/parquet"%basepath
    mcPath = "%s/DYJetsToLL/jets/parquet"%basepath

    dataColumns = ['pT', 'eta', 'weight']
    mcColumns = dataColumns[:]
    for i in range(50):
        dataColumns.append("%swt%d"%(name,i))
        mcColumns.append("%swt%d"%(name,i))
        mcColumns.append("gen%swt%d"%(name,i))

    #in future can optimize by only reading the columns we care about
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
                eta = df.eta,
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

EEC2 = get_EEC("EEC2", edges)
EEC3 = get_EEC("EEC3", edges)
EEC4 = get_EEC("EEC4", edges)
EEC5 = get_EEC("EEC5", edges)
EEC6 = get_EEC("EEC6", edges)
EECnonIRC12 = get_EEC("EECnonIRC12", edges)
EECnonIRC22 = get_EEC("EECnonIRC22", edges)
EECnonIRC13 = get_EEC("EECnonIRC13", edges)