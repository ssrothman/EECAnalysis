from bdb import effective
from get_hists import read_summary, get_EEC, get_muons, get_Zs, get_jets

import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import hist
import pandas as pd
import seaborn as sns
import os
import mplhep as hep

xlabels = {
    'EEC2' : '2-point energy correlator'
}

XSEC =  (6077.22) * 1000  #in fb
LUMI = 7.545787391 #in fb-1

def getValueErr(hist, norm='sumw'):
    values = hist.values()
    errs = np.sqrt(hist.variances())

    if norm == 'sumw':
        A = np.sum(values)
    else:
        raise NotImplementedError("Only valid 'norm' are 'sumw' and 'xsec'")
    
    return values/A, errs/A

def dataMC(hist, axis, fname, slc = slice(None,None,None)):
    '''
    Assumes identical binning between dataHist and mcHist
    NB does not check
    '''
    mcHist = hist[{'label' : "MC"}].project(axis)
    dataHist = hist[{'label' : "Data"}].project(axis)
    
    fig = plt.gcf()
    grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

    main_ax = fig.add_subplot(grid[0])
    subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
    plt.setp(main_ax.get_xticklabels(), visible=False)

    centers = dataHist.axes[0].centers
    widths = dataHist.axes[0].widths

    mcValues, mcErrs = getValueErr(mcHist)

    dataValues, dataErrs = getValueErr(dataHist)

    ratio = dataValues/mcValues 
    ratioerr = np.sqrt(np.square(dataErrs/dataValues) + np.square(mcErrs/mcValues))*ratio

    main_ax.errorbar(centers, dataValues, yerr = dataErrs, xerr = widths/2, color='k', label="Data", fmt='o')
    band(centers, mcValues, mcErrs, 'MC', ax=main_ax)
    main_ax.set_ylabel("Events [AU]")
    main_ax.legend()

    #TODO: validate the changes here
    subplot_ax.errorbar(centers, ratio, yerr=ratioerr, fmt='o', xerr=widths/2, color='k')
    subplot_ax.set_ylabel("Data/MC")
    subplot_ax.set_xlabel(dataHist.axes[0].label)
    subplot_ax.axhline(1.0, color='k', alpha=0.5, linestyle='--')
    bottom, top = subplot_ax.get_ylim()
    if bottom>0.95:
        subplot_ax.set_ylim(bottom=0.95)
    if top<1.05:
        subplot_ax.set_ylim(top=1.05)

    plt.savefig(fname, format='png', bbox_inches='tight')
    plt.clf()

def mkdir(path, exist_ok=True):
    try:
        os.mkdir(path)
    except FileExistsError as e:
        if not exist_ok:
            raise e

def band(x, y, yerr, label, ax=None):
    if ax is None:
        ax = plt.gca()
    line = ax.plot(x, y, 'o--', linewidth=1, markersize=2)
    ax.fill_between(x, y-yerr, y+yerr, label=label, color=line[0].get_color(), alpha=0.7)
    return line

def _plotEEC(hist, label, ax=None, norm='sumw', effective_lumi=0):
    if ax is None:
        ax = plt.gca()
    
    centers = hist.axes[0].centers
    edges = hist.axes[0].edges
    widths = np.log(edges[1:])/np.log(edges[:-1])

    values, errs = getValueErr(hist, norm)

    band(centers, values/widths, errs/widths, label)

def plotEEC(name, folder, norm='sumw', title=None, pTbin=None, show=False):
    '''
    norm = one of ['sumw', 'xsec']
    '''
    plt.cla()
    plt.clf()

    _, sumws = read_summary(folder)
    effective_lumi = sumws['DYJetsToLL']/XSEC
    LUMI = sumws['SingleMuon']/XSEC

    EEC = get_EEC(name, folder)

    if pTbin is None:
        EEC = EEC.project('dR', 'label')
    else:
        pTedges = EEC.axes['pT'].edges
        pTlow = pTedges[pTbin]
        pThigh = pTedges[pTbin+1]
        EEC = EEC[{'pT':pTbin}].project('dR', 'label')

    labels = ['Reco', 'Gen', 'Data']
    for label in labels:
        hist = EEC[{'label':label}]
        _plotEEC(hist, label, norm=norm, effective_lumi=effective_lumi)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("$\Delta R_{max}$")
    plt.ylabel(xlabels[name])
    plt.legend()
    if title is None:
        title = name
    if pTbin is not None:
        title += '\n$%0.2f < p_T < %0.2f$'%(pTlow, pThigh)
    plt.title(title)

    outdir = "figures/%s/EEC"%folder
    mkdir(outdir)
    if pTbin is None:
        outname = '%s.png'%name
    else:
        outname = '%s_pT%d.png'%(name, pTbin)
    plt.savefig("%s/%s"%(outdir, outname), format='png', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
        plt.cla()

def plotMuons(folder, norm='sumw', title=None, show=False):
    plt.cla()
    plt.clf()

    _, sumws = read_summary(folder)
    effective_lumi = sumws['DYJetsToLL']/XSEC
    LUMI = sumws['SingleMuon']/XSEC

    muons = get_muons(folder, overwrite=True)

    outdir = "figures/%s/muons"%folder
    mkdir(outdir)

    dataMC(muons, 'pT', '%s/muon_pT.png'%outdir)
    dataMC(muons, 'eta', '%s/muon_eta.png'%outdir)
    dataMC(muons, 'phi', '%s/muon_phi.png'%outdir)

def plotZs(folder, norm='sumw', title=None, show=False):
    plt.cla()
    plt.clf()

    _, sumws = read_summary(folder)
    effective_lumi = sumws['DYJetsToLL']/XSEC
    LUMI = sumws['SingleMuon']/XSEC

    Zs = get_Zs(folder, overwrite=True)

    outdir = "figures/%s/Zs"%folder
    mkdir(outdir)

    dataMC(Zs, 'pT', '%s/Z_pT.png'%outdir)
    dataMC(Zs, 'phi', '%s/Z_phi.png'%outdir)
    dataMC(Zs, 'y', '%s/Z_y.png'%outdir)
    dataMC(Zs, 'mass', '%s/Z_mass.png'%outdir)

def plotJets(folder, norm='sumw', title=None, show=False):
    plt.cla()
    plt.clf()

    _, sumws = read_summary(folder)
    effective_lumi = sumws['DYJetsToLL']/XSEC
    LUMI = sumws['SingleMuon']/XSEC

    jets = get_jets(folder, overwrite=True)

    outdir = "figures/%s/jets"%folder
    mkdir(outdir)

    dataMC(jets, 'pT', '%s/jet_pT.png'%outdir)
    dataMC(jets, 'phi', '%s/jet_phi.png'%outdir)
    dataMC(jets, 'eta', '%s/jet_eta.png'%outdir)

'''
for pTbin in range(10):
    folder = 'small_v3_EECs_3PU_09-27-2022_CS'
    plotEEC('EEC2', folder, title='CS', pTbin=pTbin)
    folder = 'small_v3_EECs_3PU_09-27-2022_CHS'
    plotEEC('EEC2', folder, title='CHS', pTbin=pTbin)
    folder = 'small_v3_EECs_3PU_09-27-2022_Puppi'
    plotEEC('EEC2', folder, title='Puppi', pTbin=pTbin)
'''
folder = 'small_v3_EECs_3PU_09-27-2022_Puppi'
#plotMuons(folder)
#plotZs(folder)
plotJets(folder)
#plotEEC('EEC3')
#plotEEC('EEC4')
#plotEEC('EEC5')
#plotEEC('EEC6')
#plotEEC('EECnonIRC12')
#plotEEC('EECnonIRC13')
#plotEEC('EECnonIRC22')