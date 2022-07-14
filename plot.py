import pickle
from hist import rebin, loc, sum, underflow, overflow
import mplhep as hep
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

XSEC =  (6077.22) * 1000  #in fb
#LUMI = 0.080401481 / 2#in fb-1 #fudge factor of 2
LUMI = 7.545787391 #in fb-1

#xsec twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
#pdmv 2017: https://twiki.cern.ch/twiki/bin/view/CMS/DCUserPage2017Analysis#13_TeV_pp_runs_Legacy_2017_aka_U

with open("output/Full07-12-2022.hist.pickle", 'rb') as f:
#with open("out.pickle", 'rb') as f:
    acc = pickle.load(f)

effective_lumi = acc['DYJetsToLL']['sumw']/XSEC #*10 #fudge factor of 10????
est_lumi = acc['DoubleMuon']['sumw']/XSEC

doJets = False
doEEC = False
doZ = False
doMu = False
doEECpt = False
doEECorder = True

def dataMC(MC, DATA, axis, fname):
    '''
    Assumes identical binning between dataHist and mcHist
    NB does not check
    '''
    mcHist = MC.project(axis)
    dataHist = DATA.project(axis)
    if axis == 'pT':
        mcHist = mcHist[:20]
        dataHist = dataHist[:20]
    elif axis != 'phi' and axis != 'nJets' and axis!='dR':
        mcHist = mcHist[::rebin(5)]
        dataHist = dataHist[::rebin(5)]
    
    fig = plt.gcf()
    grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

    main_ax = fig.add_subplot(grid[0])
    subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
    plt.setp(main_ax.get_xticklabels(), visible=False)

    centers = dataHist.axes[0].centers
    bins = dataHist.axes[0].edges
    widths = dataHist.axes[0].widths

    #TODO: figure out exactly how the scaling should work
    #TODO: error bars?
    print(fname)
    #print(effective_lumi/LUMI)
    #print(mcHist.sum()/dataHist.sum())
    mcValues = mcHist.values()/effective_lumi #* dataHist.sum()/mcHist.sum()
    dataValues = dataHist.values()/LUMI
    #print(mcValues.sum()/dataValues.sum())
    #print()
    ratio = dataValues/mcValues 

    hep.histplot(mcValues, bins, density=False, histtype='step', ax=main_ax, label='MC')
    hep.histplot(dataValues, bins, density=False, histtype='errorbar', ax=main_ax, color='k', label='DATA')
    main_ax.set_ylabel("Events per fb-1")
    main_ax.legend()

    subplot_ax.errorbar(centers, ratio, fmt='o', xerr=widths/2, color='k')
    subplot_ax.set_ylabel("Data/MC")
    subplot_ax.set_xlabel(dataHist.axes[0].label)
    #subplot_ax.axhline(1.0, color='k', alpha=0.5, linestyle='--')

    plt.savefig(fname, format='png', bbox_inches='tight')
    plt.clf()

if doMu:    
    dataMC(acc['DYJetsToLL']['muon'], acc['DoubleMuon']['muon'], 'pT', 'figures/muon_pT.png')
    dataMC(acc['DYJetsToLL']['muon'], acc['DoubleMuon']['muon'], 'eta', 'figures/muon_eta.png')
    dataMC(acc['DYJetsToLL']['muon'], acc['DoubleMuon']['muon'], 'phi', 'figures/muon_phi.png')

if doZ:
    dataMC(acc['DYJetsToLL']['dimuon'], acc['DoubleMuon']['dimuon'], 'pT', 'figures/dimuon_pT.png')
    dataMC(acc['DYJetsToLL']['dimuon'], acc['DoubleMuon']['dimuon'], 'y', 'figures/dimuon_y.png')
    dataMC(acc['DYJetsToLL']['dimuon'], acc['DoubleMuon']['dimuon'], 'phi', 'figures/dimuon_phi.png')
    dataMC(acc['DYJetsToLL']['dimuon'], acc['DoubleMuon']['dimuon'], 'mass', 'figures/dimuon_mass.png')

if doJets:
    dataMC(acc['DYJetsToLL']['jets'], acc['DoubleMuon']['jets'], 'pT', 'figures/jets_pT.png')
    dataMC(acc['DYJetsToLL']['jets'], acc['DoubleMuon']['jets'], 'eta', 'figures/jets_eta.png')
    dataMC(acc['DYJetsToLL']['jets'], acc['DoubleMuon']['jets'], 'phi', 'figures/jets_phi.png')
    dataMC(acc['DYJetsToLL']['nJets'], acc['DoubleMuon']['nJets'], 'nJets', 'figures/jets_nJets.png')

if doEECpt:
    for N in range(2, 7):
        for pt in range(0, 5):
            H = acc['DoubleMuon']['EEC%d'%N][:,2*pt:2*(pt+1):sum,:].project('dR')
            hist = H.values()/H.sum()

            midbins = H.axes[0].centers
            binwidths = H.axes[0].widths

            plt.errorbar(midbins, hist/binwidths, fmt='o',lw=1.5,markersize=2, label='$%d<p_T<%d$'%(100*pt, 100*(pt+1)))

        plt.legend()
        plt.xlabel("$\Delta R$")
        plt.ylabel("Projected %d-point correlator"%N)
        plt.axvline(0.4, c='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(left=1e-4)
        plt.ylim(bottom=1e-2)
        plt.savefig("figures/EEC%d_data_pt.png"%N, format='png')
        plt.clf()

if doEECorder:
    for N in range(2, 7):
        H = acc['DoubleMuon']['EEC%d'%N][:,:,:].project('dR')
        hist = H.values()/H.sum()

        midbins = H.axes[0].centers
        binwidths = H.axes[0].widths

        plt.errorbar(midbins, hist/binwidths, fmt='o',lw=1.5,markersize=2, label='%d-point'%N)

    plt.legend()
    plt.xlabel("$\Delta R$")
    plt.ylabel("Projected correlator")
    plt.axvline(0.4, c='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(left=1e-4)
    plt.ylim(bottom=1e-2)
    plt.savefig("figures/EEC_data_order.png", format='png')
    plt.clf()