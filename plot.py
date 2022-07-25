import pickle
from hist import rebin, loc, sum, underflow, overflow
import mplhep as hep
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import os

def mkdir(path, exist_ok=False):
    try:
        os.mkdir(path)
    except FileExistsError as e:
        if not exist_ok:
            raise e
    
XSEC =  (6077.22) * 1000  #in fb
#LUMI = 0.080401481 / 2#in fb-1 #fudge factor of 2
LUMI = 7.545787391 #in fb-1
LUMI_PM = 2 #in percent
XSEC_PM = 5 #in percent

ratio_pm = np.sqrt(np.square(LUMI_PM/100) + np.square(XSEC_PM/100))*100

#xsec twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
#pdmv 2017: https://twiki.cern.ch/twiki/bin/view/CMS/DCUserPage2017Analysis#13_TeV_pp_runs_Legacy_2017_aka_U

name = "Full07-23-2022"

with open("output/%s.hist.pickle"%name, 'rb') as f:
#with open("out.pickle", 'rb') as f:
    acc = pickle.load(f)

mkdir("figures/%s"%name, exist_ok=True)

kinDir = "figures/%s/KinDataMC"%name
EECdataMCdir = "figures/%s/EECDataMC"%name
EECflavordir = "figures/%s/EECflavor"%name
EECrankdir = "figures/%s/EECrank"%name

effective_lumi = acc['DYJetsToLL']['sumw']/XSEC
est_lumi = acc['DoubleMuon']['sumw']/XSEC

print("Data luminosity:",LUMI)
print("Effective MC luminosity",effective_lumi)

doJets = False
doJetsFlavor = False
doZ = False
doMu = False
doEECdataMC = False
doEECflavor = False
doEECrank = True

def dataMC(MC, DATA, axis, fname):
    '''
    Assumes identical binning between dataHist and mcHist
    NB does not check
    '''
    mcHist = MC.project(axis)
    dataHist = DATA.project(axis)
    #f axis == 'pT':
    #    mcHist = mcHist[:20]
    #    dataHist = dataHist[:20]
    #elif axis != 'phi' and axis != 'nJets' and axis!='dR':
    #    mcHist = mcHist[::rebin(5)]
    #    dataHist = dataHist[::rebin(5)]
    
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
    #print(mcHist.sum().value/dataHist.sum().value)
    mcValues = mcHist.values()/effective_lumi #* dataHist.sum().value/mcHist.sum().value
    dataValues = dataHist.values()/LUMI
    #print(mcValues.sum().value/dataValues.sum().value)
    #print()
    ratio = dataValues/mcValues 

    hep.histplot(mcValues, bins, density=False, histtype='step', ax=main_ax, label='MC')
    hep.histplot(dataValues, bins, density=False, histtype='errorbar', ax=main_ax, color='k', label='DATA')
    main_ax.set_ylabel("Events per fb-1")
    main_ax.legend()

    #TODO: validate the changes here
    subplot_ax.errorbar(centers, ratio, yerr=ratio*ratio_pm/100, fmt='o', xerr=widths/2, color='k')
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

if doMu:  
    mkdir(kinDir, exist_ok=True)  
    dataMC(acc['DYJetsToLL']['muon'], acc['DoubleMuon']['muon'], 'pT', '%s/muon_pT.png'%kinDir)
    dataMC(acc['DYJetsToLL']['muon'], acc['DoubleMuon']['muon'], 'eta', '%s/muon_eta.png'%kinDir)
    #dataMC(acc['DYJetsToLL']['muon'], acc['DoubleMuon']['muon'], 'phi', '%s/muon_phi.png'%kinDir)

if doZ:
    mkdir(kinDir, exist_ok=True)  
    dataMC(acc['DYJetsToLL']['dimuon'], acc['DoubleMuon']['dimuon'], 'pT', '%s/dimuon_pT.png'%kinDir)
    dataMC(acc['DYJetsToLL']['dimuon'], acc['DoubleMuon']['dimuon'], 'y', '%s/dimuon_y.png'%kinDir)
    #dataMC(acc['DYJetsToLL']['dimuon'], acc['DoubleMuon']['dimuon'], 'phi', '%s/dimuon_phi.png'%kinDir)
    dataMC(acc['DYJetsToLL']['dimuon'], acc['DoubleMuon']['dimuon'], 'mass', '%s/dimuon_mass.png'%kinDir)

if doJets:
    mkdir(kinDir, exist_ok=True)  
    dataMC(acc['DYJetsToLL']['jets'], acc['DoubleMuon']['jets'], 'pT', '%s/jets_pT.png'%kinDir)
    dataMC(acc['DYJetsToLL']['jets'], acc['DoubleMuon']['jets'], 'eta', '%s/jets_eta.png'%kinDir)
    #dataMC(acc['DYJetsToLL']['jets'], acc['DoubleMuon']['jets'], 'phi', '%s/jets_phi.png'%kinDir)
    dataMC(acc['DYJetsToLL']['jets'], acc['DoubleMuon']['jets'], 'nConstituents', '%s/jets_nConstituents.png'%kinDir)
    dataMC(acc['DYJetsToLL']['nJets'], acc['DoubleMuon']['nJets'], 'nJets', '%s/jets_nJets.png'%kinDir)

def band(x, y, yerr, label, ax=None):
    if ax is None:
        ax = plt.gca()
    line = ax.plot(x, y, 'o--', linewidth=1, markersize=2)
    ax.fill_between(x, y-yerr, y+yerr, label=label, color=line[0].get_color(), alpha=0.7)

if doEECdataMC:
    mkdir(EECdataMCdir, exist_ok=True)  
    for pT in range(0, 5):
        for N in range(2, 7):
            fig = plt.gcf()
            grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

            main_ax = fig.add_subplot(grid[0])
            subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
            plt.setp(main_ax.get_xticklabels(), visible=False)

            HMC = acc['DYJetsToLL']['EEC%d'%N].project('dR', 'pT')[::rebin(2),2*pT:2*(pT+1):sum]
            HDATA = acc['DoubleMuon']['EEC%d'%N].project('dR', 'pT')[::rebin(2),2*pT:2*(pT+1):sum]
            
            #print("values",HMC.values())
            #print("sum",HMC.sum().value)
            histMC = HMC.values()/HMC.sum().value
            errsMC = np.sqrt(HMC.variances())/HMC.sum().value

            histDATA = HDATA.values()/HDATA.sum().value
            errsDATA = np.sqrt(HDATA.variances())/HDATA.sum().value

            midbins = HMC.axes[0].centers
            #binwidths = H.axes[0].widths
            edges = HMC.axes[0].edges
            binwidths = np.log(edges[1:]) - np.log(edges[:-1])
            band(midbins, histMC/binwidths, errsMC/binwidths, "MC", main_ax)
            main_ax.errorbar(midbins, histDATA/binwidths, yerr=errsDATA/binwidths, label='Data', color='k', fmt='o')
            main_ax.legend()
            main_ax.set_ylabel("$\\frac{d \\sigma}{d \\log \\Delta R_{max}}$", fontsize=14)
            main_ax.axvline(0.4, c='k')
            main_ax.text(0.35, 1.5e-2, "Jet clustering radius", fontsize=8, rotation=270, va='bottom', ha='right')
            main_ax.set_xscale('log')
            main_ax.set_yscale('log')
            main_ax.set_xlim(left=1e-3)
            main_ax.set_ylim(bottom=1e-2)
            main_ax.set_title("%d-point correlator\n$%d < p_T < %d$"%(N, 100*pT, 100*(pT+1)))

            ratio = histDATA/histMC
            ratioerr = np.sqrt(np.square(errsDATA/histDATA) + np.square(errsMC/histDATA))*ratio
            subplot_ax.errorbar(midbins, ratio, yerr=ratioerr, color='k', fmt='o')
            subplot_ax.axhline(1.0, color='k', alpha=0.5, linestyle='--')
            subplot_ax.set_ylabel("Data/MC")
            subplot_ax.set_ylim(0, 2)

            subplot_ax.set_xlabel("$\Delta R_{max}$", fontsize=14)

            plt.savefig("%s/N%dpT%d.png"%(EECdataMCdir, N,pT), format='png', bbox_inches='tight')
            plt.clf()

if doJetsFlavor:
    H = acc['DYJetsToLL']['jets'].project('eta', 'flav')
    midbins = H.axes[0].centers
    binwidths = H.axes[0].widths

    hist0 = H[:,0].values()/H[:,0].sum().value
    err0 = np.sqrt(H[:,0].variances())/H[:,0].sum().value
   
    hist1 = H[:,1].values()/H[:,1].sum().value
    err1 = np.sqrt(H[:,1].variances())/H[:,1].sum().value

    hist2 = H[:,2].values()/H[:,2].sum().value
    err2 = np.sqrt(H[:,2].variances())/H[:,2].sum().value

    hist3 = H[:,3].values()/H[:,3].sum().value
    err3 = np.sqrt(H[:,3].variances())/H[:,3].sum().value

    hist4 = H[:,4].values()/H[:,4].sum().value
    err4 = np.sqrt(H[:,4].variances())/H[:,4].sum().value

    #plt.errorbar(midbins, hist0/binwidths, yerr = err0/binwidths, fmt='o',lw=1.5,markersize=2, label='undefined')
    #plt.errorbar(midbins, hist1/binwidths, yerr = err0/binwidths, fmt='o-',lw=1.5,markersize=2, label='light quark')
    #plt.errorbar(midbins, hist2/binwidths, yerr = err0/binwidths, fmt='o-',lw=1.5,markersize=2, label='gluon')
    #plt.errorbar(midbins, hist3/binwidths, yerr = err0/binwidths, fmt='o-',lw=1.5,markersize=2, label='charm')
    #plt.errorbar(midbins, hist4/binwidths, yerr = err0/binwidths, fmt='o-',lw=1.5,markersize=2, label='bottom')
    band(midbins, hist0/binwidths, err0/binwidths, "undefined")
    band(midbins, hist1/binwidths, err1/binwidths, "light quark")
    band(midbins, hist2/binwidths, err2/binwidths, "gluon")
    band(midbins, hist3/binwidths, err3/binwidths, "charm")
    band(midbins, hist4/binwidths, err4/binwidths, "bottom")
    plt.xlabel("$|\eta|$")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("test.png")
    plt.clf()

if doEECflavor:
    mkdir(EECflavordir, exist_ok=True)  
    for pT in range(0, 5):
        for N in range(2, 7):
            labels = ['Undefined truth flavor', 'Light quark', 'Gluon', 'Charm', 'Bottom']
            HIST = acc['DYJetsToLL']['EEC%d'%N].project('dR', 'pT', 'flav')[::rebin(2), :, :]
            for flav in range(1,5):
                H = HIST[:, 2*pT:2*(pT+1):sum, flav]
                hist = H.values()/H.sum().value
                errs = np.sqrt(H.variances())/H.sum().value
                midbins = H.axes[0].centers
                edges = H.axes[0].edges
                binwidths = np.log(edges[1:]) - np.log(edges[:-1])
                band(midbins, hist/binwidths, errs/binwidths, labels[flav])

            plt.title("%d-point correlator\n$%d < p_T < %d$"%(N, 100*pT, 100*(pT+1)))
            plt.legend()
            plt.xlabel("$\Delta R$")
            plt.ylabel("Projected correlator")
            plt.axvline(0.4, c='k')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(left=1e-3)
            plt.ylim(bottom=1e-2)
            plt.text(0.35, 1.5e-2, "Jet clustering radius", fontsize=8, rotation=270, va='bottom', ha='right')
            plt.savefig("%s/EEC%dpT%d_MC_flav.png"%(EECflavordir, N,pT), format='png')
            plt.clf()

if doEECrank:
    mkdir(EECrankdir, exist_ok=True)  
    for pT in range(0, 5):
        for N in range(2, 7):
            labels = ['Leading jet', 'subleading jet', 'sub-subleading jet']
            HIST = acc['DYJetsToLL']['EEC%d'%N].project('dR', 'pT', 'pTrank')[::rebin(2), :, :]
            for rank in range(0,3):
                H = HIST[:, 2*pT:2*(pT+1):sum, rank]
                hist = H.values()/H.sum().value
                errs = np.sqrt(H.variances())/H.sum().value
                midbins = H.axes[0].centers
                edges = H.axes[0].edges
                binwidths = np.log(edges[1:]) - np.log(edges[:-1])
                band(midbins, hist/binwidths, errs/binwidths, labels[rank])

            plt.title("%d-point correlator\n$%d < p_T < %d$"%(N, 100*pT, 100*(pT+1)))
            plt.legend()
            plt.xlabel("$\Delta R$")
            plt.ylabel("Projected correlator")
            plt.axvline(0.4, c='k')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(left=1e-3)
            plt.ylim(bottom=1e-2)
            plt.text(0.35, 1.5e-2, "Jet clustering radius", fontsize=8, rotation=270, va='bottom', ha='right')
            plt.savefig("%s/EEC%dpT%d_MC_rank.png"%(EECrankdir, N,pT), format='png')
            plt.clf()