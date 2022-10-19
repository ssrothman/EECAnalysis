import pickle
from hist import rebin, loc, sum, underflow, overflow
import mplhep as hep
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import os
import matplotlib.colors
import mplhep as hep

    
XSEC =  (6077.22) * 1000  #in fb
LUMI = 7.545787391 #in fb-1
lumi_err = 0.023 * LUMI

xsec_err = np.sqrt( 
    np.square(0.0149 * XSEC)  #integration 
    + np.square(0.1478*XSEC)  #pdf
    + np.square(0.02 * XSEC)) #scale

#xsec twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
#pdmv 2017: https://twiki.cern.ch/twiki/bin/view/CMS/DCUserPage2017Analysis#13_TeV_pp_runs_Legacy_2017_aka_U

name = "full_08-05-2022_PUPPI"
'''
with open("test.pickle", 'rb') as f:
#with open("out.pickle", 'rb') as f:
    acc = pickle.load(f)
'''

DATAsum = 36321468
MCsum = 102919490.0

mkdir("figures/%s"%name, exist_ok=True)

kinDir = "figures/%s/KinDataMC"%name
EECdataMCdir = "figures/%s/EECDataMC"%name
EECflavordir = "figures/%s/EECflavor"%name
EECetadir = "figures/%s/EECeta"%name
EECrankdir = "figures/%s/EECrank"%name
EECtagdir = "figures/%s/EECtag"%name
effDir = "figures/%s/efficiency"%name


effective_lumi = MCsum/XSEC
effective_lumi_err = xsec_err/XSEC * effective_lumi

print("Data luminosity:",LUMI)
print("Effective MC luminosity",effective_lumi)


def dataMC(MC, DATA, axis, fname, slc = slice(None,None,None)):
    '''
    Assumes identical binning between dataHist and mcHist
    NB does not check
    '''
    mcHist = MC.project(axis)[slc]
    dataHist = DATA.project(axis)[slc]
    
    fig = plt.gcf()
    grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

    main_ax = fig.add_subplot(grid[0])
    subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
    plt.setp(main_ax.get_xticklabels(), visible=False)

    centers = dataHist.axes[0].centers
    bins = dataHist.axes[0].edges
    widths = dataHist.axes[0].widths

    print(fname)

    mcValues = mcHist.values()/effective_lumi
    mcErrs = np.sqrt(mcHist.variances())/effective_lumi
    #mcErrs = np.sqrt( np.square(mcErrs/mcValues) + np.square(effective_lumi_err/effective_lumi)) * mcValues

    dataValues = dataHist.values()/LUMI
    dataErrs = np.sqrt(dataHist.variances())/LUMI
    #dataErrs = np.sqrt( np.square(dataErrs/dataValues) + np.square(lumi_err/LUMI))

    ratio = dataValues/mcValues 
    ratioerr = np.sqrt(np.square(dataErrs/dataValues) + np.square(mcErrs/mcValues))*ratio

    hep.histplot(mcValues, bins, density=False, histtype='step', ax=main_ax, label='MC')
    main_ax.errorbar(centers, dataValues, yerr = dataErrs, color='k', label="DATA", fmt='o')
    #hep.histplot(dataValues, bins, density=False, histtype='errorbar', ax=main_ax, color='k', label='DATA')
    main_ax.set_ylabel("Events per fb-1")
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

def doMu():
    with open("%s/DYJetsToLL/muon/muonsHist.pickle"%name, 'rb') as f:
        mcHIST = pickle.load(f)
    with open("%s/DoubleMuon/muon/muonsHist.pickle"%name, 'rb') as f:
        dataHIST = pickle.load(f)
    mkdir(kinDir, exist_ok=True)  
    dataMC(mcHIST, dataHIST, 'pT', '%s/muon_pT.png'%kinDir)
    dataMC(mcHIST, dataHIST, 'eta', '%s/muon_eta.png'%kinDir)
    dataMC(mcHIST, dataHIST, 'phi', '%s/muon_phi.png'%kinDir)

def doZ():
    with open("%s/DYJetsToLL/dimuon/dimuonsHist.pickle"%name, 'rb') as f:
        mcHIST = pickle.load(f)
    with open("%s/DoubleMuon/dimuon/dimuonsHist.pickle"%name, 'rb') as f:
        dataHIST = pickle.load(f)
    mkdir(kinDir, exist_ok=True)  
    dataMC(mcHIST, dataHIST, 'pT', '%s/dimuon_pT.png'%kinDir)
    dataMC(mcHIST, dataHIST, 'y', '%s/dimuon_y.png'%kinDir)
    dataMC(mcHIST, dataHIST, 'phi', '%s/dimuon_phi.png'%kinDir)
    dataMC(mcHIST, dataHIST, 'mass', '%s/dimuon_mass.png'%kinDir)

def doJets():
    with open("%s/DYJetsToLL/jets/jetsHist.pickle"%name, 'rb') as f:
        mcHIST = pickle.load(f)
    with open("%s/DoubleMuon/jets/jetsHist.pickle"%name, 'rb') as f:
        dataHIST = pickle.load(f)

    mkdir(kinDir, exist_ok=True)  
    dataMC(mcHIST, dataHIST, 'pT', '%s/jets_pT.png'%kinDir)
    dataMC(mcHIST, dataHIST, 'eta', '%s/jets_eta.png'%kinDir)
    dataMC(mcHIST, dataHIST, 'phi', '%s/jets_phi.png'%kinDir)
    dataMC(mcHIST, dataHIST, 'nConstituents', '%s/jets_nConstituents.png'%kinDir)
    
    with open("%s/DYJetsToLL/event/eventHist.pickle"%name, 'rb') as f:
        mcHIST = pickle.load(f)
    with open("%s/DoubleMuon/event/eventHist.pickle"%name, 'rb') as f:
        dataHIST = pickle.load(f)
    dataMC(mcHIST, dataHIST, 'nJets', '%s/jets_nJets.png'%kinDir)

def doEECdataMC():
    mkdir(EECdataMCdir, exist_ok=True)  
    for N in range(2, 7):
        with open("%s/DYJetsToLL/EEC%d/EECHist.pickle"%(name,N), 'rb') as f:
            mcHIST = pickle.load(f)
        with open("%s/DoubleMuon/EEC%d/EECHist.pickle"%(name,N), 'rb') as f:
            dataHIST = pickle.load(f)
        for pT in range(0, 5):
            fig = plt.gcf()
            grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

            main_ax = fig.add_subplot(grid[0])
            subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
            plt.setp(main_ax.get_xticklabels(), visible=False)

            HMC = mcHIST.project('dR', 'pT')[::,2*pT:2*(pT+1):sum]
            HDATA = dataHIST.project('dR', 'pT')[::,2*pT:2*(pT+1):sum]
            
            #print("values",HMC.values())
            #print("sum",HMC.sum().value)
            histMC = HMC.values()/effective_lumi#HMC.sum().value
            errsMC = np.sqrt(HMC.variances())/effective_lumi#HMC.sum().value

            histDATA = HDATA.values()/LUMI#HDATA.sum().value
            errsDATA = np.sqrt(HDATA.variances())/LUMI#HDATA.sum().value

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

def doJetsFlavor(): 
    with open("%s/DYJetsToLL/jets/jetsHist.pickle"%(name), 'rb') as f:
        mcHIST = pickle.load(f)

    HIST = mcHIST.project('eta', 'genFlav')
    labels = ['Undefined truth flavor', 'Light quark', 'Gluon', 'Charm', 'Bottom']
    dHist = HIST[:, ::sum]
    for flav in range(0,5):
        H = HIST[:, flav]
        hist = H.values()/dHist.values()
        print(H.sum().value/dHist.sum().value)
        errs = 0 #np.sqrt(H.variances())/H.sum().value
        midbins = H.axes[0].centers
        edges = H.axes[0].edges
        #binwidths = edges[1:] - edges[:-1]
        band(midbins, hist, errs, labels[flav])

    plt.xlabel("$\eta$")
    plt.ylabel("Fraction")
    plt.legend()
    plt.savefig("%s/jetFlavor.png"%kinDir)
    plt.clf()

def doEECflavor():
    mkdir(EECflavordir, exist_ok=True)  
    for N in range(2, 7):
        with open("%s/DYJetsToLL/EEC%d/EECHist.pickle"%(name,N), 'rb') as f:
            mcHIST = pickle.load(f)
        for pT in range(0, 5):
            labels = ['Undefined truth flavor', 'Light quark', 'Gluon', 'Charm', 'Bottom']
            HIST = mcHIST.project('dR', 'pT', 'genFlav')[:, :, :]
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

def doEECeta():
    mkdir(EECetadir, exist_ok=True)  
    for N in range(2, 7):
        with open("%s/DYJetsToLL/EEC%d/EECHist.pickle"%(name,N), 'rb') as f:
            mcHIST = pickle.load(f)
        with open("%s/DoubleMuon/EEC%d/EECHist.pickle"%(name,N), 'rb') as f:
            dataHIST = pickle.load(f)

        for pT in range(0, 5):
            labels = ['$%0.2f < |\eta| < %0.2f$'%(i, (i+1)) for i in range(3)]
            HIST = mcHIST.project('dR', 'pT', 'eta')[:, :, ::rebin(3)]
            HISTdata = dataHIST.project('dR', 'pT', 'eta')[:, :, ::rebin(3)]

            for eta in range(3):
                H = HIST[:, 2*pT:2*(pT+1):sum, eta]
                hist = H.values()/H.sum().value
                errs = np.sqrt(H.variances())/H.sum().value
                midbins = H.axes[0].centers
                edges = H.axes[0].edges
                binwidths = np.log(edges[1:]) - np.log(edges[:-1])
                line = band(midbins, hist/binwidths, errs/binwidths, labels[eta])

                H = HISTdata[:, 2*pT:2*(pT+1):sum, eta]
                hist = H.values()/H.sum().value
                errs = np.sqrt(H.variances())/H.sum().value
                midbins = H.axes[0].centers
                edges = H.axes[0].edges
                binwidths = np.log(edges[1:]) - np.log(edges[:-1])
                c = line[0].get_color()
                rval = int("0x%s"%c[1:3], base=16)
                gval = int("0x%s"%c[3:5], base=16)
                bval = int("0x%s"%c[5:7], base=16)
                hsv = matplotlib.colors.rgb_to_hsv([rval, gval, bval])
                hsv[-1]/=1.2
                rgb = matplotlib.colors.hsv_to_rgb(hsv)
                rstr = "%02x"%int(rgb[0]+0.5)
                gstr = "%02x"%int(rgb[1]+0.5)
                bstr = "%02x"%int(rgb[2]+0.5)
                c = "#%s%s%s"%(rstr, gstr, bstr)
                plt.errorbar(midbins, hist/binwidths, yerr=errs/binwidths, label=None, color=c, fmt='o', markersize=5)

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
            plt.savefig("%s/EEC%dpT%d_eta.png"%(EECetadir, N,pT), format='png')
            plt.clf()

def doEfficiency():
    mkdir(effDir, exist_ok=True)  

    with open("%s/DYJetsToLL/jets/effHist.pickle"%name, 'rb') as f:
        HIST = pickle.load(f)
    
    tagNames = {'cTag' : 'C tagging', 'bTag' : 'B tagging'}
    genNames = ['light quark, gluon, & undefined-flavor', "charm", "bottom"]
    for genFlav in [0, 1, 2]:
        for tag in ['cTag', 'bTag']:
            H = HIST[{'genFlav' : genFlav}].project("pT", "eta", tag)
            tagged = H[{tag: 1}]
            total = H[{tag: slice(None,None,sum)}]
            efficiency = tagged/(total)
            print(tag, genFlav)

            plt.title("%s efficiency for generated %s jets"%(tagNames[tag], genNames[genFlav]))

            efficiency.plot2d(cmap='Oranges', vmin=0, vmax=1)
            plt.xlabel("$p_{T,Jet}$")
            plt.ylabel("$|\eta_{Jet}|$")
            plt.savefig("%s/%d_%s.png"%(effDir, genFlav, tag), format='png', bbox_inches='tight')
            plt.clf()

#doEfficiency()
#doMu()
#doZ()
#doJets()
#doEECdataMC()
#doEECflavor()
#doEECeta()
doJetsFlavor()
