import pickle
from hist import rebin, loc, sum, underflow, overflow
import mplhep as hep
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

XSEC =  (6077.22/3) * 1000  #in fb
LUMI = 0.080401481 / 2#in fb-1 #fudge factor of 2
#LUMI = 9.243831866 #in fb-1

#xsec twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
#pdmv 2017: https://twiki.cern.ch/twiki/bin/view/CMS/DCUserPage2017Analysis#13_TeV_pp_runs_Legacy_2017_aka_U

#with open("output/hists.pickle", 'rb') as f:
with open("out.pickle", 'rb') as f:
    acc = pickle.load(f)

effective_lumi = acc['sumw']['DYJetsToLL']/XSEC #*10 #fudge factor of 10????
est_lumi = acc['sumw']['DoubleMuon']/XSEC
#print(effective_lumi/LUMI)
#print(acc['dimuon']['DYJetsToLL',:,:,:,:].sum()/acc['dimuon']['DoubleMuon',:,:,:,:].sum())

print("EFF",effective_lumi)
print("EST",est_lumi)
print("REAL",LUMI)
print("EFF/REAL",effective_lumi/LUMI)
print("EST/REAL",est_lumi/LUMI)
print("EFF/EST",effective_lumi/est_lumi)
print()
print("MC",acc['muon']['DYJetsToLL',:,:,:].sum())
print("DATA",acc['muon']['DoubleMuon',:,:,:].sum())
print("MC/DATA",acc['muon']['DYJetsToLL',:,:,:].sum()/acc['muon']['DoubleMuon',:,:,:].sum())

doJets = True
doEEC = False
doZ = True
doMu = True

def dataMC(hist, axis, fname):
    '''
    Assumes identical binning between dataHist and mcHist
    NB does not check
    '''

    H = hist.project("dataset", axis)
    mcHist = H['DYJetsToLL',:]
    dataHist = H['DoubleMuon',:]
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
    dataMC(acc['muon'], 'pT', 'figures/muon_pT.png')
    dataMC(acc['muon'], 'eta', 'figures/muon_eta.png')
    dataMC(acc['muon'], 'phi', 'figures/muon_phi.png')
    
if doZ:
    dataMC(acc['dimuon'], 'pT', 'figures/dimuon_pT.png')
    dataMC(acc['dimuon'], 'y', 'figures/dimuon_y.png')
    dataMC(acc['dimuon'], 'phi', 'figures/dimuon_phi.png')
    dataMC(acc['dimuon'], 'mass', 'figures/dimuon_M.png')

if doJets:
    dataMC(acc['jets'], 'pT', 'figures/jets_pT.png')
    dataMC(acc['jets'], 'eta', 'figures/jets_eta.png')
    dataMC(acc['jets'], 'phi', 'figures/jets_phi.png')
    dataMC(acc['nJets'], 'nJets', 'figures/jets_nJets.png')

if doEEC:
    '''
    H1 = acc['EEC2']["DYJetsToLL",:,:1:sum,:].project('dR')
    midbins = H1.axes[0].centers
    binwidths = H1.axes[0].widths
    hist1 = H1.values()/H1.sum()

    H2 = acc['EEC2']["DYJetsToLL",:,1:2:sum,:].project('dR')
    hist2 = H2.values()/H2.sum()

    H3 = acc['EEC2']["DYJetsToLL",:,2:3:sum,:].project('dR')
    hist3 = H3.values()/H3.sum()

    H4 = acc['EEC2']["DYJetsToLL",:,3:4:sum,:].project('dR')
    hist4 = H4.values()/H4.sum()

    H5 = acc['EEC2']["DYJetsToLL",:,4::sum,:].project('dR')
    hist5 = H5.values()/H5.sum()

    plt.xlabel('$\Delta R$')
    plt.ylabel("Projected %d-point correlator"%2)
    plt.errorbar(midbins, hist1/binwidths, fmt='o',lw=1.5,markersize=2, label='0<pT<100')
    plt.errorbar(midbins, hist2/binwidths, fmt='o',lw=1.5,markersize=2, label='100<pT<200')
    plt.errorbar(midbins, hist3/binwidths, fmt='o',lw=1.5,markersize=2, label='200<pT<300')
    plt.errorbar(midbins, hist4/binwidths, fmt='o',lw=1.5,markersize=2, label='300<pT<400')
    plt.errorbar(midbins, hist5/binwidths, fmt='o',lw=1.5,markersize=2, label='400<pT<500')
    plt.legend()
    plt.axvline(0.4, c='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("EEC2.png", format='png')
    plt.clf()

    H1 = acc['EEC2']["DYJetsToLL",:,::sum,:].project('dR')
    H2 = acc['genEEC2']["DYJetsToLL",:,::sum,:].project('dR')
    hist1 = H1.values()/H1.sum()
    hist2 = H2.values()/H2.sum()
    plt.errorbar(midbins, hist1/binwidths, fmt='o',lw=1.5,markersize=2, label='RECO')
    plt.errorbar(midbins, hist2/binwidths, fmt='o',lw=1.5,markersize=2, label='GEN')
    plt.legend()
    plt.axvline(0.4, c='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("genEEC2.png", format='png')
    plt.clf()

    H1 = acc['EEC2']["DYJetsToLL",:,::sum,:].project('dR')[::rebin(2)]
    H2 = acc['EEC3']["DYJetsToLL",:,::sum,:].project('dR')[::rebin(2)]
    H3 = acc['EEC4']["DYJetsToLL",:,::sum,:].project('dR')[::rebin(2)]
    H4 = acc['EEC5']["DYJetsToLL",:,::sum,:].project('dR')[::rebin(2)]
    H5 = acc['EEC6']["DYJetsToLL",:,::sum,:].project('dR')[::rebin(2)]
    hist1 = H1.values()/H1.sum()
    hist2 = H2.values()/H2.sum()
    hist3 = H3.values()/H3.sum()
    hist4 = H4.values()/H4.sum()
    hist5 = H5.values()/H5.sum()
    midbins = H1.axes[0].centers
    binwidths = H1.axes[0].widths

    plt.errorbar(midbins, hist1/binwidths, fmt='o--',lw=1.5,markersize=2, label='2-point')
    plt.errorbar(midbins, hist2/binwidths, fmt='o--',lw=1.5,markersize=2, label='3-point')
    plt.errorbar(midbins, hist3/binwidths, fmt='o--',lw=1.5,markersize=2, label='4-point')
    plt.errorbar(midbins, hist4/binwidths, fmt='o--',lw=1.5,markersize=2, label='5-point')
    plt.errorbar(midbins, hist5/binwidths, fmt='o--',lw=1.5,markersize=2, label='6-point')
    plt.legend()
    plt.axvline(0.4, c='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("EECall.png", format='png')
    plt.clf()
    '''
    for N in range(2,7):
        H1 = acc['EEC%d'%N]["DYJetsToLL",:,::sum,:].project('dR')[::rebin(2)]
        H2 = acc['EEC%d'%N]["DoubleMuon",:,::sum,:].project('dR')[::rebin(2)]
        hist1 = H1.values()/H1.sum()
        hist2 = H2.values()/H2.sum()
        midbins = H1.axes[0].centers
        binwidths = H1.axes[0].widths

        plt.errorbar(midbins, hist1/binwidths, fmt='o',lw=1.5,markersize=2, label='MC')
        plt.errorbar(midbins, hist2/binwidths, fmt='o',lw=1.5,markersize=2, label='Data')
        plt.legend()
        plt.xlabel("$\Delta R$")
        plt.ylabel("Projected %d-point correlator"%N)
        plt.axvline(0.4, c='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig("figures/EEC%ddataMC.png"%N, format='png')
        plt.clf()
    
    for N in range(2,7):
        H1 = acc['EEC%d'%N]["DYJetsToLL",:,::sum,:].project('dR')[::rebin(2)]
        H2 = acc['genEEC%d'%N]["DYJetsToLL",:,::sum,:].project('dR')[::rebin(2)]
        hist1 = H1.values()/H1.sum()
        hist2 = H2.values()/H2.sum()
        midbins = H1.axes[0].centers
        binwidths = H1.axes[0].widths

        plt.errorbar(midbins, hist1/binwidths, fmt='o',lw=1.5,markersize=2, label='RECO')
        plt.errorbar(midbins, hist2/binwidths, fmt='o',lw=1.5,markersize=2, label='GEN')
        plt.legend()
        plt.xlabel("$\Delta R$")
        plt.ylabel("Projected %d-point correlator"%N)
        plt.axvline(0.4, c='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig("figures/EEC%dGenReco.png"%N, format='png')
        plt.clf()

    for N in range(2,7):
        H1 = acc['EEC%d'%N]["DYJetsToLL",:,::sum,:].project('dR')[::rebin(2)]
        #H2 = acc['EEC%d'%N]["DoubleMuon",:,::sum,:].project('dR')[::rebin(2)]
        hist1 = H1.values()/H1.sum()
        #hist2 = H2.values()/H2.sum()
        midbins = H1.axes[0].centers
        binwidths = H1.axes[0].widths

        plt.errorbar(midbins, hist1/binwidths, fmt='o',lw=1.5,markersize=2, label='%d-point'%N)
        #plt.errorbar(midbins, hist2/binwidths, fmt='o',lw=1.5,markersize=2, label='Data')
    plt.legend()
    plt.xlabel("$\Delta R$")
    plt.ylabel("Projected correlator")
    plt.axvline(0.4, c='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("figures/EECorder.png", format='png')
    plt.clf()

    for N in range(0,10):
        H1 = acc['EEC2']["DYJetsToLL",:,N:(N+1):sum,:].project('dR')[::rebin(2)]
        #H2 = acc['EEC%d'%N]["DoubleMuon",:,::sum,:].project('dR')[::rebin(2)]
        hist1 = H1.values()/H1.sum()
        #hist2 = H2.values()/H2.sum()
        midbins = H1.axes[0].centers
        binwidths = H1.axes[0].widths

        plt.errorbar(midbins, hist1/binwidths, fmt='o',lw=1.5,markersize=2, label='%d < pT < %d'%(N*100, (N+1)*100))
        #plt.errorbar(midbins, hist2/binwidths, fmt='o',lw=1.5,markersize=2, label='Data')
    plt.legend(loc='lower left')
    plt.xlabel("$\Delta R$")
    plt.ylabel("2-point correlator")
    plt.axvline(0.4, c='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("figures/EECpT.png", format='png')
    plt.clf()
