import pickle
from hist import rebin, loc, sum, underflow, overflow

with open("out.pickle", 'rb') as f:
    acc = pickle.load(f)

import matplotlib.pyplot as plt

doJets = False
doEEC = False
doZ = True
doMu = True

if doMu:
    H = acc['muon']
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    H.project('pT')[:20].plot(ax=ax1)
    H.project('eta')[::rebin(2)].plot(ax=ax2)
    H.project('phi')[::rebin(2)].plot(ax=ax3)
    plt.tight_layout()
    plt.savefig("muon.png",format='png')
    plt.clf()

if doZ:
    H = acc['dimuon']
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    H.project('mass').plot(ax=ax0)
    H.project('pT')[:20].plot(ax=ax1)
    H.project('eta')[::rebin(2)].plot(ax=ax2)
    H.project('phi')[::rebin(2)].plot(ax=ax3)
    plt.tight_layout()
    plt.savefig("dimuon.png",format='png')
    plt.clf()

if doJets:
    jetHist = acc['jets']

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

    jetHist.project('pT')[:20:rebin(1)].plot(ax=ax0)
    jetHist.project('eta')[::rebin(2)].plot(ax=ax1)
    jetHist.project('phi')[::rebin(2)].plot(ax=ax2)
    acc['nJets'].plot(ax=ax3) 

    plt.tight_layout()
    plt.savefig("jets.png", format='png')
    plt.clf()

if doEEC:
    H1 = acc['EEC2'][:,:10:sum,:].project('dR')
    midbins = H1.axes[0].centers
    binwidths = H1.axes[0].widths
    hist1 = H1.values()/H1.sum()

    H2 = acc['EEC2'][:,10:20:sum,:].project('dR')
    hist2 = H2.values()/H2.sum()

    H3 = acc['EEC2'][:,20:30:sum,:].project('dR')
    hist3 = H3.values()/H3.sum()

    H4 = acc['EEC2'][:,30:40:sum,:].project('dR')
    hist4 = H4.values()/H4.sum()

    H5 = acc['EEC2'][:,40::sum,:].project('dR')
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