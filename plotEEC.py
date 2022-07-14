import matplotlib.pyplot as plt
from hist import rebin, loc, sum, underflow, overflow
import pickle
import numpy as np

with open("output/Full07-12-2022.hist.pickle", 'rb') as f:
    acc = pickle.load(f)

low=8
high=14

EEC2 = acc['DYJetsToLL']['EEC2']

H0 = EEC2[:,low:high:sum,:,:].project('dR')[::rebin(2)]
hist0 = H0.values()/H0.sum()

H1 = EEC2[:,low:high:sum,:,0].project('dR')[::rebin(2)]
hist1 = H1.values()/H1.sum()
midbins = H1.axes[0].centers
binedges = H1.axes[0].edges
binwidths = np.log(binedges[1:]) - np.log(binedges[:-1])

H2 = EEC2[:,low:high:sum,:,1].project('dR')[::rebin(2)]
hist2 = H2.values()/H2.sum()

H3 = EEC2[:,low:high:sum,:,2].project('dR')[::rebin(2)]
hist3 = H3.values()/H3.sum()

H4 = EEC2[:,low:high:sum,:,3].project('dR')[::rebin(2)]
hist4 = H4.values()/H4.sum()

H5 = EEC2[:,low:high:sum,:,4].project('dR')[::rebin(2)]
hist5 = H5.values()/H5.sum()

plt.errorbar(midbins, hist0/binwidths, fmt='o-',lw=1,markersize=3, label='Inclusive')
#plt.errorbar(midbins, hist1/binwidths, fmt='o-',lw=1.5,markersize=5, label='No gen flavor')
plt.errorbar(midbins, hist2/binwidths, fmt='o-',lw=1,markersize=3, label='Light flavor')
plt.errorbar(midbins, hist3/binwidths, fmt='o-',lw=1,markersize=3, label='C jet')
plt.errorbar(midbins, hist4/binwidths, fmt='o-',lw=1,markersize=3, label='B jet')
plt.errorbar(midbins, hist5/binwidths, fmt='o-',lw=1,markersize=3, label='Gluon jet')
plt.legend()
plt.xlabel("$\Delta R$")
plt.ylabel("$d\sigma_2/d \log\Delta R$")
plt.axvline(0.4, c='k')
plt.xscale('log')
plt.yscale('log')
plt.title("%d < pT < %d"%(low*50, high*50))
plt.xlim(left=1e-3)
plt.ylim(bottom=1e-3)
plt.savefig("figures/EECgenFlavor.png", format='png')
plt.clf()

EEC2 = acc['DYJetsToLL']['EEC2']

H0 = EEC2[:,low:high:sum,0,:].project('dR')[::rebin(2)]
hist0 = H0.values()/H0.sum()

H0 = EEC2[:,low:high:sum,0,:].project('dR')[::rebin(2)]
hist1 = H1.values()/H1.sum()
midbins = H1.axes[0].centers
binedges = H1.axes[0].edges
binwidths = np.log(binedges[1:]) - np.log(binedges[:-1])

H2 = EEC2[:,low:high:sum,:,1].project('dR')[::rebin(2)]
hist2 = H2.values()/H2.sum()

H3 = EEC2[:,low:high:sum,:,2].project('dR')[::rebin(2)]
hist3 = H3.values()/H3.sum()

H4 = EEC2[:,low:high:sum,:,3].project('dR')[::rebin(2)]
hist4 = H4.values()/H4.sum()

H5 = EEC2[:,low:high:sum,:,4].project('dR')[::rebin(2)]
hist5 = H5.values()/H5.sum()

plt.errorbar(midbins, hist0/binwidths, fmt='o-',lw=1,markersize=3, label='Inclusive')
#plt.errorbar(midbins, hist1/binwidths, fmt='o-',lw=1.5,markersize=5, label='No gen flavor')
plt.errorbar(midbins, hist2/binwidths, fmt='o-',lw=1,markersize=3, label='Light flavor')
plt.errorbar(midbins, hist3/binwidths, fmt='o-',lw=1,markersize=3, label='C jet')
plt.errorbar(midbins, hist4/binwidths, fmt='o-',lw=1,markersize=3, label='B jet')
plt.errorbar(midbins, hist5/binwidths, fmt='o-',lw=1,markersize=3, label='Gluon jet')
plt.legend()
plt.xlabel("$\Delta R$")
plt.ylabel("$d\sigma_2/d \log\Delta R$")
plt.axvline(0.4, c='k')
plt.xscale('log')
plt.yscale('log')
plt.title("%d < pT < %d"%(low*50, high*50))
plt.xlim(left=1e-3)
plt.ylim(bottom=1e-3)
plt.savefig("figures/EEC2CHS.png", format='png')
plt.clf()