import matplotlib.pyplot as plt
from hist import rebin, loc, sum, underflow, overflow
import pickle
import numpy as np

with open("output/hists_3.pickle", 'rb') as f:
    acc = pickle.load(f)

with open("output/hists_1.pickle", 'rb') as f:
    accCHS = pickle.load(f)

with open("output/hists_2.pickle", 'rb') as f:
    accCS = pickle.load(f)

with open("output/hists.pickle", 'rb') as f:
    accMU = pickle.load(f)


H1 = acc['EEC2']["DYJetsToLL",:,4:7:sum,::sum].project('dR')[::rebin(2)]
hist1 = H1.values()/H1.sum()
midbins = H1.axes[0].centers
#binwidths = H1.axes[0].widths
binedges = H1.axes[0].edges
binwidths = np.log(binedges[1:]) - np.log(binedges[:-1])

H2 = accCHS['EEC2']["DYJetsToLL",:,4:7:sum,::sum].project('dR')[::rebin(2)]
hist2 = H2.values()/H2.sum()

H3 = accCS['EEC2']["DYJetsToLL",:,4:7:sum,::sum].project('dR')[::rebin(2)]
hist3 = H3.values()/H3.sum()

H4 = accMU['EEC2']["DYJetsToLL",:,4:7:sum,::sum].project('dR')[::rebin(2)]
hist4 = H4.values()/H4.sum()

plt.errorbar(midbins, hist1/binwidths, fmt='o',lw=1.5,markersize=2, label='PUPPI')
plt.errorbar(midbins, hist2/binwidths, fmt='o',lw=1.5,markersize=2, label='CHS')
plt.errorbar(midbins, hist3/binwidths, fmt='o',lw=1.5,markersize=2, label='CS')
plt.errorbar(midbins, hist4/binwidths, fmt='o',lw=1.5,markersize=2, label='PUPPI, no Muon veto')
plt.legend()
plt.xlabel("$\Delta R$")
plt.ylabel("Projected %d-point correlator"%2)
plt.axvline(0.4, c='k')
plt.xscale('log')
plt.yscale('log')
plt.xlim(left=1e-4)
plt.ylim(bottom=1e-2)
plt.savefig("figures/EEC2CHS.png", format='png')
plt.clf()