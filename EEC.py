import boost_histogram as bh
import numpy as np
import awkward as ak
import mplhep
import more_itertools as mit
from scipy.special import factorial

def getPartitions(n):
  '''
  Generate all partitions of integer N
  Code from https://jeromekelleher.net/generating-integer-partitions.html
  Might need to cite this?

  Afaik this is the fastest known algorithm to generate partitions
  '''
  a = [0 for i in range(n + 1)]
  k = 1
  y = n - 1
  while k != 0:
    x = a[k - 1] + 1
    k -= 1
    while 2 * x <= y:
      a[k] = x
      y -= x
      k += 1
    l = k + 1
    while x <= y:
      a[k] = x
      a[l] = y
      yield a[:k + 2]
      x += 1
      y -= 1
    a[k] = x + y
    y = x + y - 1
    yield a[:k + 1]

class ProjectedEEC:
  def __init__(self, N, bins, axisMin, axisMax):
    self.N = N
    self.hist = bh.Histogram(bh.axis.Regular(bins=bins, start=axisMin, stop=axisMax, transform=bh.axis.transform.log))

  @staticmethod
  def _wt(parts, jets, combos, N):
    #jet energy weight
    jetWt = np.power(jets.pt, -N)

    #particle energy weight = product(pt_i)
    Ewt = 1
    names = [str(i) for i in range(N)]
    for name in names:
      Ewt = Ewt * parts.pt[combos[name]]

    #combinatorics
    runs = ak.run_lengths(ak.concatenate([combos[name][:,:,:,None] for name in names], axis=-1))
    comboWt = factorial(N)/ak.prod(factorial(runs), axis=-1)

    return jetWt, Ewt, comboWt

  @staticmethod
  def _maxDR(parts, N):
    combos = ak.argcombinations(parts, N, replacement=True, axis=2)
    names = [str(i) for i in range(N)]
    pairs = ak.combinations(names, 2, replacement=False, axis=0)

    maxDR = 0
    for pair in pairs:
      i = combos[pair['0']]
      j = combos[pair['1']]
      nextDR = parts[i].delta_r(parts[j])
      maxDR = ak.where(nextDR>maxDR, nextDR, maxDR)

    return combos, maxDR

  def __call__(self, parts, jets):
    combos, maxDR = self._maxDR(parts, self.N)
    jetWt, Ewt, comboWt = self._wt(parts, jets, combos, self.N)

    flatDR = ak.flatten(maxDR, axis=None)
    flatWt = ak.flatten(jetWt*Ewt*comboWt, axis=None)
    self.hist.fill(flatDR, weight=flatWt)
