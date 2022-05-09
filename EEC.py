import boost_histogram as bh
from time import time
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
    self._precomputeFactorial()

  def _precomputeFactorial(self):
    self._factorialArr = np.asarray( [factorial(i) for i in range(1, self.N+1)] )

  def _factorial(self, x):
    '''
    if type(x) is int:
      return self._factorialArr[x-1]
    elif type(x) is ak.Array:
      return ak.sum([ (x==i+1)*self._factorialArr[i] for i in range(self.N)], axis=0)
    else:
      raise ValueError("ProjectedEEC._factorial() only implemented for ints and ak.Arrays")
    '''
    '''
    return factorial(x)
    '''
    from ufunc.factorial.build.lib.npufunc_directory import npufunc
    return npufunc.factorial(x)

  def _wt(self, parts, jets, combos, N):
    print("computing weighting...")
    t0 = time()
    #jet energy weight
    jetWt = np.power(jets.pt, -N)
    print("\tjetWt took %0.3f seconds"%(time()-t0))

    t1 = time()
    #particle energy weight = product(pt_i)
    Ewt = 1
    names = [str(i) for i in range(N)]
    for name in names:
      Ewt = Ewt * parts.pt[combos[name]]
    print("\tEwt took %0.3f seconds"%(time()-t1))

    t2 = time()
    #combinatorics
    runs = ak.run_lengths(ak.concatenate([combos[name][:,:,:,None] for name in names], axis=-1))
    #would be nice to procompute factorials...
    comboWt = self._factorial(N)/ak.prod(self._factorial(runs), axis=-1)
    print("\tcomboWt took %0.3f seconds"%(time()-t2))

    print("Took %0.3f seconds"%(time()-t0))
    return jetWt, Ewt, comboWt

  def _maxDR(self, parts, N):
    print("Computing maxDR...")
    t0 = time()
    combos = ak.argcombinations(parts, N, replacement=True, axis=2)
    print("\tMaking combos took %0.3f seconds"%(time()-t0))
    names = [str(i) for i in range(N)]
    pairs = ak.combinations(names, 2, replacement=False, axis=0)

    t1 = time()
    maxDR = 0
    for pair in pairs:
      i = combos[pair['0']]
      j = combos[pair['1']]
      nextDR = parts[i].delta_r(parts[j])
      maxDR = ak.where(nextDR>maxDR, nextDR, maxDR)
    print("\tmaking maxDR took %0.3f seconds"%(time()-t1))

    print("Took %0.3f seconds"%(time()-t0))
    return combos, maxDR

  def __call__(self, parts, jets):
    combos, maxDR = self._maxDR(parts, self.N)
    jetWt, Ewt, comboWt = self._wt(parts, jets, combos, self.N)

    flatDR = ak.flatten(maxDR, axis=None)
    flatWt = ak.flatten(jetWt*Ewt*comboWt, axis=None)
    self.hist.fill(flatDR, weight=flatWt)
