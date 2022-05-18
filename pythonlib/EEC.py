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

class ProjectedEEC_naive:
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

class ProjectedEEC_optv1:
  def __init__(self, N, bins, axisMin, axisMax):
    self.N = N
    self.hist = bh.Histogram(bh.axis.Regular(bins=bins, start=axisMin, stop=axisMax, transform=bh.axis.transform.log))
  
    self._precomputeSymmetry()
  
  def _precomputeSymmetry(self):
    partitions = ak.Array(list(getPartitions(self.N)))
    num = ak.num(partitions)

    self._partitions = {}
    for i in range(1,self.N+1):
      self._partitions[i] = partitions[num==i]

    numerator = self._factorial(self.N)
    self._symFactors = {}
    for i in range(1, self.N+1):
      self._symFactors[i] = []
      for partition in self._partitions[i]:
        denominator = ak.prod(self._factorial(partition), axis=0)
        self._symFactors[i].append(numerator/denominator)

  def _factorial(self, x):
    from ufunc.factorial.build.lib.npufunc_directory.npufunc import factorial
    return factorial(x)

  def _wt(self, parts, jets, combos, N):
    M = len(combos.fields) #number of distinct particles in the correlator
    print("computing weighting for M=%d..."%M)
    t0 = time()

    names = [str(i) for i in range(M)]

    t1 = time()
    #particle energy weight = product(pt_i)
    Ewt = 0
    for partition, symFactor in zip(self._partitions[M], self._symFactors[M]):
      partWt = 0
      for composition in mit.distinct_permutations(partition):
        print("\tconsidering",composition,"...")
        compWt = 1
        for idx, power in enumerate(composition):
          compWt = compWt * np.power(parts.pt[combos[names[idx]]], power)
        partWt = partWt + compWt
      partWt = partWt * symFactor
      Ewt = Ewt + partWt
    print("\tEwt took %0.3f seconds"%(time()-t1))

    print("Took %0.3f seconds"%(time()-t0))
    return Ewt

  def _maxDR(self, parts, M):
    print("Computing maxDR for M=%d..."%M)
    t0 = time()
    combos = ak.argcombinations(parts, M, replacement=False, axis=2)
    print("\tMaking combos took %0.3f seconds"%(time()-t0))
    print("\tAnd they are %d bytes"%combos.layout.nbytes)
    names = [str(i) for i in range(M)]
    pairs = ak.combinations(names, 2, replacement=False, axis=0)

    t1 = time()
    maxDR = 0
    for pair in pairs:
      i = combos[pair['0']]
      j = combos[pair['1']]
      nextDR = parts[i].delta_r(parts[j])
      maxDR = ak.where(nextDR>maxDR, nextDR, maxDR)
      del nextDR
    print("\tmaking maxDR took %0.3f seconds"%(time()-t1))


    print("Took %0.3f seconds"%(time()-t0))
    return combos, maxDR

  def __call__(self, parts, jets):

    #jet energy weight
    jetWt = np.power(jets.pt, -self.N)

    for M in range(2, self.N+1):
      combos, maxDR = self._maxDR(parts, M)
      #combination-specific weights
      Ewt = self._wt(parts, jets, combos, M)

      flatDR = ak.flatten(maxDR, axis=None)
      flatWt = ak.flatten(jetWt*Ewt, axis=None)
      self.hist.fill(flatDR, weight=flatWt)

class ProjectedEEC_optv2:
  def __init__(self, N, bins, axisMin, axisMax):
    self.N = N
    self.hist = bh.Histogram(bh.axis.Regular(bins=bins, start=axisMin, stop=axisMax, transform=bh.axis.transform.log))
  
    self._precomputeSymmetry()
  
  def _precomputeSymmetry(self):
    partitions = ak.Array(list(getPartitions(self.N)))
    num = ak.num(partitions)

    self._partitions = {}
    for i in range(1,self.N+1):
      self._partitions[i] = partitions[num==i]

    numerator = self._factorial(self.N)
    self._symFactors = {}
    for i in range(1, self.N+1):
      self._symFactors[i] = []
      for partition in self._partitions[i]:
        denominator = ak.prod(self._factorial(partition), axis=0)
        self._symFactors[i].append(numerator/denominator)

  def _factorial(self, x):
    from ufunc.factorial.build.lib.npufunc_directory.npufunc import factorial
    return factorial(x)

  def _wt(self, parts, jets, combos, N):
    M = len(combos.fields) #number of distinct particles in the correlator
    print("computing weighting for M=%d..."%M)
    t0 = time()

    names = [str(i) for i in range(M)]

    t1 = time()
    #particle energy weight = product(pt_i)
    Ewt = 0
    for partition, symFactor in zip(self._partitions[M], self._symFactors[M]):
      partWt = 0
      for composition in mit.distinct_permutations(partition):
        print("\tconsidering",composition,"...")
        compWt = 1
        for idx, power in enumerate(composition):
          compWt = compWt * np.power(parts.pt[combos[names[idx]]], power)
        partWt = partWt + compWt
      partWt = partWt * symFactor
      Ewt = Ewt + partWt
    print("\tEwt took %0.3f seconds"%(time()-t1))

    print("Took %0.3f seconds"%(time()-t0))
    return Ewt

  def _maxDR(self, parts, localIdx, M): 
    print("Computing maxDR for M=%d..."%M)
    t0 = time()
    combos = ak.combinations(localIdx, M, replacement=False, axis=2)
    print("\tMaking combos took %0.3f seconds"%(time()-t0))
    print("\tAnd they are %d bytes"%combos.layout.nbytes)

    names = [str(i) for i in range(M)]
    pairs = ak.combinations(names, 2, replacement=False, axis=0)

    t1 = time()
    maxDR = 0
    for pair in pairs:
      i = combos[pair['0']]
      j = combos[pair['1']]
      nextDR = parts[i].delta_r(parts[j])
      maxDR = ak.where(nextDR>maxDR, nextDR, maxDR)
      del nextDR
    print("\tmaking maxDR took %0.3f seconds"%(time()-t1))

    print("Took %0.3f seconds"%(time()-t0))
    return combos, maxDR

  def __call__(self, parts, jets):
    localIdx = ak.local_index(parts)
    localIdx = ak.values_astype(localIdx, "uint8") #unsigned 16-bit intengers. Maxval 255

    #jet energy weight
    jetWt = np.power(jets.pt, -self.N)

    for M in range(2, self.N+1):
      combos, maxDR = self._maxDR(parts, localIdx, M)
      #combination-specific weights
      Ewt = self._wt(parts, jets, combos, M)

      flatDR = ak.flatten(maxDR, axis=None)
      flatWt = ak.flatten(jetWt*Ewt, axis=None)
      self.hist.fill(flatDR, weight=flatWt)

##########################################################################################

def idx1(i, n):
  return i

def idx2(i, j, n):
  return tri(n-1) - tri(n-i-1) + idx1(j-i-1, n-i-1)

def idx3(i, j, k, n):
  return pyr(n-2) - pyr(n-i-2) + idx2(j-i-1, k-i-1, n-i-1) 

def idx4(i, j, k, l, n):
  return simp4(n-3) - simp4(n-i-3) + idx3(j-i-1, k-i-1, l-i-1, n-i-1)

def idxD(indices, D):
  indices = np.asarray(indices)
  i = indices[0]
  n = indices[-1]
  if D == 1:
    return indices[0]
  else:
    return simpD(n-D+1, D) - simpD(n-i-D+1, D) + idxD(indices[1:]-i-1, D-1)

def simpD(n, D):
  '''
  Simplectic in D dimensions
  ie: 
    simpN(n, 2) are triangular numbers
    simpN(n, 3) are pyramidal numbers
    etc
  '''
  ans = 1
  for i in range(D):
    ans = ans*(n+i)
  return ans // int(factorial(D))

def simp4(n):
  return n * (n+1) * (n+2) * (n+3) // 24

def pyr(n):
  return n * (n+1) * (n+2) // 6

def tri(n):
  return n * (n+1) // 2

def pt(n):
  return n

ProjectedEEC = ProjectedEEC_optv1
