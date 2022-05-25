if __package__ or "." in __name__:
  from . import eec_back as backend
else:
  import eec_back as backend
import awkward as ak
import numpy as np
from scipy.special import comb
from time import time
import boost_histogram as bh
from numbers import Number

class ProjectedEEC:
  '''
  Wrapper class for c++ backend computing energy-energy correlators (EECs) 
  Correlators of higher order than 2 are project onto the longest side of the N-simplex

  Recommended usage is to construct with at least one axis, in which case 
  a boost_histogram object is used to bin the correlators in dR, and optionally
  any other variables you care to pass.

  If you are binning with respect to additional variables, just pass them 
  to __call__() in the same order as you listed their axes. They will 
  automagically be broadcasted to the right shapes and filled into the histogram

  If you are binning, addition and multiplication have been overloaded to 
  allow sensible addition of two ProjectedEEC objects and multiplication 
  by scalars

  If axes is None, this wrapper will instead store the dR and weight arrays
  with no binning. The overloaded arithmetic operators may not behave properly 
  in this case.

  Examples:

  #bin only with respect to dR:
  eec = ProjectedEEC(3, axes = [bh.axis.Regular(bins=75, start=1e-5, stop=1.0, transform=bh.axis.transform.log)])
  eec(particles, jets)
  print(eec.hist)

  #bin also with respect to jet pT and eta:
  eec = ProjectedEEC(3, axes = [bh.axis.Regular(bins=75, start=1e-5, stop=1.0, transform=bh.axis.transform.log), bh.axis.Regular(bins=10, start=10, end=500), bh.axis.Regular(bins=10, start=-5, end=5)])
  eec(particles, jets, jets.pt, jets.eta)
  '''

  def __init__(self, N, axes=None, hist=None):
    '''
    N: correlator order
    axes: list of bh.axis objects to bin along
      If axes is None, don't bin and instead just store the dR and weight arrays
      If axies is not None, the first axis is the dR axis, 
        and any additional axes can be for arbitrary other variables (eg pt, eta, etc)
    hist: bh.Histogram object to use as the histogram
      If hist is supplied, axes is ignored
      Intended only for use by internal methods (ie __add__, etc)
    '''
    if N<2:
      raise ValueError("Correlator order must be at least 2")

    self.N = N
    if hist is not None:
      self.hist = hist
    elif axes is not None:
      self.hist = bh.Histogram(*axes)
    else:
      self.hist = None

    self.dRs = None
    self.wts = None

  def __call__(self, parts, jets, *bin_vars):
    '''
    Computed EECs and bin appropriately

    parts: awkward array of jet constituent 4-vectors. Axes (event, jet, particle)
    jets: awkward array of jet 4-vectors. Axes (event, jet)
    bin_vars: optional additional variables to bin against
      must be passed in the same order as their axes were passed to the constructor
      bin_vars will must be broadcast-compatible with the parts array, but the actual
        broadcasting will be handed behind the scenes by this method
    '''
    #call c++ backend to compute correlator values
    dRs, wts = projectedEEC_values(parts, jets, self.N)

    if self.hist is None: #if we're not filling a histogram
      if self.dRs is None:
        self.dRs = dRs
      else:
        self.dRs = ak.concatenate((self.dRs, dRs), axis=0)
    
      if self.wts is None:
        self.wts = wts
      else:
        self.wts = ak.concatenate((self.wts, wts), axis=0)
    else: #if we are filling a histogram
      if len(bin_vars) != self.hist.ndim-1:
        raise ValueError("Wrong number of binning variables: %d. Expected %d"%(len(bin_vars), self.hist.ndim-1))

      bin_vars = ak.broadcast_arrays(*(*bin_vars, dRs))[:-1]
      self.hist.fill(ak.flatten(dRs, axis=None), 
                      *[ak.flatten(var, axis=None) for var in bin_vars], 
                      weight=ak.flatten(wts, axis=None))

  def __add__(self, other):
    if type(other) is not ProjectedEEC or self.N != other.N:
      return NotImplemented

    return ProjectedEEC(self.N, hist=self.hist+other.hist)

  def __radd__(self, other):
    return self + other
  
  def __iadd__(self, other):
    if type(other) is not ProjectedEEC or self.N != other.N:
      return NotImplemented

    self.hist += other.hist
    return self

  def __mul__(self, other):
    if not isinstance(other, Number):
      return NotImplemented

    return ProjectedEEC(self.N, hist=self.hist*other)

  def __rmul__(self, other):
    return self * other

  def __imul__(self, other):
    if not isinstance(other, Number):
      return NotImplemented

    self.hist *= other
    return self

def projectedEEC_values(parts, jet4vec, N, verbose=False):
  '''
  Inputs:
    parts: awkward array of jet constituent 4-vectors
      shape (event, jet, particle)
    jet4vec: awkward array of jet 4-vectors
      shape (event, jet)

  Outputs (dRs, wts):
    dRs: pairwise delta R between particles within a given jet
    wts: EEC weights assigned to each delta R value

    Both have shape (event, jet), and are broadcast-compatible with:
      jet quantities (ie jet pT, eta, etc)
      event quantitites (ie event weights, etc)
  '''
  if type(N) is not int:
    print("Error: N must be an integer")
    return

  if N<2:
    print("Error: N must be >=2")
    return
  
  if N>10:
    print("Error: correlators larger than 10 are not jet supported")
    return


  if verbose:
    t0 = time()

  pts = np.asarray(ak.flatten(parts.pt/jet4vec.pt, axis=None)).astype(np.float32)
  etas = np.asarray(ak.flatten(parts.eta, axis=None)).astype(np.float32)
  phis = np.asarray(ak.flatten(parts.phi, axis=None)).astype(np.float32)
  nParts = np.asarray(ak.flatten(ak.num(parts,axis=-1), axis=None)).astype(np.int32)
  jetIdxs = np.cumsum(nParts).astype(np.int32)
  nDRs = comb(nParts, 2).astype(np.int32)
  dRIdxs = np.cumsum(nDRs).astype(np.int32)
  nDR = int(dRIdxs[-1])
  jets = np.column_stack((pts, etas, phis))

  nJets = ak.num(parts, axis=1)

  if verbose:
    print('setting up inputs took %0.3f seconds'%(time()-t0))
    t0 = time()

  dRs, wts = backend.eec(jets, jetIdxs, N, nDR, nDR, dRIdxs)
  dRs = np.sqrt(dRs)
  
  if verbose:
    print("computing EEC values took %0.3f seconds"%(time()-t0))
    t0 = time()

  dRs = ak.unflatten(dRs, nDRs)
  wts = ak.unflatten(wts, nDRs)
  
  dRs = ak.unflatten(dRs, nJets)
  wts = ak.unflatten(wts, nJets)

  if verbose:
    print("unflattening took %0.3f seconds"%(time()-t0))

  return dRs, wts
