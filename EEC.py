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

