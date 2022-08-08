import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector
import uproot
from scipy.special import comb
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
import matplotlib.pyplot as plt
import hist
from processing.roccor import kScaleDT
from coffea import lookup_tools

import correctionlib
from correctionlib import schemav2 as schema
from coffea.lookup_tools.txt_converters import convert_rochester_file
import correctionlib._core as core
import numpy as np
from coffea.lookup_tools.rochester_lookup import rochester_lookup

dt = convert_rochester_file('corrections/roccor/RoccoR2017UL.txt')
coffeaRC = rochester_lookup(dt)

def wrap(*corrs):
    cset = schema.CorrectionSet(
        schema_version=schema.VERSION,
        corrections=list(corrs),
    )
    return correctionlib.CorrectionSet.from_string(cset.json())

Ms = dt['values']['M'][0][0][1].flatten()
As = dt['values']['A'][0][0][1].flatten()

cset = wrap(
    schema.Correction(
        name = "kScaleDT",
        version = 2,
        inputs = [
            schema.Variable(name='Q', type='real'),
            schema.Variable(name='pt', type='real'),
            schema.Variable(name='eta', type='real'),
            schema.Variable(name='phi', type='real'),
        ],
        generic_formulas=[
            schema.Formula(
                nodetype='formula',
                expression="1.0 / ([0] + x * [1] * y)",
                parser="TFormula", 
                variables=["Q","pt"]
            )
        ],
        output = schema.Variable(name = 'muon pt sf', type='real'),
        data=schema.MultiBinning(
            nodetype='multibinning',
            inputs=['eta', 'phi'],
            edges=[dt['edges']['scales'][i].tolist() for i in range(2)],
            content=
                [schema.FormulaRef(
                    nodetype='formularef',
                    index=0,
                    parameters=[Ms[i], As[i]]
                ) for i in range(len(Ms))],
            flow='error'
        )
    )
)

rochester_data = lookup_tools.txt_converters.convert_rochester_file(
    "corrections/roccor/RoccoR2017UL.txt", loaduncs=True)
rochester = lookup_tools.rochester_lookup.rochester_lookup(rochester_data)

from time import time

N = 100000
N2 = 1000

tUFUNC=0
tCOFFEA=0
tCORRECTIONLIB=0
badmatches1 = 0
badmatches2 = 0
for i in range(N2):
  charge = np.random.randint(0, 2, N)
  pt = np.random.random(N)*460+40
  eta = np.random.random(N)*4.8 - 2.4
  phi = np.random.random(N)*2*np.pi - np.pi

  t0 = time()
  x = kScaleDT(charge, pt, eta, phi, 0, 0)
  tUFUNC += time()-t0

  t0 = time()
  y = rochester.kScaleDT(charge, pt, eta, phi)
  tCOFFEA += time()-t0

  charge = charge.astype(np.float64)
  t0 = time()
  z = cset['kScaleDT'].evaluate(charge, pt, eta, phi)
  tCORRECTIONLIB += time()-t0

  badmatches1 += np.sum(~np.isclose(y, z, atol = 1e-5))
  badmatches2 += np.sum(~np.isclose(y, x, atol = 1e-5))

print("coffea implementation took %0.3f seconds"%tCOFFEA)
print("npy ufunc took %0.3f seconds"%tUFUNC)
print("\tufunc disagreed with coffea %d times (%0.2f%%)"%(badmatches2, badmatches1/N/N2*100))
print("correctionlib implementation took %0.3f seconds"%tCORRECTIONLIB)
print("\tcorrectionlib disagreed with coffea %d times (%0.2f%%)"%(badmatches1, badmatches2/N/N2*100))
