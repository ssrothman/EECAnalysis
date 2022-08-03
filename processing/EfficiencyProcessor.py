from cmath import log
from attr import dataclass
import awkward as ak
import numpy as np
import json
from types import SimpleNamespace
from coffea.nanoevents.methods import vector, candidate
import hist
from vector import dim
from .Histogram import Histogram

from coffea import processor
from coffea.lookup_tools import extractor, evaluator

ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)


class EECProcessor(processor.ProcessorABC):
    def __init__(self, args):
        self.args = args
    
    def process(self, data):
        #muons...
        muons = data[self.args.muons]
        
        genParts = data[self.args.genParts]
        genMuons = genParts[np.abs(genParts.pdgId) == 13]
        
        #trigger efficiency
        passTrigger = data[self.args.triggers[0]] == 1

        #fill histogram of nMuons passing trigger by n true muons

        #apply trigger
        data = data[passTrigger]
        muons = muons[passTrigger]
        genMuons = genMuons[passTrigger]

        #reco efficiency
        #fill histogram with nMuons in reco collection by nMuons in gen collection

        #id efficiency
        passId = muons.tightId == 1

        #fill histogram with nMuons passing tightId by n real muons



        jets = data[self.args.jets]


