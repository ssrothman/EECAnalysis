from cmath import log
from attr import dataclass
import awkward as ak
import numpy as np
import json
from types import SimpleNamespace
from coffea.nanoevents.methods import vector, candidate
import hist
from .Histogram import Histogram

from coffea import processor

ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)

'''

Btagging resources

https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools
https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#b_tagging_efficiency_in_MC_sampl
https://coffeateam.github.io/coffea/api/coffea.btag_tools.BTagScaleFactor.html


'''




class LumiProcessor(processor.ProcessorABC):
    def __init__(self, datasets):
        #read in json config
        with open("config.json", 'r') as f:
            self.args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))
    
        #build accumulator
        self._accumulator = processor.dict_accumulator({
            "sumw": processor.defaultdict_accumulator(float),
        })

    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, data):
        output = self.accumulator.identity()

        dataset = data.metadata['dataset']
        
        #apply cuts...
        mask = np.ones(len(data), dtype=bool)
        
        isMC = hasattr(data, "ak4GenJetsNoNu_pt")
        if isMC:
            evtwt = data.Generator_weight
        else:
            evtwt = ak.ones_like(data.nselectedPatJetsAK4PFPuppi)
        output["sumw"][dataset] += ak.sum(evtwt)

        return output

    def postprocess(self, accumulator):
        return accumulator