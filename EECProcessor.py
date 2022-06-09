import awkward as ak
import numpy as np
import json
from types import SimpleNamespace
from coffea.nanoevents.methods import vector, candidate
import fastjet

from coffea import hist, processor
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

class EECProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            "sumw": processor.defaultdict_accumulator(float),
            "mass": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$m_{\mu\mu}$ [GeV]", 60, 60, 120),
                hist.Bin("nMyJets", "$N_{Jets}$", 10, 0, 50),
                hist.Bin("nCmsJets", "$N_{Jets}$", 10, 0, 50)
            ),
            "myJets" : hist.Hist(
                "$N_{Jets}$",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_T$ [GeV]", 50, 0, 100),
                hist.Bin("npart", "N_{constituents}", 10, 0, 100),
                hist.Bin("eta", "$\eta", 10, -5, 5),
                hist.Bin("phi", "$\phi", 10, -np.pi, np.pi)
            ),
            "cmsswJets" : hist.Hist(
                "$N_{Jets}$",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_T$ [GeV]", 50, 0, 100),
                hist.Bin("npart", "$N_{constituents}$", 10, 0, 100),
                hist.Bin("eta", "$\eta$", 10, -5, 5),
                hist.Bin("phi", "$\phi$", 10, -np.pi, np.pi)
            )
        })

        #read in json config
        with open("config.json", 'r') as f:
            self.args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))

    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, data):
        output = self.accumulator.identity()

        dataset = data.metadata['dataset']
        
        #apply cuts...
        mask = np.ones(len(data), dtype=bool)
        
        for cut in self.args.evtCuts:
            if cut.minval == cut.maxval:
                mask = np.logical_and(mask, data[cut.var] == cut.minval)
            else:
                if cut.minval != -999:
                    mask = np.logical_and(mask, data[cut.var] >= cut.minval)
                if cut.maxval != -999:
                    mask = np.logical_and(mask, data[cut.var] <= cut.maxval)

        
        for cut in self.args.muCuts:
            if cut.minval == cut.maxval:
                mask = np.logical_and(mask, ak.all(data[cut.var] == cut.minval, axis=1))
            else:
                if cut.minval != -999:
                    mask = np.logical_and(mask, ak.all(data[cut.var] >= cut.minval, axis=1))
                if cut.maxval != -999:
                    mask = np.logical_and(mask, ak.all(data[cut.var] <= cut.maxval, axis=1))

        print("%d/%d (%0.2f%%) pass initial cuts"%(sum(mask), len(mask), 100*sum(mask)/len(mask)))
        data = data[mask]

        #z selection
        muons = ak.to_regular(ak.zip({
                "pt" : data[self.args.muons.pt],
                'eta' : data[self.args.muons.eta],
                'phi' : data[self.args.muons.phi],
                'mass' : data[self.args.muons.mass]
            },
            behavior = vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        ))

        mu1 = muons[:,0]
        mu2 = muons[:,1]

        mass = np.abs(mu1 + mu2)

        massWindow = np.logical_and(mass<self.args.Zsel.maxMass, mass>self.args.Zsel.minMass)
        charge = ak.prod(data[self.args.muons.charge],axis=1) == -1

        Zsel = np.logical_and.reduce((massWindow, charge))

        print("%d/%d (%0.2f%%) pass Z selection"%(sum(Zsel), len(Zsel), 100*sum(Zsel)/len(Zsel)))
        data = data[Zsel]
        mu1 = mu1[Zsel]
        mu2 = mu2[Zsel]
        mass = mass[Zsel]

        pfcands = ak.zip({
                "pt" : data[self.args.pfcands.pt] * data[self.args.pfcands.puppiWeight],
                'eta' : data[self.args.pfcands.eta],
                'phi' : data[self.args.pfcands.phi],
                'mass' : data[self.args.pfcands.mass],
            },
            behavior = candidate.behavior,
            with_name = 'PtEtaPhiMLorentzVector'
        )

        pfcands = pfcands[pfcands.pt > 0]

        pfcandmask = pfcands.pt > self.args.minPfCandPt
        pfcands = pfcands[pfcandmask]

        charge= data[self.args.pfcands.charge]

        #jet clustering
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, self.args.jetSize)
        cluster = fastjet.ClusterSequence(pfcands, jetdef)

        parts = cluster.constituents(self.args.minJetPt)
        
        #need to have the 4vector summation because ak.sum() doesn't work right for 4vecs
        jet4vec = ak.zip({
                'x' : ak.sum(parts.x, axis=-1),
                'y' : ak.sum(parts.y, axis=-1),
                'z' : ak.sum(parts.z, axis=-1),
                't' : ak.sum(parts.t, axis=-1)
            },
            behavior = vector.behavior,
            with_name = 'LorentzVector'
        )

        goodjets = np.logical_and(
            jet4vec.delta_r(mu1) > self.args.jetSize,
            jet4vec.delta_r(mu2) > self.args.jetSize)
        
        jet4vec = jet4vec[goodjets]
        parts = parts[goodjets]

        output["sumw"][dataset] += len(data)
        output["mass"].fill(
            dataset=dataset,
            mass=mass,
            nMyJets=ak.num(jet4vec),
            nCmsJets=ak.num(data['Jet_pt'])
        )
        output['myJets'].fill(
            dataset=dataset,
            pt=ak.flatten(jet4vec.pt, axis=None),
            eta=ak.flatten(jet4vec.eta, axis=None),
            phi=ak.flatten(jet4vec.phi, axis=None),
            npart=ak.flatten(ak.num(parts, axis=2), axis=None)
        )

        output['cmsswJets'].fill(
            dataset=dataset,
            pt=ak.flatten(data['Jet_pt'],axis=None),
            eta=ak.flatten(data['Jet_eta'], axis=None),
            phi=ak.flatten(data['Jet_phi'], axis=None),
            npart=ak.flatten(data["Jet_nConstituents"], axis=None)
        )

        return output

    def postprocess(self, accumulator):
        return accumulator