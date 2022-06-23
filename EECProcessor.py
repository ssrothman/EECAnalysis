from cmath import log
from attr import dataclass
import awkward as ak
import numpy as np
import json
from types import SimpleNamespace
from coffea.nanoevents.methods import vector, candidate
import hist
from Histogram import Histogram

from coffea import processor

ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)


class EECProcessor(processor.ProcessorABC):
    def __init__(self):
        #read in json config
        with open("config.json", 'r') as f:
            self.args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))
    
        #build accumulator
        self._accumulator = processor.dict_accumulator({
            "sumw": processor.defaultdict_accumulator(float),
            "nJets": Histogram(
                hist.axis.Regular(10, 0, 10, name="nJets", label="nJets")
            ),
            "jets": Histogram(
                hist.axis.Regular(100, 0, 1000, name='pT', label='$p_T$'),
                hist.axis.Regular(100, -5, 5, name='eta', label='$\eta$'),
                hist.axis.Regular(100, -np.pi, np.pi, name='phi', label='$\phi$')
            ),
            "EEC2": Histogram(
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(100, 0, 1000, name='pT', label='$p_T$'),
                hist.axis.Regular(100, 0, 5, name='eta', label='$\eta$')
            ),
            "otherEEC":Histogram(
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
            ),
            "dimuon":Histogram(
                hist.axis.Regular(100, self.args.Zsel.minMass, self.args.Zsel.maxMass, name='mass', label='$m_{\mu\mu}$'),
                hist.axis.Regular(100, 0, 1000, name='pT', label='$p_{T,\mu\mu}$'),
                hist.axis.Regular(100, -5, 5, name='eta', label='$\eta_{\mu\mu}$'),
                hist.axis.Regular(100,-np.pi,np.pi,name='phi', label='$\phi_{\mu\mu}$')
            ),
            "muon":Histogram(
                hist.axis.Regular(100, 0, 1000, name='pT', label='$p_{T,\mu\mu}$'),
                hist.axis.Regular(100, -5, 5, name='eta', label='$\eta_{\mu\mu}$'),
                hist.axis.Regular(100,-np.pi,np.pi,name='phi', label='$\phi_{\mu\mu}$')
            )
        })

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

        Z = mu1+mu2

        Z.mass

        massWindow = np.logical_and(Z.mass<self.args.Zsel.maxMass, Z.mass>self.args.Zsel.minMass)
        charge = ak.prod(data[self.args.muons.charge],axis=1) == -1

        Zsel = np.logical_and.reduce((massWindow, charge))

        print("%d/%d (%0.2f%%) pass Z selection"%(sum(Zsel), len(Zsel), 100*sum(Zsel)/len(Zsel)))
        data = data[Zsel]
        mu1 = mu1[Zsel]
        mu2 = mu2[Zsel]
        Z = Z[Zsel]

        jets = ak.zip({
            "pt" : data.selectedPatJetsAK4PFPuppi_pt,
            'eta' : data.selectedPatJetsAK4PFPuppi_eta,
            "phi" : data.selectedPatJetsAK4PFPuppi_phi,
            'mass' : data.selectedPatJetsAK4PFPuppi_mass
            },
            behavior=vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )
        
        #veto jets that are too close to the muons
        goodjets = np.logical_and(jets.delta_r(mu1)>self.args.jetSize, jets.delta_r(mu2)>self.args.jetSize)
        jets = jets[goodjets]

        output["sumw"][dataset] += len(data)
        output['nJets'].fill(
            ak.flatten(ak.num(jets.pt), axis=None),
            weight=ak.flatten(data.Generator_weight,axis=None)
        )

        castedweight = ak.broadcast_arrays(data.Generator_weight, jets.pt)[0]
        output['jets'].fill(
            pT = ak.flatten(jets.pt, axis=None),
            eta = ak.flatten(jets.eta, axis=None),
            phi = ak.flatten(jets.phi, axis=None),
            weight=ak.flatten(castedweight, axis=None)
        )

        EEC2_dRs = data.EEC2_dRs
        EEC2_wts = data.EEC2_wts
        EEC2_jetIdx = data.EEC2_jetIdx

        goodEEC = goodjets[EEC2_jetIdx]
        EEC2_dRs = EEC2_dRs[goodEEC]
        EEC2_wts = EEC2_wts[goodEEC]
        EEC2_jetIdx = EEC2_jetIdx[goodEEC]

        jetPt = data.selectedPatJetsAK4PFPuppi_pt
        jetPt = jetPt[EEC2_jetIdx]

        jetEta = np.abs(data.selectedPatJetsAK4PFPuppi_eta)
        jetEta = jetEta[EEC2_jetIdx]


        output['EEC2'].fill(
            ak.flatten(EEC2_dRs, axis=None), 
            weight=ak.flatten(EEC2_wts * data.Generator_weight, axis=None),
            pT = ak.flatten(jetPt, axis=None),
            eta = ak.flatten(jetEta, axis=None)
        )

        castedweight = ak.broadcast_arrays(data.Generator_weight, Z.pt)[0]
        output['dimuon'].fill(
            mass = Z.mass,
            pT = Z.pt,
            eta = Z.eta,
            phi = Z.phi,
            weight=castedweight
        )

        output['muon'].fill(
            pT = mu1.pt,
            eta = mu1.eta,
            phi = mu1.phi
        )

        output['muon'].fill(
            pT = mu2.pt,
            eta = mu2.eta,
            phi = mu2.phi
        )

        return output

    def postprocess(self, accumulator):
        return accumulator