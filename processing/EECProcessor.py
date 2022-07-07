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

'''

Btagging resources

https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools
https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#b_tagging_efficiency_in_MC_sampl
https://coffeateam.github.io/coffea/api/coffea.btag_tools.BTagScaleFactor.html


'''

'''
Trigger resources

https://twiki.cern.ch/twiki/bin/view/CMS/TopTrigger
https://twiki.cern.ch/twiki/bin/viewauth/CMS/MuonHLT2017
'''




class EECProcessor(processor.ProcessorABC):
    def __init__(self, datasets):
        #read in json config
        with open("config.json", 'r') as f:
            self.args = json.load(f, object_hook = lambda x : SimpleNamespace(**x))
    
        #build accumulator
        self._accumulator = processor.dict_accumulator({
            "sumw": processor.defaultdict_accumulator(float),
            "nJets": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(10, 0, 10, name="nJets", label="nJets"),
            ),
            "jets": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                #hist.axis.Regular(10, 0, 1, name='JEC', label='$\frac{\sum p_{T,Constituents}}{p_{T,Jet}}$'),
                hist.axis.Regular(100, -5, 5, name='eta', label='$\eta_{Jet}$'),
                hist.axis.Regular(10, -np.pi, np.pi, name='phi', label='$\phi_{Jet}$')
            ),
            "EEC2": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "genEEC2": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "EEC3": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "genEEC3": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "EEC4": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "genEEC4": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "EEC5": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "genEEC5": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "EEC6": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "genEEC6": Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                hist.axis.Regular(10, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
            ),
            "dimuon":Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, self.args.Zsel.minMass, self.args.Zsel.maxMass, name='mass', label='$m_{\mu\mu}$'),
                hist.axis.Regular(100, 0, 1000, name='pT', label='$p_{T,\mu\mu}$'),
                hist.axis.Regular(100, -5, 5, name='y', label='$y{\mu\mu}$'),
                hist.axis.Regular(10,-np.pi,np.pi,name='phi', label='$\phi_{\mu\mu}$')
            ),
            "muon":Histogram(
                hist.axis.StrCategory(datasets, name='dataset'),
                hist.axis.Regular(100, 0, 1000, name='pT', label='$p_{T,\mu}$'),
                hist.axis.Regular(100, -5, 5, name='eta', label='$\eta_{\mu}$'),
                hist.axis.Regular(10,-np.pi,np.pi,name='phi', label='$\phi_{\mu}$')
            )
        })

        ext = extractor()
        ext.add_weight_sets([
                            "ID NUM_TightID_DEN_genTracks_pt_abseta corrections/muonSF/RunBCDEF_SF_ID_syst.histo.root",
                            "IDstat NUM_TightID_DEN_genTracks_pt_abseta_stat corrections/muonSF/RunBCDEF_SF_ID_syst.histo.root",
                            "IDsyst NUM_TightID_DEN_genTracks_pt_abseta_syst corrections/muonSF/RunBCDEF_SF_ID_syst.histo.root",
                            
                            "ISO NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta corrections/muonSF/RunBCDEF_SF_ISO_syst.histo.root",
                            "ISOstat NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta_stat corrections/muonSF/RunBCDEF_SF_ISO_syst.histo.root",
                            "ISOsyst NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta_syst corrections/muonSF/RunBCDEF_SF_ISO_syst.histo.root",

                            "TRG IsoMu27_PtEtaBins/pt_abseta_ratio corrections/muonSF/EfficienciesAndSF_RunBtoF_Nov17Nov2017.histo.root"
            ])
        ext.finalize()

        self.SFeval = ext.make_evaluator()

    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, data):
        output = self.accumulator.identity()

        dataset = data.metadata['dataset']
        isMC = hasattr(data, "Generator_weight")

        #apply cuts...
        mask = np.ones(len(data), dtype=bool)
       
        if isMC:
            genParts = ak.zip({
                    "pt" : data.GenCands_pt,
                    'eta' : data.GenCands_eta,
                    'phi' : data.GenCands_phi,
                    'mass' : data.GenCands_mass
                },
                behavior = vector.behavior,
                with_name='PtEtaPhiMLorentzVector'
            )

            genId = data.GenCands_pdgId

            mu1 = genParts[genId==13]
            mu2 = genParts[genId==-13]

            mu1 = mu1[ak.argsort(mu1.pt, ascending=False, axis=-1)]
            mu2 = mu2[ak.argsort(mu2.pt, ascending=False, axis=-1)]
            
            dimuonMask = np.logical_and(ak.num(mu1)>0, ak.num(mu2)>0)
            massMask = (mu1[dimuonMask,0] + mu2[dimuonMask,0]).mass > 50
            output['sumw'][dataset] += ak.sum(data.Generator_weight[dimuonMask][massMask])
        else:
            output["sumw"][dataset] += len(dataset)

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

        #print("%d/%d (%0.2f%%) pass initial cuts"%(sum(mask), len(mask), 100*sum(mask)/len(mask)))
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

        muons = muons[ak.argsort(muons.pt, ascending=False)]

        mu1 = muons[:,0]
        mu2 = muons[:,1]

        Z = mu1+mu2

        Z.mass

        massWindow = np.logical_and(Z.mass<self.args.Zsel.maxMass, Z.mass>self.args.Zsel.minMass)
        charge = ak.prod(data[self.args.muons.charge],axis=1) == -1

        Zsel = np.logical_and.reduce((massWindow, charge))

        #print("%d/%d (%0.2f%%) pass Z selection"%(sum(Zsel), len(Zsel), 100*sum(Zsel)/len(Zsel)))
        data = data[Zsel]
        mu1 = mu1[Zsel]
        mu2 = mu2[Zsel]
        Z = Z[Zsel]

        #compute Z boson rapidity 
        numerator = np.sqrt(Z.mass*Z.mass + Z.pt*Z.pt*np.square(np.cosh(Z.eta))) + Z.pt * np.sinh(Z.eta)
        denominator = np.sqrt(Z.mass*Z.mass + Z.pt*Z.pt)
        Zrapid = np.log(numerator/denominator)

        jets = ak.zip({
            "pt" : data.selectedPatJetsAK4PFPuppi_pt,
            'eta' : data.selectedPatJetsAK4PFPuppi_eta,
            "phi" : data.selectedPatJetsAK4PFPuppi_phi,
            'mass' : data.selectedPatJetsAK4PFPuppi_mass
            },
            behavior=vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )
        #partPt = data.selectedPatJetsAK4PFPuppiPFCands_pt
        #partJetIdx = data.selectedPatJetsAK4PFPuppiPFCands_jetIdx

        #JEC = ak.sum(ak.where(partJetIdx==0, partPt, 0), axis=-1)[:,None]

        #for i in range(1,np.max(partJetIdx)):
        #    JEC = ak.concatenate((JEC, ak.sum(ak.where(partJetIdx==i, partPt, 0), axis=-1)[:,None]), axis=1)
        #    print(JEC)

        #JEC = JEC/jets.pt

        #veto jets that are too close to the muons
        goodjets = np.logical_and(jets.delta_r(mu1)>self.args.jetSize, jets.delta_r(mu2)>self.args.jetSize)
        jets = jets[goodjets]
        #JEC = JEC[goodjets]

        if isMC:
            genJets = ak.zip({
                "pt" : data.ak4GenJetsNoNu_pt,
                'eta' : data.ak4GenJetsNoNu_eta,
                "phi" : data.ak4GenJetsNoNu_phi,
                'mass' : data.ak4GenJetsNoNu_mass
                },
                behavior=vector.behavior,
                with_name='PtEtaPhiMLorentzVector'
            )
            goodgenjets = np.logical_and(genJets.delta_r(mu1)>self.args.jetSize, genJets.delta_r(mu2)>self.args.jetSize)

        if isMC:
            evtwt = data.Generator_weight
            #SFs
            trgSF = self.SFeval['TRG'](mu1.pt, np.abs(mu1.eta))
            idSF1 = self.SFeval['ID'](mu1.pt, np.abs(mu1.eta))
            idSF2 = self.SFeval['ID'](mu2.pt, np.abs(mu2.eta))
            isoSF1 = self.SFeval['ISO'](mu1.pt, np.abs(mu1.eta))
            isoSF2 = self.SFeval['ISO'](mu2.pt, np.abs(mu2.eta))
            evtwt = evtwt*trgSF*idSF1*idSF2*isoSF1*isoSF2
            #print("TRG",np.min(trgSF))
            #print("ID1",np.min(idSF1))
            #print("ID2",np.min(idSF2))
            #print("ISO1",np.min(isoSF1))
            #print("ISO2",np.min(isoSF2))
        else:
            evtwt = ak.ones_like(data.nselectedPatJetsAK4PFPuppi)
        output["sumw"][dataset] += ak.sum(evtwt)

        output['nJets'].fill(
            dataset = dataset,
            nJets = ak.flatten(ak.num(jets, axis=-1), axis=None),
            weight=ak.flatten(evtwt,axis=None)
        )

        castedweight = ak.broadcast_arrays(evtwt, jets.pt)[0]
        output['jets'].fill(
            dataset = dataset,
            pT = ak.flatten(jets.pt, axis=None),
            eta = ak.flatten(jets.eta, axis=None),
            phi = ak.flatten(jets.phi, axis=None),
            weight=ak.flatten(castedweight, axis=None)
        )

        self.fillProjectedEEC(output['EEC2'], 
                              dataset,
                              data.EEC2_dRs,
                              data.EEC2_wts,
                              data.EEC2_jetIdx,
                              evtwt,
                              data.selectedPatJetsAK4PFPuppi_pt,
                              data.selectedPatJetsAK4PFPuppi_eta,
                              goodjets)
        
        self.fillProjectedEEC(output['EEC3'], 
                              dataset,
                              data.EEC3_dRs,
                              data.EEC3_wts,
                              data.EEC3_jetIdx,
                              evtwt,
                              data.selectedPatJetsAK4PFPuppi_pt,
                              data.selectedPatJetsAK4PFPuppi_eta,
                              goodjets)
        
        self.fillProjectedEEC(output['EEC4'], 
                              dataset,
                              data.EEC4_dRs,
                              data.EEC4_wts,
                              data.EEC4_jetIdx,
                              evtwt,
                              data.selectedPatJetsAK4PFPuppi_pt,
                              data.selectedPatJetsAK4PFPuppi_eta,
                              goodjets)

        self.fillProjectedEEC(output['EEC5'], 
                              dataset,
                              data.EEC5_dRs,
                              data.EEC5_wts,
                              data.EEC5_jetIdx,
                              evtwt,
                              data.selectedPatJetsAK4PFPuppi_pt,
                              data.selectedPatJetsAK4PFPuppi_eta,
                              goodjets)
        
        self.fillProjectedEEC(output['EEC6'], 
                              dataset,
                              data.EEC6_dRs,
                              data.EEC6_wts,
                              data.EEC6_jetIdx,
                              evtwt,
                              data.selectedPatJetsAK4PFPuppi_pt,
                              data.selectedPatJetsAK4PFPuppi_eta,
                              goodjets)
        
        if isMC:
            self.fillProjectedEEC(output['genEEC2'], 
                                dataset,
                                data.genEEC2_dRs,
                                data.genEEC2_wts,
                                data.genEEC2_jetIdx,
                                evtwt,
                                data.ak4GenJetsNoNu_pt,
                                data.ak4GenJetsNoNu_eta,
                                goodgenjets)   

            self.fillProjectedEEC(output['genEEC3'], 
                                dataset,
                                data.genEEC3_dRs,
                                data.genEEC3_wts,
                                data.genEEC3_jetIdx,
                                evtwt,
                                data.ak4GenJetsNoNu_pt,
                                data.ak4GenJetsNoNu_eta,
                                goodgenjets)  
            
            self.fillProjectedEEC(output['genEEC4'], 
                                dataset,
                                data.genEEC4_dRs,
                                data.genEEC4_wts,
                                data.genEEC4_jetIdx,
                                evtwt,
                                data.ak4GenJetsNoNu_pt,
                                data.ak4GenJetsNoNu_eta,
                                goodgenjets)  

            self.fillProjectedEEC(output['genEEC5'], 
                                dataset,
                                data.genEEC5_dRs,
                                data.genEEC5_wts,
                                data.genEEC5_jetIdx,
                                evtwt,
                                data.ak4GenJetsNoNu_pt,
                                data.ak4GenJetsNoNu_eta,
                                goodgenjets)  

            self.fillProjectedEEC(output['genEEC6'], 
                                dataset,
                                data.genEEC6_dRs,
                                data.genEEC6_wts,
                                data.genEEC6_jetIdx,
                                evtwt,
                                data.ak4GenJetsNoNu_pt,
                                data.ak4GenJetsNoNu_eta,
                                goodgenjets)

        castedweight = ak.broadcast_arrays(evtwt, Z.pt)[0]
        output['dimuon'].fill(
            dataset=dataset,
            mass = Z.mass,
            pT = Z.pt,
            y = Zrapid,
            phi = Z.phi,
            weight=castedweight
        )

        output['muon'].fill(
            dataset=dataset,
            pT = mu1.pt,
            eta = mu1.eta,
            phi = mu1.phi
        )

        output['muon'].fill(
            dataset=dataset,
            pT = mu2.pt,
            eta = mu2.eta,
            phi = mu2.phi
        )

        return output

    def postprocess(self, accumulator):
        return accumulator
    
    def fillProjectedEEC(self, hist, dataset, dRs, wts, jetIdx, evtWt, jetPt, jetEta, goodJets):
        goodEEC = goodJets[jetIdx]

        dRs = dRs[goodEEC]
        wts = wts[goodEEC]
        jetIdx = jetIdx[goodEEC]

        jetPt = jetPt[jetIdx]
        jetEta = jetEta[jetIdx]

        hist.fill(
            dataset = dataset,
            dR = ak.flatten(dRs, axis=None),
            weight = ak.flatten(wts * evtWt, axis=None),
            pT = ak.flatten(jetPt, axis=None),
            eta = np.abs(ak.flatten(jetEta, axis=None))
        )