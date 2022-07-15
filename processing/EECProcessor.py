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
    def __init__(self, args):
        self.args = args
    
        ext = extractor()
        ext.add_weight_sets([
                            "ID NUM_TightID_DEN_genTracks_pt_abseta corrections/muonSF/RunBCDEF_ID_syst.histo.root",
                            "IDstat NUM_TightID_DEN_genTracks_pt_abseta_stat corrections/muonSF/RunBCDEF_ID_syst.histo.root",
                            "IDsyst NUM_TightID_DEN_genTracks_pt_abseta_syst corrections/muonSF/RunBCDEF_ID_syst.histo.root",
                            
                            "ISO NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta corrections/muonSF/RunBCDEF_ISO_syst.histo.root",
                            "ISOstat NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta_stat corrections/muonSF/RunBCDEF_ISO_syst.histo.root",
                            "ISOsyst NUM_TightRelIso_DEN_TightIDandIPCut_pt_abseta_syst corrections/muonSF/RunBCDEF_ISO_syst.histo.root",

                            "TRG IsoMu27_PtEtaBins/pt_abseta_ratio corrections/muonSF/EfficienciesAndSF_RunBtoF_Nov17Nov2017.histo.root"
            ])
        ext.finalize()

        self.SFeval = ext.make_evaluator()

    @property
    def accumulator(self):
        return self._accumulator
    
    def computeMask(self, data):
        #event cuts...
        evtMask = np.zeros(len(data), dtype=bool)
        
        for trg in self.args.triggers:
            evtMask = np.logical_or(evtMask, data.HLT[trg])

        muons = data[self.args.muons]

        #exactly two muons
        NMu = ak.num(muons, axis=-1)
        evtMask = np.logical_and(evtMask, NMu==2)

        #all muons pass isolation
        evtMask = np.logical_and(evtMask,
            ak.all(muons[self.args.muCuts.isoName] 
            < self.args.muCuts.maxIso, axis=-1))
        #all muons pass ID
        evtMask = np.logical_and(evtMask,
            ak.all(muons[self.args.muCuts.id]==1, axis=-1))
        
        #all muons pass min pT, max eta
        evtMask = np.logical_and(evtMask,
            ak.all(muons.pt > self.args.muCuts.minPt, axis=-1))
        evtMask = np.logical_and(evtMask,
            ak.all(np.abs(muons.eta) < self.args.muCuts.maxEta, axis=-1))

        muons = ak.pad_none(muons, 2, axis=-1)
        Zs = muons[:,0] + muons[:,1]
        evtMask = np.logical_and(evtMask, Zs.mass > self.args.Zsel.minMass)
        evtMask = np.logical_and(evtMask, Zs.mass < self.args.Zsel.maxMass)
        evtMask = np.logical_and(evtMask, Zs.charge==0)

        evtMask = ak.fill_none(evtMask, False)
        return evtMask

    def process(self, data):
        dataset = data.metadata['dataset']
        output = {}
        output [dataset] = {}

        isMC = hasattr(data, "genWeight")

        if isMC:
            output[dataset]['sumw'] = ak.sum(data.genWeight)
        else:
            output[dataset]['sumw'] = len(data)

        evtMask = self.computeMask(data)
        data = data[evtMask]

        #muons
        muons = data[self.args.muons]
        mu1 = muons[:,0]
        mu2 = muons[:,1]

        #Z bosons
        print(mu1.phi)
        print(mu2.phi)
        Z = muons[:,0] + muons[:,1]
        print(Z.phi)

        numerator = np.sqrt(Z.mass*Z.mass + Z.pt*Z.pt*np.square(np.cosh(Z.eta))) + Z.pt * np.sinh(Z.eta)
        denominator = np.sqrt(Z.mass*Z.mass + Z.pt*Z.pt)
        Z.rapid = np.log(numerator/denominator)

        #jets
        jets = data[self.args.jets]
        jet4vec = ak.zip({
            "pt" : jets.pt,
            'eta' : jets.eta,
            "phi" : jets.phi,
            'mass' : jets.mass
            },
            behavior=vector.behavior,
            with_name='PtEtaPhiMLorentzVector'
        )

        #veto jets that are too close to the muons
        goodjets = np.logical_and(jet4vec.delta_r(mu1)>self.args.jetSize, 
                                  jet4vec.delta_r(mu2)>self.args.jetSize)
        
        #apply jet arbitration
        if self.args.jetArbitration == 'all':
            pass
        elif self.args.jetArbitrarion.startsWith("lead"):
            localIdx = ak.local_index(goodjets)
            thresh=int(self.args.jetArbitration[4:])
            goodjets = np.logical_and(goodjets, localIdx<thresh)
        else:
            raise ValueError ("Invalid jetArbitration parameter %s"%self.args.jetArbitration)

        #apply jetID
        goodjets = np.logical_and(goodjets, jets.jetId==7)

        jets.bTagLoose = jets.btagDeepFlavB > 0.0532
        jets.bTagMedium = jets.btagDeepFlavB > 0.304
        jets.bTagTight = jets.btagDeepFlavB > 0.7476

        #TODO: check tag definition
        CvsL = jets.pfDeepFlavourJetTags_probc/(jets.pfDeepFlavourJetTags_probc + jets.pfDeepFlavourJetTags_probuds + jets.pfDeepFlavourJetTags_probg)
        CvsB = jets.pfDeepFlavourJetTags_probc/(jets.pfDeepFlavourJetTags_probc + jets.pfDeepFlavourJetTags_probb + jets.pfDeepFlavourJetTags_probbb + jets.pfDeepFlavourJetTags_problepb)
        jets.cTagLoose = np.logical_and(CvsL > 0.03, CvsB > 0.4)
        jets.cTagMedium = np.logical_and(CvsL > 0.085, CvsB > 0.34)
        jets.cTagTight = np.logical_and(CvsL > 0.52, CvsB > 0.05)

        jets.tagFlav = 2*jets.bTagTight + 1*jets.cTagTight

        #genJets
        if isMC:
            #compute gen flavor
            hadFlav = np.abs(jets.hadronFlavor)
            partFlav = np.abs(jets.partonFlavor)
            bJet = hadFlav == 5
            cJet = hadFlav == 4
            heavyFlav = np.logical_or(bJet, cJet)
            gJet = np.logical_and(~heavyFlav, partFlav == 21)
            lJet = np.logical_and(~heavyFlav, np.logical_or(np.logical_or(partFlav==1, partFlav==2), partFlav==3))
            NAJet = np.logical_and(hadFlav==0, partFlav==0)

            #categories should all be mutually exclusive, so we can just go one by one 
            jets.genFlav = NAJet*0 + lJet*1 + gJet*2 + cJet*3 + bJet*4

            genJets = data[self.args.jets]
            genJet4vec = ak.zip({
                "pt" : genJets.pt,
                'eta' : genJets.eta,
                "phi" : genJets.phi,
                'mass' : genJets.mass
                },
                behavior=vector.behavior,
                with_name='PtEtaPhiMLorentzVector'
            )
            goodgenjets = np.logical_and(
                genJet4vec.delta_r(mu1)>self.args.jetSize, 
                genJet4vec.delta_r(mu2)>self.args.jetSize)
        
        #scale factors
        if isMC:
            evtwt = data.genWeight
            trgSF = self.SFeval['TRG'](mu1.pt, np.abs(mu1.eta))
            idSF1 = self.SFeval['ID'](mu1.pt, np.abs(mu1.eta))
            idSF2 = self.SFeval['ID'](mu2.pt, np.abs(mu2.eta))
            isoSF1 = self.SFeval['ISO'](mu1.pt, np.abs(mu1.eta))
            isoSF2 = self.SFeval['ISO'](mu2.pt, np.abs(mu2.eta))
            evtwt = evtwt*trgSF*idSF1*idSF2*isoSF1*isoSF2
        else:
            evtwt = ak.ones_like(ak.num(jets,axis=-1))
        
        output[dataset]["nJets"] = hist.Hist(
            hist.axis.Regular(10, 0, 10, name="nJets", label="nJets"),
        )

        if isMC:
            output[dataset]["jets"] = hist.Hist(
                hist.axis.Regular(100, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(100, -5, 5, name='eta', label='$\eta_{Jet}$'),
                hist.axis.Regular(10, -np.pi, np.pi, name='phi', label='$\phi_{Jet}$', circular=True),
                hist.axis.IntCategory([0, 1, 2, 3, 4], name='flav', label='Flavor'),
                hist.axis.IntCategory([0, 1, 2, 3], name='tag', label='Tagged Flavor'),
                hist.axis.Regular(25, 0, 50, name='nConstituents', label='$N_{Constituents}$')
            )
        else:
            output[dataset]["jets"] = hist.Hist(
                hist.axis.Regular(100, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                hist.axis.Regular(100, -5, 5, name='eta', label='$\eta_{Jet}$'),
                hist.axis.Regular(10, -np.pi, np.pi, name='phi', label='$\phi_{Jet}$', circular=True),
                hist.axis.IntCategory([0, 1, 2, 3], name='tag', label='Tagged Flavor'),
                hist.axis.Regular(25, 0, 50, name='nConstituents', label='$N_{Constituents}$')
            )

        output[dataset]['dimuon'] = hist.Hist(
            hist.axis.Regular(100, self.args.Zsel.minMass, self.args.Zsel.maxMass, name='mass', label='$m_{\mu\mu}$'),
            hist.axis.Regular(100, 0, 1000, name='pT', label='$p_{T,\mu\mu}$'),
            hist.axis.Regular(100, -5, 5, name='y', label='$y{\mu\mu}$'),
            hist.axis.Regular(10,-np.pi,np.pi,name='phi', label='$\phi_{\mu\mu}$', circular=True)
        )
        output[dataset]['muon'] = hist.Hist(
            hist.axis.Regular(100, 0, 1000, name='pT', label='$p_{T,\mu}$'),
            hist.axis.Regular(100, -5, 5, name='eta', label='$\eta_{\mu}$'),
            hist.axis.Regular(10,-np.pi,np.pi,name='phi', label='$\phi_{\mu}$', circular=True)
        )
        for i in range(2, 7):
            if isMC:
                output[dataset]['EEC%d'%i] = hist.Hist(
                    hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                    hist.axis.Regular(20, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                    hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$'),
                    hist.axis.IntCategory([0,1,2,3,4], name='flav', label='Flavor'),
                    hist.axis.IntCategory([0, 1, 2, 3], name='tag', label='Tagged Flavor')
                )

                output[dataset]['genEEC%d'%i] = hist.Hist(
                    hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                    hist.axis.Regular(20, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                    hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$')
                )
            else:
                output[dataset]['EEC%d'%i] = hist.Hist(
                    hist.axis.Regular(100, 1e-5, 1.0, name='dR', label='$\Delta R$', transform=hist.axis.transform.log),
                    hist.axis.Regular(20, 0, 1000, name='pT', label='$p_{T,Jet}$'),
                    hist.axis.Regular(10, 0, 5, name='eta', label='$\eta_{Jet}$'),
                    hist.axis.IntCategory([0, 1, 2, 3], name='tag', label='Tagged Flavor')
                )

        output[dataset]['nJets'].fill(
            nJets = ak.flatten(ak.num(jets[goodjets], axis=-1), axis=None),
            weight=ak.flatten(evtwt,axis=None)
        )

        castedweight = ak.broadcast_arrays(evtwt, jets.pt)[0]
        if isMC:
            output[dataset]['jets'].fill(
                pT = ak.flatten(jets.pt[goodjets], axis=None),
                eta = ak.flatten(jets.eta[goodjets], axis=None),
                phi = ak.flatten(jets.phi[goodjets], axis=None),
                weight=ak.flatten(castedweight[goodjets], axis=None),
                flav = ak.flatten(jets.genFlav[goodjets]),
                tag = ak.flatten(jets.tagFlav[goodjets]),
                nConstituents = ak.flatten(jets.nConstituents[goodjets], axis=None)
            )
        else:
            output[dataset]['jets'].fill(
                pT = ak.flatten(jets.pt[goodjets], axis=None),
                eta = ak.flatten(jets.eta[goodjets], axis=None),
                phi = ak.flatten(jets.phi[goodjets], axis=None),
                weight=ak.flatten(castedweight[goodjets], axis=None),
                tag = ak.flatten(jets.tagFlav[goodjets], axis=None),
                nConstituents = ak.flatten(jets.nConstituents[goodjets], axis=None)
            )

        for i in range(2, 7):
            name = "EEC%d"%i
            EEC = data[name]
            self.fillProjectedEEC(output[dataset][name], 
                              EEC,
                              jets,
                              goodjets,
                              evtwt,
                              doFlav=isMC,
                              doTag = True)
        
            if isMC:
                name = "genEEC%d"%i
                EEC = data[name]
                self.fillProjectedEEC(output[dataset][name], 
                                EEC,
                                genJets,
                                goodgenjets,
                                evtwt,
                                doFlav=False,
                                doTag = False)

        castedweight = ak.broadcast_arrays(evtwt, Z.pt)[0]
        output[dataset]['dimuon'].fill(
            mass = Z.mass,
            pT = Z.pt,
            y = Z.rapid,
            phi = Z.phi,
            weight=castedweight
        )

        castedweight = ak.broadcast_arrays(evtwt, mu1.pt)[0]
        output[dataset]['muon'].fill(
            pT = mu1.pt,
            eta = mu1.eta,
            phi = mu1.phi,
            weight= castedweight
        )

        castedweight = ak.broadcast_arrays(evtwt, mu2.pt)[0]
        output[dataset]['muon'].fill(
            pT = mu2.pt,
            eta = mu2.eta,
            phi = mu2.phi,
            weight = castedweight
        )

        return output

    def postprocess(self, accumulator):
        return accumulator
    
    def fillProjectedEEC(self, hist, EEC, jets, goodJets, evtWt, doFlav, doTag):
        dRs = EEC.dRs
        wts = EEC.wts
        jetIdx = EEC.jetIdx

        goodJets = ak.pad_none(goodJets, ak.max(jetIdx, axis=None)+1, clip=True, axis=-1)
        goodJets = ak.fill_none(goodJets, False)
        goodEEC = goodJets[jetIdx]

        dRs = dRs[goodEEC]
        wts = wts[goodEEC]
        jetIdx = jetIdx[goodEEC]

        jetPt = jets.pt[jetIdx]
        jetEta = jets.eta[jetIdx]

        if doFlav:
            jetFlav = jets.genFlav[jetIdx]
            
            if doTag:
                tagFlav = jets.tagFlav[jetIdx]

                hist.fill(
                    dR = ak.flatten(dRs, axis=None),
                    weight = ak.flatten(wts * evtWt, axis=None),
                    pT = ak.flatten(jetPt, axis=None),
                    eta = np.abs(ak.flatten(jetEta, axis=None)),
                    flav = ak.flatten(jetFlav, axis=None),
                    tag = ak.flatten(tagFlav, axis=None)
                )
            else:
                hist.fill(
                    dR = ak.flatten(dRs, axis=None),
                    weight = ak.flatten(wts * evtWt, axis=None),
                    pT = ak.flatten(jetPt, axis=None),
                    eta = np.abs(ak.flatten(jetEta, axis=None)),
                    flav = ak.flatten(jetFlav, axis=None)
                )
        else:
            if doTag:
                tagFlav = jets.tagFlav[jetIdx]

                hist.fill(
                    dR = ak.flatten(dRs, axis=None),
                    weight = ak.flatten(wts * evtWt, axis=None),
                    pT = ak.flatten(jetPt, axis=None),
                    eta = np.abs(ak.flatten(jetEta, axis=None)),
                    tag = ak.flatten(tagFlav, axis=None)
                )
            else:
                hist.fill(
                    dR = ak.flatten(dRs, axis=None),
                    weight = ak.flatten(wts * evtWt, axis=None),
                    pT = ak.flatten(jetPt, axis=None),
                    eta = np.abs(ak.flatten(jetEta, axis=None))
                )