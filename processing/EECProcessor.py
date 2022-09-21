import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector, candidate
import hist
from .roccor import kScaleDT, kSpreadMC, kSmearMC
import os
from coffea import processor

import correctionlib

import pandas as pd

from .deltaR_matching import object_matching

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

MISSING = -999

def cast(x):
    z = ak.to_numpy(ak.flatten(ak.fill_none(x, MISSING), axis=None))
    if z.dtype == np.float64:
        return z.astype(np.float32)
    elif z.dtype == np.int64:
        return z.astype(np.int32)
    else:
        return z

def save_dfs_parquet(fname, dfs_dict):
    xrdpath = "root://cmseos.fnal.gov//store/user/srothman/%s"%fname
    dfs_dict.to_parquet(xrdpath)

def ak_to_pd(data):
    return pd.DataFrame.from_dict(data)

class EECProcessor(processor.ProcessorABC):
    def __init__(self, args, outname):
        self.args = args

        self.cset = correctionlib.CorrectionSet.from_file("corrections/muon_Z.json")
    
        self._output_location = outname

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
        
        #all muons pass max eta (min pT to be applied after rochester corrections)
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
        #print(data.metadata)
        #for key in data.metadata.keys():
        #    print(key, data.metadata[key])
        #if data.metadata['fileuuid'] == 'bb2a9618-f406-11ec-ad3c-0d2d12acbeef':
        #    print(data.metadata['filename'])
        #    print(data.metadata['filename'][-10:])
        #    print()
        #    print()
        #return {}
        dataset = data.metadata['dataset']
        output = {}
        output [dataset] = {}

        isMC = hasattr(data, "genWeight")

        if isMC:
           sumw = ak.sum(data.genWeight)
        else:
            sumw = len(data)

        #return {dataset: sumw}

        evtMask = self.computeMask(data)
        data = data[evtMask]
        
        #muons
        muons = data[self.args.muons]
        
        #rochester corrections
        if not isMC:
            rcSF = kScaleDT(muons.charge, muons.pt, muons.eta, muons.phi, 0, 0)
        else:
            genParts = data[self.args.genParts]
            genMuons = genParts[np.abs(genParts.pdgId)==13]

            matchGen, matchReco, dR = object_matching(genMuons, muons, self.args.muMatchingCone)            
            matchedSF = kSpreadMC(matchReco.charge, matchReco.pt, matchReco.eta, matchReco.phi, matchGen.pt, 0, 0)
        
            nmu = ak.sum(ak.num(dR))
            u = ak.unflatten(np.random.random(nmu).astype(dtype=np.float32), ak.num(dR))
            unmatchedSF = kSmearMC(muons.charge, muons.pt, muons.eta, muons.phi, muons.nTrackerLayers, u, 0, 0)
        
            rcSF = ak.where(ak.is_none(dR, axis=-1), unmatchedSF, matchedSF)
        
        muons['pt'] = muons.pt*rcSF

        #muon pt cut
        mask = ak.all(muons.pt > self.args.muCuts.minPt, axis=-1)
        data = data[mask]
        muons = muons[mask]

        mu1 = muons[:,0]
        mu2 = muons[:,1]


        #Z bosons
        Z = muons[:,0] + muons[:,1]
        
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

        #apply min pT
        goodjets = np.logical_and(goodjets, jets.pt > self.args.minJetPt)

        #apply max eta
        goodjets = np.logical_and(goodjets, np.abs(jets.eta) < self.args.maxJetEta)
        
        #apply mask
        jets = jets[goodjets]

        jetFeats = {key: jets[key] for key in jets.fields}
        jetFeats['bTag'] = jets.btagDeepFlavB > self.args.tag.bWP

        CvsL = jets.pfDeepFlavourJetTags_probc/(jets.pfDeepFlavourJetTags_probc + jets.pfDeepFlavourJetTags_probuds + jets.pfDeepFlavourJetTags_probg)
        CvsB = jets.pfDeepFlavourJetTags_probc/(jets.pfDeepFlavourJetTags_probc + jets.pfDeepFlavourJetTags_probb + jets.pfDeepFlavourJetTags_probbb + jets.pfDeepFlavourJetTags_problepb)
        jetFeats['cTag'] = np.logical_and(CvsL > self.args.tag.CvsLWP, CvsB > self.args.tag.CvsBWP)

        #jet pT rank
        jetFeats['pTrank'] = ak.local_index(jets, axis=-1)
        
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
            jetFeats['genFlav'] = NAJet*0 + lJet*1 + gJet*2 + cJet*3 + bJet*4
            
            jets = ak.zip(jetFeats)

            genJets = data[self.args.genJets]
            genJet4vec = ak.zip({
                "pt" : genJets.pt,
                'eta' : genJets.eta,
                "phi" : genJets.phi,
                'mass' : genJets.mass
                },
                behavior=vector.behavior,
                with_name='PtEtaPhiMLorentzVector'
            )

            #veto gen jets too close to the muons
            goodgenjets = np.logical_and(
                genJet4vec.delta_r(mu1)>self.args.jetSize, 
                genJet4vec.delta_r(mu2)>self.args.jetSize)
            
            #apply min pT
            goodgenjets = np.logical_and(goodgenjets, genJets.pt > self.args.minGenJetPt)
            
            #apply max eta
            goodgenjets = np.logical_and(goodgenjets, np.abs(genJets.eta) < self.args.maxGenJetEta)

            genJets = genJets[goodgenjets]
            
            genJetFeats = {key:genJets[key] for key in genJets.fields}

            #gen matching
            #*Idx is None indicates failed match
            _, _, _, jets.genIdx, _ = object_matching(genJets, jets, self.args.jetMatchingCone, return_indices=True)
            
            matchedRecoJets, matchedGenJets, _, genJetFeats['recoIdx'], _ = object_matching(jets, genJets, self.args.jetMatchingCone, return_indices=True)

            genJetFeats['genFlav'] = matchedRecoJets.genFlav
            genJetFeats['cTag'] = matchedRecoJets.cTag
            genJetFeats['bTag'] = matchedRecoJets.bTag
            genJetFeats['pTrank'] = ak.local_index(genJets, axis=-1)

            genJets = ak.zip(genJetFeats)
        else:
            jets = ak.zip(jetFeats)

        #reshape EECs and apply goodjets mask
        EECnames = []
        for i in range(2, 7): #projected
            EECnames.append("EEC%d"%i)
        
        for ps in [12, 13, 22]:
            EECnames.append("EECnonIRC%d"%ps)
        
        for i in [3,4]:
            EECnames.append("Full%dPtEEC"%i)
        
        for name in EECnames:
            EEC = data[name]
            idxs = ak.materialized(EEC.jetIdx)
            runs = ak.run_lengths(idxs)
            data[name] = ak.unflatten(EEC, ak.flatten(runs), axis=-1)
            data[name] = data[name][goodjets]
            #print()
            #print("Reshaped", name)
            if hasattr(data[name], 'jetPT'):
                assert ak.all(data[name].jetPT == jets.pt)
            #    print("correct?", ak.all(data[name].jetPT == jets.pt))
            #else:
            #    print("no jetPT attr, assuming correctness for now")
            #print()

        if isMC:
            for name in EECnames:
                name = 'gen%s'%name
                EEC = data[name]
                idxs = ak.materialized(EEC.jetIdx)
                runs = ak.run_lengths(idxs)
                data[name] = ak.unflatten(EEC, ak.flatten(runs), axis=-1)
                data[name] = data[name][goodgenjets]
                #print()
                #print("Reshaped", name)
                #if hasattr(data[name], 'jetPT'):
                #    print("correct?", ak.all(data[name].jetPT == genJets.pt))
                #else:
                #    print("no jetPT attr, assuming correctness for now")
                #print()

        #scale factors
        if isMC:
            evtwt = data.genWeight
            recoSF1 = self.cset['NUM_TrackerMuons_DEN_genTracks'].evaluate('2017_UL', np.asarray(np.abs(mu1.eta)), np.asarray(mu1.pt), 'sf')
            recoSF2 = self.cset['NUM_TrackerMuons_DEN_genTracks'].evaluate('2017_UL', np.asarray(np.abs(mu2.eta)), np.asarray(mu2.pt), 'sf')
            
            idSF1 = self.cset['NUM_TightID_DEN_TrackerMuons'].evaluate('2017_UL', np.asarray(np.abs(mu1.eta)), np.asarray(mu1.pt), 'sf')
            idSF2 = self.cset['NUM_TightID_DEN_TrackerMuons'].evaluate('2017_UL', np.asarray(np.abs(mu2.eta)), np.asarray(mu2.pt), 'sf')
            
            isoSF1 = self.cset['NUM_TightRelIso_DEN_TightIDandIPCut'].evaluate('2017_UL', np.asarray(np.abs(mu1.eta)), np.asarray(mu1.pt), 'sf')
            isoSF2 = self.cset['NUM_TightRelIso_DEN_TightIDandIPCut'].evaluate('2017_UL', np.asarray(np.abs(mu2.eta)), np.asarray(mu2.pt), 'sf')
            
            trgSF = self.cset['NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight'].evaluate('2017_UL', np.asarray(np.abs(mu1.eta)), np.asarray(mu1.pt), 'sf')

            evtwt = evtwt*trgSF*idSF1*idSF2*isoSF1*isoSF2*recoSF1*recoSF2
        else:
            evtwt = ak.ones_like(ak.num(jets,axis=-1))
        
        jetwt = ak.broadcast_arrays(evtwt, jets.pt)[0]
        muwt = ak.broadcast_arrays(evtwt, mu1.pt)[0]
        dimuonwt = ak.broadcast_arrays(evtwt, Z.pt)[0]

        variables = {
            'event' : {
                'nJets' : cast(ak.sum(goodjets, axis=-1)),
                'weight' : cast(evtwt)
            },
            'jets' : {
                'pT' : cast(jets.pt),
                'eta': cast(jets.eta),
                'phi': cast(jets.phi),
                'bTag': cast(jets.bTag),
                'cTag': cast(jets.cTag),
                'nConstituents': cast(jets.nConstituents),
                'pTrank': cast(jets.pTrank),
                'weight' : cast(jetwt),
            },
            'dimuon' : {
                'pT' : cast(Z.pt),
                'y' : cast(Z.rapid),
                'phi' : cast(Z.phi),
                'mass' : cast(Z.mass),
                'weight' : cast(dimuonwt)
            },
            'muon' : {
                'pT' : cast(ak.concatenate((mu1.pt, mu2.pt), axis=0)),
                'eta' : cast(ak.concatenate((mu1.eta, mu2.eta), axis=0)),
                'phi' : cast(ak.concatenate((mu1.phi, mu2.phi), axis=0)),
                'weight' : cast(ak.concatenate((muwt, muwt), axis=0)),
            }
        }

        if isMC:
            genjetwt = ak.broadcast_arrays(evtwt, genJets.pt)[0]

            variables['jets']['genFlav'] = cast(jets.genFlav)
            
            #genJets variables
            matchedGenJets = genJets[jets.genIdx]

            variables['jets']['genPT'] = cast(matchedGenJets.pt)
            variables['jets']['genEta'] = cast(matchedGenJets.eta)
            variables['jets']['genPhi'] = cast(matchedGenJets.phi)

            #need to also account for genJets which don't get reconstructed
            missingMask = ak.is_none(genJets.recoIdx, axis=-1)

            nMissing = ak.sum(missingMask)
            missingInt = cast(MISSING*np.ones(nMissing, dtype=np.int32))
            missingFloat = cast(MISSING*np.ones(nMissing, dtype=np.float32))

            recoFloats = ['pT', 'eta', 'phi', 'bTag', 'cTag']
            recoInts = ['nConstituents', 'genFlav', 'pTrank']
            
            for var in recoFloats:
                variables['jets'][var] = np.concatenate((variables['jets'][var], missingFloat), axis=0)
            
            for var in recoInts:
                variables['jets'][var] = np.concatenate((variables['jets'][var], missingInt), axis=0)
            
            variables['jets']['genPT'] = np.concatenate((variables['jets']['genPT'], 
                                                         cast(genJets.pt[missingMask])), axis=0)
            variables['jets']['genEta'] = np.concatenate((variables['jets']['genEta'], 
                                                         cast(genJets.eta[missingMask])), axis=0)
            variables['jets']['genPhi'] = np.concatenate((variables['jets']['genPhi'], 
                                                         cast(genJets.phi[missingMask])), axis=0)
            variables['jets']['weight'] = np.concatenate((variables['jets']['weight'], 
                                                         cast(genjetwt[missingMask])), axis=0)

        dRaxis = hist.axis.Regular(self.args.dRHist.nBins, 
                                   self.args.dRHist.min, 
                                   self.args.dRHist.max, 
                                   transform=getattr(hist.axis.transform, self.args.dRHist.transform))
        centers = dRaxis.centers
        edges = dRaxis.edges

        for name in EECnames:
            if "Full" in name:
                continue
            EEC = data[name]

            if isMC:
                matchedEEC = data['gen%s'%name][jets.genIdx]
                genEEC = data['gen%s'%name][missingMask]

            for j in range(len(centers)):
                left = edges[j]
                right = edges[j+1]
                mask = np.logical_and(left < EEC.dRs, EEC.dRs <= right)
                wts = ak.sum(EEC.wts[mask], axis=-1)
                variables['jets']['%swt%d'%(name,j)] = cast(wts)

                if isMC:
                    mask = np.logical_and(left < matchedEEC.dRs, matchedEEC.dRs <= right)
                    wts = ak.sum(matchedEEC.wts[mask], axis=-1)
                    variables['jets']['gen%swt%d'%(name,j)] = cast(wts)

                    #account for unmatched genJets
                    variables['jets']['%swt%d'%(name,j)] = np.concatenate((
                        variables['jets']['%swt%d'%(name,j)], missingFloat
                    ), axis=0)

                    mask = np.logical_and(left < genEEC.dRs, genEEC.dRs <= right)
                    wts = ak.sum(genEEC.wts[mask], axis=-1)
                    variables['jets']['gen%swt%d'%(name,j)] = np.concatenate((
                        variables['jets']['gen%swt%d'%(name,j)], cast(wts)
                    ), axis=0)

            #if isMC:
            #    name = 'genEEC%d'%i
            #    genEEC = data[name]
                #variables = addEECvars(variables, name, genEEC, genJets, goodgenjets, evtwt, doFlav=False, doTag=False, doRank=False)

        output = {}
        for key in variables.keys():
            output[key] = ak_to_pd(variables[key])

        fname = data.behavior["__events_factory__"]._partition_key.replace("/", "-")
        fname= "condor_" + fname

        for key in output.keys():
            outpath = self._output_location + dataset + "/" + key
            pqpath = outpath + "/parquet"
            if not os.path.exists(outpath):
                os.makedirs(outpath, exist_ok=True)
            if not os.path.exists(pqpath):
                os.makedirs(pqpath, exist_ok=True)
            save_dfs_parquet(pqpath + "/" + fname + '.parquet', output[key])

        return {dataset: sumw}

    def postprocess(self, accumulator):
        return accumulator