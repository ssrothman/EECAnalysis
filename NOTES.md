# Thoughts
 - How should we apply JECs to our custom-clustered jets?
      just some function, binned in pT and eta
      just need to get the right version for a given dataset
      also gonna need rho
        event level pileup thing fastJetFixedGrid
 - How should JECs be propegated to jet constituents? 
      no preexisting per-particle scheme 
 - Should we include leptons in EECs?
      dont worry about particle content
 - How should we propegate finite energy resolution?
 - How should we propegate finite angular resolution?
 - How should we propegate event weights and their uncertainties?

 fluctuate up, fluctuate down, make set of histograms
 all of the existing uncertainty schemes are per-jet not per-particle

  check with Jeff what the candidates are
    things depend on what our pileup mitigation strategy is
  We should use constituent subtraction
    rho-based area subtraction
  Should we try to separate out heavy flavor
    Ask Ian

  email Ian

  We should probably do this in c++.......

  The smarter implementation should:
    - cache dR values for (M-1)-fold combinations
    - construct (M)-fold combinations from trios of (M-1)-fold combinations
      ie the combination (0, 1, 2, 3) is most efficiently written as 
        (0, 1, 2) + (0, 1, 3) + (1, 2, 3)
      NB this is not unique
