/*
 * The plan here is to write some c++ code
 * Atm just hoping and praying that wrapping it won't be too much of a pain in the ass
 * :)
 */
#include <vector>
#include <tuple>
#include <iostream>
#include <array>
#include <math.h>
#include <stdint.h>
#include <algorithm>

//TODO: think carefully about datatype size
//      Things probably don't need to be full width
//      idx_t needs to be larger than max nParts**2
typedef uint16_t idx_t;
typedef float coord_t;
typedef std::tuple<idx_t,idx_t> pair;

void printOrd(const std::vector<idx_t> ord){
  std::cout << "(";
  idx_t i =0;
  for(i=0; i<ord.size()-1; ++i){
    std::cout << ord[i] << ", ";
  }
  std::cout << ord[ord.size()-1] << ")";
}

void part2(int n, std::vector<std::vector<std::vector<int>>>& out){
  int a[n+1];
  int k, y, x, l;
  for(int q=0; q<n+1; ++q){
    a[q] = 0;
  }
  k = 1;
  y = n - 1;
  while(k != 0){
    x = a[k - 1] + 1;
    k -= 1;
    while(2 * x <= y){
      a[k] = x;
      y -= x;
      k += 1;
    }
    l = k + 1;
    while(x <= y){
      a[k] = x;
      a[l] = y;
      //yield a[:k + 2];
      do{
        std::vector<int> next;
        next.reserve(k+1);
        for(int q=0; q<k+2; ++q){
          next.push_back(a[q]);
        } 
        out[k+1].push_back(next);
      } while (std::next_permutation(a, a+k+2));
      x += 1;
      y -= 1;
    }
    a[k] = x + y;
    y = x + y - 1;
    //yield a[:k + 1];
    do{
      std::vector<int> next;
      next.reserve(k);
      for(int q=0; q<k+1; ++q){
        next.push_back(a[q]);
      } 
      out[k].push_back(next);
    } while (std::next_permutation(a, a+k+1));
  }
}

size_t fact(idx_t n){
  size_t result=1;
  for(idx_t i=2;i<n+1;++i){
    result*=i;
  }
  return result;
}

size_t intPow(idx_t a, idx_t b){
  //mine, very stupid
  //should be upgraded to to square multiply
  size_t result=1;
  for(idx_t i=0; i<b; ++i){
    result*=a;
  }
  return result; 
}

size_t simp(idx_t n, idx_t k){
  size_t result = 1;
  for(idx_t i=0; i<k; ++i){
    result*=n+i;
  }
  result/=fact(k);
  return result;
}

size_t choose(idx_t n, idx_t k){
  if (k > n) return 0;
  if (k * 2 > n) k = n-k;
  if (k == 0) return 1;

  size_t result = n;
  for(size_t i = 2; i <= k; ++i ) {
    result *= (n-i+1);
    result /= i;
  }
  return result;
}

inline void iterate(const idx_t dims, std::vector<idx_t>& ordinates,
                      const idx_t nParts){
  // iterate over dimensions in reverse...
  for (int dim = dims - 1; dim >= 0; --dim){
    if (ordinates[dim] < nParts-dims+dim){
      ++ordinates[dim];
      for (idx_t d2=dim+1; d2<dims; ++d2){
        ordinates[d2] = ordinates[d2-1]+1;
      }
      return;
    }
    if(dim>0){
      ordinates[dim] = ordinates[dim-1]+1;
    } else{
      ordinates[dim] = 0;
    }
  }
}

size_t getIndex(const std::vector<idx_t>& ordinates, const idx_t nPart){
  size_t result=0;
  size_t partPow=1;
  for(idx_t dim=0; dim<ordinates.size(); ++dim){
    result += ordinates[dim]*partPow;
    partPow*=nPart;
  }
  return result;
}

void getPairs(const idx_t nPart, std::vector<pair>* result){
  idx_t i, j;

  result->clear();
  for(i=0; i<nPart-1; ++i){
    for(j=i+1; j<nPart; ++j){
      result->emplace_back(i,j);
    }
  }
}

coord_t dR2(coord_t eta1, coord_t phi1, coord_t eta2, coord_t phi2){
  /*
   * Compute delta R^2 between (eta1, phi1) and (eta2, phi2)
   */
  coord_t deta = eta2-eta1;
  coord_t dphi = phi2-phi1;
  if(dphi > M_PI){
    dphi = 2*M_PI - dphi;
  }else if(dphi < -M_PI){
    dphi = 2*M_PI + dphi; //correct up to sign
  }
  return deta*deta + dphi*dphi;
}

void fillDR2(const coord_t jet[][3], const idx_t nPart, 
               std::vector<coord_t>& dRs){
  /*
   *  Compute the pairwise delta r^2 between all the particles in the jet
   *  
   *  jet: (nPart x 3) array of (pT, eta, phi), where pT has been morned to jet pT
   *  nPart: number of jet constituents
   *  dRs: array to fill with dR^2 values
   */
  idx_t i, j, n=0;
  for(i=0;i<nPart-1;++i){
    for(j=i+1;j<nPart;++j){
      dRs[n++] = dR2(jet[i][1], jet[i][2], jet[j][1], jet[j][2]);
    }
  }
}

coord_t getWt(const coord_t jet[][3], const idx_t nPart, const idx_t N,
                  const std::vector<idx_t>& ord, const idx_t M){
  coord_t result = fact(M);
  for(idx_t i=0; i<M; ++i){
    result*=jet[ord[i]][0];
  }
  
  return result;
}

coord_t getWt(const coord_t jet[][3], const idx_t nPart, const idx_t N,
                  const idx_t i, const idx_t j, const idx_t M){
  return 2*jet[i][0]*jet[j][0];
}


void doM(const coord_t jet[][3], const idx_t nPart, const idx_t N, 
          const idx_t M,
          const std::vector<coord_t>& dRs, 
          const std::vector<idx_t>& cache, const idx_t L,
          std::vector<coord_t>& wts, std::vector<idx_t>& newCache,
          bool fillCache){
  /*
   * Compute the weights for correlators with M distinct particles
   * 
   * jet: (nPart x 3) array of (pT, eta, phi), where pT has been normed to jet pT
   * nPart: number of jet constituents
   * N: correlator order
   * M: number of distinct particles in the correlator
   * dRs: array of pairwise dRs
   * cache: indices from computation L<M
   * L: number of distinct particles in cache
   * wts: vector to add weights to
   * newCache: vector in which to store the new cache indices
   * fillCache: if true, fill newCache. else ignore newCache
   */

  std::cout << "doing " << M << "..." << std::endl;

  //setup new cache
  if(fillCache)
    newCache.resize(intPow(nPart, M));
  
  idx_t i, j, idx;
  if(M==2){ 
    /* 
     * Special case for M=2. Ignores cache
     * Ignore cache, L.
     * Pairwise indices are already correct for indexing dRs and wts vectors
     */
    idx = 0;
    for(i=0;i<nPart-1;++i){
      for(j=i+1;j<nPart;++j){
        wts[idx] += getWt(jet, nPart, N, i, j, M);
        if(fillCache)
          newCache[i + nPart*j] = idx;
        std::cout << "(" << i << ", " << j << ") " << sqrt(dRs[idx]) << std::endl;
        ++idx;
      } //end for j
    } // end for i
  } // end if M==2
  else {
    //looping over arbitrary dimensions is complicated...
    size_t maxIter = choose(nPart,M);
    std::vector<idx_t> ord(M); //which M-fold combination?
    for(i=0; i<M; ++i){
      ord[i]=i;
    }
    for(size_t iter=0; iter<maxIter; ++iter){//iterate over M-fold combinations
      std::vector<idx_t> ordL(L); //which subcombinations are in the cache? 
      for(i=0; i<L; ++i){
        ordL[i]=i;
      }
      size_t maxIterL = choose(M, L);
      idx_t bestIdx=0;
      coord_t bestDR=-1;
      for(size_t iterL=0; iterL<maxIterL; ++iterL){ //iterate over L-fold combinations of items of the M-fold combination
        std::vector<idx_t> sub(L); //which sub-combination?
        for(i=0; i<L; ++i){
          sub[i] = ord[ordL[i]];
        }
        idx = cache[getIndex(sub, nPart)];//get cached index
        if(dRs[idx]>bestDR){
          bestIdx=idx;
          bestDR=dRs[idx];
        }
        iterate(L, ordL, M);
      }//end iterate over L-fold combnations. We should now have identified the maxDR

      printOrd(ord);
      float newWt = getWt(jet, nPart, N, ord, M);
      std::cout << ":" << std::endl 
        << "\tdR: " << sqrt(dRs[bestIdx]) << std::endl 
        << "\twt: " << newWt << std::endl;
      wts[bestIdx] += newWt; //placeholder wts call
      if(fillCache)
        newCache[getIndex(ord, nPart)] = bestIdx;
      iterate(M, ord, nPart);
    }
  }
}

//does jet[][3] pass by pointer? by reference? I hope to god it isn't a copy
//This will eventually be wrapped in something so I won't stress about it now
void eec_onejet(const coord_t jet[][3], idx_t nPart, idx_t N){
  /*
   * Compute EEC for one jet
   * 
   * double *jet: (nPart x 3) array of (pT, eta, phi) for particles in the jet
   * int nPart: number of particles in the jet
   * int N: correlator order
   *
   * assumes pT has already been normalized by the jet pT
   */
  //there are never actually more than nPart choose 2 different delta rs at any order
  //TODO: clever wrapper class that handles indexing for you
  //TODO: only store upper triangle (excluding diagonal)
  //We can just incrememt the wt for each dR value as appropriate
  
  size_t nPart2 = nPart*nPart;
  
  idx_t nDR = choose(nPart, 2);

  std::vector<coord_t> dRs(nDR, -1.0); 
  std::vector<coord_t> wts(nDR, 0.0);

  //fill dR vector
  fillDR2(jet, nPart, dRs);

  std::vector<idx_t> cache(nPart2, 0);

  doM(jet, nPart, N, 2, dRs, cache, 0, wts, cache, true);
  doM(jet, nPart, N, 3, dRs, cache, 2, wts, cache, false);
  doM(jet, nPart, N, 4, dRs, cache, 2, wts, cache, false);
  doM(jet, nPart, N, 5, dRs, cache, 2, wts, cache, false);

  std::cout<<std::endl<<"dR\twt"<<std::endl;
  for(idx_t i=0; i<nDR; ++i){
    std::cout << sqrt(dRs[i]) << ",\t" << wts[i] << std::endl;
  }

  /*
   * Plan:
   *  For M-1:
   *    store where in the dR array each combination points
   *  For M:
   *    for each combination:
   *      identify the 3 component (M-1) combinations
   *      identify which has max dR
   *      record index in dR vector
   *      compute appropriate weights by iterating over partitions
   *      add to wTs vector in appropriate index
   *
   *  If M>some limit (depending on maximum allowable memory consumption, 5 could be a reasonable choice)
   *    Stop caching dR location of each combination
   *    Instead, fallback on largest cached dR combination list, and do something clever
   *    
   */

  /* One non-cacheing strategy:
   *    Priority queue of L-wise dRs
   *      If (L-tuple) in (M-tuple): O(M) opperation because they're both sorted
   *        stop, we're done 
   *      Else:
   *        onto next in priority queue 
   * Worst case complexity is O((nPart choose L) * M) 
   *    Want L as small as possible????
   * Larger L makes best case more likely
   *    Probably need to do experiment to learn impact on average complexity
   *
   *  This maybe could be refined with a tree
   */

  /*
   * Another non-cacheing strategy:
   *    Direct identification of which L-tuples to look at
   *    In general this is hard
   *    If M < 2L this is easy: they are:
   *      x[0:L], x[M-L:] and (x[i:i+L-1], x[-1]) for i in range(L-1))
   *      Complexity is (M-L+2)
   *      Obv want L as large as possible. If L_cap=5, can go to 9th order correlators
   *
   *
   *    That was stupid. It's just the collection of (M choose L) combinations
   *    This is 100% the thing to do
   */

  /*
   * Lets write a function that you pass the cache and the cache size (L) to
   * Keep it completely general
   * The wrapping function can decide when to stop caching
   */

  /*
   * Food for thought: 
   *  might as well compute all lower order correlators while we're at it?
   *  Histograms are very memory-efficient
   */
}

int main(){

  int N=5;
  std::vector<std::vector<std::vector<int>>> partitions;
  for(int i=0; i<N; ++i){
    std::vector<std::vector<int>> next;
    partitions.push_back(next);
  }
  part2(N, partitions);
  std::cout<<std::endl<<std::endl;
  for(int i=0; i<N; ++i){
    std::cout << "Partitions with length " << i+1 << std::endl;
    for(int j=0; j<partitions[i].size(); ++j){
      for(int k=0; k<partitions[i][j].size(); ++k){
        std::cout << partitions[i][j][k] << ", ";
      }
      std::cout << std::endl;
    }
  }
  std::cout<<"done with partitions"<<std::endl<<std::endl;

  coord_t jet[5][3];
  jet[0][0] = 1.0;
  jet[1][0] = 2.0;
  jet[2][0] = 0.5;
  jet[3][0] = 2.0;
  jet[4][0] = 3.0;

  jet[0][1] = 0.0;
  jet[1][1] = 0.1;
  jet[2][1] = 0.4;
  jet[3][1] = 1.0;
  jet[4][1] = 0.4;

  jet[0][2] = 0.0;
  jet[1][2] = 0.2;
  jet[2][2] = 0.4;
  jet[3][2] = 0.0;
  jet[4][2] = -0.5;

  eec_onejet(jet, 5, 2);

  return 0;
}
