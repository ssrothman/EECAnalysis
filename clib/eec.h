#ifndef EEC_H
#define EEC_H

//#define VERBOSE

#include <vector>
#include <tuple>
#include <iostream>
#include <array>
#include <math.h>
#include <stdint.h>
#include <algorithm>

typedef uint16_t idx_t;
typedef float coord_t;
typedef std::tuple<idx_t,idx_t> pair;
typedef std::vector<std::vector<std::vector<idx_t>>> comp_t;
typedef std::vector<std::vector<idx_t>> factor_t;

void eec_onejet(float* jet, int nPart, int nFeat, int N);

#endif
