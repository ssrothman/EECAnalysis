%module eec

%{
  #define SWIG_FILE_WITH_INIT
  #include "eec.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* jet, int nPart, int nFeat)}

%include "eec.h"
