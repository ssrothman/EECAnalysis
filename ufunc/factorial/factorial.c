#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"

/*
 * multi_type_logit.c
 * This is the C code for creating your own
 * Numpy ufunc for a logit function.
 *
 * Each function of the form type_logit defines the
 * logit function for a different numpy dtype. Each
 * of these functions must be modified when you
 * create your own ufunc. The computations that must
 * be replaced to create a ufunc for
 * a different funciton are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 *
 */

int precomputed_ints[10] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 3628800};


static PyMethodDef FactorialMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definitions must precede the PyMODINIT_FUNC. */

static void int_factorial(char **args, npy_intp *dimensions,
                              npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out=args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    int tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(int *)in;
        *((int *)out) = precomputed_ints[tmp];
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void long_factorial(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    long tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(long *)in;
        *((long *)out) = precomputed_ints[tmp];
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

/*This gives pointers to the above functions*/
PyUFuncGenericFunction funcs[2] = {&int_factorial,
                                   &long_factorial};

static char types[4] = {NPY_INT, NPY_INT,
                        NPY_LONG, NPY_LONG};
static void *data[2] = {NULL, NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    FactorialMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *factorial, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    factorial = PyUFunc_FromFuncAndData(funcs, data, types, 4, 1, 1,
                                    PyUFunc_None, "factorial",
                                    "factorial_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "factorial", factorial);
    Py_DECREF(factorial);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *factorial, *d;


    m = Py_InitModule("npufunc", FactorialMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    factorial = PyUFunc_FromFuncAndData(funcs, data, types, 4, 1, 1,
                                    PyUFunc_None, "factorial",
                                    "factorial_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "factorial", factorial);
    Py_DECREF(factorial);
}
#endif
