import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef data_to_numpy_array_with_spec(void * ptr, np.npy_intp* N, int t):
    cdef np.ndarray[np.float64_t, ndim=3] arr = np.PyArray_SimpleNewFromData(3, N, t, ptr)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


cdef extern struct Image:
    unsigned char *data
    int dx
    int dy
    int dz
    float sx
    float sy
    float sz

cdef extern struct Image_d:
    double *data
    int dx
    int dy
    int dz
    float sx
    float sy
    float sz


cdef extern Image_d smooth(Image, int)

ctypedef np.uint8_t DTYPE_t

def smoothpy(np.ndarray[DTYPE_t, ndim=3] image not None, int n):
    cdef Image img

    img.data = <unsigned char*> image.data
    img.dz = image.shape[0]
    img.dy = image.shape[1]
    img.dx = image.shape[2]

    cdef Image_d out = smooth(img, n)

    np_out = data_to_numpy_array_with_spec(out.data, image.shape, np.NPY_FLOAT64)

    return np_out
