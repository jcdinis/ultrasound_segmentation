# coding=utf-8

import numpy as np
import pylab
import skimage
from scipy import ndimage
try:
    from skimage import restoration, feature,filters
except ImportError:
    pass
from skimage.morphology import disk
try:
    from skimage import filters
except ImportError:
    pass    
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries,slic
from skimage.util import img_as_float
from scipy.ndimage.morphology import grey_opening, grey_closing

import scipy

import skimage.filters.rank as rank


def Filtro1(matrix_imagem):
    #gaussian gradiente de magnitude


    imagens_filtrada=ndimage.gaussian_gradient_magnitude(matrix_imagem,sigma=1.5)
    return imagens_filtrada

def Filtro2(matrix_imagem):
    #sobel
    imagens_filtrada=ndimage.filters.sobel(matrix_imagem)
    return imagens_filtrada


def Filtro3(matrix_imagem):
    print 'hhhhhhhhhhhhhhhhhhhhhhhhhh'
    # correlate
    #c = np.ones((3,3,3))SS
    #imagens_filtrada=ndimage.filters.correlate(matrix_imagem,c)
    imagens_filtrada=Filtro2(matrix_imagem)-Filtro1(matrix_imagem)

    gmin, gmax = 0, 255 #self.m_input.min(), self.m_input.max()
    fmin, fmax = imagens_filtrada.min(), imagens_filtrada.max()

    matrizf = (float(gmax - gmin)/float(fmax - fmin)) * (imagens_filtrada - fmin) + gmin

    print matrizf.max(), matrizf.min()

    ctr=matrizf
    k, j, i =ctr.shape
    for z in range(1, k-1):
        for y in range(1, j-1):
            for x in range(1, i-1):
                if ctr[z,y,x] != ctr[z+1,y,x] or\
                  ctr[z,y,x] != ctr[z-1,y,x] or \
                  ctr[z,y,x] != ctr[z,y+1,x] or \
                  ctr[z,y,x] != ctr[z,y-1,x] or \
                  ctr[z,y,x] != ctr[z,y,x+1] or \
                  ctr[z,y,x] != ctr[z,y,x-1]:

                    pass
                else:
                    ctr[z,y,x] = 0




    return ctr

def Filtro4(matrix_imagem):
    #prewit
    #matrix_imagem=Filtro11(matrix_imagem)
    mask=np.array([[0,1,0],[1,0,1],[0,1,0]])

    mat_empty= np.zeros_like(matrix_imagem - matrix_imagem.min(), dtype="uint16")
    for i in xrange(matrix_imagem.shape[0]):
        mat_empty[i]=filters.prewitt(matrix_imagem[i].astype("uint16") - matrix_imagem.min())
    return mat_empty.astype('int16') + matrix_imagem.min()

#imagens_filtrada=ndimage.filters.prewitt(matrix_imagem,axis=1,mode='nearest')
#return 255.0 - imagens_filtrada


def Filtro5(matrix_imagem):
    c = np.ones((3,3,3))
    #imagens_filtrada=skimage.filter.rank.bilateral(matrix_imagem, c)
    imagens_filtrada=ndimage.filters.rank_filter(matrix_imagem, rank=-6,size=5)
    return imagens_filtrada


def Filtro6(matrix_imagem):

    print matrix_imagem.min(), matrix_imagem.max()
    imagens_filtrada=scipy.ndimage.filters.gaussian_laplace(matrix_imagem, sigma=0.5)
    print imagens_filtrada.min(), imagens_filtrada.max()
    return 255.0-imagens_filtrada


def Filtro7(matrix_imagem):
    imagens_filtrada=scipy.ndimage.fourier.fourier_gaussian(matrix_imagem, sigma=0.1)
    return imagens_filtrada


def Filtro8(matrix_imagem):
    _P3 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    f=_P3
    imagens_filtrada=scipy.ndimage.morphology.morphological_gradient(matrix_imagem,footprint=f)
    return imagens_filtrada


def Filtro9(matrix_imagem):
    #winer
    #_P3 = np.array([[0,1,0],[1,1,1],[0,1,0]])/27
    #_P3 = np.array([[1,1,1],[1,1,1],[1,1,1]])/27.0
    #_P3 = np.ones((3,3,3))/27
    _P3 = scipy.ndimage.generate_binary_structure(3, 3)
    #f=1.0 + _P3 * 1.0 / 27.0
    #print f.shape, f.dtype
    #print f
    #imagens_filtrada=restoration.denoise_bilateral(matrix_imagem, 5)
    imagens_filtrada=restoration.denoise_bilateral(matrix_imagem, 9,sigma_range=0.9,sigma_spatial=25.0,bins=2)
    #imagens_filtrada=imagens_filtrada[:,:,:-1]
    return imagens_filtrada



#lllllllllllllllllllllllllllllllllllllllllll
def anisodiff3(matrix_imagem):
    stack=matrix_imagem
    #niter=25
    #kappa=50
    #gamma=0.25
    #step=(1.,1.,1.)
    #option=2
    #ploton=False

    niter=55
    kappa=100
    gamma=0.7
    step=(2.,2.,2.)
    option=2
    ploton=False

    if stack.ndim == 4:
        warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()
    for ii in xrange(niter):

        # calculate the diffs
        deltaD[:-1,: ,: ] = np.diff(stackout,axis=0)
        deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
        deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)

        # conduction gradients (only needsde to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:,: ,: ] -= D[:-1,:  ,:  ]
        NS[: ,1:,: ] -= S[:  ,:-1,:  ]
        EW[: ,: ,1:] -= E[:  ,:  ,:-1]

        # update the image
        stackout += gamma*(UD+NS+EW)
        imagens_filtrada=stackout
    return matrix_imagem


def Filtro10(matrix_imagem):
    mat_empty= np.zeros_like(matrix_imagem, dtype='float64')
    for i in xrange(matrix_imagem.shape[0]):
        #mat_empty[i] =feature.canny(matrix_imagem[i]*1.0, low_threshold=10.0, high_threshold=255.0)
        mat_empty[i] =feature.canny(matrix_imagem[i]*1.0, sigma=6)
    return mat_empty

#def Filtro10(matrix_imagem):


#imagens_filtrada=scipy.ndimage.generic_filter(matrix_imagem,footprint=f)
#   return imagens_filtrada

def Filtro11(matrix_imagem):
    #mediana
    # mat_empty= np.zeros_like(matrix_imagem - matrix_imagem.min(), dtype="uint16")
    #for i in xrange(matrix_imagem.shape[0]):
    #    mat_empty[i] =filters.rank.median(matrix_imagem[i].astype("uint16") - matrix_imagem.min(), disk(4))
    #return mat_empty.astype('int16') + matrix_imagem.min()
    return ndimage.median_filter(matrix_imagem, 4)


def Filtro12(matrix_imagem):
    #mediana,and treshold_adptive
    matrix_imagem=Filtro11(matrix_imagem)

    mat_empty= np.zeros_like(matrix_imagem)
    for i in xrange(matrix_imagem.shape[0]):
        mat_empty[i]=filters.threshold_adaptive(matrix_imagem[i],3)
    return mat_empty


def Filtro13(matrix_imagem):
    #Lee Filter
    infilter = matrix_imagem/np.float(matrix_imagem.max())
    print infilter.min(), infilter.max()
    mat_empty= np.zeros_like(matrix_imagem, dtype='float64')
    for i in xrange(matrix_imagem.shape[0]):
        mat_empty[i]=LeeFilter.LeeFilter(infilter[i])
    return mat_empty

def Filtro14(matrix_imagem):
    mat_empty= np.zeros_like(matrix_imagem)
    #img_ret=np.zeros_like(matrix_imagem)
    for i in xrange(matrix_imagem.shape[0]):
        mat_empty[i]=slic(matrix_imagem[i],n_segments=300, compactness=10.0,sigma=5)
    return mat_empty



def Filtro_opening(matrix_imagem):
    #gaussian gradiente de magnitude

    imagens_filtrada1=grey_opening(matrix_imagem,size=5)
    #imagens_filtrada2=grey_opening(imagens_filtrada1,size=5)
    imagens_filtrada=grey_closing(imagens_filtrada1,size=5)

    return imagens_filtrada


def sharpening(img, size=3, alpha=30):
    blurred_l = ndimage.gaussian_filter(img, 3)
    filter_blurred_l = ndimage.gaussian_filter(blurred_l, 1)
    sharpened = blurred_l + alpha * (blurred_l - filter_blurred_l)
    return sharpened