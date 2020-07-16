# some function for processing 2D/3D data using SimleITK and others
# vim:ts=4:et
# Copyright (C) 2013 Milos Sramek <milos.sramek@soit.sk>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/gpl.html
# - or any later version.

import SimpleITK as sitk
import numpy as np
import cv2
from scipy.sparse import csr_matrix

def plot(data):
    import matplotlib.pyplot as plt
    # data is tuple of lists
    for d in data:
        plt.plot(range(len(d)),d)
    #plt.pause(1)
    plt.show(block=True)

def disp(iimg, label = None, gray=False):
    #import guiqwt.pyplot as plt
    import matplotlib.pyplot as plt
    plt.ioff()
    #plt.imshow(iimg, interpolation='none')
    plt.imshow(iimg)
    #plt.pause(1)
    plt.show(block=True)

def toitk(image):
    """
    Convert to ITK, if necessary, and return True as the first return value if input is itk image
    """
    if isinstance(image, sitk.Image):
        return True, image
    else:
        if image.dtype=='bool':
            return False, sitk.GetImageFromArray(image.astype(np.int8))
        else:
            return False, sitk.GetImageFromArray(image)

def fromitk(itkImage, wasitk):
    """
    Convert to numpy array, if wasitk==False
    """
    if wasitk:
        return itkImage
    else:
        return sitk.GetArrayFromImage(itkImage)

    
def labelitk(image):
    """
    label objects in a binary image
    image: a numpy array
    """
    #itkImage = sitk.GetImageFromArray(image.astype(np.uint16))
    wasitk, itkImage = toitk(image)
    if itkImage.GetPixelIDValue() != sitk.sitkUInt16:
        itkImage = sitk.Cast( itkImage, sitk.sitkUInt16)

    filter = sitk.ConnectedComponentImageFilter()
    itkImage = filter.Execute(itkImage)
    return fromitk(itkImage, wasitk)

def actcontitk(image, mask, LSigma, RMSError=0.02, CScale=0.5, AScale=1, PScale=1, iterations=1000):
    """
    Geodesic active contours by ITK
    Input as numpy array ot itk image
    LSigma: smoothing sigma in blob detection in the last slice
    RMSError: RMS error (to stop propagation) {", RMSError, "}" 
    PScale: propagation scaling parameter (0,1) {", PScale, "}" 
    CScale: curvature scaling parameter (0,1) {", CScale, "}" 
    AScale: advection scaling parameter (0,1) {", AScale, "}" 
    """
    wasitk, itkImage = toitk(image)
    if not wasitk:
        mask = sitk.GetImageFromArray(mask)

    mask = sitk.Cast( mask, itkImage.GetPixelIDValue() ) * -1 + 0.5
    #geodesicActiveContour = sitk.ShapeDetectionLevelSetImageFilter()
    geodesicActiveContour = sitk.GeodesicActiveContourLevelSetImageFilter()
    geodesicActiveContour.SetPropagationScaling( PScale )
    geodesicActiveContour.SetCurvatureScaling( CScale )
    geodesicActiveContour.SetAdvectionScaling( AScale )
    geodesicActiveContour.SetNumberOfIterations( iterations )
    
    gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradientMagnitude.SetSigma(LSigma)
    featureImage = sitk.BoundedReciprocal( gradientMagnitude.Execute( itkImage ) )

    #find the right RMSError value, if tracking fails
    repeatAC = True
    while repeatAC: 
        geodesicActiveContour.SetMaximumRMSError( RMSError )
        levelset = geodesicActiveContour.Execute( mask, featureImage )
        if geodesicActiveContour.GetElapsedIterations() < iterations:
            repeatAC = False
        else:
            RMSError *=1.25
            #print "RMSError:", RMSError
    #print( "RMS Change: ", geodesicActiveContour.GetRMSChange() )
    #print( "Elapsed Iterations: ", geodesicActiveContour.GetElapsedIterations() )

    contour = sitk.BinaryThreshold( levelset, -1000, 0 )
    return fromitk(contour, wasitk)

def gaussitk(image, sigma):
    wasitk, itkImage = toitk(image)
    filter = sitk.SmoothingRecursiveGaussianImageFilter()
    filter.SetSigma(float(sigma))
    itkImage = filter.Execute(itkImage)
    return fromitk(itkImage, wasitk)

def gaussgraditk(image, sigma):
    """ smooth by gaussian and compute gradient magnitude """
    wasitk, itkImage = toitk(image)
    filter = sitk.SmoothingRecursiveGaussianImageFilter()
    filter.SetSigma(float(sigma))
    itkImage = filter.Execute(itkImage)
    gradmax = sitk.GradientMagnitudeImageFilter()
    itkImage = gradmax.Execute(itkImage)
    return fromitk(itkImage, wasitk)

def distanceitk(image, signed=False):
    wasitk, itkImage = toitk(image)
    if signed:
        itkImage = sitk.DanielssonDistanceMap(itkImage)
    else:
        itkImage = sitk.SignedDanielssonDistanceMap(itkImage)
    return fromitk(itkImage, wasitk)

def dilateitk(image, radius):    #not working?
    wasitk, itkImage = toitk(image)
    filter = sitk.DilateObjectMorphologyImageFilter()
    filter.SetKernelRadius(radius)
    itkImage = filter.Execute(itkImage)
    return fromitk(itkImage, wasitk)

def erodeitk(image, radius):    #not working?
    wasitk, itkImage = toitk(image)
    #itkImage = sitk.GetImageFromArray(image.astype(np.uint8))
    filter = sitk.ErodeObjectMorphologyImageFilter()
    filter.SetKernelRadius(radius)
    itkImage = filter.Execute(itkImage)
    return fromitk(itkImage, wasitk)

def statisticsitk(image, mask=None):
    dummy, itkImage = toitk(image)
    if mask is None:
        dummy, itkMask = toitk(np.ones(
                (itkImage.GetWidth(), itkImage.GetHeight()), dtype=np.uint8))
    else:
        dummy, itkMask = toitk(mask)
    #itkImage = sitk.GetImageFromArray(image.astype(np.uint16))
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(itkImage, itkMask)
    #for labelCode in stats.GetValidLabels():
        #print labelCode, stats.GetMean(labelCode), stats.GetSigma(labelCode)
    return stats

#not tested
def anisoitk(image, iter):
    itkImage = sitk.GetImageFromArray(image.astype(float))
    anizodif = sitk.GradientAnisotropicDiffusionImageFilter()
    anizodif.SetNumberOfIterations( iter );
    anizodif.SetTimeStep( 0.06 );
    anizodif.SetConductanceParameter( 3 );
    itkImage = anizodif.Execute(itkImage)
    return sitk.GetArrayFromImage(itkImage).astype(image.dtype)

def watersheditk(image, sigma, level, wsLines=False, isColor=False):
    """ partition the image in homogeneous areas
    Parameters:
    sigma: gaussian smoothing
    level: merge level
    """

    if isColor:
        gradients=[gaussgraditk(sitk.GetImageFromArray(image[...,i].astype(float)), sigma) for i in range(image.shape[-1])]
        itkImage = sitk.NaryMaximum(gradients)

    else:
        itkImage = sitk.GetImageFromArray(image.astype(float))
        itkImage = gaussgraditk(itkImage, sigma)

    wsFilter = sitk.MorphologicalWatershedImageFilter()
    if wsLines:
        wsFilter.MarkWatershedLineOn()
    else :
        wsFilter.MarkWatershedLineOff()
    wsFilter.SetLevel(level)
    itkImage = wsFilter.Execute(itkImage)
    return sitk.GetArrayFromImage(itkImage)

def getColRanges(image, mask, tsigma=3.0):
    """
    get color ranges for area identified by 'mask'
    image: single band 2D or 3D image
    """
    wasitk, itkImage = toitk(image)
    wasitk, itkMask = toitk(mask)
    stats = statisticsitk(itkImage, itkMask)
    #print "S:", stats.GetMean(1), stats.GetSigma(1)
    lb = stats.GetMean(1) - tsigma*stats.GetSigma(1)
    ub = stats.GetMean(1) + tsigma*stats.GetSigma(1)
    return lb, ub

def selectLargestRegion(img):
    """
    select the largest region in each band of the image
    """
    if img.ndim == 2:
        nslices = 1
    elif img.ndim == 3:
        nslices = img.shape[2]
    else:#FIXME raise and exception
        return img

    for i in range(nslices):
        if img.ndim == 2:
            aux = img.astype(np.uint8)
        else:
            aux = img[:,:,i].astype(np.uint8)
    
        clist,hier = cv2.findContours(aux,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        maxcontour=clist[0]
        maxcontourarea=cv2.contourArea(clist[0])
        for cont in clist:
            if(cv2.contourArea(cont) > maxcontourarea):
                maxcontour = cont
                maxcontourarea = cv2.contourArea(cont)
        
        aux[:] = 0
        cv2.drawContours(aux,[maxcontour],0,255, -1)
        #print cv2.minAreaRect(maxcontour)
        if img.ndim == 2:
            img = aux
        else:
            img[:,:,i] = aux;
    return img

def selectLargestRegions(img, size_ratio):
    """
    Select regions, which are at least as large as 'size_ratio' multiple of the largest one
    """
    aux = img.astype(np.uint8)
    clist,hier = cv2.findContours(aux,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # create a list to sort
    idcont = []
    n = 0;
    for cont in clist:
        idcont.append((n, cv2.contourArea(cont)))
        n += 1
    idcont = sorted(idcont, key=lambda x: x[1], reverse=True)

    aux[:] = 0
    for c, s in idcont:
        if s < size_ratio * idcont[0][1]:
            break
        cv2.drawContours(aux,[clist[c]],0,255, -1)
    return aux

def imHist(im):
    """
    compute histogram of a single band image
    returns array of frequencies
    """
    maxid = np.amax(im)
    hg = cv2.calcHist([im.astype(np.float32)],[0], None, [int(maxid+1)],[0, int(maxid+1)]).astype(np.int)
    return hg.flatten() # hg is a maxval x 1 2D array, make it 1D

def showHist(img, mask=None):
    if mask == None:
        plt.hist(img.flatten(), bins=256, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Uniform')
    else:
        a = csr_matrix(mask).nonzero()
        plt.hist(img[a], bins=256, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Uniform')
    plt.show()

def maskImage(img, mask):
    aux = img.copy()
    for i in range(img.shape[2]):
        aux[...,i] = img[...,i]*mask  
    return aux


def fillHoles(mask):
    """ fill holes in foreground regions 
    """
    mask = (mask.astype(np.uint8))
    clist,hier = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    mask[:] = 0
    for cont in clist:
        cv2.drawContours(mask,[cont],0,1,-1)
    return mask

def rgb2luv(img):
    ''' according to fiji'''
    imgf=img.astype(np.float32)/255
    unp = 0.19784
    vnp = 0.4683
    imgf = np.piecewise(imgf, 
        [imgf > 0.04045, imgf <= 0.04045], 
        [lambda x: np.exp(np.log((x + 0.055)/1.055)*2.4), lambda x: x/12.92])

    XYZ=imgf.copy()
    XYZ[...,0] = 0.4124 * imgf[...,0] + 0.3576 * imgf[...,1] + 0.1805 * imgf[...,2]
    XYZ[...,1] = 0.2126 * imgf[...,0] + 0.7152 * imgf[...,1] + 0.0722 * imgf[...,2]
    XYZ[...,2] = 0.0193 * imgf[...,0] + 0.1192 * imgf[...,1] + 0.9505 * imgf[...,2]

    yyn = XYZ[...,1]/100.0
    yyn = np.piecewise(yyn, 
        [yyn > 0.008856, yyn <= 0.008856],
        [lambda x: np.exp(np.log(x)/3.0), lambda x: (7.787 * x) + (16.0/116.0)])

    xyz = XYZ[...,0] + 15.0* XYZ[...,1] + 3.0* XYZ[...,2]
    xyz = np.piecewise(xyz, [xyz != 0, xyz == 0], [lambda x: 1.0/x, 0])

    up = 4.0* XYZ[...,0] * xyz
    vp = 9.0* XYZ[...,1] * xyz

    imgf[...,0] = 116.0*yyn - 16 
    imgf[...,1] = 13 * imgf[...,0] * (up-unp) 
    imgf[...,2] = 13 * imgf[...,0] * (vp-vnp) 
    #return (50*(imgf+2)).astype(np.uint8)
    return imgf
