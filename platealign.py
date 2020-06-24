#!/usr/bin/python3
# vim:ts=4:et
# Stage 1: identify dishes in each time step, align them and create the NNN dfirectory and the NNN/plates-NNN.tif file 

# Copyright (C) 2013 Milos Sramek <milos.sramek@soit.sk>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/gpl.html
# - or any later version.

from tifffile import TiffWriter, TiffFile
import sys, glob, shutil, os, getopt
import re
import SimpleITK as sitk
import numpy as np
from skimage.segmentation import clear_border
from skimage import measure
import cv2, imutils
import scipy.ndimage as ndi
import configparser, imageio
from time import gmtime, strftime
import ipdb

def plot(data):
    plt.plot(data)
    plt.show()

def disp(iimg, label = None, gray=False):
    """ Display an image using pylab
    """
    import pylab, matplotlib
    matplotlib.interactive(True)
    matplotlib.pyplot.imshow(iimg, interpolation='none')
 
def loadTiff(ifile):
    try:
        with TiffFile(str(ifile)) as tfile:
            #nz, ny, nx = tfile.series[0]['shape']
            #ipdb.set_trace()
            nz, ny, nx = tfile.series[0].shape
            if len(tfile.pages) == 1: #directly one volume, tiff volume exported by fiji
                vol = tfile.pages[0].asarray()
            else:
                vol=np.zeros((nz,ny,nx),dtype=np.int16)
                for ip in range(0,nz):
                    vol[ip,...] = tfile.pages[ip].asarray()
        return vol
    except IOError as err:
        print ("%s: Error -- Failed to open '%s'"%(sys.argv[0], str(ifile)))
        sys.exit(0)

def transformItk(fixed, moving, transform):
    if moving.ndim == 3:
        output = moving.copy()
        for n in range(3):
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitk.GetImageFromArray(fixed[:,:,n]));
            resampler.SetDefaultPixelValue(output[:,:,n].mean())
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(transform)
            output[:,:,n] = sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(output[:,:,n])))
        return output
    else:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk.GetImageFromArray(fixed));
        resampler.SetDefaultPixelValue(moving.mean())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)
        return sitk.GetArrayFromImage(resampler.Execute(sitk.GetImageFromArray(moving)))

def to2DMax(img3):
    return img3.max(axis=2)

def to2DGradMax(img3):
    img3 = sitk.GetImageFromArray(img3.astype(np.float32))
    img3 = sitk.GradientMagnitudeRecursiveGaussian(img3, 1)
    return sitk.GetArrayFromImage(img3).max(axis=2)

def command_iteration(method) :
        if (method.GetOptimizerIteration()==0):
            print("Estimated Scales: ", method.GetOptimizerScales())
        print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
                                               method.GetMetricValue(),
                                               method.GetOptimizerPosition()))

def register(img1, img2, sub=4, ssigma=23, algmax="max"):
    if img1.ndim == 3:
        if algmax == "max": 
            img1=to2DMax(img1)
            img2=to2DMax(img2)
        else:
            img1=to2DGradMax(img1)
            img2=to2DGradMax(img2)

    img1 = sitk.GetImageFromArray(img1.astype(np.float32))
    img2 = sitk.GetImageFromArray(img2.astype(np.float32))

    fixed=sitk.SmoothingRecursiveGaussian(img1,ssigma)
    moving=sitk.SmoothingRecursiveGaussian(img2,ssigma)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                               minStep=1e-4,
                                               numberOfIterations=50,
                                               gradientMagnitudeTolerance=1e-8 )
    R.SetOptimizerScalesFromIndexShift()
    R.SetInterpolator(sitk.sitkLinear)
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform())
    R.SetInitialTransform(tx)
    R.SetShrinkFactorsPerLevel(shrinkFactors = [8])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[8])

    outTx = R.Execute(fixed, moving)
    return outTx

def getBBox(img, sub=4):
    nzero = np.nonzero(img)
    return (
            (sub*int((nzero[0].min()+sub/2)/sub),sub*int((nzero[0].max()+sub/2)/sub)), 
            (sub*int((nzero[1].min()+sub/2)/sub),sub*int((nzero[1].max()+sub/2)/sub))
        )

def cropPlate(img, sdef):
    return img[slice(*sdef[0]),slice(*sdef[1])]

def maskPlate3(img, mask):
    for n in range(img.shape[2]):
        band = img[:,:,n]
        band[np.nonzero(mask==0)] = band[np.nonzero(mask!=0)].mean()
    return img


def rolling_ball_filter(img, sub, ball_radius, top=False):
    """Rolling ball filter implemented with morphology operations """
    img=img[::sub,::sub,:]
    se  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*ball_radius,2*ball_radius))
    img = img.copy()
    if img.ndim == 3:
        for b in range(3):
            iband = img[:,:,b]
            if not top:
                iband = ndi.grey_erosion(iband, structure=se)
                img[:,:,b] = ndi.grey_dilation(iband, structure=se)
            else:
                iband = ndi.grey_dilation(iband, structure=se)
                img[:,:,b] = ndi.grey_erosion(iband, structure=se)
    else:
        if not top:
            img = ndi.grey_erosion(img, structure=se)
            img = ndi.grey_dilation(img, structure=se)
        else:
            img = ndi.grey_dilation(img, structure=se)
            img = ndi.grey_erosion(img, structure=se)
    return ndi.zoom(img, (sub,sub,1), order=1) 

# crop to nearest dimension, which is multiple of n
def trimShape(img, n):
    ts = (n*int(img.shape[0]/n), n*int(img.shape[1]/n), img.shape[2])
    return img[:ts[0], :ts[1], :]

def normalizeColor(img):
    img = img.copy()
    if img.ndim == 3:
        maxval = img.min()
        minval = img.max()
        for b in range(3):
            simg = ndi.gaussian_filter(img[::4,::4,b], 1.0)
            maxval = max(maxval,simg.max())
            minval = min(minval,simg.min())
    else:
        simg = gaussian_filter(img[::4,::4], 1.0)
        minval = simg.min()
        maxval = simg.max()
    img[np.nonzero(img<minval)] = minval
    img -= minval
    img[np.nonzero(img>(maxval-minval))] = maxval - minval
    img /= (maxval-minval)
    return img

def img3mask(img, mask):
    img[:,:,0] = (mask>0)*img[:,:,0] 
    img[:,:,1] = (mask>0)*img[:,:,1] 
    img[:,:,2] = (mask>0)*img[:,:,2] 
    return img

def detectDishExe(mask):
        mask = clear_border(mask)
        labels, nlabels = measure.label(mask, return_num=True)
        lsizes = np.bincount(labels.flat)
        #get the largest region
        maxlabel = 1+np.argmax(lsizes[1:])
        mask = labels == maxlabel
        mask=ndi.binary_fill_holes(mask)
        if lsizes[maxlabel] > mask.size/3: 
            return mask.astype(np.uint8)
        return None

def detectDishFinal(mask):
    # cleanup, along rows and columns for speedup
    mm = ndi.binary_closing(mask,np.ones((1,50)))
    mm = ndi.binary_closing(mm,np.ones((50,1))) #weird border effects
    mask[25:-25,25:-25] = mm[25:-25,25:-25]
    return mask.astype(np.uint8)

def detectDish(plate, fname, sub=4, relthrmax=15, relthrmin=4):
    """
    sub: subdivision in rolling_ball_filter
    relthr: reative threshold of the dish in respect to surrounding
    """
    global reportLog
    reportData={}
    fname = fname.split("/")[-1]
    #rb=rolling_ball_filter(plate, sub, 5).min(axis=2)
    rb = ndi.grey_erosion(plate.min(axis=2),structure=np.ones((5,5)))
    #relthrmax=5
    #ipdb.set_trace()
    for thr in range(relthrmax,relthrmin,-2):   #try several threshods in the case the dish inside is connected to image border
        m = rb > rb.mean()-thr
        mask = detectDishExe(m)
        #ipdb.set_trace()
        reportData["Relative threshold"]=thr
        if not mask is None : 
            reportData["Success"]="yes"
            reportLog[fname]=reportData
            return detectDishFinal(mask)
        ks=3
        m1 = ndi.binary_erosion(m, np.ones((2*ks+1,2*ks+1))) 
        for k in range(ks): # fill border, otherwise detectDishExe will always succeed
            m1[k,:]=m1[ks,:]
            m1[:,k]=m1[:,ks]
            m1[-(k+1),:]=m1[-(ks+1),:]
            m1[:,-(k+1)]=m1[:,-(ks+1)]
        mask = detectDishExe(m1)
        if not mask is None : 
            reportData["Erosion"]="%dx%d"%(2*ks+1,2*ks+1)
            reportData["Success"]="yes"
            reportLog[fname]=reportData
            return detectDishFinal(ndi.binary_dilation(mask, np.ones((2*ks+1,2*ks+1))))
        else:
            m1[-1:] = 0     #the dish perhaps touches the bottom, so delete the bottom line
            mask = detectDishExe(m1 )
            if not mask is None : 
                reportData["Erosion"]="%dx%d"%(2*ks+1,2*ks+1)
                reportData["Delete last line:"]="yes"
                reportData["Success"]="yes"
                reportLog[fname]=reportData
                return detectDishFinal(ndi.binary_dilation(mask, np.ones((2*ks+1,2*ks+1))))
    #still no successs, be more aggresive
    dd2=int(6000/2-500)  #dish dimension is 6000, border is 500
    nz = np.nonzero(m)
    cpos=(int(nz[0].mean()), int(nz[1].mean()))
    m2 = m1.copy()
    m2[:cpos[0]-dd2,:] = ndi.binary_opening(m2[:cpos[0]-dd2,:],np.ones((1,50)))
    m2[cpos[0]+dd2:,:] = ndi.binary_opening(m2[cpos[0]+dd2:,:],np.ones((1,50)))
    m2[:,:cpos[0]-dd2] = ndi.binary_opening(m2[:,:cpos[0]-dd2],np.ones((50,1)))
    m2[:,cpos[0]+dd2:] = ndi.binary_opening(m2[:,cpos[0]+dd2:],np.ones((50,1)))
    mask = detectDishExe(m2)
    if not mask is None : 
        reportData["Opening"]="50, 50"
        reportData["Success"]="yes"
        reportLog[fname]=reportData
        return detectDishFinal(mask)
    m1[-1:] = 0     #the dish perhaps touches the bottom, so delete the bottom line
    mask = detectDishExe(m1 )
    #ipdb.set_trace()
    reportData["Delete last line:"]="yes"
    if not mask is None : 
        reportData["Success"]="yes"
        reportLog[fname]=reportData
        return detectDishFinal(mask)
    reportData["Success"]="no"
    reportLog[fname]=reportData
    return None

def drawLine(img, p0, p1, lval=2, thick=5):
    cv2.line(img,(p0[0][0], p0[0][1]), (p1[0][0], p1[0][1]), lval, 5) 



def segLenght(s, e):
    s = s[0]
    e = e[0]
    return np.sqrt((s[0]-e[0])*(s[0]-e[0]) + (s[1]-e[1])*(s[1]-e[1]))
 
def segAngle(s, e):
    d1 = s[0][0]-e[0][0] 
    d2 = s[0][1]-e[0][1]
    if np.abs(d1) > np.abs(d2):
        return 180*np.arctan(-d2/d1)/np.pi
    else:
        return 180*np.arctan(d1/d2)/np.pi

def getMaskRotationCont(img):
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    approx=cv2.approxPolyDP(contours[0],20,True)
    #disp(cv2.drawContours(img.copy(), [approx], -1 ,2,30))
    lengths = [segLenght(s, e) for s, e in zip(approx[0:-2], approx[1:-1])]
    pairs = [(s, e) for s, e in zip(approx[0:-2], approx[1:-1])]
    angles = []
    for i in range(3):
        m = np.argmax(lengths)
        mpair = pairs[m]
        lengths[m]=0
        angles.append(segAngle(mpair[0],mpair[1]))
    M = cv2.moments(approx)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY) , np.array(angles).mean()

def getFilesAndCreateDir(inDirName, outDirName, sid):
    fnames = glob.glob("%s/*%s.tif"%(inDirName, sid))
    spath = "%s/%s"%(outDirName, sid)
    if fnames:
        if  os.path.exists(spath):
            try:  
                shutil.rmtree(spath)
            except OSError:  
                print ("Deletion of the directory %s failed" % spath)
                sys.exit(1)
        try:  
            os.mkdir(spath)
        except OSError:  
            print ("Creation of the directory %s failed" % spath)
            sys.exit(1)
    fdict = {}
    for f in fnames:
        fdict[int(re.findall(r"day([0-9]*)_",f)[0])] = f
    # return sorted file list
    return [value for (key, value) in sorted(fdict.items())]

def procPlate(n, fname, m0r, sub, bbox):
    plate = trimShape(loadTiff(fname), sub)
    m = detectDish(plate,fname)
    if m is None: 
        ##with open("%s/%s/result.txt"%(dirname,sid), 'w') as reportfile: reportLog.write(reportfile)
        return None
    otrans = register(m0r, m, sub, algmax="max",ssigma=7)
    mt = transformItk(m0r, m, otrans)
    platet = transformItk(plate, plate, otrans)
    return (n, cropPlate(img3mask(platet,mt), bbox))

def procPlateSet(inDirName, outDirName, sid, sub=4):
    global reportLog
    reportLog["Directory Name"] = inDirName
    reportLog["Start time"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    files = getFilesAndCreateDir(inDirName, outDirName, sid)
    ## create reference mask based on the first plate
    plate0 = trimShape(loadTiff(files[0]), sub)
    m0 = detectDish(plate0,files[0])
    if m0 is None: 
        #with open("%s/%s/result.txt"%(inDirName,sid), 'w') as reportfile: reportLog.write(reportfile)
        return
    #align dish with image borders
    #ipdb.set_trace()
    rotCenter, rotAngle = getMaskRotationCont(m0)
    M = cv2.getRotationMatrix2D(rotCenter,-rotAngle,1)
    m0r = cv2.warpAffine(m0,M,(m0.shape[1],m0.shape[0]))
    bbox = getBBox(m0r) #bounding box to crop all plates
    plate0r = cv2.warpAffine(plate0,M,(plate0.shape[1],plate0.shape[0]))
    plates = [cropPlate(img3mask(plate0r, m0r), bbox)]

    #for n, fname in enumerate(files[9:]):
    for n, fname in enumerate(files[1:]):
        plates.append(procPlate(n, fname, m0r, sub, bbox)[1])
    plates = np.array(plates)
    reportLog["End time"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    return plates, reportLog

reportLog={}
desc="Identify dish and align plates"
inDirName="."
outDirName=None

dishId=None    # a NNN identifier od a dish

def usage(desc):
    global inDirName, outDirName, dishId
    print(sys.argv[0]+":",   desc)
    print("Usage: ", sys.argv[0], "[switches]")
    print("Switches:")
    print("\t-h .......... this usage")
    print("\t-d name ..... directory with plant datasets (%s)"%inDirName)
    print("\t-o name ..... directory to store the result to (in a NNN subdirecory) (%s)"%"same as input")
    print("\t-p NNN ...... ID of a dish (NNN) to process (all dishes)")

def parsecmd(desc):
    global inDirName, outDirName, dishId
    try:
        opts, Names = getopt.getopt(sys.argv[1:], "hd:p:o:", ["help"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err)) # will print something like "option -a not recognized"
        sys.exit()
    for o, a in opts:
        if o in ("-h", "--help"):
            usage(desc)
            sys.exit()
        elif o in ("-d"):
            inDirName = a
        elif o in ("-o"):
            outDirName = a
        elif o in ("-p"):
            dishId = a

def saveAll(outDirName, dishId, plates, reportLog):
    #report = configparser.ConfigParser()
    #report["platealign.py"] = reportLog
    #with open("%s/%s/result.txt"%(outDirName, dishId), 'w') as reportfile: report.write(reportfile)
    with TiffWriter("%s/%s/plates-%s.tif"%(outDirName, dishId, dishId)) as tif: tif.save(plates)
    imageio.imwrite("%s/%s/plates-%s.png"%(outDirName, dishId, dishId), plates.max(axis=0)[::4,::4,:])

def main():
    global inDirName, outDirName, dishId

    parsecmd(desc)
    outDirName = outDirName if outDirName else inDirName

    if dishId:
        print("Input directory:  %s"%(inDirName))
        print("Output directory: %s/%s"%(outDirName,dishId))
        plates, report = procPlateSet(inDirName, outDirName, dishId)
        saveAll(outDirName, dishId, plates, report)
    else:
        for p in range(200):
            dishId = "%03d"%p
            reportLog={}
            # check if dishId images exist
            pnames = glob.glob("%s/*%s.tif"%(inDirName, dishId))
            if pnames == []: continue   # no such plant
            fnames = glob.glob("%s/%s/plates-%s.png"%(outDirName, dishId, dishId))
            if fnames: 
                print("%s/%s/plates-%s.png exists, skipping"%(outDirName, dishId, dishId))
                continue # plates.tif exists
            print("Input directory:  %s"%(inDirName))
            print("Output directory: %s/%s"%(outDirName,dishId))
            plates, report = procPlateSet(inDirName, outDirName, dishId)
            saveAll(outDirName, dishId, plates, report)

if __name__ == "__main__":
    main()
