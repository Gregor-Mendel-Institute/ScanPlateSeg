#!/usr/bin/python3
# vim:ts=4:et
#Stage 2: identify seeds. Requires file plates-001.tif. Creates files seeds-001.tif and seeds-mask-001.tif 

# Copyright (C) 2013 Milos Sramek <milos.sramek@soit.sk>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/gpl.html
# - or any later version.

from tifffile import TiffWriter, TiffFile
import sys, glob, shutil, os, getopt
import numpy as np
import phlib
from skimage import measure
import cv2, imutils
import scipy.ndimage as ndi
from time import gmtime, strftime
#import ipdb

def plot(data):
    # data is tuple of lists
    for d in data:
        plt.plot(range(len(d)),d)
    plt.show()

def disp(iimg, label = None, gray=False):
    """ Display an image using pylab
    """
    import pylab, matplotlib
    matplotlib.interactive(True)
    matplotlib.pyplot.imshow(iimg, interpolation='none')

def maskPlate3(img, mask):
    for n in range(img.shape[2]):
        band = img[:,:,n]
        band[np.nonzero(mask==0)] = band[np.nonzero(mask!=0)].mean()
    return img

def trimShape(img, n):
    ts = (n*int(img.shape[0]/n), n*int(img.shape[1]/n), img.shape[2])
    return img[:ts[0], :ts[1], :]

def img3mask(img, mask):
    img=img.copy()
    img[:,:,0] = (mask>0)*img[:,:,0] 
    img[:,:,1] = (mask>0)*img[:,:,1] 
    img[:,:,2] = (mask>0)*img[:,:,2] 
    return img

def img3overlay(img, mask):
    mask = (mask + ndi.binary_dilation(mask))==1
    nz = mask.nonzero()
    img[:,:,2][nz]=255
    return img
 
def loadTiff(ifile):
    try:
        with TiffFile(str(ifile)) as tfile:
            vol = tfile.asarray()
        return vol
    except IOError as err:
        print ("%s: Error -- Failed to open '%s'"%(sys.argv[0], str(ifile)))
        sys.exit(0)

def getPlateBackground(img, sub=4, sigma=2, level=0.15):
    ws = phlib.watersheditk(img[::sub,::sub],sigma,level,False)
    # label of the largest region, i.e. the plate background
    bc = np.bincount(ws.flat)
    lmax = bc.argmax()
    ws = ndi.zoom(ws, sub, order=0)
    return (ws==lmax)[:img.shape[0], :img.shape[1]] # sometines is larger

def drawCnts(img, cnts):
    img=np.zeros(img.shape[:2],np.uint8)
    for cnt, n in zip(cnts,range(len(cnts))):
        cv2.drawContours(img, [cnt], 0 ,10+n,cv2.FILLED)
    return img

def getdistvar(xPos, ex):
    if ex < 0:
        epos = xPos
    else:
        epos = np.concatenate((xPos[:ex],xPos[ex+1:]))
    diffs = epos[1:]-epos[:-1]
    return diffs.var()

def checkBlobPos(xPos):
    xPos=np.array(xPos)
    origdvar = getdistvar(xPos, -1)
    dvar = []
    for p in range(1,len(xPos)-1): 
        dvar.append(getdistvar(xPos,p))
    #if blobs are correct, removing any one of then deteriorated the distribution
    # otherwise there is at least one, which improves it
    return min(dvar) > origdvar

def checkBlobs(labels):
    # labels: a row of blobs
    # returns True, if their positions are OK
    # the first and last blobs are not checked 
    rslt=[]
    cnts, hrch = cv2.findContours((labels>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #sort cnts according to x position
    cnts = sorted(cnts, key=lambda x: cv2.moments(x)["m10"]/cv2.moments(x)["m00"])
    xPos=[]
    for cnt in cnts:
        x,y,w,h=cv2.boundingRect(cnt)
        xPos.append(int(x+w/2))
    #ipdb.set_trace()
    return checkBlobPos(xPos)

def getoutlier(xPos, val):
    #ipdb.set_trace()
    #check if insdide is OK, then the bordering ones
    if checkBlobPos(xPos[:-1]): return len(xPos)-1  # the last one is bad
    if checkBlobPos(xPos[1:]): return 0  # the first one is bad
    dvar=[]
    for p in range(len(xPos)): 
        dvar.append(getdistvar(xPos,p))
    cand = np.argmin(dvar)    #candidate
    mdist = np.median(xPos[1:] - xPos[:-1])
    if xPos[cand] - xPos[cand-1] < mdist/2: 
        rtrn = cand if val[cand-1] > val[cand]  else cand-1
    else:
        rtrn = cand+1 if val[cand] > val[cand+1]  else cand
    return rtrn
    

# prune region list solely by analyzing blob positions
def pruneRegionListPos(plate, labels):
    cnts, hrch = cv2.findContours((labels>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #sort cnts according to x position
    cnts = sorted(cnts, key=lambda x: cv2.moments(x)["m10"]/cv2.moments(x)["m00"])
    origlen = len(cnts)
# exclude blobs in wrong positions
    # get blob positions
    xPos=[]
    val=[]
    for cnt in cnts:
        x,y,w,h=cv2.boundingRect(cnt)
        xPos.append(int(x+w/2))
        val.append(np.median(plate[y:y+h, x:x+w]))
    xPos=np.array(xPos)
    val=np.array(val)
    # removing a blob at wrong position minimizes variance of position distances
    #xPos=np.concatenate((np.array([100]),xPos[:5], np.array([(xPos[4]+xPos[5])/2]),xPos[5:7], np.array([(xPos[7]+xPos[8])/2]),xPos[7:]))
    #ipdb.set_trace()
    while len(xPos) > 12:
        ex = getoutlier(xPos, val)
        xPos = np.concatenate((xPos[:ex], xPos[ex+1:]))
        val = np.concatenate((val[:ex], val[ex+1:]))
        cnts = cnts[:ex]+cnts[ex+1:]
    return cnts, "pruneRegionListPos removed_regions_by_position_analysis: %d"%(origlen-12)


def _pruneRegionListCont(plate, labels):
    cnts, hrch = cv2.findContours((labels>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lrBorder = 0.1 #ignore blobs in left and right border in data collection
    rslt=[]
    #sort cnts according to x position, good for debugging
    cnts = sorted(cnts, key=lambda x: cv2.moments(x)["m10"]/cv2.moments(x)["m00"])
    props=[] #blob properties to classify
    refcnts=[]
    refprops=[] #properties of reference blobs 
    yRowPos=0   # row mean position
    xPos=[]

    excludePos(cnts, labels>0)
    getTriples(cnts,labels>0)

    # get mean vertical position and a list of horizontal positions
    for cnt in cnts:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        yRowPos += cY   #for mean row position
        xPos.append(cX) # to compute distance to the nearest
    xPos = np.array(xPos)

    # collect data
    #ipdb.set_trace()
    for cnt, pos in zip(cnts, xPos):
        _,(a,b),_=cv2.fitEllipse(cnt)  #get semiaxes lengths
        apos = np.abs(xPos-pos) 
        apos[np.argmin(apos)]=labels.shape[1] #rewrite 0 by something big
        mindist = apos.min()
        area = cv2.contourArea(cnt)
        x,y,w,h=cv2.boundingRect(cnt)
        proplist = [y+h/2,area,a/b]
        props.append(proplist)
        if pos > lrBorder * plate.shape[1] and pos < (1-lrBorder)*plate.shape[1]: 
            refprops.append(proplist)
            refcnts.append(cnt)

    props = np.array(props)
    refprops = np.array(refprops)
    refpmean = refprops.mean(axis=0)
    refpcov = np.cov(refprops.T)
    dist = spatial.distance.cdist(props, [refpmean], metric='mahalanobis', V=refpcov)
    dist = dist.reshape(dist.shape[0])
    #ipdb.set_trace()
    if len(dist) > 12:
        rslt.append(["pruneRegionListCont removed_regions_by_Mahalanobis_distance",len(dist)-12])  
        thr = np.sort(dist)[12]
        rsltcnts = [cnt for cnt, d in zip(cnts,dist) if d < thr]

    #ipdb.set_trace()
    #disp(drawCnts(plate, rsltcnts))
    return rsltcnts, rslt

def getTriples(cnts,mask):
    nz = np.nonzero(mask)
    sy = slice(nz[0].min(), nz[0].max())
    sx = slice(nz[1].min(), nz[1].max())
    dist=ndi.morphology.distance_transform_edt(mask[sy,sx]==0)
    # detect triples
    #get three most prominent maxima, 
    distmax = dist.max(axis=0)
    locmax = distmax == ndi.morphology.grey_dilation(distmax, size=101) #large kernel to filter out noise
    locmaxval = distmax[np.nonzero(locmax)]
    #get three most prominent maxima
    max3=[]
    max3.append(np.argmax(locmaxval))
    locmaxval[max3[-1]] = 0
    max3.append(np.argmax(locmaxval))
    locmaxval[max3[-1]] = 0
    max3.append(np.argmax(locmaxval))
    locmaxval[max3[-1]] = 0
    max3 = sorted(max3)

    #ipdb.set_trace()
    pass

# get bonding box in the horizontal direction
def getXBBox(sv):
    borderfract=8   #upper and lower 1/borderfract is dish
    medianmult = 10 #threshold factor above median to identify border (3 is not enough, regards seeds as border)
    border = int(sv.shape[0]/borderfract)
    # properties of sv depend on the calling function
    xsum = sv[border:-border,...].sum(axis=0)
    xsum -= np.median(xsum) # move the profile above zero, so that its maximum is positive
    xthr = medianmult*np.max(xsum[int(len(xsum)/4):3*int(len(xsum)/4)])
    xlabels = measure.label(xsum<xthr)
    xmax = np.bincount(xlabels).argmax()
    xnz = np.nonzero(xlabels == xmax) 
    xs = slice(np.min(xnz), np.max(xnz))
    #ipdb.set_trace()
    return (slice(sv.shape[0]),xs)

def findSeeds(plates, gsigma, mergelevel):
    dogsigma = 10
    dogthresh = 15
    confDict={}
    confDict["dog_sigma"]=str(dogsigma)

    #detect seeds by the DOG filter
    bgmask = segBgGrayWS(plates[0], gsigma=3, mergelevel=0.7)
    gplate = platesToGray(plates[:1], bgmask).max(axis=0).astype(np.float)  
    xBBox = getXBBox(gplate)# strip left and right border - a lot of junk there
    dogplate=(ndi.gaussian_filter(gplate[xBBox], dogsigma)-ndi.gaussian_filter(gplate[xBBox], 1.6*dogsigma))
    dogmask = findSeedsDog(dogplate, dogthresh)

    mask1 = ndi.binary_dilation(dogmask,np.ones((25,1)))
    #ipdb.set_trace()
    #find two highest peeks in blobcnt
    # count number of blobs along rows
    blobcnt=np.array([measure.label(mm, return_num=True)[1] for mm in mask1])
    # label individual regions
    collabels, nlabels = measure.label(blobcnt>0, return_num=True)
    # find maximum value in each region
    lmax = [blobcnt[np.nonzero(collabels==m+1)].max() for m in range(nlabels)] 
    #select the two with maximum value
    l1 = np.argmax(lmax)
    lmax[l1]=0
    l2 = np.argmax(lmax)    
    seedcnts = []
    for ll, row in zip((l1, l2),("row1", "row2")):
        # create mask of the row
        seedrow = np.array((collabels==ll+1))
        rowmask = dogmask.copy()
        # mask out other regions
        for n in range(rowmask.shape[1]):
            rowmask[:,n] *= seedrow
        labels, nlabels = measure.label(rowmask, return_num=True)
        #ipdb.set_trace()
        log = "Initial nLabels: %d"%nlabels
        dt = dogthresh
        while dt > dogthresh/2 and nlabels < 12 or (nlabels==12 and not checkBlobs(rowmask)):
            # we gradually decrease dt to get larger regions
            dt *= 0.8
            rowmask = findSeedsDog(dogplate, dt)
            for n in range(rowmask.shape[1]):
                rowmask[:,n] *= seedrow
            labels, nlabels = measure.label(rowmask, return_num=True)
        log = "%s, nlabels < 12, dogthresh=%f"%(log, dt) 
        if dt < dogthresh/2:
            cnts, hrch = cv2.findContours((labels>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        elif nlabels == 12:
            cnts, hrch = cv2.findContours((labels>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            log = "%s, nlabels == 12"%(log)
        elif nlabels > 12:
            cnts, rslt = pruneRegionListPos(dogplate, labels)
            log = "%s, nlabels > 12, %s"%(log, rslt)
            pass
        confDict[row] = log
        seedcnts += cnts
    pass
    confDict["success"]="yes"
    rowmask = np.zeros(plates[0].shape[:2], np.uint8)
    rowmask[xBBox]=cv2.drawContours(rowmask[xBBox], seedcnts, -1 ,1,cv2.FILLED)
    #disp(rowmask)
    return confDict, rowmask

def getPlateBackgroundWS(img, sub=4, sigma=2, level=0.15):
    ws = phlib.watersheditk(img[::sub,::sub],sigma,level,False)
    # label of the largest region, i.e. the plate background
    bc = np.bincount(ws.flat)
    lmax = bc.argmax()
    ws = ndi.zoom(ws, sub, order=0)
    return ws==lmax

#find dish backgroud mask
def segBgGrayWS(plate, gsigma=3, mergelevel=3):
    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    bg=getPlateBackgroundWS(gray, 2, gsigma, mergelevel)
    bg = ndi.binary_fill_holes(bg)
    return bg

def imHist(im):
    """
    compute histogram of a single band image
    returns array of frequencies
    """
    maxid = np.amax(im)
    hg = cv2.calcHist([im.astype(np.float32)],[0], None, [int(maxid+1)],[0, int(maxid+1)]).astype(np.int)
    return hg.flatten() # hg is a maxval x 1 2D array, make it 1D

def regstat(img, mask):
    """ compute mean vector and covariance matrix of the regions defined by mask """
    nzero = mask.nonzero()
    return  img[nzero].mean(axis=0), np.cov(img[nzero].T)

#convert plates to gray and normalize them to common mean and sdev
def platesToGray(plates, mask):
    gplates = np.zeros(plates.shape[:3], np.uint8)
    means=[]
    sdevs=[]
    for p in range(plates.shape[0]):
        gplates[p] = cv2.cvtColor(plates[p], cv2.COLOR_RGB2GRAY)
        mean, cov = regstat(gplates[p],mask)
        means.append(mean)
        sdevs.append(np.sqrt(cov))
    means=np.array(means)
    ntarget = np.argmin(np.abs(means-np.median(means)))
    hists=[]
    maxvals=[]
    for p in range(gplates.shape[0]):
        if np.abs(means[p]-means[ntarget])/means[ntarget] > 0.05:
            gplates[p,...] = normalizeGray(gplates[p], means[p], sdevs[p], means[ntarget], sdevs[ntarget]) 
        imhist=imHist(gplates[p])
        hists.append(imhist)
        cntperc= 99 * mask.size/100
        cnt=0
        for ii in range(len(imhist)):
            cnt += imhist[ii]
            if cnt > cntperc: break
        maxvals.append(ii)

    for gp in range(gplates.shape[0]):
        gplates[gp]=stretchPlate(gplates[gp],np.mean(means),np.mean(maxvals))

    return gplates

def stretchPlate(img, minval, maxval):
    minval=int(minval)
    iii=img.copy()
    iii[img<minval]=minval
    iii = 255.*(iii-minval)/(maxval-minval)
    iii[iii>255]=255
    return iii.astype(np.uint8)

# identify seedss in the DOG image, exclude some according to size and shape
def findSeedsDog(dogplate, dogthresh=3):
    # seed shape parameters
    circ_thr = 6.5
    d_min = 20
    d_max = 100
    borderfract=8   #upper and lower 1/borderfract is dish
    dogmask = (dogplate > dogthresh).astype(np.uint8)
    border = int(dogmask.shape[0]/borderfract)
    dogmask[:border,:] = 0
    dogmask[-border:,:] = 0
    #dogmask = ndi.binary_closing(dogmask, np.ones((60, 1)))
    #dogmask = ndi.binary_closing(dogmask, np.ones((1, 60))).astype(np.uint8)
    contours = cv2.findContours(dogmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
    seedcnts=[]
    for cnt in contours:
        #rect =cv2.boundingRect(cnt)
        x,y,w,h=cv2.boundingRect(cnt)
        center = (int((x+x+w)/2),int((y+y+h)/2))
        d = int((w+h)/2)    #diameter estimate
        di = min((w,h))    #min diameter estimate
        da = max((w,h))    #max diameter estimate
        cv2.rectangle(dogmask,(x,y),(x+w, y+h),2,5)
        if w/h < circ_thr and h/w < circ_thr and di > d_min and da < d_max:
            _,(a,b),_=cv2.fitEllipse(cnt)  #get semiaxes lengths
            cv2.circle(dogmask,center,d,3,5)
            if a/b < circ_thr and b/a < circ_thr:
                cv2.circle(dogmask,center,2*d,3,5)
                seedcnts.append(cnt)

    #ipdb.set_trace()
    seedmask = np.zeros(dogmask.shape,np.uint8)
    seedmask=cv2.drawContours(seedmask, seedcnts, -1 ,1,cv2.FILLED)
    return seedmask

def procPlateSet(plates, gsigma=3, mergelevel=4):
    reportLog={}
    reportLog["Start time"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    #load all plates from a single file
    # make single band based on hsv
    rslt, mask = findSeeds(plates,gsigma, mergelevel)
    for key in rslt:
        reportLog[key] = rslt[key]
    reportLog["End time"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    return mask, reportLog

mergelevel = 4
gsigma=3
desc="Identify seeds in plate images"
dirName="."
dishId=None
subStart=0

def usage(desc):
    print(sys.argv[0]+":",   desc)
    print("Usage: ", sys.argv[0], "[switches]")
    print("Switches:")
    print("\t-h ............... this usage")
    print("\t-d name .......... directory with plant datasets (%s)"%dirName)
    print("\t-p NNN ........... ID of a dish (NNN) to process (all dishes)")

def parsecmd(desc):
    global dirName, dishId, subStart
    try:
        opts, Names = getopt.getopt(sys.argv[1:], "hd:ms:p:", ["help"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err)) # will print something like "option -a not recognized"
        sys.exit()
    for o, a in opts:
        if o in ("-h", "--help"):
            usage(desc)
            sys.exit()
        elif o in ("-d"):
            dirName = a
        elif o in ("-p"):
            dishId = a
        elif o in ("-s"):
            subStart = int(a)


def main():
    global dirName, dishId, subStart
    parsecmd(desc)
    if dishId:
        print("Work directory: %s/%s"%(dirName,dishId))
        platesfile = "%s/%s/plates-%s.tif"%(dirName,dishId,dishId)
        plates = loadTiff(platesfile)
        mask, report = procPlateSet(plates, gsigma=gsigma, mergelevel=mergelevel)
        with TiffWriter("%s/%s/seeds-%s.tif"%(dirName,dishId,dishId)) as tif:
            tif.save(img3mask(plates[0],mask),compress=5)
        with TiffWriter("%s/%s/seeds-mask-%s.tif"%(dirName,dishId,dishId)) as tif:
            tif.save(img3mask(plates[0]+1,1-mask),compress=5)
    else:
        for p in range(subStart, 200):
            dishId = "%03d"%p
            fnames = glob.glob("%s/%s"%(dirName, dishId))
            if fnames == []: continue   # no such plant
            fnames = glob.glob("%s/%s/seeds-%s.tif"%(dirName, dishId, dishId))
            print("Work directory: %s/%s"%(dirName,dishId))
            platesfile = "%s/%s/plates-%s.tif"%(dirName,dishId,dishId)
            plates = loadTiff(platesfile)
            mask, report = procPlateSet(plates, gsigma=gsigma, mergelevel=mergelevel)
            with TiffWriter("%s/%s/seeds-%s.tif"%(dirName,dishId,dishId)) as tif:
                tif.save(img3mask(plates[0],mask),compress=5)
            with TiffWriter("%s/%s/seeds-mask-%s.tif"%(dirName,dishId,dishId)) as tif:
                tif.save(img3mask(plates[0]+1,1-mask),compress=5)

if __name__ == "__main__":
    main()
