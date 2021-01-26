#!/usr/bin/python3
# vim:ts=4:et
#Stage 3: Export regions. Requires file plates-001.tif and seeds-mask-001.tif. Creates 24 files seeds-001-...-.tif and plant-regions-001.png 

# Copyright (C) 2013 Milos Sramek <milos.sramek@soit.sk>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/gpl.html
# - or any later version.

from importlib import reload  
from tifffile import TiffWriter, TiffFile
#import SimpleITK as sitk
import numpy as np
import sys, glob, re, os, getopt, csv
import cv2, math
import ipdb
import phlib
reload(phlib)
from phlib import disp,plot
#from skimage import morphology, filters
from skimage import measure
#import imageio
import scipy.ndimage as ndi
#import scipy.stats as stats
#from scipy.signal import medfilt
#import guiqwt.pyplot as plt
import matplotlib.pyplot as plt

def img3mask(img, mask):
    if len(img) != len(mask):
        print("incorrect dimensions")
        return
    img = np.array(img)
    mask = np.array(mask)
    if img.ndim == 2:
        img = (mask>0)*img
    elif img.ndim == 3 and img.shape[-1] > 3:
        for n in img.shape[0]:
            img[n] = (mask[n]>0)*img[n] 
    elif img.ndim == 3 and img.shape[-1] == 3:
        img[:,:,0] = (mask>0)*img[:,:,0] 
        img[:,:,1] = (mask>0)*img[:,:,1] 
        img[:,:,2] = (mask>0)*img[:,:,2] 
    else:
        for n in img.shape[0]:
            img[n, :,:,0] = (mask[n]>0)*img[n, :,:,0] 
            img[n, :,:,1] = (mask[n]>0)*img[n, :,:,1] 
            img[n, :,:,2] = (mask[n]>0)*img[n, :,:,2] 

    return img

def loadTiff(ifile):
    try:
        with TiffFile(str(ifile)) as tfile:
            vol = tfile.asarray()
        return vol
    except IOError as err:
        print ("%s: Error -- Failed to open '%s'"%(sys.argv[0], str(ifile)))
        sys.exit(0)

def regstat(img, mask):
    """ compute mean vector and covariance matrix of the regions defined by mask"""
    nzero = mask.nonzero()
    return  img[nzero].mean(axis=0), np.cov(img[nzero].T)

def getPlateBackgroundWS(img, sigma=2, level=0.15):
    ws = phlib.watersheditk(img,sigma,level,False)
    # label of the largest region, i.e. the plate background
    bc = np.bincount(ws.flat)
    lmax = bc.argmax()
    return ws==lmax

def getLargest(mask):
    bc = np.bincount(mask.flat)
    lmax = bc.argmax()
    return mask==lmax

#convert plates to gray and normalize them to common mean and sdev
def platesToGray(plates, masks):
    gplates = np.zeros(plates.shape[:3], np.uint8)
    means=[]
    sdevs=[]
    for p in range(plates.shape[0]):
        #gplates[p] = cv2.cvtColor(plates[p], cv2.COLOR_RGB2GRAY)
        gplates[p] = plates[p][...,0]
        mean, cov = regstat(gplates[p],masks[p])
        means.append(mean)
        sdevs.append(np.sqrt(cov))
    means=np.array(means)

    ntarget = np.argmin(np.abs(means-np.median(means)))
    for p in range(gplates.shape[0]):
        gplates[p,...] = normalizeGray(gplates[p], means[p], sdevs[p], means[ntarget], sdevs[ntarget]) 
    return gplates

#convert plates to gray and normalize them to common mean and sdev
def normalizeGrays(gplates, masks):
    means=[]
    sdevs=[]
    for p in range(gplates.shape[0]):
        mean, cov = regstat(gplates[p],masks[p])
        means.append(mean)
        sdevs.append(np.sqrt(cov))
    means=np.array(means)

    ntarget = np.argmin(np.abs(means-np.median(means)))
    for p in range(gplates.shape[0]):
        gplates[p,...] = normalizeGray(gplates[p], means[p], sdevs[p], means[ntarget], sdevs[ntarget]) 
    return gplates

# normalize gray image with smean and scov to image with tmean and tcov
# https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
def normalizeGray(source, smean, scov, tmean, tcov):
    target = source-smean
    target *= tcov/scov
    target += tmean
    return target

def getLargest (mask):
    labels, nlabels = measure.label(mask, return_num=True)
    lsizes = np.bincount(labels.flat)
    #get the largest region
    maxlabel = 1+np.argmax(lsizes[1:])
    return labels == maxlabel

def select_overlaps(mask, prevmask, plantnum=-1, platenum=-1):
    ''' select the region in mask with overlaps in prevmask'''
    labels, nlabels = measure.label(mask, return_num=True)
    ovlaps = np.unique(labels*prevmask)[1:] # the first one is background
    # make premask large, if they do not overlap (happens for seeds)
    while not ovlaps.any(): 
        #ipdb.set_trace()
        print("select_overlaps: dilation of prevmask")
        prevmask = ndi.binary_dilation(prevmask, np.ones((7,1)))
        prevmask = ndi.binary_dilation(prevmask, np.ones((1,7)))
        ovlaps = np.unique(labels*prevmask)[1:] # the first one is background
    
    #remove regions too small <500, a typical seed is > 500
    # Example:  apogwas2//021,5
    if None and len(ovlaps) > 1:
        aux=[]
        for reg in ovlaps:
            if (labels==reg).sum() > 500:
                aux.append(reg)
            else:
                print("select_overlaps: removed small region")
        ovlaps=aux

    #ipdb.set_trace()
    # select all overlapping regions
    gmask = labels.copy()
    gmask[:]=0
    for lbl in ovlaps:
        gmask += (labels == lbl)

    if plantnum >= 0:
        gprof=gmask.sum(axis=0)
        pprof=prevmask.sum(axis=0)
        #print( gprof.max(), 2* pprof.max())
        if gprof.max() > 2* pprof.max():
            if plantnum in (0, 12): # left side images
                print("Plant %2d,%d fix left plant"%(plantnum, platenum))
                gmask = fix_left_plant(gmask, prevmask)
                #gmask2 = fix_border_plant(gmask, prevmask)
                pass
            elif plantnum in (11, 23): # left side images
                print("Plant %2d,%d fix right plant"%(plantnum, platenum))
                gmask = fix_right_plant(gmask, prevmask)
        #ipdb.set_trace()
    return gmask

def select_largest_overlap(mask, prevmask):
    ''' select region in mask with larges overlap in prevmask'''
    labels, nlabels = measure.label(mask, return_num=True)
    largest_overlap = getLargest(mask*prevmask)
    plant_label = (largest_overlap*labels).max()
    return labels == plant_label

# segment plate by thresholding based on background statistics
def segPlateStat(gplate, bgmask=None, thrsigma=4):
    if gplate.ndim > 2: gplate=cv2.cvtColor(gplate, cv2.COLOR_RGB2GRAY)
    gplate = phlib.gaussitk(gplate, 4)
    if not bgmask.any(): bgmask = gplate >=0;
    #estimate statistical parameters of the whole image
    mean, cov = regstat(gplate,bgmask)
    #estimate statistical parameters of what we think is background
    mean, cov = regstat(gplate,bgmask*(gplate < mean + np.sqrt(cov)))
    #return getLargest(bgmask *(gplate > mean + thrsigma*np.sqrt(cov)))
    #ipdb.set_trace()
    return gplate > mean + thrsigma*np.sqrt(cov)

def drawHoughLines(gmask, lines):
    cdst = cv2.cvtColor(200*gmask, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + gmask.shape[0]*(-b)), int(y0 + gmask.shape[0]*(a)))
            pt2 = (int(x0 - gmask.shape[0]*(-b)), int(y0 - gmask.shape[0]*(a)))
            cv2.line(cdst, pt1, pt2, (0,200,0), 3, cv2.LINE_AA)
    return cdst


# a leftmost image can touch something 'big' on the left, usually a vertical strip (or strips)
def fix_left_plant(gmask, prevmask):
    # we assume that the incorrect mask touches top, botton or left border
    if not (gmask[0].any() or gmask[-1].any() and gmask[:,0].any()):
        return gmask
    gmask = gmask.astype(np.uint8)
    # detect vertical strips as lines to estimate their angle
    lines = cv2.HoughLines(gmask, 1, np.pi / 180, int(gmask.shape[0]/2), None, 0, 0)
    #cdst = drawHoughLines(gmask, lines)

    # convert angles > pi/2 to negative
    angles = [ll[0][1] if ll[0][1] < np.pi/2 else ll[0][1] - np.pi for ll in lines]
    rotangle = 180*np.mean(angles)/np.pi
    # analyze only in the vertivcal range of nonzero prevmask values
    nz = np.nonzero(prevmask)
    pmiy = nz[0].min()
    pmay = nz[0].max()
    # align strips vertically
    gmask = ndi.rotate(gmask,rotangle,reshape=False)
    # compute foreground pixels in vertical columns, the strips go top to bottom
    gprof=gmask[pmiy:pmay,:].sum(axis=0)
    gprof = gprof > 0.8*(pmay-pmiy)
    cutpos = np.nonzero(gprof)[0].max() # the rightmost value, we hope this is where the plant touches it
    gmask[:,:cutpos] = 0
    # remove noise along the border
    gmask = ndi.binary_opening(gmask, np.ones((1,5))).astype(np.uint8)
    #rotate back
    gmask = ndi.rotate(gmask,-rotangle,reshape=False)
    return select_overlaps(gmask, prevmask)

# a rightmost image can touch something 'big' on the right, usually a vertical strip (or strips)
def fix_right_plant(gmask, prevmask):
    # we assume that the incorrect mask touches top, botton or right border
    if not (gmask[0].any() or gmask[-1].any() and gmask[:,0].any()):
        return gmask
    gmask = gmask.astype(np.uint8)
    # detect vertical strips as lines to estimate their angle
    lines = cv2.HoughLines(gmask, 1, np.pi / 180, int(gmask.shape[0]/2), None, 0, 0)
    #cdst = drawHoughLines(gmask, lines)

    # convert angles > pi/2 to negative
    angles = [ll[0][1] if ll[0][1] < np.pi/2 else ll[0][1] - np.pi for ll in lines]
    rotangle = 180*np.mean(angles)/np.pi
    # analyze only in the vertivcal range of nonzero prevmask values
    nz = np.nonzero(prevmask)
    pmiy = nz[0].min()
    pmay = nz[0].max()
    # align strips vertically
    gmask = ndi.rotate(gmask,rotangle,reshape=False)
    # compute foreground pixels in vertical columns, the strips go top to bottom
    gprof=gmask[pmiy:pmay,:].sum(axis=0)
    gprof = gprof > 0.8*(pmay-pmiy)
    cutpos = np.nonzero(gprof)[0].min() # the leftmost value, we hope this is where the plant touches it
    gmask[:,cutpos:] = 0
    # remove noise along the border
    gmask = ndi.binary_opening(gmask, np.ones((1,5))).astype(np.uint8)
    #rotate back
    gmask = ndi.rotate(gmask,-rotangle,reshape=False)
    return select_overlaps(gmask, prevmask)

# a rightmost image can touch something 'tall' on the right
def fix_border_plant(gmask, prevmask):
    # we assume that the incorrect mask touches top, bottom or right border
    if not (gmask[0].any() or gmask[-1].any() and gmask[:,-1].any()):
        return gmask
    omask = ndi.binary_opening(gmask, np.ones((25,1)))
    omask = getLargest(omask)
    #ipdb.set_trace()
    omask = select_overlaps(gmask-ndi.binary_dilation(omask, np.ones((2,2))), prevmask)
    return omask

def procplant(plates, plantnum, seedmask):
    gmasks=[seedmask]
    # dilate for the cases when the seed moves a bit prior to germination
    # Example: seed 0 in apogwas2/005   np.ones((19,19)
    # Example: seed 14 in apogwas2/021  np.ones((29,29)
    prevmask = ndi.binary_dilation(seedmask, np.ones((29,29)))
    prevmasksum = seedmask.sum()
    failed = False
    for plnum in range(1,len(plates)):
        plate = plates[plnum]
        gplate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        gplate = phlib.gaussitk(gplate, 2)
        #rb=phlib.rolling_ball_filter(gplate,2,19)
        rb=phlib.rolling_ball_filter(gplate,4,9)

        threshold = 4
        repeat = True
        while repeat:
            #print("Plant %2d,%d: Using threshold %f"%(plantnum, plnum, threshold))
            gmaskall = (gplate.astype(np.float) - rb) > threshold
            gmask = select_overlaps(gmaskall, prevmask, plantnum, plnum)
            #ipdb.set_trace()
            gprof=gmask.sum(axis=0)
            pprof=prevmask.sum(axis=0)
            threshold += 1
            repeat = gprof.max() > 3* pprof.max() and threshold < 7
            #ipdb.set_trace()
            pass
        #ipdb.set_trace()
        gmasks.append(gmask)
        # the plant should normally not shrink, so shrinking is suspicious
        if gmask.sum() < 0.8*prevmasksum:
            print("Plant %2d,%d: Plant detection failed"%(plantnum, plnum))
            failed = True 
        else:
            prevmask=gmask
            prevmasksum = prevmask.sum()
        pass
    masksums = [m.sum() for m in gmasks]
    #print(masksums)

    #ipdb.set_trace()
    return np.array(gmasks).astype(np.uint8), failed
   
desc="segment individual plants in plate data"
dirName="."
dirName="/media/milos/SAN128/data/Patrick/batch1/apogwas2/"
dishId=None
plantNum=None
subStart=0
rWidth = 120
rebuildAll=False

def usage(desc):
    global dirName, dishId, rWidth
    print(sys.argv[0]+":",   desc)
    print("Usage: ", sys.argv[0], "[switches]")
    print("Switches:")
    print("\t-h ............... this usage")
    print("\t-d name .......... directory with plant datasets (%s)"%dirName)
    print("\t-p subdir_name[,plant#] ... process subdirectory with plant data (all subdirs)")
    print("\t-s INT ........... subdirectory number to start from (all subdirs)")
    print("\t-w INT ........... region width in %% of interseed distance (%d %%)"%rWidth)
    print("\t-r ............... rebuild all")

def parsecmd(desc):
    global dirName, dishId, plantNum, subStart, rWidth, rebuildAll
    try:
        opts, Names = getopt.getopt(sys.argv[1:], "hrd:s:p:", ["help"])
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
        elif o in ("-w"):
            rWidth = int(a)
        elif o in ("-r"):
            rebuildAll=True

def main():
    global dirName, dishId, rebuildAll
    parsecmd(desc)
    plantNum=None

    # create lists of directories and file to process

    dishIds = {}    #dishes to process
    reprocessname=None
    if dishId:
        if "," in dishId:
            dishId, plantNum = dishId.split(",")
            reprocessname = glob.glob("%s/%s/plant-%03d-%02d_*.tif"%(dirName, dishId, int(dishId), int(plantNum)))[0]
        plantnames=sorted(glob.glob("%s/%s/plant-*.tif"%(dirName, dishId)))
        dishIds[dishId] = plantnames
    else:
        for p in range(subStart, 200):
            dishId = "%03d"%p
            if glob.glob("%s/%s"%(dirName, dishId)) == []: continue   # no such dish
            if not rebuildAll:
                if os.path.isfile("%s/%s/pmask-%s.tif"%(dirName,dishId,dishId)):
                    print("Skipping %s"%"%s/%s"%(dirName,dishId))
                    continue
            plantnames=sorted(glob.glob("%s/%s/plant-*.tif"%(dirName, dishId)))
            dishIds[dishId] = plantnames
    #ipdb.set_trace()

    for dishId in dishIds:
        seedsmask = loadTiff( "%s/%s/seeds-mask-%s.tif"%(dirName,dishId,dishId))
        plantnames = dishIds[dishId]
        plantmasks_name = "%s/%s/pmask-%s.tif"%(dirName,dishId,dishId)
        plantoverview_name = "%s/%s/pmask-ovl-%s.tif"%(dirName,dishId,dishId)

        #ipdb.set_trace()
        plantoverview=seedsmask.copy()
        plantmasks = np.zeros((11,seedsmask.shape[0],seedsmask.shape[1])).astype(np.uint8)
       
        for plantname in plantnames:
            #ipdb.set_trace()
            failed = False
            ulx, uly, lrx, lry=np.array(re.findall(r"\d+",plantname.split("/")[-1])[2:]).astype(np.int)
            plant = loadTiff(plantname)
            pmaskname = plantname.replace("plant-","pmask-")
            # Reload, if plant mask exists
            if not rebuildAll and os.path.isfile(pmaskname) and plantname != reprocessname:
                print("Reloaded %s"%pmaskname)
                masks = loadTiff(pmaskname)
            else:
                print("Processing %s"%pmaskname)
                #ipdb.set_trace()
                pnum = int(re.findall(r"plant-[0-9]*-([0-9]*)",plantname.split("/")[-1])[0])
                masks, failed = procplant(plant, pnum, seedsmask[uly:lry,ulx:lrx][...,0] == 0)
                if failed:
                    with TiffWriter(plantname.replace("plant-","pmask-").replace(".tif","-failed.tif")) as tif:
                        tif.save(masks,compress=5)
                else:
                    with TiffWriter(plantname.replace("plant-","pmask-")) as tif:
                        tif.save(masks,compress=5)


            #rslt = evaluate_masks(masks)
            plantmasks[...,uly:lry,ulx:lrx] += 255*masks
            plantoverview[uly:lry,ulx:lrx,...] = plant.max(axis=0)
            if failed:
                plantoverview[uly:uly+10,ulx:lrx,...] = (255,0,0)
                plantoverview[lry-10:lry,ulx:lrx,...] = (255,0,0)
            #plantoverview[uly:uly+30,ulx:lrx,...] = 128+(128*np.array(rslt)).astype(np.int)
            #ipdb.set_trace()
            pass

        plantoverview = phlib.img3overlay(plantoverview, plantmasks.max(axis=0))
        with TiffWriter(plantmasks_name) as tif:
            tif.save(plantmasks,compress=5)
        #overview image
        with TiffWriter(plantoverview_name) as tif:
            #tif.save(plantoverview,compress=5)
            tif.save(plantoverview)
            #tif.save(phlib.img3overlay(plantoverview,plantmasks.sum(axis=0)),compress=5)
    #ipdb.set_trace()
    pass

if __name__ == "__main__":
    main()
