#!/usr/bin/python3
# vim:ts=4:et
#Stage 3: Export regions. Requires file plates-001.tif and seeds-mask-001.tif. Creates 24 files seeds-001-...-.tif and plant-regions-001.png 

# Copyright (C) 2013 Milos Sramek <milos.sramek@soit.sk>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/gpl.html
# - or any later version.

from importlib import reload  
from collections import defaultdict
from tifffile import TiffWriter, TiffFile
#import SimpleITK as sitk
import numpy as np
import sys, glob, re, os, getopt, csv, tempfile
import cv2, math, imageio
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

import plateplantseg
reload(plateplantseg)

#type	batch	set_nr.	plate_nr.	plate_id	acc_id	row	column
type,batch,set_nr,plate_nr,plate_id,acc_id,row,column = range(8)
def loadCsv(ifile):
    #ipdb.set_trace()
    acc=defaultdict(list)
    with open(ifile, 'rt', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"',quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if row[0] == "type": continue
            acc[row[acc_id]].append(row)
    return acc

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
    minsize = 400   # minimal blob area to not to be regarded as noise (minimal seed size)

    labels, nlabels = measure.label(mask, return_num=True)
    ovlaps = np.unique(labels*prevmask)[1:] # the first one is background

    # if area of overlapping reagions is too small (overlapping with a noise blob), 
    #   make prevmask larger to find the plant
    # Example:  apogwas2//021,22
    sumovlaps=0
    for lbl in ovlaps:
        sumovlaps += (labels == lbl).sum()

    # check in a loop
    while sumovlaps < minsize: 
        print(f"Plant {plantnum},{platenum} select_overlaps: dilation of prevmask")
        prevmask = ndi.binary_dilation(prevmask, np.ones((7,1)))
        prevmask = ndi.binary_dilation(prevmask, np.ones((1,7)))
        ovlaps = np.unique(labels*prevmask)[1:] # the first one is background
        sumovlaps=0
        for lbl in ovlaps:
            sumovlaps += (labels == lbl).sum()
    
    #remove regions too small <minsize, a typical seed is > minsize
    # Example:  apogwas2//021,5
    if len(ovlaps) > 1:
        aux=[]
        for reg in ovlaps:
            regsize = (labels==reg).sum()
            if regsize > minsize:
                aux.append(reg)
            else:
                print(f"Plant {plantnum},{platenum} select_overlaps: removed blob, size {regsize}")
        ovlaps=aux

    #ipdb.set_trace()
    # select all overlapping regions
    gmask = labels.copy()
    gmask[:]=0
    for lbl in ovlaps:
        gmask += (labels == lbl)
        # the problems occur for large platenums and height increase may be large for platenum == 1
        # thus, check only id platenum > 1
        if platenum > 1 and plantnum in (0, 12, 13, 23): # left side images
            # if gmask height increases too much, we have the border problem. So fix it
            gmaskheight = np.nonzero(gmask)[0].max() - np.nonzero(gmask)[0].min()
            pmaskheight = np.nonzero(prevmask)[0].max() - np.nonzero(prevmask)[0].min()
            #ipdb.set_trace()
            if gmaskheight > 2* pmaskheight:
                if plantnum in (0, 12): # left side images
                    print("Plant %2d,%d fix left plant"%(plantnum, platenum))
                    gmask = fix_left_plant(gmask, prevmask)
                elif plantnum in (11, 23): # right side images
                    print("Plant %2d,%d fix right plant"%(plantnum, platenum))
                    gmask = fix_right_plant(gmask, prevmask)
                pass
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

def linfit(x, data):
    if len(x) == 2:
        m = (data[1]-data[0])/(x[1]-x[0])
        c = ((data[1]+data[0])-m*(x[1]+x[0]))/2
        return x, data, m, c, 0
    else:
        A = np.vstack([x, np.ones(len(x))]).T
        (m, c), res = np.linalg.lstsq(A, data, rcond=None)[:2]
        return x, data, m, c, np.sqrt(res[0])

def linplot(pdata):
    plt.clf()
    #pdata; [[x, data, m, c], [...], ...)
    for (x, data, m, c) in pdata:
        _ = plt.plot(x, data, 'o', label='Original data', markersize=10)
        _ = plt.plot(x, m*x + c, 'r', label='Fitted line')
    plt.show()


def linplotarray(pdata):
    plt.clf()
    ymax = np.max([np.max(pd[1]) for pd in pdata]) 
    plt.ylim(0, 1.1*ymax)
    for (x, data, m, c) in pdata:
        _ = plt.plot(x, data, 'o', label='Original data', markersize=10)
        _ = plt.plot(x, m*x + c, 'r', label='Fitted line')
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    return image

def procplant(plant_name, mask_name):
    masks  = loadTiff(mask_name)
    plates = loadTiff(plant_name)
    #ipdb.set_trace()
    maskheight = [np.nonzero(m)[0].max() - np.nonzero(m)[0].min() if m.max() > 0 else 0 for m in masks]
    return_state = plateplantseg.classifyGrowth(plates.shape[1], maskheight)

    # create plant growth image
    cmasks = np.concatenate(masks, axis=1)[::2,::2]
    nz = np.nonzero(cmasks)
    cmasks = cmasks[nz[0].min()-5 : nz[0].max()+5,:]
    cplant = np.concatenate(plates, axis=1)[::2,::2]
    cplant = cplant[nz[0].min()-5 : nz[0].max()+5,:]
    oplant = phlib.img3overlay(cplant, cmasks)
    return return_state+[oplant], maskheight

desc="Create report for individual accessions as defined by a csv/tsv file"
dirName="."
dirName="/media/milos/SAN128/data/Patrick/all/"
tsvName="apogwas.csv"
dishId=None
accIds=[]
plantNum=None
subStart=0
rWidth = 120
rebuildAll=False
reportWriter=None

def usage(desc):
    global dirName, accIds, rWidth
    print(sys.argv[0]+":",   desc)
    print("Usage: ", sys.argv[0], "[switches]")
    print("Switches:")
    print("\t-h ............... this usage")
    print("\t-d name .......... directory with plant datasets (%s)"%dirName)
    print("\t-a id,id,......... list aof accession ids, separated by a comma")
    print("\t-r ............... rebuild all")

def parsecmd(desc):
    global dirName, rebuildAll, accIds
    try:
        opts, Names = getopt.getopt(sys.argv[1:], "hrd:s:a:", ["help"])
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
        elif o in ("-a"):
            accIds = a.split(",")
        elif o in ("-s"):
            subStart = int(a)
        elif o in ("-w"):
            rWidth = int(a)
        elif o in ("-r"):
            rebuildAll=True

def main():
    global dirName, accIds, rebuildAll, reportWriter
    parsecmd(desc)

    accessions = loadCsv(f"{dirName}/{tsvName}")
    for accession in accessions:
        #check first, if all plates exist (important in testing)
        if not accession in accIds: continue

        plant_dirs = {} 
        for acs in accessions[accession]:
            pdirectory = "%s/apogwas%s/%03d"%(dirName,acs[batch],int(acs[plate_id]))
            if os.path.isdir(pdirectory):
                plant_dirs[pdirectory] = acs
            pass
        if plant_dirs and len(plant_dirs) == 8:
            print(f"Processing accession {accession}") 
            controls=[]
            apos=[]
            for ppd in plant_dirs:
                pd = plant_dirs[ppd]
                pos = 3*(4*(int(pd[row])-1) + int(pd[column]) -1)
                for n in range(3):
                    mask_name = glob.glob("%s/apogwas%s/%03d/pmask-*-%02d_*.tif"%(dirName,pd[batch],int(pd[plate_id]), pos+n))[0]
                    plant_name =glob.glob("%s/apogwas%s/%03d/plant-*-%02d_*.tif"%(dirName,pd[batch],int(pd[plate_id]), pos+n))[0]
                    print(ppd, pd, pos+n, mask_name)
                    rslt, maskheight = procplant(plant_name, mask_name)
                    if pd[type] == "control":
                        controls.append([["batch%s"%pd[batch], "%03d/%d"%(int(pd[plate_id]),pos+n)]+rslt, maskheight])
                    else:
                        apos.append([["batch%s"%pd[batch], "%03d/%d"%(int(pd[plate_id]),pos+n)]+rslt, maskheight])
                    pass
                #reportWriter.writerow(retval)
            reportWriter = plateplantseg.ODSWriter()
            hdr=["Batch","Plate Id","Type","Growth rate","From day", "Residuals", "Growth plot","Plant growth, days 0 – 10"]

            reportWriter.addtable("control", hdr)
            ok_data=[]
            pnames=[]
            for rr in controls: 
                reportWriter.writerow(rr[0])
                if not "error" in rr[0][2]:
                    ok_data.append(rr[1])
                    pnames.append("%s/%s"%(rr[0][0],rr[0][1]))
            oo=np.array(ok_data).T
            omean=oo.mean(axis=1)
            osdev=np.sqrt(oo.var(axis=1))
            ot = np.hstack((oo,omean.reshape(omean.shape[0],1),osdev.reshape(omean.shape[0],1)))

            hdr_data=["Day"]+[p for p in pnames] + ["Mean", "SDev",".""."]
            reportWriter.addtable("control-data", hdr_data)
            for n, rr in enumerate(ot): 
                reportWriter.writerow([n]+[r for r in rr])

            reportWriter.addtable("apo", hdr)
            ok_data=[]
            pnames=[]
            for rr in apos: 
                reportWriter.writerow(rr[0])
                if not "error" in rr[0][2]:
                    ok_data.append(rr[1])
                    pnames.append("%s/%s"%(rr[0][0],rr[0][1]))
            oo=np.array(ok_data).T
            omean=oo.mean(axis=1)
            osdev=np.sqrt(oo.var(axis=1))
            ot = np.hstack((oo,omean.reshape(omean.shape[0],1),osdev.reshape(omean.shape[0],1)))

            hdr_data=["Day"]+[p for p in pnames] + ["Mean", "SDev",".""."]
            reportWriter.addtable("apo-data", hdr_data)
            for n, rr in enumerate(ot): 
                reportWriter.writerow([n]+[r for r in rr])

            reportWriter.save("%s/acc-report-%s.ods"%(dirName,accession))
            #ipdb.set_trace()
            pass
    pass
if __name__ == "__main__":
    main()
