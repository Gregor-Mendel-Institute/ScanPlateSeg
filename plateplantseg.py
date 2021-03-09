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
from odf.opendocument import OpenDocumentSpreadsheet
from odf.style import Style, TextProperties, ParagraphProperties, TableColumnProperties, TableCellProperties,TableRowProperties
from odf.text import P, A
from odf.table import Table, TableColumn, TableRow, TableCell
#from odf.draw import Frame, Image
from odf import draw
from odf.office import Annotation

class ODSWriter:

    def __init__(self):
        self.doc = OpenDocumentSpreadsheet()
        #styles
        self.itemRowStyle1 = Style(name="itemRowStyle", family="table-row")
        self.itemRowStyle1.addElement(TableRowProperties(rowheight="7mm"))
        self.doc.automaticstyles.addElement(self.itemRowStyle1)

        self.itemRowStyle3 = Style(name="itemRowStyle", family="table-row")
        self.itemRowStyle3.addElement(TableRowProperties(rowheight="30mm"))
        self.doc.automaticstyles.addElement(self.itemRowStyle3)

        self.colStyle30 = Style(name="colStyle30", family="table-column")
        self.colStyle30.addElement(TableColumnProperties(columnwidth="25mm"))
        self.doc.automaticstyles.addElement(self.colStyle30)

        self.colStyle40 = Style(name="colStyle40", family="table-column")
        self.colStyle40.addElement(TableColumnProperties(columnwidth="40mm"))
        self.doc.automaticstyles.addElement(self.colStyle40)

        self.colStyle50 = Style(name="colStyle50", family="table-column")
        self.colStyle50.addElement(TableColumnProperties(columnwidth="50mm"))
        self.doc.automaticstyles.addElement(self.colStyle50)

        self.colStyle200 = Style(name="colStyle200", family="table-column")
        self.colStyle200.addElement(TableColumnProperties(columnwidth="200mm"))
        self.doc.automaticstyles.addElement(self.colStyle200)


        self.cellStyle1 = Style(name="cellStyle1",family="table-cell", parentstylename='Standard', displayname="middle")
        self.cellStyle1.addElement(ParagraphProperties(textalign="center"))
        self.cellStyle1.addElement(TableCellProperties(verticalalign="middle"))
        self.cellStyle1.addElement(TableCellProperties(wrapoption="wrap"))
        self.doc.automaticstyles.addElement(self.cellStyle1)

        self.hdrStyle = Style(name="hdrStyle",family="table-cell", parentstylename='Standard', displayname="middle")
        self.hdrStyle.addElement(ParagraphProperties(textalign="center"))
        self.hdrStyle.addElement(TextProperties(fontweight="bold"))
        self.hdrStyle.addElement(TableCellProperties(verticalalign="middle"))
        self.doc.automaticstyles.addElement(self.hdrStyle)


    def addtable(self, name, hdr): 
        self.table = Table(name=str(name))
        self.doc.spreadsheet.addElement(self.table)

        for h in hdr[:-2]:
            tcol = TableColumn(stylename=self.colStyle30)
            self.table.addElement(tcol)
        tcol = TableColumn(stylename=self.colStyle40)
        self.table.addElement(tcol)
        tcol = TableColumn(stylename=self.colStyle200)
        self.table.addElement(tcol)

        self.exrow=1
        tr = TableRow()
        for item in hdr:
            tc = TableCell(stylename="hdrStyle") #empty cell
            tr.addElement(tc)
            p = P(text=item)
            tc.addElement(p)
        self.table.addElement(tr)
        return

    def writerow(self, items): 
        self.exrow += 1
            #pass
        # If there is and image in the row, make the row high
        textrow = True
        for item in items:
            if isinstance(item, np.ndarray):
                textrow = False
                break


        if textrow:
            tr = TableRow(stylename=self.itemRowStyle1)
        else:
            tr = TableRow(stylename=self.itemRowStyle3)

        cells = "ABCDEFGHIJKLM"
        for n in range(len(items)):
            if isinstance(items[n], (int, np.int64)):
                tc = TableCell(valuetype="float", value=str(items[n]), stylename="cellStyle1")
                p = P(text=items[n])
            elif isinstance(items[n], float):
                tc = TableCell(valuetype="float", value=str("%4.1f"%items[n]), stylename="cellStyle1")
                p = P(text=items[n])
            elif isinstance(items[n], np.ndarray):
                tc = TableCell(stylename="cellStyle1")
                fname = tempfile.mktemp(".jpg")
                sf=0.08
                im = items[n]
                imageio.imwrite(fname, items[n])
                f = draw.Frame(endcelladdress="import.%s%d"%(cells[n],self.exrow),endx="%dmm"%int(sf*im.shape[1]), endy="%dmm"%int(sf*im.shape[0]))
                tc.addElement(f)
                href=self.doc.addPicture(fname)
                i = draw.Image(href=href, type="simple", show="embed", actuate="onLoad")
                f.addElement(i)
                p = P(text="")
                i.addElement(p)
            else:
                tc = TableCell(stylename="cellStyle1") #empty cell
                p = P(text=items[n])
            tc.addElement(p)
            tr.addElement(tc)
        self.table.addElement(tr)
        return

    def save(self, ofname):
        self.doc.save(ofname)

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
    iters = 15      # number of repetitions in dilation (dilation size iter*7) (required by the plant 013/9)
    fail_growth_factor = 1.5  #growth error if mask grows more that fail_growth_factor*pmaskheight"

    labels, nlabels = measure.label(mask, return_num=True)
    ovlaps = np.unique(labels*prevmask)[1:] # the first one is background

    # if area of overlapping reagions is too small (overlapping with a noise blob), 
    #   make prevmask larger to find the plant
    # Example:  apogwas2//021,22
    sumovlaps=0
    for lbl in ovlaps:
        sumovlaps += (labels == lbl).sum()

    # check in a loop
    while sumovlaps < minsize and iters: 
        iters -= 1
        #print(f"Plant {plantnum},{platenum} select_overlaps: dilation of prevmask")
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
            #else:
                #print(f"Plant {plantnum},{platenum} select_overlaps: removed blob, size {regsize}")
        ovlaps=aux

    #ipdb.set_trace()
    # select all overlapping regions
    gmask = labels.copy()
    gmask[:]=0
    for lbl in ovlaps:
        gmask += (labels == lbl)
        # the problems occur for large platenums and height increase may be large for platenum == 1
        # thus, check only id platenum > 1
        if platenum > 1 and plantnum in (0, 11, 12, 23): # left side images
            # if gmask height increases too much, we have the border problem. So fix it
            gmaskheight = np.nonzero(gmask)[0].max() - np.nonzero(gmask)[0].min()
            pmaskheight = np.nonzero(prevmask)[0].max() - np.nonzero(prevmask)[0].min()
            if gmaskheight > fail_growth_factor * pmaskheight:
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
    if not (gmask[0].any() or gmask[-1].any() or gmask[:,0].any()):
        return gmask
    gmask = gmask.astype(np.uint8)

    # detect vertical strips as lines to estimate their angle
    lines = cv2.HoughLines(gmask, 1, np.pi / 180, int(gmask.shape[0]/2), None, 0, 0)
    if lines is None:
        return gmask

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
    if not gprof.any():
        # unclear case, return somethinng which would be late classified as failure
        return gmask
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
    if not (gmask[0].any() or gmask[-1].any() or gmask[:,-1].any()):
        return gmask
    gmask = gmask.astype(np.uint8)
    # detect vertical strips as lines to estimate their angle
    lines = cv2.HoughLines(gmask, 1, np.pi / 180, int(gmask.shape[0]/2), None, 0, 0)
    if lines is None:
        return gmask

    # convert angles > pi/2 to negative
    angles = [ll[0][1] if ll[0][1] < np.pi/2 else ll[0][1] - np.pi for ll in lines]
    rotangle = 180*np.mean(angles)/np.pi
    # analyze only in the vertical range of nonzero prevmask values
    nz = np.nonzero(prevmask)
    pmiy = nz[0].min()
    pmay = nz[0].max()
    # align strips vertically
    gmask = ndi.rotate(gmask,rotangle,reshape=False)
    # compute foreground pixels in vertical columns, the strips go top to bottom
    gprof=gmask[pmiy:pmay,:].sum(axis=0)
    gprof = gprof > 0.8*(pmay-pmiy)
    if not gprof.any():
        # unclear case, return somethinng which would be late classified as failure
        return gmask
    cutpos = np.nonzero(gprof)[0].min() # the leftmost value, we hope this is where the plant touches it
    gmask[:,cutpos:] = 0
    # remove noise along the border
    gmask = ndi.binary_opening(gmask, np.ones((1,5))).astype(np.uint8)
    #rotate back
    gmask = ndi.rotate(gmask,-rotangle,reshape=False)
    return select_overlaps(gmask, prevmask)

# classify plant growth using a piecewise linear model
def classifyGrowth(box_height, plant_heights_in):
    # free parameters
    NormalGrowthFactor = 0.8
    NoGerminationSizeThreshold = 50
    NotGrowingSpeedThresh = 10  #distinguish between growing and not growing plant
    RegularGrowthFactor = 0.75

    plant_heights_in = np.array(plant_heights_in)
    ix = np.array(range(len(plant_heights_in)))

    # fit linear model to all
    x, data, slope_all, intercept_all, allres = linfit(ix, plant_heights_in)
    allplot = linplotarray([[x, data, slope_all, intercept_all]])

    # exclude day 0 (owing to a large height diffrerence, which sshadows other changes in growth rate
    # remove outliers
    plant_heights = ndi.median_filter(plant_heights_in, size=3)
    plant_heights =plant_heights[1:]
    ix =ix[1:]
    #print(0,allres)
    reslist=[]
    slopes1=[]
    slopes2=[]
    # find a piecewise linear model with smallest residuals
    # split data in two parts 1 and 2 and fit linear models to them
    bplots=[]
    for bp in range(2, len(plant_heights)-2, 1):
        x1, data1, slope1, intercept1, res1 = linfit(ix[:bp], plant_heights[:bp])
        x2, data2, slope2, intercept2, res2 = linfit(ix[bp-1:], plant_heights[bp-1:])
        #linplot([ [x1, data1, slope1, c1], [x2, data2, slope2, c2] ])
        bplots.append(linplotarray([ [0, plant_heights_in[0]], [x1, data1, slope1, intercept1], [x2, data2, slope2, intercept2] ]))
        # estimate residuals as sum of partial residuals 
        reslist.append(res1+res2)
        # used to classify growth type
        slopes1.append(slope1)
        slopes2.append(slope2)

    print(plant_heights_in)
    #characteristics used to classify growth type
    xmin = np.argmin(reslist)
    slopes1mean = np.mean(slopes1[:xmin+1])
    if xmin == 0:
        heights1mean = plant_heights[0]
    else:
        heights1mean = np.mean(plant_heights[:xmin])
    heights2mean = np.mean(plant_heights[xmin:])
    slopes2mean = np.mean(slopes2[xmin:])
    rmin = min(reslist)
    #ipdb.set_trace()
    if np.min(plant_heights) == 0 or slope_all < 0:
        print(f"Detection error (vanished)" )
        return ["Detection error (vanished)" , None, None, None, None, None, allplot]
    elif np.max(plant_heights) < NoGerminationSizeThreshold:
        print(f"Not germinated")
        return ["Not germinated", plant_heights_in[1], None, None, None, None, allplot]
    elif np.max(plant_heights) > 0.95* box_height:
        print(f"Detection error (too large)" )
        return ["Detection error (too large)" , plant_heights_in[1], None, None, None, allplot]
    # if significant change in growth rate
    elif np.min(reslist) < NormalGrowthFactor*allres:
        if heights1mean < NoGerminationSizeThreshold and slopes2mean > NotGrowingSpeedThresh :
            print(f"Late germination, day {xmin+1}")
            return ["Late germination", slopes2mean, None, xmin+1, rmin, bplots[xmin]]
        else:
            #ipdb.set_trace()
            if slopes2mean < NotGrowingSpeedThresh:
                print(f"Stopped growing, day {xmin+2}")
                return ["Stopped growing", plant_heights_in[1], slopes2mean, 0, xmin+2, rmin, bplots[xmin]]
            elif slopes2mean < RegularGrowthFactor*slopes1mean:
                print(f"Normal growth, slowdown, day {xmin+2}, {slopes2mean/slopes1mean}")
                return ["Normal growth, slowdown", plant_heights_in[1], slopes2mean, slopes2mean/slopes1mean, xmin+2, rmin, bplots[xmin]]
            elif RegularGrowthFactor*slopes2mean > slopes1mean:
                print(f"Normal growth, acceleration, day {xmin+2, {slopes2mean/slopes1mean}}")
                return ["Normal growth, acceleration", plant_heights_in[1], slopes2mean,slopes2mean/slopes1mean, xmin+2, rmin, bplots[xmin]]
            else:
                print(f"Normal growth, regular")
                return ["Normal growth, regular", plant_heights_in[1], (slopes2mean+slopes1mean)/2, 1, 1, rmin, bplots[xmin]]
    # no significant change in growth rate
    else:
        if slope_all < NotGrowingSpeedThresh:
            print(f"Stopped growing")
            return ["Stopped growing", plant_heights_in[1], slope_all, None, 0, rmin, allplot]
        else:
            print(f"Normal growth, regular")
            return ["Normal growth, regular", plant_heights_in[1], slope_all, 1, 0, rmin, allplot]
    #pass
def linfit(x, data):
    if len(x) == 2:
        m = (data[1]-data[0])/(x[1]-x[0])
        c = ((data[1]+data[0])-m*(x[1]+x[0]))/2
        return x, data, m, c, 0
    else:
        A = np.vstack([x, np.ones(len(x))]).T
        (m, c), res = np.linalg.lstsq(A, data, rcond=None)[:2]
        # normalize res, check if all data points are equal 0
        if data.any():
            return x, data, m, c, np.sqrt(res[0]/data.mean())
        else:
            return x, data, m, c, res[0]

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
    for item in pdata:
        if len(item) == 4:
            (x, data, m, c) = item
            _ = plt.plot(x, data, 'o', label='Original data', markersize=10)
            _ = plt.plot(x, m*x + c, 'r', label='Fitted line')
        elif len(item) == 2:
            #_ = plt.plot([item[0],item[0]], [item[1],item[1]], 'o', markersize=6)
            _ = plt.plot([item[0]], [item[1]], 'o', markersize=15)
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    return image

def procplant(plates, plantnum, seedmask):
    gmasks=[seedmask]
    # dilate for the cases when the seed moves a bit prior to germination
    # Example: seed 0 in apogwas2/005   np.ones((19,19)
    # Example: seed 14 in apogwas2/021  np.ones((29,29)
    prevmask = ndi.binary_dilation(seedmask, np.ones((29,29)))
    prevmasksum = seedmask.sum()
    for plnum in range(1,len(plates)):
        plate = plates[plnum]
        gplate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        gplate = phlib.gaussitk(gplate, 2)
        #rb=phlib.rolling_ball_filter(gplate,2,19)
        rb=phlib.rolling_ball_filter(gplate,4,9)

        threshold = 4
        gmaskall = (gplate.astype(np.float) - rb) > threshold
        #ipdb.set_trace(s)
        # break thin horizontal structures
        gmaskall = ndi.binary_opening(gmaskall, np.ones((7,1)))

        gmask = select_overlaps(gmaskall, prevmask, plantnum, plnum)
        #ipdb.set_trace()
        gmasks.append(gmask)
        # the plant should normally not shrink, so shrinking is suspicious. Keep the last good mask
        gmasksum = gmask.sum()
        if gmasksum < 0.8*prevmasksum:
            #print("Plant %2d,%d: Plant shrinking by %02f%%"%(plantnum, plnum, 100*(prevmasksum-gmasksum)/prevmasksum))
            pass
        else:
            prevmask=gmask
            prevmasksum = gmasksum
        pass
    #failure, if the mask touches upper border
    gmasks = np.array(gmasks).astype(np.uint8)
    #ipdb.set_trace()
    if gmasks.max(axis=0)[0,:].any():
        print(f"Detection error (too large)" )
        return gmasks, ["Detection error (too large)" , None, None, None, None]
    maskheight = [np.nonzero(m)[0].max() - np.nonzero(m)[0].min() if m.max() > 0 else 0 for m in gmasks]
    return gmasks, classifyGrowth(plates.shape[1], maskheight)
   
desc="segment individual plants in plate data"
dirName=os.environ.get('APOGWAS_PATH')
dishId=None
plantNum=None
subStart=0
rWidth = 120
rebuildAll=False
batchNum=1

def usage(desc):
    global dirName, dishId, rWidth, batchNum
    print(sys.argv[0]+":",   desc)
    print("Usage: ", sys.argv[0], "[switches]")
    print("Switches:")
    print("\t-h ............... this usage")
    print("\t-d path........... directory with plant datasets {taken from the APOGWAS_PATH environment variable}")
    print(f"\t-b 1,2,3,4,5 .... batch number {batchNum}")
    print("\t-p subdir_name[,plant#] ... process subdirectory with plant data (all subdirs)")
    print("\t-s INT ........... subdirectory number to start from (all subdirs)")
    print("\t-w INT ........... region width in %% of interseed distance (%d %%)"%rWidth)
    print("\t-r ............... rebuild all")

def parsecmd(desc):
    global dirName, dishId, subStart, rWidth, rebuildAll,batchNum
    try:
        opts, Names = getopt.getopt(sys.argv[1:], "hrd:s:p:b:", ["help"])
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
        elif o in ("-b"):
            batchNum = int(a)
        elif o in ("-w"):
            rWidth = int(a)
        elif o in ("-r"):
            rebuildAll=True

def main():
    global dirName, dishId, rebuildAll
    parsecmd(desc)
    plantNum=None

    dishFiles = {}    #dishes to process, organized by dishId
    reprocessname=None
    if dishId:
        if "," in dishId:
            dishId, plantNum = dishId.split(",")
            dirPath = "%s/batch%d/%s"%(dirName, batchNum, dishId)
            globname = "%s/plant-%03d-%02d_*.tif"%(dirPath, int(dishId), int(plantNum))
            reprocessname = glob.glob(globname)
            if not reprocessname:
                print ("%s: Error -- No files found for '%s'"%(sys.argv[0], globname))
                sys.exit(0)

            dishFiles[dishId] = reprocessname
        else:
            dirPath = "%s/batch%d/%s"%(dirName, batchNum, dishId)
            plantnames=sorted(glob.glob("%s/plant-*.tif"%dirPath))
            dishFiles[dishId] = plantnames
    else:
        for p in range(subStart, 200):
            dishId = "%03d"%p
            dirPath = "%s/batch%d/%s"%(dirName, batchNum, dishId)
            if glob.glob(dirPath) == []: continue   # no such dish
            if not rebuildAll:
                if os.path.isfile("%s/pmask-%s.tif"%(dirPath,dishId)):
                    print("Skipping %s"%dirPath)
                    continue
            plantnames=sorted(glob.glob("%s/plant-*.tif"%dirPath))
            dishFiles[dishId] = plantnames
    #ipdb.set_trace()

    #process dishes one by one
    for dishId in dishFiles:
        dirPath = "%s/batch%d/%s"%(dirName, batchNum, dishId)
        seedsmask = loadTiff( "%s/seeds-mask-%s.tif"%(dirPath,dishId))
        # gimp make this sometimes with alpha channel
        if seedsmask.shape[-1] == 4:
            seedsmask = seedsmask[...,:3]
        plantnames = dishFiles[dishId]
        plantmasks_name = "%s/pmask-%s.tif"%(dirPath,dishId)
        plantoverview_name = "%s/pmask-ovl-%s.tif"%(dirPath,dishId)


        if reprocessname:
            plantoverview = loadTiff(plantoverview_name)
            plantmasks = loadTiff(plantmasks_name)
        else:
            plantoverview=seedsmask.copy()
            plantmasks = np.zeros((11,seedsmask.shape[0],seedsmask.shape[1])).astype(np.uint8)
       
        reportWriter = ODSWriter()
        reportWriter.addtable(dishId, ["Plant number","Type","Day 1 height", "Growth rate","Accel. factor", "From day", "Residuals", "Growth plot","Plant growth, days 0 – 10"])

        for plantname in plantnames:
            #ipdb.set_trace()
            ulx, uly, lrx, lry=np.array(re.findall(r"\d+",plantname.split("/")[-1])[2:]).astype(np.int)
            plant = loadTiff(plantname)
            pmaskname = plantname.replace("plant-","pmask-")
            # Reload, if plant mask exists
            print("Processing %s"%pmaskname)
            pnum = int(re.findall(r"plant-[0-9]*-([0-9]*)",plantname.split("/")[-1])[0])
            masks, return_state = procplant(plant, pnum, seedsmask[uly:lry,ulx:lrx][...,0] == 0)

            # save masks file
            with TiffWriter(plantname.replace("plant-","pmask-")) as tif:
               tif.save(masks,compress=5)

            #update plantoverview and plantmasks buffers
            plantmasks[...,uly:lry,ulx:lrx] = np.maximum(plantmasks[...,uly:lry,ulx:lrx], 255*masks)
            #plantoverview[uly:lry,ulx:lrx,...] = phlib.img3overlay(plant.max(axis=0), masks.max(axis=0))
            plantoverview[uly:lry,ulx:lrx,...] = np.maximum(plantoverview[uly:lry,ulx:lrx,...], phlib.img3overlay(plant.max(axis=0), masks.max(axis=0)))
            # draw color marks 
            cmasks = np.concatenate(masks, axis=1)[::2,::2]
            nz = np.nonzero(cmasks)
            m_from = max(nz[0].min()-5, 0)
            m_to = min(nz[0].max()+5, cmasks.shape[0])
            cmasks = cmasks[m_from:m_to,:]
            cplant = np.concatenate(plant, axis=1)[::2,::2]
            cplant = cplant[m_from:m_to,:]
            oplant = phlib.img3overlay(cplant, cmasks)
            #ipdb.set_trace()
            if "Detection error" in return_state[0]: 
                hcolor = (255,0,0)
            #elif "Normal growth" in return_state: hcolor = (0,255,0)
            elif "Not germinated" in return_state[0]: 
                hcolor = (255,255,0)
            elif "Stopped growing" in return_state[0]: 
                hcolor = (0, 0, 255)
            elif "Late germination" in return_state[0]: 
                hcolor = (0, 255, 0)

            reportWriter.writerow([pnum]+return_state+[oplant])

            if not "Normal growth" in return_state[0]:  
                plantoverview[uly:uly+20,ulx:lrx,...] = hcolor
                plantoverview[lry-20:lry,ulx:lrx,...] = hcolor

        # create report
        reportWriter.save("%s/plant-report-%s.ods"%(dirPath,dishId))
        #ipdb.set_trace()
        with TiffWriter(plantmasks_name) as tif:
            tif.save(plantmasks,compress=5)
        #overview image
        with TiffWriter(plantoverview_name) as tif:
            #tif.save(plantoverview,compress=5)
            tif.save(plantoverview)
    #ipdb.set_trace()
    pass

if __name__ == "__main__":
    main()
