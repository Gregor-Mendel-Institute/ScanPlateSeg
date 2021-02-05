#!/usr/bin/python3
# vim:ts=4:et
#Stage 3: Export regions. Requires file plates-001.tif and seeds-mask-001.tif. Creates 24 files seeds-001-...-.tif and plant-regions-001.png 

# Copyright (C) 2013 Milos Sramek <milos.sramek@soit.sk>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/gpl.html
# - or any later version.

from tifffile import TiffWriter, TiffFile
import SimpleITK as sitk
import numpy as np
import sys, glob, re, os, getopt, csv
import ipdb
import phlib
from skimage import morphology, filters
from skimage import measure
import cv2, imageio
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as stats
from scipy.signal import medfilt
#import guiqwt.pyplot as plt
import matplotlib.pyplot as plt

def plot(data):
    import matplotlib.pyplot as plt
    # data is tuple of lists
    for d in data:
        plt.plot(range(len(d)),d)
    plt.pause(1)
    plt.show(block=True)

def disp(iimg, label = None, gray=False):
    #import guiqwt.pyplot as plt
    import matplotlib.pyplot as plt
    plt.ioff()
    #plt.imshow(iimg, interpolation='none')
    plt.imshow(iimg)
    plt.pause(1)
    plt.show(block=True)
 
def loadTiff(ifile):
    try:
        with TiffFile(str(ifile)) as tfile:
            vol = tfile.asarray()
        return vol
    except IOError as err:
        print ("%s: Error -- Failed to open '%s'"%(sys.argv[0], str(ifile)))
        sys.exit(0)
   
def procPlateSet(dirname, sid, plates, seeds, rWidth):
    '''
    simplified version, uses the original masks to evaluate growth
    rootthrsigma: estimated value 4.0
    '''

    contours = cv2.findContours(seeds.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
    centers=[]
    for cnt in contours:
        #rect =cv2.boundingRect(cnt)
        x,y,w,h=cv2.boundingRect(cnt)
        centers.append([int((x+x+w)/2),int((y+y+h)/2)])

    #sort and split
    # Y direction
    centers.sort(key = lambda centers: centers[1])
    y0 = int(np.mean([c[1] for c in centers[:12]]))
    y1 = int(np.mean([c[1] for c in centers[12:]]))
    #positions
    x0 = sorted([c[0] for c in centers[:12]])
    x1 = sorted([c[0] for c in centers[12:]])
    #mean half distance between seeds
    xd0 = int((x0[-1] - x0[0])/2/(len(x0) - 1))
    xd1 = int((x1[-1] - x1[0])/2/(len(x1) - 1))

    #load all plates
    pmax = plates.max(axis=0)

    #save individual plants
    ytop = 450
    ybot = 1800
    cnt=0
    colors=((200,200,0),(200,0,200),(0,200,200))
    reportLog={}
    for yy, xx, xd in zip([y0, y1], [x0,x1], [xd0, xd1]):
        for px in xx: 
            rw = int(xd*rWidth/100) #region width
            ulx, uly, lrx, lry = px-rw, yy-ytop, px+rw, yy+ybot
            cv2.rectangle(pmax, (ulx,uly-20*(cnt%2)), (lrx, lry+20*(cnt%2)), colors[cnt%3], 9)
            subname = "%s/%s/plant-%s-%02d_%04d-%04d_%04d-%04d.tif"%(dirname, sid, sid, cnt, ulx, uly, lrx, lry)  
            #print(subname)
            reportLog["Plant %2d"%cnt] = str((ulx, uly, lrx, lry))
            subplates=plates[:,uly:lry,ulx:lrx,:]
            with TiffWriter(subname) as tif:
                tif.save(subplates, compress=5)
            cnt += 1
    
    subname = "%s/%s/plant-regions-%s.png"%(dirname, sid,sid)
    imageio.imwrite(subname, pmax[::4,::4])
    return reportLog

   
desc="segment individual plants in plate data"
dirName="."
dirName="/media/milos/SAN128/data/Patrick/batch1/apogwas2/"
dishId=None
subStart=0
rWidth = 120
rebuildAll = False

def usage(desc):
    global dirName, dishId, rWidth
    print(sys.argv[0]+":",   desc)
    print("Usage: ", sys.argv[0], "[switches]")
    print("Switches:")
    print("\t-h ............... this usage")
    print("\t-d name .......... directory with plant datasets (%s)"%dirName)
    print("\t-p subdir_name ... process subdirectory with plant data (all subdirs)")
    print("\t-s INT ........... subdirectory number to start from (all subdirs)")
    print("\t-w INT ........... region width in %% of interseed distance (%d %%)"%rWidth)
    print("\t-r ............... rebuild all")

def parsecmd(desc):
    global dirName, dishId, subStart, rWidth, rebuildAll
    try:
        opts, Names = getopt.getopt(sys.argv[1:], "hrd:s:p:w:", ["help"])
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
    global dirName, dishId
    parsecmd(desc)

    if dishId:
        print("%s/%s"%(dirName,dishId))
        # delete existing plant files
        if rebuildAll:
            fnames = glob.glob("%s/%s/plant-%s*.tif"%(dirName, dishId,dishId))
            for fname in fnames: os.remove(fname)
            fnames = glob.glob("%s/%s/pmask-%s*.tif"%(dirName, dishId,dishId))
            for fname in fnames: os.remove(fname)
        seeds = loadTiff("%s/%s/seeds-mask-%s.tif"%(dirName,dishId,dishId))
        mask = seeds[...,0] == 0
        plates = loadTiff("%s/%s/plates-%s.tif"%(dirName,dishId,dishId))
        procPlateSet(dirName, dishId, plates, mask, rWidth)
    else:
        for p in range(subStart, 200):
            dishId = "%03d"%p
            if rebuildAll:
                fnames = glob.glob("%s/%s/plant-%s*.tif"%(dirName, dishId,dishId))
                for fname in fnames: os.remove(fname)
                fnames = glob.glob("%s/%s/pmask-%s*.tif"%(dirName, dishId,dishId))
                for fname in fnames: os.remove(fname)
            fnames = glob.glob("%s/%s"%(dirName, dishId))
            if fnames == []: continue   # no such plant
            fnames = glob.glob("%s/%s/plant-%s*.tif"%(dirName, dishId,dishId))
            print("%s/%s"%(dirName,dishId))
            if fnames: 
                print("Skipping %s/%s"%(dirName,dishId))
                continue # plates.tif exists
            else:
                print("Processing %s/%s"%(dirName,dishId))
            #load all plates from a single file
            seeds = loadTiff( "%s/%s/seeds-mask-%s.tif"%(dirName,dishId,dishId))
            mask = seeds[...,0] == 0
            plates = loadTiff("%s/%s/plates-%s.tif"%(dirName,dishId,dishId))
            procPlateSet(dirName, dishId, plates, mask, rWidth)

if __name__ == "__main__":
    main()
