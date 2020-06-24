#!/usr/bin/python3
# vim:ts=4:et
# process data sets related to one dish and  
# Identify dishes in each time step and align them 
#
#and the outDirName/NNN directory and store the results there

# Copyright (C) 2013 Milos Sramek <milos.sramek@soit.sk>
# Licensed under the GNU LGPL v3 - http://www.gnu.org/licenses/gpl.html
# - or any later version.

from tifffile import TiffWriter, TiffFile
import sys, glob, shutil, os, getopt,time
import imutils
import configparser, imageio
import ipdb
import platealign
import platesegseed
import platesplit

desc="Identify dish and align plates"
inDirName="."
outDirName=None

dishId=None    # a NNN identifier od a dish
verbose=False
rWidth = 120    # % 
namePrefix=""

def loadTiff(ifile):
    try:
        with TiffFile(str(ifile)) as tfile:
            vol = tfile.asarray()
        return vol
    except IOError as err:
        return None

def usage(desc):
    global inDirName, outDirName, dishId, verbose, rWidth,namePrefix
    print(sys.argv[0]+":",   desc)
    print("Usage: ", sys.argv[0], "[switches]")
    print("Switches:")
    print("\t-h .......... this usage")
    print("\t-h .......... be verbose")
    print("\t-d name ..... directory with plant datasets (%s)"%inDirName)
    print("\t-o name ..... directory to store the result to (in a NNN subdirecory) (%s)"%"same as input")
    print("\t-e string ... file name prefix (%s)"%namePrefix)
    print("\t-p NNN ...... ID of a dish (NNN) to process (all dishes)")
    print("\t-w INT ...... region width in %% of inter seed distance (%d %%)"%rWidth)

def parsecmd(desc):
    global inDirName, outDirName, dishId, verbose, rWidth, namePrefix
    try:
        opts, Names = getopt.getopt(sys.argv[1:], "hvd:p:o:w:e:", ["help"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err)) # will print something like "option -a not recognized"
        sys.exit()
    for o, a in opts:
        if o in ("-h", "--help"):
            usage(desc)
            sys.exit()
        elif o in ("-v"):
            verbose=True
        elif o in ("-d"):
            inDirName = a
        elif o in ("-e"):
            namePrefix = a
        elif o in ("-o"):
            outDirName = a
        elif o in ("-p"):
            dishId = a
        elif o in ("-w"):
            rWidth = int(a)

def procAll(inDirName, outDirName, dishId, prefix):
    global verbose, rWidth
    reg_file="%s/%s/plant-regions-%s.png"%(outDirName,dishId,dishId)
    if os.path.isfile(reg_file):
        if verbose:
            print("Skipping dish %s (all done, file %s exists)"%(dishId, reg_file))
        return
    now = time.strftime("%Y-%b-%d_%H:%M:%S")
    reportLogs = configparser.ConfigParser()
    try:
        reportLogs.read("%s/%s/plateprocsplit.txt"%(outDirName,dishId))
    except:
        pass
    if verbose:
        print("Input directory:  %s"%(inDirName))
        print("Output directory/set: %s/%s"%(outDirName,dishId))
        if prefix: print("File name prefix:  %s"%(prefix))

    # align the dishes, first check if a file with aligned dishes exists
    plates_file = "%s/%s/plates-%s.tif"%(outDirName,dishId,dishId)
    plates = loadTiff(plates_file)
    if plates is None:
        if verbose:
            print("Detecting and aligning dishes for set %s in %s"%(dishId,inDirName))
        plates, reportLog = platealign.procPlateSet(inDirName, outDirName, dishId, prefix)
        with TiffWriter("%s/%s/plates-%s.tif"%(outDirName, dishId, dishId)) as tif: tif.save(plates)
        imageio.imwrite("%s/%s/plates-%s.png"%(outDirName, dishId, dishId), plates.max(axis=0)[::4,::4,:])
    else:
        if verbose:
            print("Reusing aligned plates: %s"%plates_file)
        reportLog = {"Reusing aligned plates": "%s"%plates_file}
    reportLogs["platealign %s"%now] = reportLog

    # identify seeds, first check if (inverted) file with seed masks exists
    mask_file = "%s/%s/seeds-mask-%s.tif"%(outDirName,dishId,dishId)
    invmask = loadTiff(mask_file)
    if invmask is None:
        if verbose:
            print("Detecting seeds in %s"%plates_file)
        mask, reportLog = platesegseed.procPlateSet(plates)
        with TiffWriter("%s/%s/seeds-mask-%s.tif"%(outDirName,dishId,dishId)) as tif:
            tif.save(platesegseed.img3mask(plates[0]+1,1-mask),compress=5)
    else:
        if verbose:
            print("Reusing seed mask: %s"%mask_file)
        mask = invmask[...,0] > 0
        reportLog = ("Reusing seed mask: %s"%mask_file)
    #write in both cases, eventually to reflect manual changes in the seeds-mask file
    with TiffWriter("%s/%s/seeds-%s.tif"%(outDirName,dishId,dishId)) as tif:
        tif.save(platesegseed.img3mask(plates[0],mask),compress=5)

    #Save regions
    if verbose:
        print("Saving regions to %s/%s"%(outDirName, dishId))
    reportLog = platesplit.procPlateSet(outDirName, dishId, plates, mask, rWidth)
    reportLogs["platesplit %s"%now] = reportLog

    with open("%s/%s/plateprocsplit.txt"%(outDirName, dishId), 'w') as reportfile: reportLogs.write(reportfile)

def main():
    global inDirName, outDirName, dishId, namePrefix

    parsecmd(desc)
    outDirName = outDirName if outDirName else inDirName

    if dishId:
        procAll(inDirName, outDirName, dishId, namePrefix)
    else:
        for p in range(200):
            dishId = "%03d"%p
            # check if dishId images exist
            pnames = glob.glob("%s/*%s.tif"%(inDirName, dishId))
            if pnames == []: continue   # no such plant
            procAll(inDirName, outDirName, dishId, namePrefix)

if __name__ == "__main__":
    main()
