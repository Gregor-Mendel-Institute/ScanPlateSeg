# ScanPlateSeg

Scripts for processing of scanned dishes with 2 rows with 12 seedlings over 10 days, starting with a plate with nongerminated seeds.

# The workflow
## Input data  
A directory with files named apogwas1\_set1\_**day7\_**20170602**\_003.tif** (bold parts are required). Here 003 identifies dish.
## The script
Run 
`plateproc.py -d input_directory {-o output_directory}`

A dish is skipped if the file `plant-regions.png` exists in its output subdirectory, 
## Output
For each detected dish a subdirectory named by its identifier (e. g. 003) is created in output directory. Files this directory

* plates-001.png: overview image of the aligned plates (projection in the Z direction)
* seeds-001.tif and seeds-mask-001.tif: masks and inverted masks of seeds drom day 0 image. 
* plant-regions.png: overview of defined region
* region data: A tiff file with 11 bands (days 0 -- 10) with a name plant-001-18_2855-3182_3391-5432.tif. The fields:
** -001-: dish identifier, equal to  subdirectory name
** -18\_: number of plant in the dish, counted from the upper left corner
** \_2855-3182\_3391-543: UpperLeftX-UpperLeftY\_LowerRightX-LowerRightY
