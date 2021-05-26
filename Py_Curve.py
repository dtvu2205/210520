# IMPORTING LIBRARY
import csv
import numpy as np
from osgeo import gdal, gdal_array



# HELPER FUNCTION
def pick(c, r, mask): # (c_number, r_number, an array of 1 amd 0) 
    filled = set()
    fill = set()
    fill.add((c, r))
    width = mask.shape[1]-1
    height = mask.shape[0]-1
    picked = np.zeros_like(mask, dtype=np.int8)
    while fill:
        x, y = fill.pop()
        if y == height or x == width or x < 0 or y < 0:
            continue
        if mask[y][x] == 1:
            picked[y][x] = 1
            filled.add((x, y))
            west = (x-1, y)
            east = (x+1, y)
            north = (x, y-1)
            south = (x, y+1)
            if west not in filled:
                fill.add(west)
            if east not in filled:
                fill.add(east)
            if north not in filled:
                fill.add(north)
            if south not in filled:
                fill.add(south)
    return picked



# INPUT PARAMETERS
# Bounding box of the reservoir [ulx, uly, lrx, lry]
bbox = [527809.019982, 2822840.736178, 624828.419682, 2732481.336478] 
# A point in the reservoir extent [lat, lon]
point = [607370, 2735550]
xp = round(abs(point[0]-bbox[0])/30)
yp = round(abs(point[1]-bbox[1])/30)
# Maximum reservoir water level
max_wl = 1240
curve_ext = max_wl + 20 # to expand the curve 



# CREATING E-A-S RELATIONSHOP
# clipping DEM by the bounding box
dem = gdal.Open("DEM.TIF") 
dem = gdal.Translate("DEM_Clipped.TIF", dem, projWin = bbox)
dem = None 

# isolating the reservoir
dem_bin = gdal_array.LoadFile("DEM_Clipped.TIF")
dem_bin[np.where(dem_bin > curve_ext)] = 0
dem_bin[np.where(dem_bin > 0)] = 1
res_iso = pick(xp, yp, dem_bin)

# finding the lowest DEM value in the reservoir extent
res_dem = gdal_array.LoadFile("DEM_Clipped.TIF")
res_dem[np.where(res_iso == 0)] = 9999
min_dem = np.min(res_dem)

# caculating reservoir surface area and storage volume 
# coresponding to each water level
results = [["Level (m)", "Area (skm)", "Storage (mcm)"]]
pre_area = 0
tot_stor = 0 
for i in range(min_dem, curve_ext): 
    level = i
    water_px = gdal_array.LoadFile("DEM_Clipped.TIF")
    water_px[np.where(res_iso == 0)] = 9999
    water_px[np.where(res_dem > i)] = 0 
    water_px[np.where(water_px > 0)] = 1
    area = np.sum(water_px)*9/10000
    storage = (area + pre_area)/2
    tot_stor += storage
    pre_area = area   
    results = np.append(results, [[level, round(area,4), round(tot_stor,4)]], 
                        axis=0)

# saving output as a csv file
with open("Curve.csv","w", newline='') as my_csv:
    csvWriter = csv.writer(my_csv)
    csvWriter.writerows(results)