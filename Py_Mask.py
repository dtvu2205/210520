# IMPORTING LIBRARY
import time
start_time = time.time()
import os
import csv
import math
import numpy as np
from osgeo import gdal, gdal_array



# HELPER FUNCTIONS
def pick(c, r, mask): # (column, row, an array of 1 amd 0) 
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

def expand(array, n): # (an array of 1 and 0, number of additional pixels)
    expand = array - array
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == 1:
                for k in range(max(0, i-n), min(i+n, len(array)-1)):
                    for l in range(max(0, j-n), min(j+n, len(array[i])-1)):
                        expand[k][l] = 1
                continue
            else:
                continue
    return expand



# INPUT PARAMETERS
# Bouding box of the reservoir [ulx, uly, lrx, lry]
bbox = [527809.019982, 2822840.736178, 624828.419682, 2732481.336478] 
# A point in the reservoir extent [lat, lon]
point = [607370, 2735550]
xp = round(abs(point[0]-bbox[0])/30)
yp = round(abs(point[1]-bbox[1])/30)
# Maximum reservoir water level
max_wl = 1240 



# CLIP LANDSAT IMAGES BY THE BOUNDING BOX 
print("Clipping Landsat images by the bounding box ...")
clip_count = 0 
os.chdir('./Landsat_8') 
directory = os.getcwd()
for filename in os.listdir(directory):
    if filename.startswith("L"): # LC08, LO08 = Landsat8, LE07 = Landsat7, LT05 = Landsat5
        ls_img = gdal.Open(filename)
        ls_img = gdal.Translate(filename[3:5]+filename[7:10]+filename[17:26]+
                                "Clipped"+filename[40:], ls_img, projWin = bbox)
        ls_img = None
        clip_count += 1
        continue
    else:
        continue
os.chdir('..')
os.chdir('./Landsat_7')     
directory = os.getcwd()
for filename in os.listdir(directory):
    if filename.startswith("L"): # LC08, LO08 = Landsat8, LE07 = Landsat7, LT05 = Landsat5
        ls_img = gdal.Open(filename)
        ls_img = gdal.Translate(filename[3:5]+filename[7:10]+filename[17:26]+
                                "Clipped"+filename[40:], ls_img, projWin = bbox)
        ls_img = None
        clip_count += 1
        continue
    else:
        continue    
os.chdir('..')
os.chdir('./Landsat_5')     
directory = os.getcwd()
for filename in os.listdir(directory):
    if filename.startswith("L"): # LC08, LO08 = Landsat8, LE07 = Landsat7, LT05 = Landsat5
        ls_img = gdal.Open(filename)
        ls_img = gdal.Translate(filename[3:5]+filename[7:10]+filename[17:26]+
                                "Clipped"+filename[40:], ls_img, projWin = bbox)
        ls_img = None
        clip_count += 1
        continue
    else:
        continue
os.chdir('..')      
print("Clipped "+str(int(clip_count))+" images")
print(" ")



# NDWI CALCULATION
print("Calculating NDVI ...")
class_count = 0 
os.chdir('./Landsat_8')
directory = os.getcwd()
for filename in os.listdir(directory):
    if filename.endswith("Clipped_B3.TIF"):   
        B3 = filename
        B5 = filename[:22]+"B5.TIF"
        B12 = filename[:22]+"BQA.TIF"          
        grn = gdal_array.LoadFile(B3).astype(np.float32)
        nir = gdal_array.LoadFile(B5).astype(np.float32)
        bqa = gdal_array.LoadFile(B12).astype(np.float32)
        ndwi_raw = ((grn-nir)/(grn+nir+0.00000000000000000001))
        #output = gdal_array.SaveArray(raw_ndwi.astype(gdal_array.numpy.float32),
        #                              B3[:13]+"_NDWI_RAW.TIF", format="GTiff", 
        #                              prototype=B3)
        #output = None
        # Remove no-data, cloud and shadow pixels
        ndwi = ndwi_raw
        ndwi[np.where(grn == 0)] = -0.5 # no-data pixels
        ndwi[np.where(nir == 0)] = -0.5 # no-data pixels
        ndwi[np.where(bqa >= 2800)] = -0.5 # cloud and shadow pixels (Landsat8, BQA >= 2800)
        output = gdal_array.SaveArray(ndwi.astype(gdal_array.numpy.float32), 
                                      B3[:13]+"_NDWI.TIF", format="GTiff", 
                                      prototype=B3)
        output = None
        os.remove(B3)
        os.remove(B5)
        class_count += 1
        continue
    else:
        continue
os.chdir('..')
os.chdir('./Landsat_7')         
directory = os.getcwd()
for filename in os.listdir(directory):
    if filename.endswith("Clipped_B2.TIF"): 
        B2 = filename
        B4 = filename[:22]+"B4.TIF"  
        B9 = filename[:22]+"BQA.TIF"  
        grn = gdal_array.LoadFile(B2).astype(np.float32)
        nir = gdal_array.LoadFile(B4).astype(np.float32)
        bqa = gdal_array.LoadFile(B9).astype(np.float32)
        ndwi_raw = ((grn-nir)/(grn+nir+0.00000000000000000001))
        #output = gdal_array.SaveArray(ndwi.astype(gdal_array.numpy.float32), 
        #                              B2[:13]+"_NDWI_RAW.TIF", format="GTiff", 
        #                              prototype=B2)
        #output = None
        # Remove no-data, cloud and shadow pixels
        ndwi = ndwi_raw 
        ndwi[np.where(grn == 0)] = -0.5 # no-data pixels
        ndwi[np.where(nir == 0)] = -0.5 # no-data pixels
        ndwi[np.where(bqa >= 752)] = -0.5 # cloud and shadow pixels (Landsat7, BQA >= 752)
        output = gdal_array.SaveArray(ndwi.astype(gdal_array.numpy.float32), 
                                      B2[:13]+"_NDWI.TIF", format="GTiff", 
                                      prototype=B2)
        output = None
        os.remove(B2)
        os.remove(B4)
        os.remove(B9)
        class_count += 1
        continue
    else:
        continue
os.chdir('..')
os.chdir('./Landsat_5')         
directory = os.getcwd()
for filename in os.listdir(directory):
    if filename.endswith("Clipped_B2.TIF"): 
        B2 = filename
        B4 = filename[:22]+"B4.TIF"  
        B9 = filename[:22]+"BQA.TIF"  
        grn = gdal_array.LoadFile(B2).astype(np.float32)
        nir = gdal_array.LoadFile(B4).astype(np.float32)
        bqa = gdal_array.LoadFile(B9).astype(np.float32)
        ndwi_raw = ((grn-nir)/(grn+nir+0.00000000000000000001))
        #output = gdal_array.SaveArray(ndwi.astype(gdal_array.numpy.float32), 
        #                              B2[:13]+"_NDWI_RAW.TIF", format="GTiff", 
        #                              prototype=B2)
        #output = None
        # Remove no-data, cloud and shadow pixels
        ndwi = ndwi_raw
        ndwi[np.where(grn == 0)] = -0.5 # no-data pixels
        ndwi[np.where(nir == 0)] = -0.5 # no-data pixels
        ndwi[np.where(bqa >= 752)] = -0.5 # cloud and shadow pixels (Landsat5, BQA >= 752)
        output = gdal_array.SaveArray(ndwi.astype(gdal_array.numpy.float32), 
                                      B2[:13]+"_NDWI.TIF", format="GTiff", 
                                      prototype=B2)
        output = None
        os.remove(B2)
        os.remove(B4)
        class_count += 1
        continue
    else:
        continue    
os.chdir('..')  
print("Classified "+str(class_count)+" images")
print(" ")            
        
        
        
# CREATE DEM-BASED MAX WATER EXTENT MASK    
# DEM is preprocessed to have the same cell size and alignment with Landsat images 
print("Creating DEM-based max water extent mask ...") 
dem = gdal.Open("DEM.TIF") 
dem = gdal.Translate("DEM_Clipped.TIF", dem, projWin = bbox)
dem = None 
dem_clip = gdal_array.LoadFile("DEM_Clipped.TIF").astype(np.float32)
water_px = dem_clip
water_px[np.where(dem_clip <= max_wl)] = 1
water_px[np.where(dem_clip > max_wl)] = 0
picked_wp = pick(xp, yp, water_px)
dem_mask = expand(picked_wp, 3)
dm_sum = np.sum(dem_mask)     
output = gdal_array.SaveArray(dem_mask.astype(gdal_array.numpy.float32), 
                              "DEM_Mask.TIF", format="GTiff", 
                              prototype="DEM_Clipped.TIF")
output = None
print("Created DEM-based max water extent mask")
print(" ")        
        
 
        
# CREATE LANDSAT-BASED MAX WATER EXTENT MASK
print("Creating Landsat-based max water extent mask ...")
count = dem_clip - dem_clip
img_used = 0
img_list = [["Landsat", "Type", "Date"]] 
os.chdir('./Landsat_8')
directory = os.getcwd()
for filename in os.listdir(directory):
    if filename.endswith("NDWI.TIF"):
        B12 = filename[:14]+"Clipped_BQA.TIF"
        bqa = gdal_array.LoadFile(B12).astype(np.float32)
        cl_px = bqa
        cl_px[np.where(bqa < 2800)] = 0
        cl_px[np.where(bqa >= 2800)] = 1
        cl_px[np.where(dem_mask != 1)] = 0
        cl_ratio = np.sum(cl_px)/dm_sum
        if cl_ratio < 0.2:
            ndwi = gdal_array.LoadFile(filename).astype(np.float32)
            water = ndwi        
            water[np.where(ndwi >= 0)] = 1 # 0 = suggested threshold for Landsat 8
            water[np.where(ndwi<0)] = 0
            count += water
            img_used += 1
            img_list = np.append(img_list, [[filename[0], filename[2:4], 
                                             filename[5:13]]], axis=0)            
            os.remove(B12)
            continue
        else:
            os.remove(B12)
            continue
        continue
    else:
        continue
os.chdir('..')
os.chdir('./Landsat_5')      
directory = os.getcwd()
for filename in os.listdir(directory):
    if filename.endswith("NDWI.TIF"):
        B9 = filename[:14]+"Clipped_BQA.TIF"
        bqa = gdal_array.LoadFile(B9).astype(np.float32)
        cl_px = bqa
        cl_px[np.where(bqa < 752)] = 0
        cl_px[np.where(bqa >= 752)] = 1
        cl_px[np.where(dem_mask != 1)] = 0
        cl_ratio = np.sum(cl_px)/dm_sum
        if cl_ratio < 0.2:
            ndwi = gdal_array.LoadFile(filename).astype(np.float32)
            water = ndwi        
            water[np.where(ndwi >= 0.1)] = 1 # 0.1 = suggested threshold for Landsat 5
            water[np.where(ndwi<0.1)] = 0
            count += water
            img_used += 1
            img_list = np.append(img_list, [[filename[0], filename[2:4], 
                                             filename[5:13]]], axis=0)
            os.remove(B9)
            continue        
        else:
            os.remove(B9)
            continue
        continue
    else:
        continue        
os.chdir('..')        
output = gdal_array.SaveArray(count.astype(gdal_array.numpy.float32), "Count.TIF", 
                              format="GTiff", prototype="DEM_Clipped.TIF")
output = None        
max_we = count
max_we[np.where(count < 1)] = 0
max_we[np.where(count >= 1)] = 1
ls_mask = pick(xp, yp, max_we)
output = gdal_array.SaveArray(ls_mask.astype(gdal_array.numpy.float32), 
                              "Landsat_Mask.TIF", 
                              format="GTiff", prototype="DEM_Clipped.TIF")
output = None
with open("Landsat_Mask.csv","w", newline='') as my_csv:
    csvWriter = csv.writer(my_csv)
    csvWriter.writerows(img_list)
print("Created Landsat-based max water extent mask from "+str(img_used)+" images")
print(" ")



# CREATE EXPANDED MASK (by 3 pixels surrounding each of water pixels)
print("Creating expanded mask ...")
mask_1 = gdal_array.LoadFile("Landsat_Mask.TIF").astype(np.float32)
mask_2 = gdal_array.LoadFile("DEM_Mask.TIF").astype(np.float32)
sum_mask = mask_1 + mask_2
mask = sum_mask
mask[np.where(sum_mask <= 1)] = 0
mask[np.where(sum_mask > 1)] = 1
exp_mask = expand(mask, 3) 
output = gdal_array.SaveArray(exp_mask.astype(gdal_array.numpy.float32), 
                              "Expanded_Mask.TIF", 
                              format="GTiff", prototype="DEM_Clipped.TIF")
output = None
print("Created expanded mask")
print(" ")



# CREATE 50-ZONE MAP (FREQUENCE MAP)
print("Creating 50-zone map (frequence map) ...")
count = gdal_array.LoadFile("Count.TIF").astype(np.float32)
freq = count*100/np.amax(count)
zone = mask*np.ceil(freq/2)
output = gdal_array.SaveArray(zone.astype(gdal_array.numpy.float32), "Zone_Mask.TIF", 
                              format="GTiff", prototype="DEM_Clipped.TIF")
output = None
print("Created 50-zone map")
print(" ")
print("Done")



# SHOW THE RUNNING TIME
time = time.time()-start_time
mins = math.trunc(time/60)
secs = round(time-mins*60)
print("Running time: "+str(mins)+" min "+str(secs)+" sec")