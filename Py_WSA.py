# IMPORTING LIBRARY
import time
start_time = time.time()
import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from osgeo import gdal_array



# SELECT THE FOLDER OF LANDSAT IMAGES
LS = 8 # 5 = Landsat_5, 7 = Landsat_7, 8 = Landsat_8



# IMPROVE NDWI-BASED LANDSAT IMAGE CLASSIFICATION
results = [["Landsat", "Type", "Date", "Threshold", "R_50", "N_10", "S_zone", 
            "Quality", "Bf_area", "Af_area", "Fn_area"]]
drtr = os.getcwd()
os.chdir('./Landsat_'+str(LS))
directory = os.getcwd()
for filename in os.listdir(directory):
    if filename.endswith("NDWI.TIF"):        
        print(filename[:13])
        ndwi = gdal_array.LoadFile(filename).astype(np.float32)
                
        # Clip NDWI rasters by the expanded mask 
        exp_mask = gdal_array.LoadFile(drtr+"\Expanded_Mask.TIF").astype(np.float32)
        clip_ndwi = ndwi
        clip_ndwi[np.where(exp_mask == 0)] = -0.5
        
        # K-means clustering clipped NDWI raters to 3 clusters 
        # (water, wet non-water, and dry non-water) (1 more cluster for the value of -0.5)
        rows = len(clip_ndwi)
        columns = len(clip_ndwi[0])
        x = clip_ndwi.ravel()
        km = KMeans(n_clusters=4)
        km.fit(x.reshape(-1,1)) 
        z = km.cluster_centers_
        z1 = max(z)
        z2 = -1
        z3 = -1
        for i in range(0, 4):
            if z[i] < z1 and z[i] > z2:
                z2 = z[i]
        for i in range(0, 4):
            if z[i] < z2 and z[i] > z3:
                z3 = z[i]        
        threshold = round(float((z1+z2)/2),3)
        print("   K-Means clustering threshold = "+str(threshold))
        plt.figure(figsize=[30,15])
        plt.hist(x, bins=200, range=[-0.49, 0.5], color='c')
        plt.axvline(z[0], color='navy', linestyle='dashed', linewidth=2)
        plt.axvline(z[1], color='navy', linestyle='dashed', linewidth=2)
        plt.axvline(z[2], color='navy', linestyle='dashed', linewidth=2)
        plt.axvline(z[3], color='navy', linestyle='dashed', linewidth=2)
        plt.axvline((z1+z2)/2, color='red', linestyle='dashed', linewidth=2)
        plt.axvline((z2+z3)/2, color='red', linestyle='dashed', linewidth=2)
        plt.title(filename[:13], fontsize=30)
        plt.xlabel('NDWI', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.show()
        labels = np.reshape(km.labels_,(-1,columns)) 
        water_label = 0
        for i in range(0, 4):
            if z[i] == max(z):
                water_label = i 
        water_cluster = labels - labels
        water_cluster[np.where(labels == water_label)] = 1
        #output = gdal_array.SaveArray(water_cluster.astype(gdal_array.numpy.float32), 
        #                              "Before_"+filename[:13]+".TIF", 
        #                              format="GTiff", prototype=filename)
        #output = None
        
        # Assess image quality
        zone_mask = gdal_array.LoadFile(drtr+"\Zone_Mask.TIF").astype(np.float32)
        count_zm = np.zeros(50)
        for i in range(0, 50):
            count_zm[i] = np.count_nonzero(zone_mask == i+1)
        cluster_zone = zone_mask
        cluster_zone[np.where(water_cluster == 0)] = 0
        count_cl = np.zeros(50)
        ratio = np.zeros(50)
        N_10 = 0
        for i in range(0, 50):
            count_cl[i] = np.count_nonzero(cluster_zone == i+1)
            ratio[i] = count_cl[i]/(count_zm[i]+0.00000000000000000001)
            if ratio[i] >= 0.1:
                N_10 += 1
        print("   Ratio of zone 50 = "+str(round(ratio[49],3)))
        print("   No. of zones having >=10% water pixels = "+str(int(N_10)))
          
        # Improve image classification
        ratio_nm = ratio*100/(max(ratio)+0.00000000000000000001)
        x_axis = np.zeros(50)
        for i in range(0, 50):
            x_axis[i] = i + 1
        xx = np.vstack((x_axis, ratio_nm)).T
        kkm = KMeans(n_clusters=2).fit(xx)
        llb = kkm.labels_
        minx0 = 50
        minx1 = 50
        for i in range(0, 50):
            if llb[i] == 0:
                if x_axis[i] < minx0:
                    minx0 = x_axis[i]
            elif llb[i] == 1:
                if x_axis[i] < minx1:
                    minx1 = x_axis[i]                 
        s_index = max(minx0, minx1)
        if minx0 == s_index:
            water_id = 0
        elif minx1 == s_index:
            water_id = 1 
        print("   Additional water pixels start from zone "+str(int(s_index)))    
        colors = ['navy' if x==water_id else 'lightblue' for x in llb]
        plt.figure(figsize=[30,15])
        plt.bar(x_axis, ratio, color=colors)
        plt.ylim(top=1)
        plt.axvline(x=s_index,color='red',linestyle='--')
        plt.title(filename[:13], fontsize=30)
        plt.xlabel('Zone', fontsize=30)
        plt.ylabel('Ratio', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.show()
         
        recall_zm = gdal_array.LoadFile(drtr+"\Zone_Mask.TIF").astype(np.float32)
        add = recall_zm
        add[np.where(recall_zm < s_index)] = 0
        improved = added_cluster = water_cluster + add
        improved[np.where(added_cluster > 1)] = 1
        bf_area = np.count_nonzero(water_cluster == 1)*0.0009 
        af_area = np.count_nonzero(improved == 1)*0.0009        
        print("   Water surface area:")
        print("      Before improvement: "+str(round(bf_area,3))+" km2")
        print("      After improvement: "+str(round(af_area,3))+" km2")       
        if bf_area == 0:
            fn_area = bf_area
            qual = 0
            print("      Image cannot be improved")
        else:
            if threshold < -0.5:
                fn_area = bf_area
                qual = 0
                print("      Image cannot be improved")
            else:
                if ratio[49] == 0:
                    fn_area = bf_area
                    qual = 0
                    print("      Image cannot be improved")
                else:
                    if N_10 == 0:
                        fn_area = bf_area
                        qual = 0
                        print("      Image cannot be improved")
                    else:
                        fn_area = af_area
                        qual = 1
        print("      Final area: "+str(round(fn_area,3))+" km2")
        print("   ")
        #output = gdal_array.SaveArray(improved.astype(gdal_array.numpy.float32), 
        #                              "Improved_"+filename[:13]+".TIF", 
        #                              format="GTiff", prototype=filename)
        #output = None 
        date = str(filename[5:13])
        results = np.append(results, [[str(filename[0]), str(filename[2:4]), 
                                       date[0:4]+"-"+date[4:6]+"-"+date[6:8], 
                                       round(threshold,3), round(ratio[49],3), 
                                       int(N_10), int(s_index), int(qual), 
                                       round(bf_area,3), round(af_area,3), 
                                       round(fn_area,3)]], axis=0)
        continue
    else:
        continue
os.chdir('..')
    


# EXPORT RESULTS AS A CSV FILE
print("Exporting results as a csv file ...")
with open("WSA_LS"+str(LS)+".csv","w", newline='') as my_csv:
    csvWriter = csv.writer(my_csv)
    csvWriter.writerows(results)
print("  ")
print("Done")    
print("  ")   
   

    
# SHOW THE RUNNING TIME
time = time.time()-start_time
mins = math.trunc(time/60)
secs = round(time-mins*60)
print("Running time: "+str(mins)+" min "+str(secs)+" sec")