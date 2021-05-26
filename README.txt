1. Py_Curve.py
	This code is to create the Water Level - Water Surface Area - Storage Volume curves of the reservoir from the DEM data.

	Step 1. Name the DEM data file (input data) as DEM.TIF and save it in the same folder with the Py_Curve.py file
	Step 2. Insert input parameters: 
		- Bounding box of the reservoir: bbox = [ulx, uly, lrx, lry]
		- A point in the reservoir extent: point = [lat, lon]
		- Maximum reservoir water level
	Step 3. Run the code, get the output as Curve.csv



2. Py_Mask.py
	This code is to create the expanded mask and zone mask. 

	Step 1. Store Landsat images in the folders Landsat_5, Landsat_7, and Landsat_8 
		The folders of landsat images are put in the in the same folder with the Py_Mask.py file
	Step 2. Name the DEM data file (input data) as DEM.TIF and save it in the same folder with the Py_Mask.py file
		Note that the DEM data is preprocessed to have the same cell size and alignment with Landsat images
	Step 3. Insert input parameters: 
		- Bounding box of the reservoir: bbox = [ulx, uly, lrx, lry]
		- A point in the reservoir extent: point = [lat, lon]
		- Maximum reservoir water level
	Step 4. Run the code, get the outputs as Expanded_Mask.TIFF, Zone_Mask.TIFF, and Landsat_Mask.csv 
		(the list of Landsat images used to create the masks)



3. Py_WSA.py
	This code is to calculate water surface area corresponding to each Landsat image. 
	Input is the NDWI layers, which are previously calculated by Py_Mask.py and stored in the folders Landsat_5, Landsat_7, 
	and Landsat_8.
	It also requires the expanded mask and zone mask previously created by Py_Mask.py and stored in the same folder with 
	the Py_WSA.py file.

	1. Select the folder of Landsat images (Landsat_5, Landsat_7, and Landsat_8)
	2. Run the code, get the outputs as WSA_LS5.csv, WSA_LS7.csv, and WSA_LS8.csv
