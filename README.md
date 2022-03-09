# GUVs_Confocal_Analysis
Detection and analysis of GUVs from confocal images

This python code automatically detects, tracks and measures the pixel intensity within the GUVs lumen. Extraction of vesicles from confocal images (or image stack) is performed through detecting the signal of the GUVs equator using the Hough Circle Transform algorithm. To ensure a consistent analysis of the same vesicle throughout the duration of the experiment, all detected GUVs are tracked and a unique ID is assigned for each one of them. Vesicles tracking is performed by calculating the minimal Euclidean distance between a vesicle’s centroid and all other detected centroids in previous frames. Finally, the GUVs radius and lumenal intensity are measured for every image in the acquired measurement video. 
