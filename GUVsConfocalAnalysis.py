# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:05:54 2020

@author: Ran Tivony 
"""


# =============================================================================
# GUVs detection and analysis in a stack of tiff images
# =============================================================================


from scipy.ndimage import filters
import numpy as np
import cv2
import statistics as st
import tifffile as tif
import tkinter as tk
from tkinter import filedialog
import os

# write to excel using pandas

 
# =============================================================================
# Reading a stack Tiff file to an array
# =============================================================================

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()   # Opens a dialog window

file_directory=os.path.dirname(file_path)  # Return the path directories without file name

img=tif.imread(file_path)   # Reads the file

# Finds number of images in stack (nimg) and image dimensions (nrows,ncols)
nimg, nrows, ncols=img[:,:,:].shape 

ans1=input('Upload another file? (y/n):   ')


if ans1=='y':
   root = tk.Tk()
   root.withdraw()
   file_path = filedialog.askopenfilename()   # Opens a dialog window
   img2=tif.imread(file_path)   # Reads the file

elif ans1!='y' and ans1!='n': 
    while ans1!='y' or ans1!='n':
        ans1=input('Chose a correct answer. Upload another file? (y/n):   ')
        if ans1=='y':
           root = tk.Tk()
           root.withdraw()
           file_path = filedialog.askopenfilename()   # Opens a dialog window
           img2=tif.imread(file_path)   # Reads the file
           break

if ans1=='n': pass
    

# =============================================================================
# Converting RGB to gray scale
# =============================================================================

ans=input('Convert the image to gray? (y/n):   ')

if ans =='y' and ans1 == 'n':
    img_gray=np.zeros([nimg,nrows,ncols],dtype=np.uint8)
    r=np.zeros([nimg,nrows,ncols],dtype=np.uint8)
    g=np.zeros([nimg,nrows,ncols],dtype=np.uint8)
    b=np.zeros([nimg,nrows,ncols],dtype=np.uint8)

    for i in range(nimg):
        r[i,:,:]=img[i,:,:,0]
        g[i,:,:]=img[i,:,:,1]
        b[i,:,:]=img[i,:,:,2]
        img_gray[i,:,:]=0.3*r[i,:,:]+0.59*g[i,:,:]+0.11*b[i,:,:]
        img_gray=img_gray.astype('uint8')
        
else: img_gray=img

'''
    # saving 3D array to a tiff stack 
    with tif.TiffWriter(os.path.join(file_directory,'GrayScale_Img.tif')) as f:
        for i in range(img_gray.shape[0]):
            f.save(img_gray[i])
'''

if ans =='y' and ans1 == 'y':
    img_gray=np.zeros([nimg,nrows,ncols],dtype=np.uint8)
    img_gray2=np.zeros([nimg,nrows,ncols],dtype=np.uint8)
    r=np.zeros([nimg,nrows,ncols],dtype=np.uint8)
    g=np.zeros([nimg,nrows,ncols],dtype=np.uint8)
    b=np.zeros([nimg,nrows,ncols],dtype=np.uint8)

    for i in range(nimg):
        r[i,:,:]=img[i,:,:,0]
        g[i,:,:]=img[i,:,:,1]
        b[i,:,:]=img[i,:,:,2]
        img_gray[i,:,:]=0.3*r[i,:,:]+0.59*g[i,:,:]+0.11*b[i,:,:]
        img_gray=img_gray.astype('uint8')
    for i in range(nimg):
        r[i,:,:]=img2[i,:,:,0]
        g[i,:,:]=img2[i,:,:,1]
        b[i,:,:]=img2[i,:,:,2]
        img_gray2[i,:,:]=0.3*r[i,:,:]+0.59*g[i,:,:]+0.11*b[i,:,:]
        img_gray2=img_gray2.astype('uint8')
        
elif ans1=='y': img_gray2=img2



# =============================================================================
# Analysis
# =============================================================================

# =============================================================================
# # 1. Image processing (filtering and morphological transformations) 
# =============================================================================

gray_copy=np.zeros([nimg,nrows,ncols], dtype=np.uint8)
gauss_gray=np.zeros([nimg,nrows,ncols], dtype=np.uint8)
im_bw=np.zeros([nimg,nrows,ncols], dtype=np.uint8)

kernel=np.ones([5,5], dtype=np.uint8)  # Creating a kernel for dilation and erosion operations

for i in range(nimg):
    gray_copy[i,:,:]=img[i,:,:]
   
    # Applying gaussian Blurring
    gauss_gray[i,:,:]=filters.gaussian_filter(img_gray[i,:,:],sigma=2)
    

# =============================================================================
# # 2. GUVs detection for each image in stack
# =============================================================================

'''cv2.HoughCircles(image, detection method, dp=inverse resolution ratio accumulator to image, 
    min dist between centres, higher threshold Canny (the lower one is twice smaller), 
    accumulator threshold, minRadius of circle, maxRadius of circles)'''
    
circles_stack=np.zeros([img_gray.shape[0],200,3], dtype=int)
ncircles=np.zeros([nimg,1], dtype=int)

# Parameters for the HoughCircles algorithm:

MinCntrDis=36  # Minimum distance between GUVs centres - in pixels - below GUVs will be detected
HighCannyThresh=20   # Higher threshold for Canny edge detector (the lower one is twice smaller)
AccumulatorThresh=12  # HoughCircles accumulator threshold below which GUVs will get excluded
minR=15 # Minimum GUV radius (in pixels) to be detected
maxR=25   # Maximum GUV radius (in pixels) to be detected

ans1=input('Use only GUVs coordinates found in frame 1? (y/n):   ')

if ans1=='y': nimgs=1

else: nimgs=nimg

for i in range(nimgs):
    # Finding GUVs from gauss_gray through edge detection using HoughCircles
    circles = cv2.HoughCircles(gauss_gray[i,:,:],cv2.HOUGH_GRADIENT,1,MinCntrDis,param1=HighCannyThresh,param2=AccumulatorThresh,minRadius=minR,maxRadius=maxR)
    # transforming circles to a 2D array
    circles = np.array(circles[0]) 
    # Rounding up the values and transform to integer type 
    circles = np.uint16(np.around(circles)) 
    # finding the number of detected GUVs in each image
    ncircles[i]=circles.shape[0]
    # Storing all detected circles (centre coordinates and radius) in a 3D array circles_stack
    for j in range(circles.shape[0]): # scan over the number of detected circles (i.e. number of rows)
        for n in range(circles.shape[1]): # scan over the number of coloumns
            circles_stack[i,j,n]=circles[j][n]


# =============================================================================
# # 3. GUVs tracking using shortest distance between centroids
# =============================================================================
'''            
GUVs stack is an array that stores all detected GUVs with their 
centre coordinates and radius after they were tracked and repositioned. 
'''

# Initializing the dictionary GUV with GUVs found in frame 1
# The dictionary keys are defined as (GUV number, Frame number)
dis_thrs=1 # The distance threshold above which the same number will be assign to a GUV in the next frame.  
GUV={}  # Defining 'GUV' as a dictionary
GUV_num=0
for r in circles_stack[0]:
    if r[0]!=0: 
        GUV.update({(0,GUV_num):r})   # A dictionary of all GUVs in frame 1
        GUV_num=GUV_num+1   
           
centroid_dist=np.empty([circles_stack.shape[0],circles_stack.shape[1],circles_stack.shape[1]]) 
centroid_dist[:]=np.nan
for frame in range(circles_stack.shape[0]-1):
    col=0   # col states the GUV which is being tracked and compared with all GUVs from previous frames 
    for cntr1 in circles_stack[frame+1]:
        radius=cntr1[2]    # Radius of the currently tracked GUV 
        if cntr1[0]!=0:   # Checks if the array circles_stack[frame+1] is not empty (i.e. has a value of zero)

            for key, cntr2 in GUV.items():
                if key[0]==frame: # Comparing a GUV in circles_stack to GUVs from dictionary on previous frame 
                    centroid_dist[frame+1,key[1],col]=(((cntr1[0]-cntr2[0])**2)+((cntr1[1]-cntr2[1])**2))**0.5

            min_dist=np.nanmin(centroid_dist[frame+1,:,col])
            index=np.where(centroid_dist[frame+1,:,col]==min_dist)

            if min_dist<=dis_thrs*radius:  # If the GUV moves a distace smaller than its radius than it will be tracked (i.e. it is the same GUV as in previous frame)
                GUV[frame+1,index[0][0]]=circles_stack[frame+1,col]

            # In case that no min was found the GUV is examined against GUVs from two frames before 
            elif min_dist>dis_thrs*radius: 
                m=1
                while m<=frame:
                    for key, cntr2 in GUV.items():
                        if key[0]==frame-m:
                            centroid_dist[frame+1,key[1],col]=(((cntr1[0]-cntr2[0])**2)+((cntr1[1]-cntr2[1])**2))**0.5
                    min_dist=np.nanmin(centroid_dist[frame+1,:,col])
                    index=np.where(centroid_dist[frame+1,:,col]==min_dist)
                    if min_dist<=dis_thrs*radius:  # If the GUV moves a distace smaller than its radius than it will be tracked (i.e. it is the same GUV as in previous frame)
                        GUV[frame+1,index[0][0]]=circles_stack[frame+1,col]
                        break
                    else: m=m+1
                    if m>frame: 
                        GUV[(frame+1,col+50)]=circles_stack[frame+1,col]    # This is for newly detected GUVs to store them somewhere in the array so they will not get lost
                        break

        col=col+1  # Moving to the next GUV in circles_stack[frame+1]             


# Creating a 3D array GUVs_stack from dictionary for monitoring purposes 
GUVs_stack=np.zeros([img_gray.shape[0],200,3], dtype=int)
for key, cntr2 in GUV.items():
    GUVs_stack[(key[0],key[1])]=cntr2


# =============================================================================
# # 4. Extracting the averaged intensity values from GUVs 
# =============================================================================

# Creating a mask of zeros to find the pxl intensity inside the band/lumen:

mask1=np.zeros([nimg, nrows, ncols], dtype=int) # creating a 3d array of zeros with dimensions of the analyzed image
mask2=np.zeros([nimg, nrows, ncols], dtype=int) # creating a 3d array of zeros with dimensions of the analyzed image

GUV_lumen=np.zeros([nrows, ncols], dtype=int) # creating a 2d array of zeros for masking the lumen area of a detected GUV
GUV_band=np.zeros([nrows, ncols], dtype=int) # creating a 2d array of zeros for masking the membrane area (i.e. the band) of a detected GUV

# Creating arrays for storing the mean and median intensity values:
av_stack=np.empty([200, nimg], dtype=float) 
av_stack[:]=np.nan   # All cells in av_stack are initialized as nan (not a number)

median_stack=np.empty([200, nimg], dtype=float) 
median_stack[:]=np.nan   # All cells in av_stack are initialized as nan (not a number)


font=cv2.FONT_HERSHEY_SIMPLEX  # Defines a font type for labels

# Drawing a band around each detected GUV using data stored in circles: 
band_size=int(input('Enter band size (pixels):  '))  # Thickness of the band in munber of pixels
band=int(band_size/2)
   
ans=input('Analyse GUVs membrane or lumen? (membrane-1; lumen-2):   ')

# =============================================================================
# Fluorescence intensity analysis on a stack of images
# =============================================================================

# Measuring the intensity in the GUVs membrane:

if ans=='1' and ans1=='n':
   for j in range(nimg):   # scans through all images in stack
       # Creating an empty list for storing the pixel intesnsity values for each detected GUV 
       pxl_int=[] 
       n=0
       for key,cntr in GUV.items(): # scans through all circles found in each image
           if key[0]==j and cntr[0]!=0:  # draws a circle only where the cell in array cicrles_stack is not zero (i.e. empty) 
              # cv2.circle(img_arr,(i[0],i[1]),i[2],255,band_size) # A circular band is marked in the original image
              cv2.circle(GUV_band[:,:],(cntr[0],cntr[1]),cntr[2],255,band_size) # A circular band (pxl value 255) is marked in the mask
              label=str(key[1]+1)
              cv2.putText(gray_copy[j,:,:], label, (cntr[0]-5,cntr[1]+5), font, 0.5, 255, 1)  # Label each circle with its row number
              cv2.circle(gray_copy[j,:,:],(cntr[0],cntr[1]),cntr[2]-band,255,1)
              cv2.circle(gray_copy[j,:,:],(cntr[0],cntr[1]),cntr[2]+band,255,1)
              indx=np.where(GUV_band[:,:]==255)
              pxl_int.append(gauss_gray[j][indx]) 
              av_stack[key[1],j]=st.mean(pxl_int[n])
              median_stack[key[1],j]=st.median(pxl_int[n])
              mask1[j][indx]=GUV_band[indx]
              GUV_band=np.zeros([nrows, ncols], dtype=int) # Setting the GUVs mask to zero 
              n=n+1

# Measuring the intensity in the GUVs lumen:
    
elif ans=='2' and ans1=='n':
     for j in range(nimg):   # scans through all images in stack
         # Creating an empty list for storing the pixel intesnsity values for each detected GUV 
         pxl_int=[] 
         n=0
         for key,cntr in GUV.items(): # scans through all circles found in each image
           if key[0]==j and cntr[0]!=0:  # draws a circle only where the cell in array GUVs_stack is not zero (i.e. empty) 
              # cv2.circle(img_arr,(i[0],i[1]),i[2],255,band_size) # A circular band is marked in the original image
              cv2.circle(GUV_lumen[:,:],(cntr[0],cntr[1]),cntr[2]-5,255,-1) # A circular band (pxl value 255) is marked in the mask
              #cv2.circle(mask[j,:,:],(i[0],i[1]),2,100,2) # The center of the circle is marked
              label=str(key[1]+1)
              cv2.putText(gray_copy[j,:,:], label, (cntr[0]-5,cntr[1]+5), font, 0.5, 255, 1)  # Label each circle with its row number
              cv2.circle(gray_copy[j,:,:],(cntr[0],cntr[1]),cntr[2]-band,255,1)
              cv2.circle(gray_copy[j,:,:],(cntr[0],cntr[1]),cntr[2]+band,255,1)
              indx=np.where(GUV_lumen[:,:]==255)
              pxl_int.append(img_gray2[j][indx]) 
              av_stack[key[1],j]=st.mean(pxl_int[n])
              median_stack[key[1],j]=st.median(pxl_int[n])
              mask2[j][indx]=GUV_lumen[indx]
              GUV_lumen=np.zeros([nrows, ncols], dtype=int) # Setting the GUVs mask to zero 
              n=n+1
              
              

# =============================================================================
# Saving the data 
# =============================================================================


# Saving the average intensities of GUVs membrane (from av_stack) to file
np.savetxt(os.path.join(file_directory,'GUVs_Mean_Intensity.dat'), av_stack, fmt='%1.0f', delimiter='\t')

# Saving the average intensities of GUVs membrane (from av_stack) to file
np.savetxt(os.path.join(file_directory,'GUVs_Median_Intensity.dat'), median_stack, fmt='%1.0f', delimiter='\t')


# Saving the radius of detected GUVs in every frame  
np.savetxt(os.path.join(file_directory,'GUVs_radius_pixels.dat'), GUVs_stack[:,:,2], fmt='%1.0f', delimiter='\t')

# =============================================================================
# # Saving the lumen mask obtained for every image in stack
# with tif.TiffWriter(os.path.join(file_directory,'mask_stack.tif')) as f:
#      for i in range(mask2.shape[0]):
#          f.save(mask2[i])        
# =============================================================================

# Saving the mask obtained for every image in stack
with tif.TiffWriter(os.path.join(file_directory,'gray_copy.tif')) as f:
    for i in range(mask1.shape[0]):
        f.save(gray_copy[i])      



