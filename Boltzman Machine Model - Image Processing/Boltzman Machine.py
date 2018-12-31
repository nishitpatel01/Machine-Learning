# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:32:47 2018

@author: NishitP
"""


# import required packages for this exercise
import numpy as np
import pandas as pd
import gzip
import numpy as np
import math
from struct import unpack
from numpy import zeros, float32
from copy import deepcopy
from pylab import imshow, show, cm


# PART 1
# reading dataset
images = "train-images-idx3-ubyte.gz"

def get_labeled_data(imagefile):

    images = gzip.open(imagefile, 'rb')

    images.read(4) 
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    x = zeros((20, rows, cols), dtype=float32)  
    for i in range(20):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
    return x


dt = get_labeled_data(images) / 255
pixel = deepcopy(dt)


# binarize the image
for k in range((20)):
    for i in range(pixel.shape[1]):
         for j in range(pixel.shape[2]):
               if pixel[k][i][j] <= 0.5:
                    pixel[k][i][j] = -1
               else:
                    pixel[k][i][j] = 1
                         
noise_pixel = deepcopy(pixel)


# PART 2
# read noise coordinates for adding noise in specific locations
noise_coord = pd.read_csv("NoiseCoordinates.csv")
noise_coord

def create_noise(img_r,img_c,main_idx) :
    
    img_df_row = noise_coord.loc[noise_coord['Row Description'] == img_r] 
    img_df_col = noise_coord.loc[noise_coord['Row Description'] == img_c] 
    
    img_df_row = img_df_row.iloc[:,1:16]
    img_df_col = img_df_col.iloc[:,1:16]
    
    img_df_row = img_df_row.values.tolist()
    img_df_col = img_df_col.values.tolist()

    for r,c in zip(img_df_row,img_df_col) :
        for i in range(len(img_df_row[0])) :
            row = r[i]
            col = c[i]
            noise_pixel[main_idx][row,col] = -pixel[main_idx][row,col]
      

# adding pre determined 
create_noise("Image 0 Row","Image 0 Column",0)
create_noise("Image 1 Row","Image 1 Column",1)
create_noise("Image 2 Row","Image 2 Column",2)
create_noise("Image 3 Row","Image 3 Column",3)
create_noise("Image 4 Row","Image 4 Column",4)
create_noise("Image 5 Row","Image 5 Column",5)
create_noise("Image 6 Row","Image 6 Column",6)
create_noise("Image 7 Row","Image 7 Column",7)
create_noise("Image 8 Row","Image 8 Column",8)
create_noise("Image 9 Row","Image 9 Column",9)
create_noise("Image 10 Row","Image 10 Column",10)
create_noise("Image 11 Row","Image 11 Column",11)
create_noise("Image 12 Row","Image 12 Column",12)
create_noise("Image 13 Row","Image 13 Column",13)
create_noise("Image 14 Row","Image 14 Column",14)
create_noise("Image 15 Row","Image 15 Column",15)
create_noise("Image 16 Row","Image 16 Column",16)
create_noise("Image 17 Row","Image 17 Column",17)
create_noise("Image 18 Row","Image 18 Column",18)
create_noise("Image 19 Row","Image 19 Column",19)



def view_image(image, label=""):  #Also from the website where we found the way to load
    """View a single image."""    # the dataset
    imshow(image, cmap=cm.gray)
show()


view_image(pixel[10])
view_image(noise_pixel[10])


# PART 3
# read initial parameter grid from csv file provided
start_params = pd.read_csv("InitialParametersModel.csv", header = None)
start_params = start_params.as_matrix()
start_params.size
shape = start_params


Q_mat_0 = deepcopy(start_params)
Q_mat_1 = deepcopy(start_params)
Q_mat_2 = deepcopy(start_params)
Q_mat_3 = deepcopy(start_params)
Q_mat_4 = deepcopy(start_params)
Q_mat_5 = deepcopy(start_params)
Q_mat_6 = deepcopy(start_params)
Q_mat_7 = deepcopy(start_params)
Q_mat_8 = deepcopy(start_params)
Q_mat_9 = deepcopy(start_params)
Q_mat_10 = deepcopy(start_params)
Q_mat_11 = deepcopy(start_params)
Q_mat_12 = deepcopy(start_params)
Q_mat_13 = deepcopy(start_params)
Q_mat_14 = deepcopy(start_params)
Q_mat_15 = deepcopy(start_params)
Q_mat_16 = deepcopy(start_params)
Q_mat_17 = deepcopy(start_params)
Q_mat_18 = deepcopy(start_params)
Q_mat_19 = deepcopy(start_params)

Q_mats = [Q_mat_0,Q_mat_1,Q_mat_2,Q_mat_3,Q_mat_4,Q_mat_5,Q_mat_6,Q_mat_7,Q_mat_8,Q_mat_9,Q_mat_10,Q_mat_11,Q_mat_12,Q_mat_13,Q_mat_14,
          Q_mat_15,Q_mat_16,Q_mat_17,Q_mat_18,Q_mat_19]



#read update coordinate files to get pixesl to be updates
update_coord = pd.read_csv("UpdateOrderCoordinates.csv")
update_coord


# placeholder for denoised version of images
denoised_image = np.zeros((20, pixel.shape[1], pixel.shape[2]))

size = pixel.shape[1]
size 


#new method for part 3
#function to find pixel neighbours
def get_neighbour(img_r, img_c):
    if(img_r in range(1,size-1) and img_c in range(1,size-1)):  #  4 neighbours
        neighbours = np.zeros((4,2))     
        neighbours[0][0] = img_r 
        neighbours[0][1] = img_c + 1
        neighbours[1][0] = img_r
        neighbours[1][1] = img_c - 1
        neighbours[2][0] = img_r - 1
        neighbours[2][1] = img_c 
        neighbours[3][0] = img_r + 1
        neighbours[3][1] = img_c 
    elif img_r == 0 and img_c == 0:   #2 neighbours
        neighbours  = np.zeros((2,2))     
        neighbours[0][0] = img_r 
        neighbours[0][1] = img_c + 1
        neighbours[1][0] = img_r + 1
        neighbours[1][1] = img_c
    elif img_r == 0 and img_c == size-1:
        neighbours = np.zeros((2,2))
        neighbours[0][0] = img_r 
        neighbours[0][1] = img_c - 1
        neighbours[1][0] = img_r + 1
        neighbours[1][1] = img_c
    elif img_r == size-1 and img_c == 0:
        neighbours = np.zeros((2,2))
        neighbours[0][0] = img_r - 1
        neighbours[0][1] = img_c 
        neighbours[1][0] = img_r 
        neighbours[1][1] = img_c + 1
    elif img_r == size-1 and img_c == size-1:
        neighbours = np.zeros((2,2))
        neighbours[0][0] = img_r - 1
        neighbours[0][1] = img_c 
        neighbours[1][0] = img_r   
        neighbours[1][1] = img_c - 1
    elif img_c == 0:
        neighbours = np.zeros((3,2))
        neighbours[0][0] = img_r - 1
        neighbours[0][1] = img_c 
        neighbours[1][0] = img_r   
        neighbours[1][1] = img_c + 1
        neighbours[2][0] = img_r + 1   
        neighbours[2][1] = img_c 
    elif img_r == 0:
        neighbours = np.zeros((3,2))
        neighbours[0][0] = img_r + 1
        neighbours[0][1] = img_c 
        neighbours[1][0] = img_r   
        neighbours[1][1] = img_c + 1
        neighbours[2][0] = img_r    
        neighbours[2][1] = img_c - 1
    elif img_r == size-1:
        neighbours = np.zeros((3,2))
        neighbours[0][0] = img_r - 1
        neighbours[0][1] = img_c 
        neighbours[1][0] = img_r   
        neighbours[1][1] = img_c + 1
        neighbours[2][0] = img_r    
        neighbours[2][1] = img_c - 1
    else :
        neighbours = np.zeros((3,2))
        neighbours[0][0] = img_r - 1
        neighbours[0][1] = img_c 
        neighbours[1][0] = img_r   
        neighbours[1][1] = img_c - 1
        neighbours[2][0] = img_r + 1  
        neighbours[2][1] = img_c         
    return neighbours


theta_ij_hh = 0.8
theta_ij_hx = 2

def denoise_new(img_r,img_c,img_idx):
    for i in range(10):
        img_df_row = update_coord.loc[update_coord['Row Description'] == img_r] 
        img_df_col = update_coord.loc[update_coord['Row Description'] == img_c] 

        img_df_row = img_df_row.iloc[:,1:785]
        img_df_col = img_df_col.iloc[:,1:785]
      
        img_df_row = img_df_row.values.tolist()
        img_df_col = img_df_col.values.tolist()
        
        for r,c in zip(img_df_row,img_df_col):
            for j in range(len(img_df_row[0])):
                row = r[j]
                col = c[j]
                neighbours = get_neighbour(row,col)
                numerator = 0
                denominator = 0
                for n in range(neighbours.shape[0]):                  
                    pij = Q_mats[img_idx][(int)(neighbours[n,0]),(int)(neighbours[n,1])]
                    numerator += theta_ij_hh * (2 * pij - 1)
                    denominator += -theta_ij_hx * (2 * pij - 1)

                numerator += theta_ij_hx * noise_pixel[img_idx][row][col]
                denominator += (-1) * theta_ij_hx * noise_pixel[img_idx][row][col]
                denominator = np.exp(denominator)                
                numerator = np.exp(numerator)
                
                denominator = numerator + denominator
                Q_mats[img_idx][row,col] = numerator / denominator
                
                if Q_mats[img_idx][row,col] < 0.5:
                     denoised_image[img_idx][row][col] = 0
                else:
                     denoised_image[img_idx][row][col] = 1    

                
denoise_new("Image 0 Row","Image 0 Column",0)   
denoise_new("Image 1 Row","Image 1 Column",1)
denoise_new("Image 2 Row","Image 2 Column",2)
denoise_new("Image 3 Row","Image 3 Column",3)
denoise_new("Image 4 Row","Image 4 Column",4)
denoise_new("Image 5 Row","Image 5 Column",5)
denoise_new("Image 6 Row","Image 6 Column",6)
denoise_new("Image 7 Row","Image 7 Column",7)
denoise_new("Image 8 Row","Image 8 Column",8)
denoise_new("Image 9 Row","Image 9 Column",9)
denoise_new("Image 10 Row","Image 10 Column",10)
denoise_new("Image 11 Row","Image 11 Column",11)
denoise_new("Image 12 Row","Image 12 Column",12)
denoise_new("Image 13 Row","Image 13 Column",13)
denoise_new("Image 14 Row","Image 14 Column",14)
denoise_new("Image 15 Row","Image 15 Column",15)
denoise_new("Image 16 Row","Image 16 Column",16)
denoise_new("Image 17 Row","Image 17 Column",17)
denoise_new("Image 18 Row","Image 18 Column",18)
denoise_new("Image 19 Row","Image 19 Column",19)           

view_image(noise_pixel[0])
view_image(denoised_image[0])


#part 4
# calculating energy for images with initial Q
def calculate_energy(img_idx):
    Qs = start_params
    size = Qs.shape[0]
    E_LogQ = 0
    E_LogP = 0
    c = 0.0000000001
    hh = 0
    hx = 0
    for i in range(size):
        for j in range(size):
           E_LogQ +=  start_params[i][j] * np.log(start_params[i][j] + c) 
           E_LogQ += ((1 - start_params[i][j]) * np.log((1 - (start_params[i][j])) + c))
           
           neighbours = get_neighbour(i,j)
           for n in range(neighbours.shape[0]):
               hj = start_params[(int)(neighbours[n,0]),(int)(neighbours[n,1])]
               hi = start_params[i][j]
               hh += theta_ij_hh * (2 * hi -1) * (2 * hj -1)
           
           hx += theta_ij_hx * (2 * hi -1) * noise_pixel[img_idx][i][j]
           E_LogP = hh + hx
           Energy = E_LogQ - E_LogP
               
    print(Energy)
    
    
# method for updating Q matrix once
def denoise_energy(img_r,img_c,img_idx):
        img_df_row = update_coord.loc[update_coord['Row Description'] == img_r] 
        img_df_col = update_coord.loc[update_coord['Row Description'] == img_c] 

        img_df_row = img_df_row.iloc[:,1:785]
        img_df_col = img_df_col.iloc[:,1:785]
      
        img_df_row = img_df_row.values.tolist()
        img_df_col = img_df_col.values.tolist()
                
        for r,c in zip(img_df_row,img_df_col):
            for j in range(len(img_df_row[0])):
                row = r[j]
                col = c[j]
                neighbours = get_neighbour(row,col)
                numerator = 0
                denominator = 0
                for n in range(neighbours.shape[0]):                  
                    pij = Q_mats[img_idx][(int)(neighbours[n,0]),(int)(neighbours[n,1])]
                    numerator += theta_ij_hh * (2 * pij - 1)
                    denominator += -theta_ij_hx * (2 * pij - 1)

                numerator += theta_ij_hx * noise_pixel[img_idx][row][col]
                denominator += (-1) * theta_ij_hx * noise_pixel[img_idx][row][col]
                denominator = np.exp(denominator)                
                numerator = np.exp(numerator)
                
                denominator = numerator + denominator
                Q_mats[img_idx][row,col] = numerator / denominator
                
                
#energy after one itiration
def calculate_log_Q_one_itiration(img_idx):
    Qs = Q_mats[img_idx]
    size = Qs.shape[0]
    E_LogQ = 0
    E_LogP = 0
    c = 0.0000000001
    hh = 0
    hx = 0
    for i in range(size):
        for j in range(size):
           E_LogQ +=  Q_mats[img_idx][i][j] * np.log( Q_mats[img_idx][i][j] + c) 
           E_LogQ += ((1 -  Q_mats[img_idx][i][j]) * np.log((1 - (Q_mats[img_idx][i][j])) + c))
           
           neighbours = get_neighbour(i,j)
           for n in range(neighbours.shape[0]):
               hj =  Q_mats[img_idx][(int)(neighbours[n,0]),(int)(neighbours[n,1])]
               hi =  Q_mats[img_idx][i][j]
               hh += theta_ij_hh * (2 * hi -1) * (2 * hj -1)
           
           hx += theta_ij_hx * (2 * hi -1) * noise_pixel[img_idx][i][j]
           E_LogP = hh + hx
           Energy = E_LogQ - E_LogP
               
    print(Energy)                  
                     

           
#def calculate_log_Q_one_itiration_with_order(img_r,img_c,img_idx):
#    img_df_row = update_coord.loc[update_coord['Row Description'] == img_r] 
#    img_df_col = update_coord.loc[update_coord['Row Description'] == img_c] 
#
#    img_df_row = img_df_row.iloc[:,1:785]
#    img_df_col = img_df_col.iloc[:,1:785]
#  
#    img_df_row = img_df_row.values.tolist()
#    img_df_col = img_df_col.values.tolist()
#    
#    E_LogQ = 0
#    E_LogP = 0
#    c = 0.0000000001
#    hh = 0
#    hx = 0
#    
#    for r,c in zip(img_df_row,img_df_col):
#        for j in range(len(img_df_row[0])):
#           row = r[j]
#           col = c[j]
#           
#           E_LogQ +=  Q_mats[img_idx][row][col] * np.log( Q_mats[img_idx][row][col] + c) 
#           E_LogQ += ((1 -  Q_mats[img_idx][row][col]) * np.log((1 - ( Q_mats[img_idx][row][col])) + c))
#           
#           neighbours = get_neighbour(row,col)
#           for n in range(neighbours.shape[0]):
#               hj =  Q_mats[img_idx][(int)(neighbours[n,0]),(int)(neighbours[n,1])]
#               hi =  Q_mats[img_idx][row][col]
#               hh += theta_ij_hh * (2 * hi -1) * (2 * hj -1)
#           
#           hx += theta_ij_hx * (2 * hi -1) * noise_pixel[img_idx][row][col]
#           E_LogP = hh + hx
#           Energy = E_LogQ - E_LogP
#               
#    print(Energy)                  
                  

                                  
calculate_energy(10)   
calculate_energy(11)   

calculate_log_Q_one_itiration(10)   
calculate_log_Q_one_itiration(11)   

denoise_energy("Image 9 Row","Image 9 Column",9)
denoise_energy("Image 10 Row","Image 10 Column",10)
denoise_energy("Image 11 Row","Image 11 Column",11)


a = start_params
b = Q_mats[10]
(a != b).sum()

# part 5 - reconstruction image dump
a_10 = denoised_image[10]
a_11 = denoised_image[11]
a_12 = denoised_image[12]
a_13 = denoised_image[13]
a_14 = denoised_image[14]
a_15 = denoised_image[15]
a_16 = denoised_image[16]
a_17 = denoised_image[17]
a_18 = denoised_image[18]
a_19 = denoised_image[19]
np.savetxt("img_10.csv", a_10,fmt='%i', delimiter=",")
np.savetxt("img_11.csv", a_11,fmt='%i', delimiter=",")
np.savetxt("img_12.csv", a_12,fmt='%i', delimiter=",")
np.savetxt("img_13.csv", a_13,fmt='%i', delimiter=",")
np.savetxt("img_14.csv", a_14,fmt='%i', delimiter=",")
np.savetxt("img_15.csv", a_15,fmt='%i', delimiter=",")
np.savetxt("img_16.csv", a_16,fmt='%i', delimiter=",")
np.savetxt("img_17.csv", a_17,fmt='%i', delimiter=",")
np.savetxt("img_18.csv", a_18,fmt='%i', delimiter=",")
np.savetxt("img_19.csv", a_18,fmt='%i', delimiter=",")






