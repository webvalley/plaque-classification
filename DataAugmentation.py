import os
import numpy as np
import skimage.filters as filters
from skimage.transform import rotate
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import random
import imageio

'''
Random Rotation
'''
def rot(img, ang):
   #angle = random.randrange(0, 180)
   return(rotate(img, ang))
'''
Flip Horizontal
'''
def hflip(img):
    return np.fliplr(img)

'''
Flip Vertical
'''
def vflip(img):
    return np.flipud(img)
'''
Flip Diagonal
'''
def dflip(img):
    return vflip(hflip(img))
'''
Gaussian Blur
'''
def gauss(img, sig):
    return filters.gaussian(img, sigma = sig)

plaque_images = "./Data/Annotated/"
patch_images = "./Data/Patches/"
aug_images =["./Data/Aug0/", "./Data/Aug1/"]
aug_num = 0
png_files = [(os.path.join(patch_images, file_path) for file_path in os.listdir(patch_images)), (os.path.join(plaque_images, file_path) for file_path in os.listdir(plaque_images))]
for folder in png_files:
    aug_num += 1
    for counter, file_path in enumerate(folder):
        if('.png' in file_path):
            #print(file_path) 
            img = np.array(imageio.imread(file_path))   
            np.save(aug_images[aug_num -1] + file_path.split("/")[-1][:-4]+".npy", img)
            img_rot90 = rot(img, 90)
            np.save(aug_images[aug_num-1] + file_path.split("/")[-1][:-4]+"_rot90.npy", img_rot90)
            img_rot270 = rot(img, 270)
            np.save(aug_images[aug_num-1] + file_path.split("/")[-1][:-4]+"_rot270.npy", img_rot270)
            img_hflip = hflip(img)
            np.save(aug_images[aug_num-1] + file_path.split("/")[-1][:-4]+"_hflip.npy", img_hflip)
            img_vflip = vflip(img)
            np.save(aug_images[aug_num-1] + file_path.split("/")[-1][:-4]+"_vflip.npy", img_vflip)
            img_dflip = dflip(img)
            np.save(aug_images[aug_num-1] + file_path.split("/")[-1][:-4]+"_dflip.npy", img_dflip)
            img_gauss = gauss(img, 10)
            np.save(aug_images[aug_num-1] + file_path.split("/")[-1][:-4]+"_gauss10.npy", img_gauss)
            #img_gauss = gauss(img_, 20)
            #np.save(aug_images + file_path.split("/")[-1][:-4]+"_gauss20.npy", img_gauss)
