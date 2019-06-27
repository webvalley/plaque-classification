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
def rand_rot(img):
   angle = random.randrange(0, 180)
   return(rotate(img, angle, resize=True))
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
Gaussian Blur
'''
def gauss(img, sig):
    return filters.gaussian(img, sigma = sig)

raw_images = "./Data/Raw/"
aug_images = "./Data/Aug/"
png_files = [os.path.join(raw_images, file_path) for file_path in os.listdir(raw_images)]
for counter, file_path in enumerate(png_files):
    print(file_path)
    img = np.array(imageio.imread(file_path))   
    np.save(aug_images + file_path.split("/")[-1][:4]+".npy", img)
    img_rot = rand_rot(img)
    np.save(aug_images + file_path.split("/")[-1][:4]+"_rot.npy",img_rot)
    img_hflip = hflip(img_rot)
    np.save(aug_images + file_path.split("/")[-1][:-4]+"_hflip.npy",img_hflip)
    img_vlip = vflip(img_hflip)
    np.save(aug_images + file_path.split("/")[-1][:-4]+"_vlip.npy",img_vlip)
    img_gauss = gauss(img_vlip, 10)
    np.save(aug_images + file_path.split("/")[-1][:-4]+"_gauss.npy",img_gauss)


