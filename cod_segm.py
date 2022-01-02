import os
import glob
import math
import random
import pydicom #working with dicom
import pywt #working with wavelets

import matplotlib.pyplot as plt
import numpy as np

from skimage import restoration, exposure, morphology, measure

#function -> show image
def plot_image(img, title=None):
    plt.imshow(img, cmap="gray"), plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_rois(img):
    fig, axs = plt.subplots(3, 3)
    axs[0,0].imshow(img[0], cmap="gist_gray")
    axs[0,0].axis('off')
    axs[0,1].imshow(img[1], cmap="gist_gray")
    axs[0,1].axis('off')
    axs[0,2].imshow(img[2], cmap="gist_gray")
    axs[0,2].axis('off')
    axs[1,0].imshow(img[3], cmap="gist_gray")
    axs[1,0].axis('off')
    axs[1,1].imshow(img[4], cmap="gist_gray")
    axs[1,1].axis('off')
    axs[1,2].imshow(img[5], cmap="gist_gray")
    axs[1,2].axis('off')
    axs[2,0].imshow(img[6], cmap="gist_gray")
    axs[2,0].axis('off')
    axs[2,1].imshow(img[7], cmap="gist_gray")
    axs[2,1].axis('off')
    axs[2,2].imshow(img[8], cmap="gist_gray")
    axs[2,2].axis('off')
    plt.show()
    
#function -> wiener filter
def wiener_filter(img):
        psf = np.ones((5, 5)) / 25
        img_wiener, _ = restoration.unsupervised_wiener(img, psf)
        return img_wiener

#function -> wavelet filter
def wavelet_filter(img):
    wave_name = 'coif5'
    k = 3
    coeffs = pywt.wavedec2(img, wavelet=wave_name, level=1)
    cA, (cH, cV, cD) = coeffs

    cH_var = np.var(cH)
    cH1 = math.sqrt(cH_var)*math.sqrt(math.pi/2)
    cH2 = math.sqrt(((4-math.pi)/2)*cH_var)
    cH_t = cH1 + (k*cH2)
    cH_new = pywt.threshold(data=cH, value=cH_t, mode='soft', substitute=0)

    cV_var = np.var(cV)
    cV1 = math.sqrt(cV_var)*math.sqrt(math.pi/2)
    cV2 = math.sqrt(((4-math.pi)/2)*cV_var)
    cV_t = cV1 + (k*cV2)
    cV_new = pywt.threshold(data=cV, value=cV_t, mode='soft', substitute=0)

    cD_var = np.var(cD)
    cD1 = math.sqrt(cD_var)*math.sqrt(math.pi/2)
    cD2 = math.sqrt(((4-math.pi)/2)*cD_var)
    cD_t = cD1 + (k*cD2)
    cD_new = pywt.threshold(data=cD, value=cD_t, mode='soft', substitute=0)

    return (pywt.waverec2(coeffs=[cA, (cH_new, cV_new, cD_new)], wavelet=wave_name ))

def wavelet_clahe(img):
    wave_name = 'coif5'
    k = 3
    coeffs = pywt.wavedec2(img, wavelet=wave_name, level=1)
    cA, (cH, cV, cD) = coeffs

    cH_var = np.var(cH)
    cH1 = math.sqrt(cH_var)*math.sqrt(math.pi/2)
    cH2 = math.sqrt(((4-math.pi)/2)*cH_var)
    cH_t = cH1 + (k*cH2)
    cH_new = pywt.threshold(data=cH, value=cH_t, mode='soft', substitute=0)

    cV_var = np.var(cV)
    cV1 = math.sqrt(cV_var)*math.sqrt(math.pi/2)
    cV2 = math.sqrt(((4-math.pi)/2)*cV_var)
    cV_t = cV1 + (k*cV2)
    cV_new = pywt.threshold(data=cV, value=cV_t, mode='soft', substitute=0)

    cD_var = np.var(cD)
    cD1 = math.sqrt(cD_var)*math.sqrt(math.pi/2)
    cD2 = math.sqrt(((4-math.pi)/2)*cD_var)
    cD_t = cD1 + (k*cD2)
    cD_new = pywt.threshold(data=cD, value=cD_t, mode='soft', substitute=0)

    cA_max = np.max(cA)
    cA1 = cA/cA_max
    cA2 = exposure.equalize_adapthist(cA1, clip_limit=0.01)
    cA_new = cA2*cA_max

    return (pywt.waverec2(coeffs=[cA_new, (cH_new, cV_new, cD_new)], wavelet=wave_name ))

#function -> segmemntation
def segmentation(img):
    se = morphology.square(3)
    imD = morphology.dilation(img, footprint=se)
    imR = morphology.reconstruction(imD, img, method='erosion', footprint=se )
    
    intM = morphology.local_maxima(imR)
    intM = morphology.closing(intM, footprint=se)

    labels, num = measure.label(intM, return_num=True)

    return intM
 
img_dir = 'C:\\Users\\Equipacare\\Desktop\\image code\\images'
data_path = os.path.join(img_dir, '*dcm')
files = glob.glob(data_path)
data = [] #list of original images
roi_data = []
for file in files:
    img = pydicom.dcmread(file)
    img = img.pixel_array
    data.append(img)
    len(data) #number of images -> each element in the list is a image

    #create a list of ROIs for each image
    row, col = img.shape
    for x in range(0, row, 50):
        for y in range(0, col, 50):
            roi = img[x:x+100, y:y+100]
            #check if max value in roi is 0
            if np.max(roi) !=0:
                roi_data.append(roi)
                #how to get x and y for each roi?
                #x -> row position
                #y -> col position
    len(roi_data) #number of ROIs -> each element in the list is a image

    #create a list of ROIs for each processing
    l_wiener = []
    l_wavelet = []
    l_clahe = []
    for roi in roi_data:
        #wiener filter
        l_wiener.append(wiener_filter(roi))
        #wavelet filter
        l_wavelet.append(wavelet_filter(roi))
        #wavelet + clahe
        l_clahe.append(wavelet_clahe(roi))
    
    l_intM = [] 
    for roi in l_wiener:
        roi_segm = segmentation(roi)
        l_intM.append(roi_segm)

    test = random.sample(l_intM, 9)
    plot_rois(test)

