import os
import glob
import math
import random
import pydicom #working with dicom
import pywt #working with wavelets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage import restoration, exposure, morphology, measure, filters, segmentation, color
from skimage import img_as_float64, img_as_ubyte, img_as_uint
from scipy import ndimage, signal

df = pd.DataFrame(columns=['area'])

#function -> show image
def plot_image(img, title=None):
    plt.imshow(img, cmap="gray", vmin=0, vmax=4095), plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_rois(list):
    img = random.sample(list, 9)
    fig, axs = plt.subplots(3, 3)
    axs[0,0].imshow(img[0], cmap="gray", vmin=0, vmax=4095)
    axs[0,0].axis('off')

    axs[0,1].imshow(img[1], cmap="gray", vmin=0, vmax=4095)
    axs[0,1].axis('off')

    axs[0,2].imshow(img[2], cmap="gray", vmin=0, vmax=4095)
    axs[0,2].axis('off')

    axs[1,0].imshow(img[3], cmap="gray", vmin=0, vmax=4095)
    axs[1,0].axis('off')

    axs[1,1].imshow(img[4], cmap="gray", vmin=0, vmax=4095)
    axs[1,1].axis('off')

    axs[1,2].imshow(img[5], cmap="gray", vmin=0, vmax=4095)
    axs[1,2].axis('off')

    axs[2,0].imshow(img[6], cmap="gray", vmin=0, vmax=4095)
    axs[2,0].axis('off')

    axs[2,1].imshow(img[7], cmap="gray", vmin=0, vmax=4095)
    axs[2,1].axis('off')
    
    axs[2,2].imshow(img[8], cmap="gray", vmin=0, vmax=4095)
    axs[2,2].axis('off')
    plt.show()

#function -> wiener filter
#ERRADO
def wiener_filter(img):
    psf = np.ones((5, 5))
    img = signal.convolve2d(img, psf, 'same')
    rng = np.random.default_rng()
    img += 0.1 * img.std() * rng.standard_normal(img.shape)
    img_wiener = restoration.wiener(img, psf, 1100, clip=True)
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
def w_segmentation(roi):
    img = roi['img_roi']
    se = np.ones((3,3), np.uint16)
    imD = morphology.dilation(img, footprint=se)
    imR = morphology.reconstruction(imD, img, method='erosion', footprint=se )
    
    intM = morphology.local_maxima(imR)
    intM = morphology.closing(intM, footprint=se)

    #selecting markers
    #checking local_maxima information
    m_labels, m_num = measure.label(intM, return_num=True, connectivity=1) #con. 1 -> 4 neighborhood, 
    m_props = measure.regionprops(label_image=m_labels, intensity_image=img)

    valid_labels = set()
    for marker in m_props:
        m_area = marker.area <= 30
        m_mj_length =  marker.axis_major_length <= 50
        if m_area and m_mj_length:
            valid_labels.add(marker.label)

    #checking ROI
    if len(valid_labels) < 3:
        return 
    else:
        result_dict = {'num_markers': len(valid_labels)} 
        markers = np.in1d(m_labels, list(valid_labels)).reshape(m_labels.shape)
        markers = ndimage.label(markers)[0]
        gmag = filters.laplace(img, ksize=3)
        r_watershed = segmentation.watershed(gmag, markers, watershed_line=False)

        w_labels, w_num = measure.label(r_watershed, return_num=True, connectivity=1)
        w_props = measure.regionprops(label_image=w_labels, intensity_image=img)
        
        valid_regions = set()
        for region in w_props:
            r_area = region.area <= 30
            r_mj_length = region.axis_major_length <= 50
            if r_area and r_mj_length:
                valid_regions.add(region.label)

            #checking ROI
        if len(valid_regions) < 3:
            return 
        else: 
            result = np.in1d(w_labels, list(valid_regions)).reshape(w_labels.shape)
            result = ndimage.label(result)[0]
            return result
 
img_dir = 'C:\\Users\\Equipacare\\Desktop\\image code\\images'
data_path = os.path.join(img_dir, '*dcm')
files = glob.glob(data_path)
data = [] #list of original images
for file in files:
    dicom_file = pydicom.dcmread(file)
    img_dict = {'name': os.path.basename(file),
                'image': dicom_file.pixel_array}
    img = dicom_file.pixel_array

    #create a list of ROIs for each image
    row, col = img.shape
    roi_index = 0
    roi_data = []
    for x in range(0, row, 50):
        for y in range(0, col, 50):
            roi = img[x:x+100, y:y+100]
            #check if max value in roi is 0
            if np.max(roi) !=0:
                roi_dict = {'index': roi_index,
                            'img_roi': roi,
                            'x': x,
                            'y': y}
                roi_data.append(roi_dict)
                roi_index += 1
    img_dict.update({'roi_list': roi_data})
    data.append(img_dict) 

    l_watershed = [] 
    for roi in roi_data:
        roi_segm = w_segmentation(roi) #roi is a dict
        if roi_segm is not None:
            l_watershed.append(roi_segm)
        else:
            continue

