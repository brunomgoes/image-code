import glob
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom  # working with dicom
import pywt  # working with wavelets
from skimage import filters, measure, morphology, segmentation

def divide_rois(img):
    img_copy = np.copy(img)
    img_copy = img_copy/4095

    roi_list = []
    n_row, n_col = img_copy.shape
    
    for x in range(0, n_col, 50):
        for y in range(0, n_row, 50):
            roi_array = img_copy[y:y+150, x:x+150]
            roi = {
                'roi_array': roi_array,
                'x': x,
                'y': y
            }
            if (np.percentile(roi_array, 75) != 0):
                roi_list.append(roi)

    # tem que pensar o que fazer nas bordas da imagem
    return roi_list

def wavelet_filter(img, wave_name):
    k = 3
    cA, (cH, cV, cD) = pywt.wavedec2(img, wavelet=wave_name, level=1)

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

    result = pywt.waverec2(coeffs=[cA, (cH_new, cV_new, cD_new)], wavelet=wave_name)

    return (result.astype('uint16'))

def micro_segm(roi_array):
    roi = np.copy(roi_array)

def extract_features(roi_array):
    roi_dict = {
            'index': i,
            'mean': np.mean(roi_array),
            'median': np.median(roi_array),
            'std': np.std(roi_array),
            'var': np.var(roi_array)
            }
            
    return(roi_dict)

img_dir = 'C:\\Users\\br_go\\Desktop\\image-code\\images'
data_path = os.path.join(img_dir, '*dcm')
files = glob.glob(data_path)
data = [] #list of original images

for file in files:
    dicom_file = pydicom.dcmread(file)
    img_array = dicom_file.pixel_array
    data.append(img_array) 
    

j = 0
for img in data:
    roi_data = divide_rois(img)
    # roi_wave_data = []
    roi_info_list = []

    i = 0
    for roi in roi_data:
        roi_array = roi['roi_array']*4095
        # plt.imshow(roi_array.astype('uint16'), cmap='gray', vmin=0, vmax=4095)
        # plt.axis('off')
        # plt.show()

        # roi_wave = wavelet_filter(roi['roi_array'], wave_name='coif5')
        # roi_wave_data.append(roi_wave)

        roi_info = extract_features(roi['roi_array'])
        roi_info_list.append(roi_info)
        i += 1

    # img_roi_df = pd.DataFrame(roi_info_list)
    # img_roi_df.to_csv('{0}'.format(j),)
    # j += 1