import glob
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom  # working with dicom
import pywt  # working with wavelets
from scipy import ndimage, signal
from skimage import (color, exposure, filters, img_as_float64, img_as_ubyte,
                     img_as_uint, measure, morphology, restoration,
                     segmentation)

#function -> show image
def plot_markers(img, title=None):
    plt.imshow(img, cmap='gray'), plt.title(title)
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
def w_segmentation(roi, index=None): #roi is a dict
    img = roi['img_roi']
    se = np.ones((3,3), np.uint16)
    imD = morphology.dilation(img, footprint=se)
    imR = morphology.reconstruction(imD, img, method='erosion', footprint=se )
    imR = imR.astype(int)

    intM = morphology.local_maxima(imR)
    intM = morphology.closing(intM, footprint=se)

    #selecting markers
    #checking local_maxima information
    m_labels, m_num = measure.label(intM, return_num=True, connectivity=2) #con. 1 -> 4 neighborhood
    m_props = measure.regionprops(label_image=m_labels, intensity_image=img)

    valid_labels = set()
    for marker in m_props:
        m_area = marker.area <= 25
        m_mj_length =  marker.axis_major_length <= 10
        m_imax = marker.intensity_max >= 2000
        if (m_area and m_mj_length) and m_imax:
            valid_labels.add(marker.label)

    #checking ROI
    if len(valid_labels) < 3:
        return 
    else:
        markers = np.in1d(m_labels, list(valid_labels)).reshape(m_labels.shape)
        ms_label, ms_num = measure.label(markers, return_num=True, connectivity=1) #con. 1 -> 4 neighborhood
        m_props = measure.regionprops(label_image=ms_label, intensity_image=img)
        result_dict = {'num_markers': len(valid_labels),
                        'markers_props': m_props} 
        
        #df markers
        #m_props_table = measure.regionprops_table(label_image=ms_label, intensity_image=img,
        #                                        properties=('intensity_min', 'axis_minor_length'))
        #m_props_table = pd.DataFrame(m_props_table)

        gmag = filters.sobel(img)
        r_watershed = segmentation.watershed(gmag, markers=ms_label, watershed_line=True)
        #se não usar as linhas pra separar as regiões, dá erro na hora de selecionar
        #talvez investigar outra forma de selecionar

        w_props = measure.regionprops(label_image=r_watershed, intensity_image=img)

        #df watershed
        w_props_table = measure.regionprops_table(label_image=r_watershed, intensity_image=img,
                                                properties=('label',
                                                            'area',
                                                            'area_bbox',
                                                            'area_convex'))
        w_props_table = pd.DataFrame(w_props_table)

        valid_regions = set()
        for region in w_props:
            w_area = (region.area <= 50)
            w_area_convex = (region.area_convex <= 50)
            w_mj_length =  (region.axis_major_length <= 25)
            if ((w_area and w_area_convex) and w_mj_length):
                valid_regions.add(region.label)
        
        #checking ROI
        if len(valid_regions) <= 3:
            return 
        else:
            #regions é um array do tipo bool
            regions = np.isin(r_watershed, list(valid_regions)).reshape(r_watershed.shape)
            rg_label, rg_num = measure.label(regions, return_num=True, connectivity=1) #con. 1 -> 4 neighborhood
            r_props = measure.regionprops(label_image=rg_label, intensity_image=img)

            #df regions
            r_props_table = measure.regionprops_table(label_image=rg_label, intensity_image=img,
                                                properties=('label',
                                                            'area',
                                                            'area_bbox',
                                                            'area_convex'))
            r_props_table = pd.DataFrame(r_props_table)
            
            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(m_labels, cmap='gray')
            ax[0,0].set_title('Markers')
            ax[0,0].axis('off')
            ax[0,1].imshow(ms_label, cmap='gray')
            ax[0,1].set_title('Selected Markers')
            ax[0,1].axis('off')
            ax[1,0].imshow(r_watershed, cmap='gray')
            ax[1,0].set_title('Regions')
            ax[1,0].axis('off')
            ax[1,1].imshow(rg_label, cmap='gray')
            ax[1,1].set_title('Selected Regions')
            ax[1,1].axis('off')
            #plt.savefig('{0}.jpeg'.format(img_index))
            #como eliminar alguns formatos de marcador? usar esqueleto?

        return (r_props_table)

img_dir = 'C:\\Users\\Equipacare\\Desktop\\image code\\images'
data_path = os.path.join(img_dir, '*dcm')
files = glob.glob(data_path)
data = [] #list of original images

df_regions = pd.DataFrame()

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

    img_index = 0
    for roi in roi_data:        
        df_result = w_segmentation(roi, img_index) #roi is a dict
        df_regions = df_regions.append(df_result)
        print('{0}'.format(img_index))
        img_index += 1

df_regions.plot()
print('end')