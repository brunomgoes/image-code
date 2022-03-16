import glob
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom  # working with dicom
import pywt  # working with wavelets
from scipy import ndimage as ndi, signal
from skimage import (color, exposure, feature, filters, img_as_float64,
                     img_as_ubyte, img_as_uint, measure, morphology,
                     restoration, segmentation)


#function -> show image
def plot_image(img, title=None):
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

    return (pywt.waverec2(coeffs=[cA, (cH_new, cV_new, cD_new)], wavelet=wave_name))

def wavelet_clahe(img):
    wave_name = 'coif5'
    k = 3
    coeffs = pywt.wavedec2(img, wavelet=wave_name, level=1)
    cA, (cH, cV, cD) = coeffs
    #por que cA vira uma array com max > 1?

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
def w_segmentation(roi, index=None, title=None): #roi is a dict
    img = roi

    se = np.ones((3,3), np.uint16)
    imD = morphology.dilation(img, footprint=se)
    imD1 = imD*img_max
    imR = morphology.reconstruction(imD, img, method='erosion', footprint=se )
    imR1 = imR*img_max

    seed = np.copy(imR)
    seed[1:-1, 1:-1] = imR.min()
    mask = imR

    dilated = morphology.reconstruction(seed, mask, method='dilation')
    dilated1 = dilated*img_max

    im_sub = imR - dilated

    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(10, 5))
    ax[0,0].imshow(imD1.astype('uint'), cmap='gray', vmin=0, vmax=4095)
    ax[0,0].axis('off')
    ax[0,0].set_title('dilated')
    ax[1,0].imshow(dilated1, cmap='gray', vmin=0, vmax=4095)
    ax[1,0].axis('off')
    ax[1,0].set_title('regional maxima')
    ax[0,1].imshow(imR1.astype('uint'), cmap='gray', vmin=0, vmax=4095)
    ax[0,1].axis('off')
    ax[0,1].set_title('closing by reconstruction')
    ax[1,1].imshow(imR1-dilated1, cmap='gray', vmin=0, vmax=4095)
    ax[1,1].axis('off')
    ax[1,1].set_title('imcbr - regional maxima')
    plt.tight_layout()
    plt.show()

    threshold_global_otsu = filters.threshold_otsu(im_sub)
    intM = im_sub >= threshold_global_otsu

    #plot_image(intM)

    #selecting markers
    #checking local_maxima information
    m_labels, m_num = measure.label(intM, return_num=True, connectivity=2) #con. 1 -> 4 neighborhood
    m_props = measure.regionprops(label_image=m_labels, intensity_image=img)

    valid_labels = set()
    for marker in m_props:
        m_area = marker.area <= 25
        m_area_bbox = marker.area_bbox <= 25
        m_area_convex = marker.area_convex <= 25
        m_mj_length =  marker.axis_major_length <= 10
        m_imax = marker.intensity_max >= 0.49
        m_mean = marker.intensity_mean >= 0.49
        if (((m_area and m_area_bbox) and (m_mj_length and m_area_convex)) and 
            ((m_imax and m_mean))):
            valid_labels.add(marker.label)

    #checking ROI
    if len(valid_labels) < 3:
        return 
    else:
        markers = np.in1d(m_labels, list(valid_labels)).reshape(m_labels.shape)
        ms_label, ms_num = measure.label(markers, return_num=True, connectivity=1) #con. 1 -> 4 neighborhood
        m_props = measure.regionprops(label_image=ms_label, intensity_image=img)
        #result_dict = {'num_markers': len(valid_labels),
        #                'markers_props': m_props} 
        
        fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10, 5))
        ax[0].imshow(intM, cmap='gray')
        ax[0].axis('off')
        ax[1].imshow(markers, cmap='gray')
        ax[1].axis('off')
        plt.tight_layout()
        plt.show()

        #df markers
        m_props_table = measure.regionprops_table(label_image=ms_label, intensity_image=img,
                                                properties=('intensity_min',
                                                            'intensity_max',
                                                            'intensity_mean',
                                                            'area',
                                                            'area_convex',
                                                            'area_bbox',
                                                            'extent',
                                                            'solidity'))
        m_props_table = pd.DataFrame(m_props_table)

        gmag = filters.sobel(img)
        r_watershed = segmentation.watershed(gmag, markers=ms_label, watershed_line=True)
        #se não usar as linhas pra separar as regiões, dá erro na hora de selecionar
        #talvez investigar outra forma de selecionar

        w_props = measure.regionprops(label_image=r_watershed, intensity_image=img)

        #df watershed
        w_props_table = measure.regionprops_table(label_image=r_watershed, intensity_image=img,
                                                properties=('intensity_min',
                                                            'intensity_max',
                                                            'intensity_mean',
                                                            'area',
                                                            'area_convex',
                                                            'area_bbox',
                                                            'extent',
                                                            'solidity'))
        w_props_table = pd.DataFrame(w_props_table)
        #w_props_table['area'].hist(bins=500)

        valid_regions = set()
        for region in w_props:
            w_area = region.area <= 25
            w_area_bbox = region.area_bbox <= 25
            w_area_convex = region.area_convex <= 25
            w_mj_length =  region.axis_major_length <= 25
            w_imax = region.intensity_max >= 0.48
            w_mean = region.intensity_mean >= 0.48
            if (((w_area and w_area_bbox) and (w_mj_length and w_area_convex)) and 
                ((w_imax and w_mean))):
                valid_regions.add(region.label)
        
        #checking ROI
        if len(valid_regions) <= 3:
            return 
        else:
            #regions é um array do tipo bool
            regions = np.isin(r_watershed, list(valid_regions)).reshape(r_watershed.shape)
            rg_label, rg_num = measure.label(regions, return_num=True, connectivity=1) #con. 1 -> 4 neighborhood
            #r_props = measure.regionprops(label_image=rg_label, intensity_image=img)

            #df regions
            r_props_table = measure.regionprops_table(label_image=rg_label, intensity_image=img,
                                                properties=('intensity_min',
                                                            'intensity_max',
                                                            'intensity_mean',
                                                            'area',
                                                            'area_convex',
                                                            'area_bbox',
                                                            'extent',
                                                            'solidity'))
            r_props_table = pd.DataFrame(r_props_table)
            
            fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10, 5))
            ax[0].imshow(segmentation.mark_boundaries(img, label_img=ms_label, color=(139,0,0)), cmap='gray', vmin=0, vmax=4095)
            ax[0].axis('off')
            ax[0].set_title('markers')
            ax[1].imshow(segmentation.mark_boundaries(img, label_img=rg_label, color=(139,0,0)), cmap='gray', vmin=0, vmax=4095)
            ax[1].axis('off')
            ax[1].set_title('regions')
            plt.tight_layout()
            #plt.show()
            plt.suptitle(title)
            plt.savefig('{0}.jpeg'.format(img_index))
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
    img_array = dicom_file.pixel_array
    img_max = img_array.max()
    img = img_array/img_max

    # #create a list of ROIs for each image
    # row, col = img.shape
    # roi_index = 0
    # roi_data = []
    # roi_median = []
    # for x in range(0, row, 50):
    #     for y in range(0, col, 50):
    #         roi = img[x:x+100, y:y+100]
    #         #check if max value in roi is 0
    #         if np.median(roi) > (1500/img_max):
    #             roi_dict = {'index': roi_index,
    #                         'img_roi': roi,
    #                         'x': x,
    #                         'y': y}
    #             roi_data.append(roi_dict)
    #             roi_index += 1

    # img_dict.update({'roi_list': roi_data})
    # data.append(img_dict) 

    img_index = 0
    roi_1 = img[700:890, 100:600]
    roi_2 = img[960:1420, 100:330]
    roi_data = [roi_1, roi_2]
    for roi in roi_data:        
        #pre processing
        #original
        df_result1 = w_segmentation(roi, img_index, 'original') #roi is a dict
        print('{0}'.format(img_index))
        img_index += 1
       
        #wavelet
        #pywt.families()
        #pywt.wavelist('coif')
        im_wave = wavelet_filter(roi, 'coif5')
        df_result2 = w_segmentation(im_wave, img_index, 'wavelet') #roi is a dict
        print('{0}'.format(img_index))
        img_index += 1

        #wavelet + clahe
        im_wave_clahe = wavelet_clahe(roi)
        df_result3 = w_segmentation(im_wave_clahe, img_index, 'wave + clahe') #roi is a dict
        print('{0}'.format(img_index))
        img_index += 1

        #média
        im_mean = roi*img_max
        im_mean = filters.rank.mean(im_mean.astype('uint16'), np.ones((3,3), np.uint16))
        im_mean = im_mean/img_max
        df_result_4 = w_segmentation(im_mean, img_index, 'mean')
        print('{0}'.format(img_index))
        img_index += 1

        #mediana
        im_median = roi*img_max
        im_median = filters.rank.median(im_median.astype('uint16'), np.ones((3,3), np.uint16))
        im_median = im_median/img_max
        df_result_5 = w_segmentation(im_median, img_index, 'median')
        print('{0}'.format(img_index))
        img_index += 1

        #wiener
        im_wiener = signal.wiener(roi, (3, 3))
        df_result_6 = w_segmentation(im_wiener, img_index, 'wiener')
        print('{0}'.format(img_index))
        img_index += 1