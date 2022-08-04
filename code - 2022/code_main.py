import glob
import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom  # working with dicom
import pywt  # working with wavelets
from scipy import ndimage
from skimage import exposure, filters, measure, morphology, segmentation, util


def divide_rois(img): # in: image float64; out: list of dict
    img_copy = np.copy(img)

    roi_list = [] # lista com todas as rois
    n_row, n_col = img_copy.shape
    
    for x in range(0, n_col, 50): # range de 50 em 50 pixels
        for y in range(0, n_row, 50):
            roi_array = img_copy[y:y+150, x:x+150] # roi de 150 x 150 px
            roi = {
                'roi_array': roi_array,
                'x': x, # col
                'y': y # row
            }
            if (np.percentile(roi_array, 75) != 0): # seleciona os rois pelo percentil
                roi_list.append(roi)

    # tem que pensar o que fazer nas bordas da imagem
    return roi_list

def wavelet_filter(img_array, wave_name): # in: image float64, out img float64
    img = np.copy(img_array)
    
    k = 3
    cA, (cH, cV, cD) = pywt.wavedec2(img, wavelet=wave_name, level=1) # decompondo a imagem -> wavelet

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

    result = pywt.waverec2(coeffs=[cA, (cH_new, cV_new, cD_new)], wavelet=wave_name) # reconstruindo a imagem

    return (result) # para visualizar a imagem precisa passar para ubit16

def wavelet_clahe(img_array, wave_name): # in: image float64, out img float64
    img = np.copy(img_array)
    
    k = 3
    cA, (cH, cV, cD) = pywt.wavedec2(img, wavelet=wave_name, level=1) # decompondo a imagem -> wavelet

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

    cA_max = cA.max()
    cA_norm = cA/cA_max #a entrada na CLAHE precisa estar entre -1 e 1
    cA_clahe = exposure.equalize_adapthist(cA_norm, clip_limit=0.01) # como adicionar o kernel_size?
    cA_clahe = cA_clahe*cA_max

    result = pywt.waverec2(coeffs=[cA_clahe, (cH_new, cV_new, cD_new)], wavelet=wave_name) # reconstruindo a imagem

    return (result) # para visualizar a imagem precisa passar para ubit16

def extract_features(roi_array):
    roi_dict = {
            'index': i,
            'mean': np.mean(roi_array),
            'median': np.median(roi_array),
            'std': np.std(roi_array),
            'var': np.var(roi_array)
            }

    return(roi_dict)

def extract_region_features(img_array):
    return(None)

def segm_1(img_array): # in: image float64, out: label
    img = np.copy(img_array)

    se = np.ones((3,3), np.uint16)
    imD = morphology.dilation(img, footprint=se)
    imR = morphology.reconstruction(seed=imD, mask=img, method='erosion', footprint=se)

    # fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,20))
    # ax[0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[0].axis('off')
    # ax[0].set_title('original')
    # ax[1].imshow(imD*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[1].axis('off')
    # ax[1].set_title('dilation')
    # ax[2].imshow(imR*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[2].axis('off')
    # ax[2].set_title('reconstruction by erosion')
    # plt.show()

    seed = np.copy(imR)
    seed[1:-1, 1:-1] = imR.min() #set its border to be the pixel values in the original image
    mask = imR

    dilated = morphology.reconstruction(seed, mask, method='dilation')
    im_sub = (imR - dilated)

    # fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,20))
    # ax[0].imshow(seed*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[0].axis('off')
    # ax[0].set_title('recon. by erosion')
    # ax[1].imshow(dilated*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[1].axis('off')
    # ax[1].set_title('background')
    # ax[2].imshow(im_sub*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[2].axis('off')
    # ax[2].set_title('rbe - background')
    # plt.show()

    thresh = filters.threshold_otsu(im_sub)
    intM = im_sub >= thresh # imagem binaria com os marcadores
    m_label, m_num = measure.label(intM, return_num=True, connectivity=1) #con. 1 -> 4 neighborhood

    gmag = filters.sobel(img)
    r_watershed = segmentation.watershed(gmag, markers=m_label, watershed_line=True)

    # fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,15))
    # ax[0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[0].axis('off')
    # ax[0].set_title('original')
    # ax[1].imshow(segmentation.mark_boundaries(img, m_label))
    # ax[1].axis('off')
    # ax[1].set_title('markers')
    # ax[2].imshow(segmentation.mark_boundaries(img, r_watershed))
    # ax[2].axis('off')
    # ax[2].set_title('watershed')
    # plt.tight_layout()
    # plt.show()
    #.savefig('{0}.png'.format(index))

    return r_watershed

def segm_2(img_array): # tese - evanivaldo - in: image float64, out: binary
    img = np.copy(img_array)
    
    img_grad = np.gradient(img)
    img_norm = np.linalg.norm(img_grad, axis=0)

    r1 = 1/(1 + img_norm)
    r1 = util.invert(r1)

    img_sobel = filters.sobel(img)

    # fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,20))
    # ax[0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[0].axis('off')
    # ax[0].set_title('original')
    # ax[1].imshow(r1*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[1].axis('off')
    # ax[1].set_title('filtro diferencial')
    # ax[2].imshow(img_sobel*4095, cmap='gray', vmin=0, vmax=4095)
    # ax[2].axis('off')
    # ax[2].set_title('sobel')
    # plt.show()

    seg_1 = r1 >= 0.018
    seg_2 = img_sobel >= 0.018

    markers = np.logical_and(seg_1, seg_2)

    # fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,20))
    # ax[0].imshow(seg_1, cmap='gray', vmin=0, vmax=1)
    # ax[0].axis('off')
    # ax[0].set_title('filtro diferencial - threshold')
    # ax[1].imshow(seg_2, cmap='gray', vmin=0, vmax=1)
    # ax[1].axis('off')
    # ax[1].set_title('sobel - threshold')
    # ax[2].imshow(markers, cmap='gray', vmin=0, vmax=1)
    # ax[2].axis('off')
    # ax[2].set_title('diferencial AND sobel')
    # plt.show()

    kernel = np.array([[1, 0, 1, 0, 1],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0],
                   [1, 0, 1, 0, 1]])

    morf_1 = ndimage.convolve(markers, kernel) # filtro asterisco tamanho mínimo, como funciona?

    se = np.array([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]])

    morf_2 = morphology.dilation(markers, footprint=se)
    morf_2 = morphology.reconstruction(seed=morf_2, mask=markers, method='erosion', footprint=se)

    # fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,20))
    # ax[0].imshow(markers, cmap='gray', vmin=0, vmax=1)
    # ax[0].axis('off')
    # ax[0].set_title('marcadores')
    # ax[1].imshow(morf_1, cmap='gray', vmin=0, vmax=1)
    # ax[1].axis('off')
    # ax[1].set_title('filtro asterisco')
    # ax[2].imshow(morf_2, cmap='gray', vmin=0, vmax=1)
    # ax[2].axis('off')
    # ax[2].set_title('dilatação - recon. por erosão')
    # plt.show()

    return morf_2

def segm_3(img_array): # usar o resultado de segm_2 como marcador para watershed
    img = np.copy(img_array)

    return img

img_dir = 'C:\\Users\\br_go\\Desktop\\image-code\\images'
data_path = os.path.join(img_dir, '*dcm')
files = glob.glob(data_path)
data = [] #list of original images

for file in files:
    dicom_file = pydicom.dcmread(file)
    img_array = dicom_file.pixel_array
    img_array =  img_array/4095 # normaliza a imagem
    data.append(img_array) 

j = 0
for img in data:
    im_wave = wavelet_filter(img, wave_name='coif5')
    im_wave_clahe = wavelet_clahe(img, wave_name='coif5')
    im_segm_1 = segm_2(img)

    # im_wave = im_wave*4095 # a imagem deixa de ser normalizada
    # im_wave_clahe = im_wave_clahe*4095
    # plt.imshow(im_wave_clahe.astype('uint16'), cmap='gray', vmin=0, vmax=4095)
    # plt.axis('off')
    # plt.show()
    
    roi_data = divide_rois(img)
    roi_wave_data = []
    roi_wave_clahe_data = []
    roi_info_list = []
    roi_wave_info_list = []
    roi_wave_clahe_info_list = []

    i = 0
    for roi in roi_data:
        roi_array = roi['roi_array']*4095
        # plt.imshow(roi_array.astype('uint16'), cmap='gray', vmin=0, vmax=4095)
        # plt.axis('off')
        # plt.show()

        # roi_wave = wavelet_filter(roi['roi_array'], wave_name='coif5')
        # roi_wave_data.append(roi_wave)
        # roi_wave_clahe = wavelet_clahe(roi['roi_array'], wave_name='coif5')
        # roi_wave_clahe_data.append(roi_wave_clahe)

        # roi_info_list.append(extract_features(roi['roi_array']))
        # roi_wave_info_list.append(extract_features(roi_wave))
        # roi_wave_clahe_info_list.append(extract_features(roi_wave_clahe))

        i += 1

    # img_roi_df = pd.DataFrame(roi_info_list)
    # img_roi_df.to_csv('{0}-original'.format(j),)

    # img_roi_wave_df = pd.DataFrame(roi_wave_info_list)
    # img_roi_wave_df.to_csv('{0}-wavelet'.format(j),)

    # img_roi_wave_clahe_df = pd.DataFrame(roi_wave_clahe_info_list)
    # img_roi_wave_clahe_df.to_csv('{0}-wavelet-clahe'.format(j),)
    j += 1
