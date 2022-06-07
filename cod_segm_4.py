import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom  # working with dicom
import pywt  # working with wavelets
from skimage import (filters, measure, morphology, segmentation)

def f_segmentation(roi, wave_name, index=None, title=None):
    roi_copy = np.copy(roi)
    roi_copy = roi_copy/4095

    k = 3
    cA, (cH, cV, cD) = pywt.wavedec2(roi_copy, wavelet=wave_name, level=1)

    img = np.copy(cA)

    se = np.ones((3,3), np.uint16)
    imD = morphology.dilation(img, footprint=se)
    imR = morphology.reconstruction(seed=imD, mask=img, method='erosion', footprint=se)

    seed = np.copy(imR)
    seed[1:-1, 1:-1] = imR.min() #set its border to be the pixel values in the original image
    mask = imR

    dilated = morphology.reconstruction(seed, mask, method='dilation')
    im_sub = (imR - dilated)

    thresh = filters.threshold_otsu(im_sub)
    intM = im_sub >= thresh #imagem binaria com os marcadores

    #selecting markers
    #checking local_maxima information
    m_label, m_num = measure.label(intM, return_num=True, connectivity=1) #con. 1 -> 4 neighborhood
    m_props = measure.regionprops(label_image=m_label, intensity_image=img)
    m_props_table = measure.regionprops_table(label_image=m_label, intensity_image=img,
                                            properties=('intensity_min',
                                                        'intensity_max',
                                                        'intensity_mean',
                                                        'area',
                                                        'area_convex',
                                                        'area_bbox',
                                                        'extent',
                                                        'solidity'))

    m_df = pd.DataFrame(m_props_table)

    valid_markers = set()
    for marker in m_props:
        m_area = marker.area <= 100
        if m_area:
            valid_markers.add(marker.label)

    markers = np.in1d(m_label, list(valid_markers)).reshape(m_label.shape)
    ms_label = measure.label(markers, connectivity=1)

    ms_props_table = measure.regionprops_table(label_image=ms_label, intensity_image=img,
                                        properties=('intensity_min',
                                                    'intensity_max',
                                                    'intensity_mean',
                                                    'area',
                                                    'area_convex',
                                                    'area_bbox',
                                                    'extent',
                                                    'solidity'))
    
    ms_df = pd.DataFrame(ms_props_table)

    gmag = filters.sobel(img)
    r_watershed = segmentation.watershed(gmag, markers=ms_label, watershed_line=True)
    w_props = measure.regionprops(label_image=r_watershed, intensity_image=img)
    w_props_table = measure.regionprops_table(label_image=r_watershed, intensity_image=img,
                                            properties=('intensity_min',
                                                        'intensity_max',
                                                        'intensity_mean',
                                                        'area',
                                                        'area_convex',
                                                        'area_bbox',
                                                        'extent',
                                                        'solidity'))

    w_df = pd.DataFrame(w_props_table)

    valid_regions = set()
    for region in w_props:
        w_area = region.area <= 100
        if w_area:
            valid_regions.add(region.label)

    regions = np.in1d(r_watershed, list(valid_regions)).reshape(r_watershed.shape)
    r_label = measure.label(regions, connectivity=1)

    fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,15))
    ax[0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    ax[0].axis('off')
    ax[0].set_title('original')
    ax[1].imshow(segmentation.mark_boundaries(img, ms_label))
    ax[1].axis('off')
    ax[1].set_title('markers')
    ax[2].imshow(segmentation.mark_boundaries(img, r_label))
    ax[2].axis('off')
    ax[2].set_title('watershed')
    plt.tight_layout()
    plt.suptitle(title)
    #plt.show()
    plt.savefig('{0}.png'.format(index))

    return r_watershed

img_dir = 'C:\\Users\\br_go\\Desktop\\image-code\\images'
data_path = os.path.join(img_dir, '*dcm')
files = glob.glob(data_path)
data = [] #list of original images

for file in files:
    dicom_file = pydicom.dcmread(file)
    img_array = dicom_file.pixel_array
    data.append(img_array) 

roi_0 = data[0][490:640, 1050:1200]
roi_1 = data[1][2000:2150, 905:1055]
roi_2 = data[2][1325:1475, 330:480] 
roi_3 = data[3][2660:2810, 800:950]
roi_4 = data[4][830:980, 760:910]
roi_5 = data[5][1415:1565, 265:415]

roi_data = [roi_0, roi_1, roi_2, roi_3, roi_4, roi_5]

i = 0
for roi in roi_data:
    #original
    result_1 = f_segmentation(roi, wave_name='coif5', index=i, title='original')
    i += 1