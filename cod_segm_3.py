import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom  # working with dicom
import pywt  # working with wavelets
from scipy import ndimage
from skimage import (feature, filters, measure, morphology, segmentation)

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

def f_segmentation(roi, index=None, title=None):
    img = np.copy(roi)
    img = img/4095

    im_canny = feature.canny(img, sigma=0.1)

    kernel = np.array([[1, 0, 1, 0, 1],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [1, 0, 1, 0, 1]])

    morf_1 = ndimage.convolve(im_canny, kernel)

    se = np.array([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]])

    morf_2 = morphology.dilation(morf_1, footprint=se)
    morf_2 = morphology.reconstruction(seed=morf_2, mask=im_canny, method='erosion', footprint=se)

    skiz = morphology.skeletonize(morf_2)
    m_labels = measure.label(skiz, connectivity=1)

    gmag = filters.sobel(img)
    r_watershed = segmentation.watershed(gmag, markers=m_labels, watershed_line=True)

    fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(20,15))
    ax[0].imshow(img*4095, cmap='gray', vmin=0, vmax=4095)
    ax[0].axis('off')
    ax[0].set_title('original')
    ax[1].imshow(segmentation.mark_boundaries(img, m_labels))
    ax[1].axis('off')
    ax[1].set_title('markers')
    ax[2].imshow(segmentation.mark_boundaries(img, r_watershed))
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
    result_1 = f_segmentation(roi, index=i, title='original')
    i += 1

    #wavelet
    im_wave = wavelet_filter(roi, 'coif5')
    result_2 = f_segmentation(im_wave, index=i, title='wavelet')
    i += 1
