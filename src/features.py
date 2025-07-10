import cv2
import numpy as np
from skimage.feature import hog

def compute_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_vec = hog(
        gray, orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_list = []
    for ch, bins in zip([0,1,2], [8,4,4]):
        h = cv2.calcHist([hsv], [ch], None, [bins], [0,256]).flatten()
        h = h / (h.sum() + 1e-6)
        hist_list.append(h)

    return np.hstack([hog_vec, np.concatenate(hist_list)])


# HOG öznitelik çıkarımı
# Kaynak: Dalal, N. & Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection".
# DOI: https://doi.org/10.1109/CVPR.2005.177
# https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

