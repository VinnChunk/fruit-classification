# -*- coding: utf-8 -*-
"""preprocess.py"""

import cv2
import numpy as np

from config import IMG_SIZE_PREPROCESS


def preprocess_image(image_path):
    """
    Melakukan preprocessing pada satu gambar.
    Output berupa numpy array float32 dengan nilai 0-1.
    """

    # Membaca gambar
    img = cv2.imread(image_path)

    if img is None:
        return None

    # Resize
    img = cv2.resize(
        img,
        (IMG_SIZE_PREPROCESS, IMG_SIZE_PREPROCESS)
    )

    # Gaussian Blur
    img = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    # Konversi ke HSV
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    # Histogram Equalization pada channel V
    hsv[:, :, 2] = cv2.equalizeHist(
        hsv[:, :, 2]
    )

    # Tetap menggunakan HSV sebagai output preprocessing
    img = hsv

    # Normalisasi ke rentang 0-1
    img = img.astype(np.float32) / 255.0

    return img