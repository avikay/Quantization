# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:58:48 2020

@author: Avinash
"""

import numpy as np
import cv2

img = cv2.imread('ronaldo.jpg')

samples = img.reshape((-1,3))
samples = np.float32(samples)

k_parameters = 4

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER, 10, 1.0)

ret, labels, center = cv2.kmeans(samples, k_parameters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
out_img = center[labels.flatten()].reshape(img.shape)

cv2.imshow("Result", out_img)

cv2.waitKey(0)
cv2.destroyAllWindows()