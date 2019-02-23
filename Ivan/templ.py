import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

# Directory with base images
COLOR = -1 # 0 for gray, -1 for color
IM_DIR = './our_dataset/'
DET_DIR = join(IM_DIR, 'detected/')
template = join(IM_DIR, 'phone.jpg')

# Template image
template = cv2.imread(template, COLOR)

# Comparison methods
methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]

# Every image
for image_name in [f for f in listdir(IM_DIR) if isfile(join(IM_DIR, f))]:
    if image_name[:-4] in ['cookie', 'phone', 'thing']:
        continue
     # Open and show image
    image = cv2.imread(join(IM_DIR, image_name), COLOR)

    # Find flower
    for i, method in enumerate(methods[2:], 2):
        # Apply filter
        result = cv2.matchTemplate(image, template, method)

        # Return best coincidences
        threshold = 0.2
        loc = np.where(result >= threshold)

        # Size retrieval depending on color
        if COLOR == 0:
            w, h = template.shape
        else:
            w, h = template.shape[:-1]

        # Detected image
        detected = image.copy()
        for pt in zip(*loc[::-1]):
            cv2.rectangle(detected, pt, (pt[0] + h, pt[1] + w), (0, 255, 0), 1)

        # Show result
        detected_name = DET_DIR + image_name[:-4] + '_' + str(i) + image_name[-4:]
        cv2.imwrite(detected_name, detected)
