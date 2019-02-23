import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

# Directory with base images
COLOR = 0 # 0 for gray, -1 for color
IM_DIR = '/home/ivan23kor/Hackathon/templ/Severstal/'
DET_DIR = IM_DIR + 'detected/'
template = 'envelope.png'

# Template image
template = cv2.imread(template, COLOR)

# Comparison methods
methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]


# Every image
for image_name in [f for f in listdir(IM_DIR) if isfile(join(IM_DIR, f))]:
     # Open and show image
    image = cv2.imread(IM_DIR + image_name, COLOR)

    # Find flower
    for i, method in enumerate(methods[2:], 2):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(template,None)
        kp2, des2 = orb.detectAndCompute(image,None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        print(matches)
        # Draw first 10 matches.
        img3 = image.copy()
        img3 = cv2.drawMatches(template, kp1, image, kp2, matches[:10], img3, flags=2)
        plt.imshow(img3),plt.show()

        # # Show result
        # detected_name = DET_DIR + image_name[:-4] + '_' + str(i) + image_name[-4:]
        # cv2.imwrite(detected_name, detected)