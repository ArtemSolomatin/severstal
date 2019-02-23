import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

IM_DIR = './our_dataset/'
template = IM_DIR + 'phone.jpg'
COLOR = -1 # 0 for gray, -1 for color
MATCH_DIR = IM_DIR + 'matched/'

# Template image
template = cv2.imread(template, COLOR)

# Every image but templates
for image_name in [f for f in listdir(IM_DIR) if isfile(join(IM_DIR, f))]:
    if image_name[:-4] in ['cookie', 'phone', 'thing']:
        continue

     # Open and show image
    image = cv2.imread(IM_DIR + image_name, COLOR)

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x:x.distance)
    # Draw first 10 matches.
    match = image.copy()
    match = cv2.drawMatches(template, kp1, image, kp2, matches[:5], match, flags=2)
    cv2.imwrite(MATCH_DIR + image_name[:-4] + '_match.jpg', match)
