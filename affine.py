# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import numpy as np


def orb(image1, image2, orb_size=500):
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(orb_size)
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
     
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
     
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
 
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return points1, points2


def rotate_images(image1, image2):
    """
    Rotate images according to the fixed points
    Returns:
        rotated_im - image1 rotated to fit image2
        h - transform matrix
    """
    # Get fixed points
    points1, points2 = orb(image1, image2)

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
    # Use homography
    height, width = image2.shape[:2] # channels are useless
    rotated_im = cv2.warpPerspective(image1, h, (width, height))

    return rotated_im, h

def smart_subtract(image1, image2):
    subtracted = cv2.subtract(image2, image1)
    return subtracted

def show_img(image, name='Image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img1 = cv2.imread('1.jpg')
    img2 = cv2.imread('2.jpg')
    rotated_im, h = rotate_images(img1, img2)
    subtracted = cv2.subtract(img2, rotated_im)
    subtracted = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(subtracted, 50, 255, cv2.THRESH_TOZERO)
    subtracted_back = cv2.subtract(rotated_im, img2)
    subtracted_back = cv2.cvtColor(subtracted_back, cv2.COLOR_BGR2GRAY)
    ret_back, thresh_back = cv2.threshold(subtracted_back, 50, 255, cv2.THRESH_TOZERO)
    added = cv2.add(thresh, thresh_back)
    show_img(added)
