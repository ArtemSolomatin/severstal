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
 
  # best 10%
  numGoodMatches = int(len(matches) * 0.1)
  matches = matches[:numGoodMatches]
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  return points1, points2


def rotate_images(image1, image2):
  # Get fixed points
  points1, points2 = orb(image1, image2)

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = image2.shape
  rotated_im = cv2.warpPerspective(image1, h, (width, height))
   
  return rotated_im, h


if __name__ == '__main__':
  img1 = cv2.imread('1.jpg')
  img2 = cv2.imread('2.jpg')

  rotated_im, h = rotate_images(img1, img2)

  subtracted = cv2.subtract(img2, rotated_im)

  plt.imshow(subtracted)
  plt.show()
