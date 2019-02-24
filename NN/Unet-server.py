# coding: utf-8
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from albumentations import IAAPerspective, Resize, Compose
from skimage.measure import label
from sklearn.model_selection import train_test_split
from PIL import Image


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


def generate_diff(img1, img2, mask1=None, mask2=None):
    rotated_im, h = rotate_images(img1, img2)
    height, width = img2.shape[:2]

    subtracted = cv2.subtract(img2, rotated_im)
    subtracted_back = cv2.subtract(rotated_im, img2)
    added = cv2.add(subtracted, subtracted_back)
    ret, th1 = cv2.threshold(rotated_im, 0, 255, cv2.THRESH_BINARY)
    added[np.where(th1 < 255)] = 0

    added = cv2.erode(added, (5,5), 7)
    added = cv2.dilate(added,(2,2),iterations = 4)
    added = np.expand_dims(added, axis=2)
    if mask1 is not None:
        rotated_mask = cv2.warpPerspective(mask1, h, (width, height))
        full_mask = (rotated_mask + mask2)/2
        full_mask = np.expand_dims(full_mask, axis=2)
        return added, full_mask
    return added

def augment(aug, image, mask):
    res = Compose(aug)(image=image, mask=mask)
    return res['image'] ,res['mask']


def thresh(intensity):
    return int(intensity < 230)*255


def watermark_with_transparency(image, new_object, input_size):
    width, height = image.size
    mask_width, mask_height = new_object.size

    position = random.randint(0,width - mask_width), random.randint(0,height - mask_height)

    mask = new_object.point(thresh)

    new_image = Image.new('L', (width, height))
    new_mask = Image.new('L', (width, height))

    new_mask.paste(mask, position)
    new_image.paste(image, (0,0))
    new_image.paste(new_object, position, mask=mask)

    new_image, new_mask = augment(
        [IAAPerspective(p=1), Resize(p=1, height=input_size[0], width=input_size[1])],
        np.array(new_image),
        np.array(new_mask)
    )

    return new_image, new_mask

def my_generator(img_dir, mask_dir, image_list, batch_size):

    ids_train_split = range(len(image_list))
    while True:
        random.shuffle(image_list)
        for start in range(0, len(image_list), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]

            for id in ids_train_batch:
                img_path = os.path.join(img_dir, image_list[id])
                mask_path = os.path.join(mask_dir, image_list[id])

                image = Image.open(img_path)
                mask = Image.open(mask_path)

                image = np.expand_dims(image, axis=2)
                mask = np.expand_dims(mask, axis=2)

                x_batch.append(image)
                y_batch.append(mask)


            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.int) / 255

            yield x_batch, y_batch > 0

from sklearn.model_selection import train_test_split
from segmentation_models import Unet, Linknet
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

WEIGHTS='imagenet'
SEED=42
OPTIMIZER='adam'
LOSS='binary_crossentropy'
ES_PATIENCE=10
LR_PATIENCE=5
FACTOR=0.5
EPOCHS=10
BACKBONE_NAME = 'resnet18'


early_stopping = EarlyStopping(monitor='val_my_iou', mode='max', patience=ES_PATIENCE, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou', mode='max', factor=FACTOR, patience=LR_PATIENCE, verbose=1)
model_checkpoint = ModelCheckpoint('../models/Unet_{}.hd5'.format(BACKBONE_NAME), monitor='val_my_iou', mode='max', save_best_only=True, verbose=1)
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

callbacks = [early_stopping, model_checkpoint, reduce_lr, tbCallBack]

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        metric.append(iou)
    return np.mean(metric)


def my_iou(label, pred, thres=0.5):
    return tf.py_func(get_iou_vector, [label, pred > thres], tf.float64)

def mask_pred(img1, img2, original_size):
    mask = generate_diff(img1, img2)
    mask = np.expand_dims(mask, axis=0) / 255
    pred = model.predict(mask)
    mask = np.array(Image.fromarray(mask[0,:,:,0]).resize(original_size))
    pred = np.array(Image.fromarray(pred[0,:,:,0]).resize(original_size))
    return mask, pred

def predict(model, path_1, path_2):
    # Save original size
    test_1 = Image.open(path_1)
    test_2 = Image.open(path_2)
    original_size = test_1.size

    # Convert input to grayscale
    test_1_grey = np.array(test_1.resize((512, 512)).convert('L'))
    test_2_grey = np.array(test_2.resize((512, 512)).convert('L'))

    # Find mask of lost things
    mask, pred = mask_pred(test_1_grey, test_2_grey, original_size)

    return test_1, test_2, mask, pred

def draw(test_1, test_2, mask, pred, porog):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(21, 14))
    ax[0][0].imshow(test_1)
    ax[0][1].imshow(test_2)
    ax[1][0].imshow(mask)
    ax[1][1].imshow(pred > porog)
    plt.show()

def fit():
    img_dir = '../input/for_unet/x'
    mask_dir = '../input/for_unet/y'
    BATCH_SIZE = 16

    train_image, valid_image = train_test_split(
        os.listdir(img_dir),
        test_size=0.2,
        random_state=42,
    )

    TRAIN_STEPS_PER_EPOCH = len(train_image)//BATCH_SIZE + 1
    VALID_STEPS_PER_EPOCH = len(valid_image)//BATCH_SIZE + 1

    train_generator = my_generator(img_dir, mask_dir, train_image, BATCH_SIZE)
    valid_generator = my_generator(img_dir, mask_dir, valid_image, BATCH_SIZE)


def model_init(path_1, path_2):
    model = Unet(BACKBONE_NAME, input_shape=(None, None, 1), classes=1, encoder_weights=None)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[my_iou])
    history = model.fit_generator(train_generator,
                                steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                                validation_data=valid_generator,
                                validation_steps=VALID_STEPS_PER_EPOCH,
                                callbacks=callbacks,
                                epochs=50)

def boxes_from_mask(img, get_boxes=True):
    """
    create boxes from the mask
    :param img: np.array: img for post process
    :param get_boxes: bool: if True the function return boxes False - function return mask
    :return: list of boxes or mask
    """
    kernel = np.ones((3, 3), np.uint8)
    # noise is removed
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    erosion = cv2.erode(opening,kernel,iterations = 1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening,kernel,iterations = 3)
    labels = label(dilation, background=0)
    boxes = []
    for i in np.unique(labels)[1:]:
        arr = np.array(labels == i, dtype=np.uint8)
        im2, contours, hierarchy = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
    result = []
    for box in boxes:
        box = box.reshape((-1))
        box = [box[0], box[1], box[2], box[3], box[6], box[7], box[4], box[5]]
        result.append(box)
    if get_boxes is False:
        return get_mask(img, result)
    return result

def draw_boxes(img, boxes):
    """
    draws boxes on the image
    :param img:  cv2 image
    :param boxes: list: list of boxes,
    each box is a list of 8 int coordinates:
                         [x1, y1,       x2, y2,     x3, y3,    x4, y4]
                         upper left, upper right, lower left, lower right
    :return: None
    """
    print(boxes)
    plt.figure(figsize=(20,30))
    plt.imshow(img, 'Greys_r')
    for box in boxes:
        x_ = [box[0], box[2], box[6], box[4], box[0]]
        y_ = [box[1], box[3], box[7], box[5], box[1]]
        plt.plot(x_, y_, '-', color = 'b')
    plt.show()

if __name__ == '__main__':
    model = load_model('../saved_models/Unet_resnet18.hd5', custom_objects={'my_iou': my_iou})

    test_1, test_2, mask, pred = predict(model, '../input/sleep/1.jpg',
                                         '../input/sleep/2.jpg')
    # draw(test_1, test_2, mask, pred, 0.8)

    boxes = boxes_from_mask(pred)
    draw_boxes(test_1, boxes)