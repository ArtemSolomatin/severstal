#!/usr/bin/env python
# coding: utf-8

# In[1]:


""" experiment: use of binary masks for text detection"""
import sys
sys.path.append("/home/dokholyan/Projects/idog/")
import cv2
import tensorflow as tf
import numpy as np
from numpy.random import choice
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from albumentations import IAAPerspective, Resize, Compose
from tqdm import tqdm_notebook 


from sklearn.utils import shuffle as sk_shuffle
from tensorflow.data import Dataset
from tensorflow.estimator import RunConfig
from glob import glob
from PIL import Image

from idog.detection.unet import make_unet


# In[2]:


from skimage.measure import label
def post_process(img, get_boxes=False):
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

def draw_boxes(img_name, boxes):
    """
    draws boxes on the image
    :param img_name:  path of the image file
    :param boxes: list: list of boxes,
    each box is list of 8 int coordinates [x1, y1, x2, y2, x3,y3, x4, y4] upper left, upper right, lower left, lower right
    :return: None
    """
    img = np.array(Image.open(img_name))
    #img = cv2.imread(img_name)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    for box in boxes:
        x_ = [box[0], box[2], box[6], box[4], box[0]]
        y_ = [box[1], box[3], box[7], box[5], box[1]]
        plt.plot(x_, y_, '-', color = 'g', linewidth = 5)
    plt.show()


# In[3]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

def augment(aug, image, mask):
    res = Compose(aug)(image=image, mask=mask)
    return res['image'] ,res['mask']


def watermark_with_transparency(image, new_object, input_size):
    random_angle = random.randint(-30, 30)
    new_object = imutils.rotate_bound(new_object, random_angle)
    image = Image.fromarray(image)
    new_object = Image.fromarray(new_object)

    width, height = image.size
    mask_width, mask_height = new_object.size
    
    position = random.randint(0,width - mask_width), random.randint(0,height - mask_height)

    mask = np.array(new_object)
    mask_1 = mask < 240
    mask_2 = mask > 5
    mask = np.array(mask_1 & mask_2, dtype='uint8')*255
    #return mask
    mask = Image.fromarray(mask[:,:,0])

    new_image = Image.new('RGB', (width, height), (0,0,0))
    new_mask = Image.new('RGB', (width, height), (0,0,0))
    
    new_mask.paste(mask, position)
    new_image.paste(image, (0,0))
    new_image.paste(new_object, position, mask=mask)
    
    A = np.array(new_image)
    A = A[position[1]:position[1]+mask_width,position[0]:position[0]+mask_height]
    A = cv2.GaussianBlur(A, (9,9),0)
    
    new_image = np.array(new_image)
    new_image[position[1]:position[1]+mask_width,position[0]:position[0]+mask_height]
    
    
    new_image, new_mask = augment(
        [IAAPerspective(p=1), Resize(p=1, height=input_size[0], width=input_size[1])],
        np.array(new_image),
        np.array(new_mask)
    )
    new_mask = Image.fromarray(new_mask).convert('L')
    new_mask = np.array(new_mask)
    new_mask = cv2.dilate(new_mask,(5,5),iterations = 3)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, (4, 4), 5)
    new_mask = cv2.erode(new_mask,(3, 3),iterations = 3)
    return new_image, new_mask


# In[4]:


def generator(image_paths, watermark_paths, image_shape=(512, 512), shuffle=True):
    """
    Generator for Estimtor, create binary mask for it

    :param image_paths: list: list of full path of the binary masks
    :param watermark_paths: list: list of full path to pickle file with boxes
    :param threshold: int: treshhold for binary masks
    :param image_shape: (int, int): the size of image we want
    :param shuffle: bool: shuffle or not data
    :return: tuple(image,mask)
    """
    if shuffle is True:
        image_paths = sk_shuffle(image_paths)

    for image_path in image_paths:

        if type(image_path) == bytes:
            image_path = image_path.decode()

        watermark_path = choice(watermark_paths, 1)[0]
        if type(watermark_path) == bytes:
            watermark_path = watermark_path.decode()
        
        image = np.array(Image.open(image_path))
        
        new_object = np.array(Image.open(watermark_path))
        relate = float(new_object.shape[1]) / float(new_object.shape[0])
        random_H = random.randint(100, 140)
        random_W = int(random_H*relate)
        
        # new_object.resize((random_W, random_H, 3))
        new_object = np.array(Image.open(watermark_path).resize((random_W, random_H)))
        res, mask =  watermark_with_transparency(image, new_object, image_shape)
        #img_2, mask_2 =  watermark_with_transparency(image, new_object, image_shape)
        
        #es, mask = generate_diff(img_1, img_2, mask_1, mask_2)
        res = np.array(res)/255
        mask = mask[:,:,np.newaxis]
        mask = np.array(mask)/255
        mask = np.array(mask>0)
        yield res, mask 
        
        
#         image, mask = watermark_with_transparency(image_path, watermark_path)
#         original_image = Image.open(image_path).convert('L')
        
#         image = np.array(image)
#         mask = np.array(mask)
#         original_image = np.array(original_image)
        
#         mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
#         image = cv2.resize(image, (image_shape[1], image_shape[0]))
#         original_image = cv2.resize(original_image, (image_shape[1], image_shape[0]))

#         mask = mask[:, :, np.newaxis]
#         image = image[:, :, np.newaxis]
#         original_image = original_image[:, :, np.newaxis]
#         result = np.concatenate((original_image,image, image), axis=2)
        
#         yield result.astype(np.float32)/255, mask.astype(np.float32)/255


def input_fn(image_paths, watermark_paths, num_epochs=2, batch_size=5):
    """
    input function for Estimator, uses the generator for this

    :param binary_mask_paths: list: list of full paths of the images
    :param boxes_paths: list: list of full path to pickle file with boxes
    :param threshold: int: treshhold for binary masks
    :param num_epochs: int: the number of epoch
    :param batch_size: int: batch size
    :return: tensorflow dataset for Estimator
    """
    dataset = Dataset.from_generator(generator=generator, output_types=(tf.float32, tf.float32),
                                     output_shapes=(tf.TensorShape([512, 512, 3]), tf.TensorShape([512, 512, 1])),
                                     args=(image_paths, watermark_paths))
    dataset = dataset.repeat(num_epochs).batch(batch_size)

    return dataset


# In[5]:


import tensorflow as tf
import numpy as np

from tensorflow.nn import sigmoid
from idog.detection.unet import get_loss, IOU


def create_summary(data):
    """
    Create summary for the model

    :param data: dict: two required keys - scalar and image, values -
            lists of tuple with display name and elements of tf.graph
    :return:
    """
    for i in data["scalar"]:
        tf.summary.scalar(i[0], i[1])
    for i in data["image"]:
        tf.summary.image(i[0], i[1])


def make_unet_estimator(features, labels, mode, params):
    """
    Creation of tf.estimator.Estimator for U-net

    :param features: tf.Tensor: images
    :param labels: tf.Tensor: masks
    :param mode: tf.estimator.ModeKeys
    :param params: dict: params of the model
    :return: tf.estimator.EstimatorSpec
    """
    orig_images = features
    true_masks = labels
    model = params["model"]
    net = model(orig_images, **params["model_params"])
    predictions = {
        "predicted_soft_masks": sigmoid(net),
        "predicted_masks": tf.round(sigmoid(net))
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions)

    loss = get_loss(net, true_masks, lam=params["IOU_weight"])

    if params["create_summary"]:
        data = {
            "scalar": [("Loss", loss), ("IOU", IOU(net, true_masks))],
            "image": [('Original image', orig_images),
                      ("Original_masks", true_masks), ("Predicted_masks", sigmoid(net))]
        }
        create_summary(data)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(params["learning_rate"], global_step,
                                                   params["lr_decay_steps"], params["lr_decay_rate"], staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        with tf.control_dependencies(update_ops):
            train_op = optim.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"IOU": tf.metrics.mean(IOU(net, labels))}

    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


# In[6]:


watermark_paths = ['/home/dokholyan/Projects/new_experiments/milk_test/milk.jpg']
image_paths = glob('./images/*')
number_for_train = -100
train_paths_images = image_paths[:number_for_train]
test_paths_images = image_paths[number_for_train:]

train_paths_watermark = watermark_paths[:]
test_paths_watermark = watermark_paths[:]

model_params = {"num_blocks": 4,
                "num_filters": 8,
                "batch_normalization": True,
                "training": True}
params = {"model": make_unet,
          "model_params": model_params,
          "IOU_weight": 1,
          "learning_rate": 0.001,
          "lr_decay_steps": 1000,
          "lr_decay_rate": 0.96,
          "create_summary": True}


strategy = tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
config = RunConfig(save_summary_steps=40,
                   train_distribute=strategy,
                   save_checkpoints_steps = 200,
                   keep_checkpoint_max = 60,
                   eval_distribute=strategy,
                   )
tf.logging.set_verbosity(tf.logging.INFO)
segmentation_model = tf.estimator.Estimator(
    model_fn=make_unet_estimator,
    model_dir="/home/dokholyan/Projects/new_experiments/milk_blocks_4_fea_8_IOU_01_norm_Tr/",
    params=params,
    config=config
)

train_params = {'image_paths': train_paths_images,
          'watermark_paths': train_paths_watermark,
          'batch_size': 10,
          'num_epochs': 500
          }
val_params = {'image_paths': test_paths_images,
                'watermark_paths': test_paths_watermark,
                'batch_size': 3,
                'num_epochs': 2
                }

train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(**train_params), max_steps = 75000)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(**val_params), throttle_secs=100, steps=200)
tf.estimator.train_and_evaluate(segmentation_model, train_spec, eval_spec)

# segmentation_model.train(input_fn=lambda: input_fn(**params))


# In[414]:


A = generator(image_paths, watermark_paths)


# In[415]:


a, b =next(A)
plt.imshow(a)


# In[397]:


plt.imshow(a)


# In[368]:


milk_image_path = '/home/dokholyan/Projects/new_experiments/milk_test/milk.jpg'


# In[369]:


image_paths = glob('./images/*')


# In[370]:


random_H = random.randint(119, 120)
random_W = random_H*2
image = np.array(Image.open(image_paths[0]))
new_object = np.array(Image.open(milk_image_path).resize((random_W, random_H)))


# In[371]:


relate = float(new_object.shape[1]) // float(new_object.shape[0])


# In[ ]:





# In[378]:


print (image.shape, new_object.shape)
res = watermark_with_transparency(image, new_object, (512, 512))


# In[373]:


import imutils


# In[374]:


new_object = imutils.rotate_bound(new_object, 45)


# In[375]:


plt.imshow(new_object)


# In[ ]:





# In[376]:


plt.figure(figsize=(20, 10))
plt.imshow(res[1], 'Greys_r')


# In[7]:


def image_generator(image_paths, image_shape=(512, 512), shuffle=True):
    """
    Generator for Estimtor, create binary mask for it

    :param image_paths: list: list of full path of the binary masks
    :param watermark_paths: list: list of full path to pickle file with boxes
    :param threshold: int: treshhold for binary masks
    :param image_shape: (int, int): the size of image we want
    :param shuffle: bool: shuffle or not data
    :return: tuple(image,mask)
    """
    for image_path in image_paths:

        if type(image_path) == bytes:
            image_path = image_path.decode()

        image = cv2.resize(np.array(Image.open(image_path)),(image_shape[1], image_shape[0]))
        image = np.array(image)/255
        yield image

def input_fn_images(image_paths,epoch=1, batch_size=1, image_shape=(512, 512)):
    """
    input function for Estimator
    :param image_paths: list: list of path of png file with images
    :param epoch: int: number of epoch
    :param batch_size: int: batch size
    :param image_shape: (int, int): the size of image we want
    :param padding: bool: use padding or not
    :return: dataset for Estimator
    """
    dataset = Dataset.from_generator(generator=image_generator, output_types=(tf.float32),
                                     output_shapes=(tf.TensorShape([512, 512, 3])),
                                     args=((image_paths, image_shape)))
    dataset = dataset.repeat(epoch).batch(batch_size)

    return dataset


# In[8]:


def predict(estimator, image_paths, post_process, predict_keys='predicted_masks', checkpoint_path=None,):
    """
    create text detector prediction
    :param estimator: tensorflow estimator
    :param image_paths: list: list of paths to images file
    :param post_process: function: function of post process
    :param predict_keys: str: predict_keys for estimator
    :param checkpoint_path: str: checkpoint path for estimator
    :param return_mask: bool: return or not binary mask prediction
    :param padding: bool: use padding or not
    :return: list: list of prediction boxes or (boxes, masks)
    """
    boxes = []
    params = {'image_paths': image_paths}
    pred = estimator.predict(
        input_fn=lambda: input_fn_images(**params),
        predict_keys=predict_keys,
        checkpoint_path=checkpoint_path
    )
    for image_path in tqdm_notebook(image_paths, total=len(image_paths)):
        pred_sample = np.array(next(pred)[predict_keys], dtype=np.uint8)[:, :, 0]
        image = cv2.imread(image_path, 2)
        pred_sample = cv2.resize(pred_sample, (image.shape[1], image.shape[0]))
        boxes.append(post_process(pred_sample, True))
    return boxes


# In[ ]:


params = {'image_paths': image_paths,
          'image_2_paths': image_2_paths}
pred = segmentation_model.predict(
        input_fn=lambda: input_fn_images(**params),
        predict_keys='predicted_soft_masks',
        checkpoint_path='/home/dokholyan/Projects/new_experiments/blocks_4_fea_8_IOU_01_norm_Tr/model.ckpt-5300'
    )


# In[9]:


test_images = glob('/home/dokholyan/Projects/new_experiments/milk_test/*')


# In[10]:


boxes = predict(segmentation_model,test_images , post_process,predict_keys='predicted_soft_masks',
                checkpoint_path = '/home/dokholyan/Projects/new_experiments/milk_blocks_4_fea_8_IOU_01_norm_Tr/model.ckpt-1000')


# In[11]:


i = 1
draw_boxes(test_images[i], boxes[i])


# In[57]:


def fit(pattern_path, model_dir):
    watermark_paths = pattern_path
    image_paths = glob('./images/*')
    number_for_train = -100
    train_paths_images = image_paths[:number_for_train]
    test_paths_images = image_paths[number_for_train:]

    train_paths_watermark = watermark_paths[:]
    test_paths_watermark = watermark_paths[:]

    model_params = {"num_blocks": 4,
                    "num_filters": 8,
                    "batch_normalization": True,
                    "training": True}
    
    params = {"model": make_unet,
              "model_params": model_params,
              "IOU_weight": 1,
              "learning_rate": 0.001,
              "lr_decay_steps": 1000,
              "lr_decay_rate": 0.96,
              "create_summary": True}


    strategy = tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
    config = RunConfig(save_summary_steps=40,
                       train_distribute=strategy,
                       save_checkpoints_steps = 200,
                       keep_checkpoint_max = 60,
                       eval_distribute=strategy,
                       )
    tf.logging.set_verbosity(tf.logging.INFO)
    segmentation_model = tf.estimator.Estimator(
        model_fn=make_unet_estimator,
        model_dir=model_dir,
        params=params,
        config=config
    )

    train_params = {'image_paths': train_paths_images,
              'watermark_paths': train_paths_watermark,
              'batch_size': 10,
              'num_epochs': 2
              }
    val_params = {'image_paths': test_paths_images,
                    'watermark_paths': test_paths_watermark,
                    'batch_size': 3,
                    'num_epochs': 1
                    }

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(**train_params), max_steps = 75000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(**val_params), throttle_secs=100, steps=200)
    tf.estimator.train_and_evaluate(segmentation_model, train_spec, eval_spec)

    segmentation_model.train(input_fn=lambda: input_fn(**params)), 
    


# In[13]:


pattern_path = ['/home/dokholyan/Projects/new_experiments/red_bull_test/red_bull.jpg']
fit(pattern_path, "/home/dokholyan/Projects/new_experiments/red_bull_blocks_4_fea_8_IOU_01_norm_Tr/")


# In[14]:


test_images = glob('/home/dokholyan/Projects/new_experiments/red_bull_test/*')
boxes = predict(segmentation_model,test_images , post_process,predict_keys='predicted_soft_masks',
                checkpoint_path = '/home/dokholyan/Projects/new_experiments/red_bull_blocks_4_fea_8_IOU_01_norm_Tr/model.ckpt-1200')


# In[37]:


draw_boxes(test_images[8], [])


# In[25]:


i = 1
draw_boxes(test_images[i], boxes[i])


# In[ ]:


def save_image_and_boxes(img_name, boxes, savedir):
    """
    draws boxes on the image
    :param img_name:  path of the image file
    :param boxes: list: list of boxes,
    each box is list of 8 int coordinates [x1, y1, x2, y2, x3,y3, x4, y4] upper left, upper right, lower left, lower right
    :return: None
    """
    img = np.array(Image.open(img_name))
    #img = cv2.imread(img_name)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    for box in boxes:
        x_ = [box[0], box[2], box[6], box[4], box[0]]
        y_ = [box[1], box[3], box[7], box[5], box[1]]
        plt.plot(x_, y_, '-', color = 'g', linewidth = 5)
    plt.savefig(savedir)


# In[58]:


def RESULT(pattern_path, test_images, model_dir, savedir='./savedir/', save=False):
    pattern_path = [pattern_path]
    fit(pattern_path, model_dir)
    boxes = predict(segmentation_model,test_images , post_process,predict_keys='predicted_soft_masks',
                checkpoint_path = model_dir)
    if save is True:
        for i, image, box in zip(range(len(test_images)), test_images, boxes):
            if boxes !=[]:
                save_image_and_boxes(image, box, savedir+str(i)+'.jpg')
                #save image
    return boxes


# In[59]:


boxes = RESULT(pattern_path='/home/dokholyan/Projects/new_experiments/mobile_test/modile.jpg', 
      test_images = glob('/home/dokholyan/Projects/new_experiments/mobile_test/*'),
      model_dir = '/home/dokholyan/Projects/new_experiments/mobile_block_4_features_8', 
      save=False)


# In[60]:


test_images = glob('/home/dokholyan/Projects/new_experiments/mobile_test/*')
#boxes = predict(segmentation_model,test_images , post_process,predict_keys='predicted_soft_masks',
                checkpoint_path = '/home/dokholyan/Projects/new_experiments/mobile_block_4_features_8/model.ckpt-440')


# In[69]:


i = 0
draw_boxes(test_images[i], boxes[i])

