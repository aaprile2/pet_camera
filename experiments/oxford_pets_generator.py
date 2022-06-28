''' CUSTOM DATA GENERATOR CLASS '''
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras 
from tensorflow.keras.layers import add, Conv2D, BatchNormalization, Activation, SeparableConv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D 
from tensorflow.keras import Model
import numpy as np


# Prepare Sequence class to load & vectorize batches of data
class OxfordPets(keras.utils.Sequence):
  # Initialization
  def __init__(self, batch_size, img_size, img_arr, mask_arr):
    self.batch_size = batch_size
    self.img_size = img_size
    self.img_arr = img_arr
    self.mask_arr = mask_arr

  # Gets number of iterations per epoch
  def __len__(self):
    return len(self.img_arr) // self.batch_size

  def __getitem__(self, ind):
    # Returns tuple (input, target) corresponding to batch number index
    i = ind * self.batch_size
    batch_imgs = self.img_arr[i : i + self.batch_size]
    batch_masks = self.mask_arr[i : i + self.batch_size]

    # Initialize 
    x = np.zeros((self.batch_size, ) + self.img_size + (3, ), dtype = 'float32')

    # Load and process input images
    for i in range(len(batch_imgs)):
      x[i] = batch_imgs[i]

    # Initialize
    y = np.zeros((self.batch_size, ) + self.img_size + (1, ), dtype = 'uint8')

    # Load and process masks
    for i in range(len(batch_masks)):
      y[i] = np.expand_dims(batch_masks[i], 2)

    # Map values back to [0, 1, 2]
    y -= 1
    
    return x, y