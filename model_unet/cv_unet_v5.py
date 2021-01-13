#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import datetime
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU, ReLU
import cv2
from keras.losses import binary_crossentropy
from PIL import Image

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb    
import skimage.io as io

from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend  import clear_session
from tensorflow.compat.v1.keras.backend  import get_session
import tensorflow as tf

'''
# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

   # print(gc.collect()) # if it's done something you should see a number being outputted
    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
'''

class network():
    def __init__(self, BATCH_SIZE, normalisation, X_CHANNEL, Y_CHANNEL, PIXEL, lr, EPOCH, smooth):
        # self.pathX = pathX
       # self.pathY = pathY
        self.BATCH_SIZE = BATCH_SIZE
        self.normalisation = normalisation
        self.X_CHANNEL = X_CHANNEL
        self.Y_CHANNEL = Y_CHANNEL
        self.PIXEL = PIXEL
        self.EPOCH = EPOCH
        self.smooth = smooth
        self.lr = lr

    def generator(self, pathX, pathY, NUM):
        while 1:
            X_train_files = os.listdir(pathX)
            X_train_files.sort()
            Y_train_files = os.listdir(pathY)
            Y_train_files.sort()
            a = (np.arange(1, NUM))
            # print(a)
            # cnt = 0
            X = []
            Y = []
            for i in range(self.BATCH_SIZE):
                index = np.random.choice(a)
                img = cv2.imread(pathX + '/' +X_train_files[index], 1)
                if self.normalisation:
                    img = img / 255  # normalization
                img = np.array(img).reshape(self.PIXEL, self.PIXEL, self.X_CHANNEL)
                X.append(img)
                img1 = cv2.imread(pathY + '/' + Y_train_files[index], 2)
                if self.normalisation:
                    img1 = img1 / 255  # normalization
                img1 = np.array(img1).reshape(self.PIXEL, self.PIXEL, self.Y_CHANNEL)
                Y.append(img1)
                #cnt += 1
            X = np.array(X)
            Y = np.array(Y)
            yield X, Y


    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        y_true_f = K.cast(y_true_f, dtype='float32')
        intersection = K.sum(y_true_f * y_pred_f)
        res=(2. * intersection + self.smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + self.smooth)
        return res

    def mean_iou(self, y_true, y_pred):
        num_labels = K.int_shape(y_pred)[-1] - 1
        mean_iou = K.variable(0)
        for label in range(num_labels):
            mean_iou = mean_iou + iou(y_true, y_pred, label)
        return mean_iou / num_labels

    def dice_coef_loss(self, y_true, y_pred):
        return 1-self.dice_coef(y_true, y_pred)
    
    
    def true_positive_rate(self, y_true, y_pred):
        return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
    
    def binary_accuracy(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

    def get_unet(self, pretrained_weights = None):
        inputs = Input((self.PIXEL, self.PIXEL, 3)) # 1200*1200
        conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)  # 600*600
     
        conv2 = BatchNormalization(momentum=0.99)(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization(momentum=0.99)(conv2)
        conv2 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = Dropout(0.02)(conv2)
        pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  # 300*300
     
        conv3 = BatchNormalization(momentum=0.99)(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization(momentum=0.99)(conv3)
        conv3 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = Dropout(0.02)(conv3)
        pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  # 150*150
     
        conv4 = BatchNormalization(momentum=0.99)(pool3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization(momentum=0.99)(conv4)
        conv4 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = Dropout(0.02)(conv4)
        pool4 = AveragePooling2D(pool_size=(2, 2))(conv4) #75*75
     
        conv5 = BatchNormalization(momentum=0.99)(pool4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization(momentum=0.99)(conv5)
        conv5 = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Dropout(0.02)(conv5)
        pool4 = AveragePooling2D(pool_size=(3, 3))(conv4) #25*25
     
        # conv5 = Conv2D(35, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # drop4 = Dropout(0.02)(conv5)
        pool4 = AveragePooling2D(pool_size=(2, 2))(pool3)  # pool3=150*150,pool4=75*75
        pool5 = AveragePooling2D(pool_size=(3, 3))(pool4)  # pool5=25*25
     
        conv6 = BatchNormalization(momentum=0.99)(pool5)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
     
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = (UpSampling2D(size=(3, 3))(conv7))  # 75*75
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
        merge7 = concatenate([pool4, conv7], axis=3)
     
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        up8 = (UpSampling2D(size=(2, 2))(conv8))  # 4
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
        merge8 = concatenate([pool3, conv8], axis=3)
     
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        up9 = (UpSampling2D(size=(2, 2))(conv9))  # 8
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
        merge9 = concatenate([pool2, conv9], axis=3)
     
        conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        up10 = (UpSampling2D(size=(2, 2))(conv10))  # 16
        conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up10)
     
        conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
        up11 = (UpSampling2D(size=(2, 2))(conv11))  # 32
        conv11 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up11)
     
        # conv12 = Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
        conv12 = Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
        conv12 = Conv2D(1, 1, 1, activation='sigmoid')(conv12)
        model = Model(inputs, conv12)
        if pretrained_weights:
            model.load_weights(pretrained_weights)

        print(model.summary())
     
        model.compile(optimizer=Adam(lr=self.lr, decay=1e-6), loss=self.dice_coef_loss, metrics=[self.dice_coef, self.binary_accuracy, self.mean_iou])
        return model
    
    def save_model(self, history, model):
        #save your training model
        cur_dir=os.getcwd()
        paprameter_file = 'model_parameter'
        if not os.path.isdir(cur_dir+'/'+paprameter_file):
            os.mkdir(cur_dir+'/'+paprameter_file)
        model.save(cur_dir+'/'+paprameter_file+'/'+'test1.h5')
    
        #save your loss data
        loss = np.array((history.history['loss']))
        np.save(cur_dir+'/'+paprameter_file+'/'+'test1.npy', loss)



    def visualisation(path_pred, index):
        pred_x_files = os.listdir(path_pred)
        a = (np.arange(1, self.X_NUM))
        X = []
        X_visialisation = []
        img = cv2.imread(path_pred + '/' + pred_x_files[index], 1)
        if self.normalisation:
            img = img / 255  # normalization
        img = np.array(img).reshape(self.PIXEL, self.PIXEL, self.X_CHANNEL)
        X.append(img)
    
        X = np.array(X)
        X_visialisation.append(img)
        X_visialisation = np.array(X_visialisation)
        predd = model.predict(X)
        predd_visualisation = model.predict(X_visialisation)
          
    
       # fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 24))
        predd_visualisation=predd_visualisation.squeeze(axis=0)
        predd_visualisation=predd_visualisation.squeeze(axis=2)
       # ax1.imshow(img)
       # ax2.imshow(predd_visualisation,cmap='binary')
       # ax2.set_title('Prediction')    
        cur_dir=os.getcwd()
        pred_pic_file = 'pred_pic'
        if not os.path.isdir(cur_dir+'/'+pred_pic_file):
            os.mkdir(cur_dir+'/'+pred_pic_file)


        im = Image.fromarray(predd_visualisation)   
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(cur_dir+'/'+pred_pic_file+'/'+pred_x_files[index]+"_pred.jpeg") 
       

        im = Image.fromarray(img)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(cur_dir+'/'+pred_pic_file+'/'+pred_x_files[index]+"_label.jpeg")
        


    def test_generator(self, path_pred):
        images = os.listdir(path_pred)
        images.sort()
        total = len(images)
        i = 0
        print('-'*30)
        print('Creating test images')
        print('-'*30)
        for image_name in images:
            img = io.imread(path_pred + '/' + image_name)
            img = img[:, :, :3]
            img = img.reshape(self.PIXEL,self.PIXEL, 3)
            img = np.reshape(img,(1,)+img.shape)
            yield img
        print('test_generator done')



def save_results(save_path, npyfile, names):
    """ Save Results
    Function that takes predictions from U-Net model
    and saves them to specified folder.
    """

    for i,item in enumerate(npyfile):
        img = normalize_mask(item)
        img = (img * 255).astype('uint8')
        io.imsave(os.path.join(save_path,"pred_"+names[i]),img)

def normalize_mask(mask):
    """ Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask




def plot(history):

    print('Start Plot')
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    dice_coef = history.history['dice_coef']
    val_dice_coef = history.history["val_dice_coef"]
    loss_bacc = history.history["binary_accuracy"]
    val_bacc = history.history["val_binary_accuracy"]

    epochs = range(1, len(loss) + 1)
    plt.style.use('ggplot')

    plt.figure()
    plt.plot(epochs, loss, color="red", label="training dice_coef loss")
    plt.plot(epochs, val_loss, color="blue", label="validation dice_coef loss")
    plt.title("Training and Validation dice coef loss")
    plt.legend()
    plt.show()
    plt.savefig("dice_coef_loss_plot.png")

    plt.figure()
    plt.plot(epochs, dice_coef, color="red", label="training dice coef")
    plt.plot(epochs, val_dice_coef, color="blue", label="validation dice coef")
    plt.title("Training and Validation dice_coef")
    plt.legend()
    plt.show()
    plt.savefig("dice coef_plot.png")

    plt.figure()
    plt.plot(epochs, loss_bacc, color="red", label="training binary acc")
    plt.plot(epochs, val_bacc, color="blue", label="validation binary acc")
    plt.title("Training and Validation binary acc")
    plt.legend()
    plt.show()
    plt.savefig("binary acc_plot.png")
    





'''
def create_test_data(path, PIXEL):
  images = os.listdir(path)
  total = len(images)
  imgs = np.ndarray((total, PIXEL, PIXEL, 3), dtype=np.uint8) # input dim: 12000*1200*3
  imgs_id = np.ndarray((total, ), dtype=np.int32)
  i = 0
  print('-'*30)
  print('Creating test images...')
  print('-'*30)
  for image_name in images:
    img_id = int(image_name.split('.')[0])
    img = cv2.imread(path + '/' + image_name, 1)
    img = np.array([img])
    imgs[i] = img
    imgs_id[i] = img_id
    if i % 100 == 0:
        print('Done: {0}/{1} images'.format(i, total))
    i += 1
  print('Loading done.')
  np.save('imgs_test.npy', imgs)
  np.save('imgs_id_test.npy', imgs_id)
  print('Saved to .npy files done.')



def load_test_data():
  imgs_test = np.load('imgs_test.npy')
  imgs_id = np.load('imgs_id_test.npy')
  return imgs_test, imgs_id
'''
'''
create_test_data(path_pred, PIXEL)
imgs_test, imgs_id = load_test_data()
'''



