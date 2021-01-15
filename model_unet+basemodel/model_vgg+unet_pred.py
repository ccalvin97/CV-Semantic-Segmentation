# coding: utf-8
# -*- coding: utf-8 -*-
import os
################ CPU only ##############
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
########################################
from keras_segmentation.models.unet import vgg_unet
epochs = 5
n_classes = 2
model = vgg_unet(n_classes=n_classes ,  input_height=512, input_width=512)
'''
model.train( train_images =  "/home/GDDC-CV1/Desktop/data_512/train_x_png/",train_annotations = "/home/GDDC-CV1/Desktop/data_512/train_y_png/",
checkpoints_path = "vgg_unet" , epochs=epochs)
'''

input_image = "/home/GDDC-CV1/Desktop/data_1024/pred_x_png/0043000000100_.png"
out = model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/model_unet+basemodel/pred_out/0043000000100_.png")



input_image = "/home/GDDC-CV1/Desktop/data_1024/pred_x_png/0043000022200_.png"
out = model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/model_unet+basemodel/pred_out/0043000022200_.png")



input_image = "/home/GDDC-CV1/Desktop/data_1024/pred_x_png/1401070129_.png"
out = model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/model_unet+basemodel/pred_out/1401070129_.png")


input_image = "/home/GDDC-CV1/Desktop/data_1024/val_x_png/austin16_20_.png"
out = model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/model_unet+basemodel/pred_out/austin16_20_.png")

input_image = "/home/GDDC-CV1/Desktop/data_1024/val_x_png/test_215_.png"
out = model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/model_unet+basemodel/pred_out/test_215_.png")






