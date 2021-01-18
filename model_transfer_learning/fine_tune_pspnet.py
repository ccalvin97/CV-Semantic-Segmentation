#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-


from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import  resnet_pspnet_VOC12_v0_1
from keras_segmentation.models.pspnet import resnet50_pspnet
import pdb

#### local path
pathX='/home/GDDC-CV1/Desktop/data_1024/train_x_png/'
pathY='/home/GDDC-CV1/Desktop/data_1024/train_y_png/'
epoch=5


#### physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: f114:00:00.0, compute capability: 7.0) ####
pretrained_model = resnet_pspnet_VOC12_v0_1()

new_model = resnet50_pspnet(n_classes=2)
transfer_weights( pretrained_model , new_model  ) 
new_model.train(
    train_images =  pathX,
    train_annotations = pathY,
    checkpoints_path = "/home/GDDC-CV1/Desktop/CV-Semantic-Segmentation/model_transfer_learning/checkpoint/pspnet_weight" , epochs=epoch
)


input_image = "/home/GDDC-CV1/Desktop/data_1024/pred_x_png/0043000000100_.png"
out = new_model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/pred_out/0043000000100_.png")



input_image = "/home/GDDC-CV1/Desktop/data_1024/pred_x_png/0043000022200_.png"
out = new_model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/pred_out/0043000022200_.png")



input_image = "/home/GDDC-CV1/Desktop/data_1024/pred_x_png/1401070129_.png"
out = new_model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/pred_out/1401070129_.png")


input_image = "/home/GDDC-CV1/Desktop/data_1024/val_x_png/austin16_20_.png"
out = new_model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/pred_out/austin16_20_.png")

input_image = "/home/GDDC-CV1/Desktop/data_1024/val_x_png/test_215_.png"
out = new_model.predict_segmentation(
    inp=input_image,
    out_fname="/home/GDDC-CV1/Desktop/pred_out/test_215_.png")












