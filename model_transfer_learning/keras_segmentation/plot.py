#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb



def plot(history):

    print('Start Plot')
    pdb.set_trace()
    tp_rate = history.history["tp_rate"]
    val_tp_rate = history.history["val_tp_rate"]
    tn_rate = history.history["tn_rate"]
    val_tn_rate = history.history["val_tn_rate"]
    MeanIoU = history.history["mean_io_u"]
    val_MeanIoU = history.history["val_mean_io_u"]
    dice_coef = history.history['dice_coef']
    val_dice_coef = history.history["val_dice_coef"]
    loss_bacc = history.history["accuracy"]
    val_bacc = history.history["val_accuracy"]

    epochs = range(1, len(MeanIoU) + 1)
    plt.style.use('ggplot')

    plt.figure()
    plt.plot(epochs, tp_rate, color="yellow", label="tp_rate")
    plt.plot(epochs, val_tp_rate, color="orange", label="val tp_rate")
    plt.plot(epochs, tn_rate, color="black", label="tn_rate")
    plt.plot(epochs, val_tn_rate, color="gray", label="val tn_rate")
    plt.plot(epochs, MeanIoU, color="red", label="train MeanIoU")
    plt.plot(epochs, val_MeanIoU, color="darkred", label="val MeanIoU")
    plt.plot(epochs, dice_coef, color="blue", label="train dice coef")
    plt.plot(epochs, val_dice_coef, color="skyblue", label="val dice coef")
    plt.plot(epochs, loss_bacc, color="green", label="train binary acc")
    plt.plot(epochs, val_bacc, color="lightgreen", label="val binary acc")
    plt.title("Training and validation")
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig("training_plot.png")

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.figure()
    plt.plot(epochs, loss, color="green", label="loss")
    plt.plot(epochs, val_loss, color="lightgreen", label="val loss")
    plt.title("Training and validation")
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig("training_plot.png")

