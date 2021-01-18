#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb



def plot(history):

    print('Start Plot')
    precision = history.history["precision"]
    val_precision = history.history["val_precision"]
    recall = history.history["recall"]
    val_recall = history.history["val_recall"]
    MeanIoU = history.history["mean_io_u"]
    val_MeanIoU = history.history["val_mean_io_u"]
    dice_coef = history.history['dice_coef']
    val_dice_coef = history.history["val_dice_coef"]
    loss_bacc = history.history["accuracy"]
    val_bacc = history.history["val_accuracy"]

    epochs = range(1, len(MeanIoU) + 1)
    plt.style.use('ggplot')

    plt.figure()
    plt.plot(epochs, precision, color="yellow", label="precision")
    plt.plot(epochs, val_precision, color="orange", label="val precision")
    plt.plot(epochs, precision, color="black", label="recall")
    plt.plot(epochs, val_precision, color="gray", label="val recall")
    plt.plot(epochs, MeanIoU, color="red", label="train MeanIoU")
    plt.plot(epochs, val_MeanIoU, color="darkred", label="val MeanIoU")
    plt.plot(epochs, dice_coef, color="blue", label="train dice coef")
    plt.plot(epochs, val_dice_coef, color="skyblue", label="val dice coef")
    plt.plot(epochs, loss_bacc, color="green", label="train binary acc")
    plt.plot(epochs, val_bacc, color="lightgreen", label="val binary acc")
    plt.title("Training and validation")
    plt.legend()
    plt.show()
    plt.savefig("training_plot.png")

