#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb



def plot(history):
    pdb.set_trace()
    print('Start Plot')
<<<<<<< HEAD
    tn_rate = history.history["tn_rate"]
    val_tn_rate = history.history["val_tn_rate"]
=======
    pdb.set_trace()
    tp_rate = history.history["tp_rate"]
    val_tp_rate = history.history["val_tp_rate"]
    tn_rate = history.history["tn_rate"]
    val_tn_rate = history.history["val_tn_rate"]
    MeanIoU = history.history["mean_io_u"]
    val_MeanIoU = history.history["val_mean_io_u"]
>>>>>>> b3e5f5edc11d62e581b8bc2960704e89fb1c1cf1
    dice_coef = history.history['dice_coef']
    val_dice_coef = history.history["val_dice_coef"]
    loss_bacc = history.history["accuracy"]
    val_bacc = history.history["val_accuracy"]
    tp = history.history["tp_rate"]
    val_tp= history.history["val_tp_rate"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(val_bacc) + 1)
    plt.style.use('ggplot')

    plt.figure()
<<<<<<< HEAD
    plt.plot(epochs, val_tn_rate, color="black", label="val tn_rate", alpha=0.7)
    plt.plot(epochs, val_dice_coef, color="blue", label="val dice coef", alpha=0.7)
    plt.plot(epochs, val_bacc, color="green", label="val binary acc", alpha=0.7)
    plt.plot(epochs, val_tp, color="yellow", label="val tp_rate", alpha=0.7)
    plt.plot(epochs, val_loss, color="red", label="val loss")
    plt.title("Validation")
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig("val_loss.png")

    plt.figure()
    plt.plot(epochs, tn_rate, color="black", label="tn_rate", alpha=0.7)
    plt.plot(epochs, dice_coef, color="blue", label="train dice coef", alpha=0.7)
    plt.plot(epochs, loss_bacc, color="green", label="train binary acc", alpha=0.7)
    plt.plot(epochs, tp, color="yellow", label="tp_rate", alpha=0.7)
    plt.plot(epochs, loss, color="red", label="loss")
    plt.title("Training")
=======
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
>>>>>>> b3e5f5edc11d62e581b8bc2960704e89fb1c1cf1
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig("training_loss.png")

