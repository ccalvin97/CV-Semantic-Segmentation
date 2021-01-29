# Image Segmentation Keras : Implementation of UNet, PSPNet.

## Contribution   
kuancalvin2016@gmail.com

## Models

Following models are supported and tested:

| model_name       | Base Model        | Segmentation Model |
|------------------|-------------------|--------------------|
| resnet50_pspnet  | Resnet-50         | PSPNet             |
| vgg_unet         | VGG 16            | U-Net              |


Example results for the pre-trained models provided :

Input Image            |  Output Segmentation Image
:-------------------------:|:-------------------------:
![](sample_images/1_input.jpg)  |  ![](sample_images/1_output.png)
![](sample_images/3_input.jpg)  |  ![](sample_images/3_output.png)


## Getting Started

### Prerequisites

* Keras 2.0
* opencv for python
* Theano / Tensorflow / CNTK

```shell
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
```

### Installing

Install the module

```shell
pip install keras-segmentation
```

or 

```shell
pip install git+https://github.com/divamgupta/image-segmentation-keras
```

### or

```shell
git clone https://github.com/divamgupta/image-segmentation-keras
cd image-segmentation-keras
python setup.py install
```


## Pre-trained models:
```python
from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12

model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

model = pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# load any of the 3 pretrained models

out = model.predict_segmentation(
    inp="input_image.jpg",
    out_fname="out.png"
)

```


### Preparing the data for training

You need to make two folders

*  Images Folder - For all the training images
* Annotations Folder - For the corresponding ground truth segmentation images

The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

Example code to generate annotation images :

```python
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
```

Only use bmp or png format for the annotation images.

## Download the sample prepared dataset

Download and extract the following:

https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing

You will get a folder named dataset1/


## Using the python module

You can import keras_segmentation in  your python script and use the API

```python
from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=51 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )

```


## Usage via command line
You can also use the tool just using command line

### Visualizing the prepared data

You can also visualize your prepared annotations for verification of the prepared data.


```shell
python -m keras_segmentation verify_dataset \
 --images_path="dataset1/images_prepped_train/" \
 --segs_path="dataset1/annotations_prepped_train/"  \
 --n_classes=50
```

```shell
python -m keras_segmentation visualize_dataset \
 --images_path="dataset1/images_prepped_train/" \
 --segs_path="dataset1/annotations_prepped_train/"  \
 --n_classes=50
```



### Training the Model

To train the model run the following command:

```shell
python -m keras_segmentation train \
 --checkpoints_path="path_to_checkpoints" \
 --train_images="dataset1/images_prepped_train/" \
 --train_annotations="dataset1/annotations_prepped_train/" \
 --val_images="dataset1/images_prepped_test/" \
 --val_annotations="dataset1/annotations_prepped_test/" \
 --n_classes=50 \
 --input_height=320 \
 --input_width=640 \
 --model_name="vgg_unet"
```

Choose model_name from the table above



### Getting the predictions

To get the predictions of a trained model

```shell
python -m keras_segmentation predict \
 --checkpoints_path="path_to_checkpoints" \
 --input_path="dataset1/images_prepped_test/" \
 --output_path="path_to_predictions"

```



### Model Evaluation 

To get the IoU scores 

```shell
python -m keras_segmentation evaluate_model \
 --checkpoints_path="path_to_checkpoints" \
 --images_path="dataset1/images_prepped_test/" \
 --segs_path="dataset1/annotations_prepped_test/"
```



## Fine-tuning from existing segmentation model

The following example shows how to fine-tune a model with 10 classes .

```python
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50

pretrained_model = pspnet_50_ADE_20K()

new_model = pspnet_50( n_classes=51 )

transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

new_model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)


```
