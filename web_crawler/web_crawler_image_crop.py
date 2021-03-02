#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
import re
import pdb
from PIL import Image
from PIL import ImageFile 
from os import mkdir
from os.path import isdir
from skimage import io
import argparse

def main():
    # Get arguments
    parser = argparse.ArgumentParser(description='welcome to the web crawling programme for splitting image to 1024*1024')
    parser.add_argument('-path', '--path', type=str,
        help='path of input data dir  ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel')
    parser.add_argument('-path_out', '--path_out', type=str,
        help='data out abs address for picture ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')   
    parser.add_argument('-file_type', '--file_type', type=str,
            help='file type for norm expression, ex: png, jpeg ')

    args = parser.parse_args()
    

    case=2
    colour_mode = -1
    # 1-> colour mode, 
    # 0-> gray mode, 
    #-1-> alpha mode

    address_list=[]
    pattern=re.compile(r'.*.'+args.file_type)
    for home, dirs, files in os.walk(args.path):
        for filename in files:
            if pattern.findall(filename):
                address_list.append(os.path.join(home, filename))

    for image in address_list:
        a, b = os.path.splitext(image)
        img_name=image.split('\\')[-1].split('.')[0]+'_'+image.split('\\')[-1].split('.')[1]
        if case ==1: 
            img = io.imread(image)
            img = np.array(img) 
            # This line is for picture has [F, F, T ,,,,,]
            #img = 255 * np.array(img).astype('uint8')
        elif case == 2:
            img=cv2.imread(image, colour_mode)
        else:
            img = Image.open(image)
  
        if img.shape[0] ==1024:
            cv2.imwrite( args.path_out + '\\' + img_name + '.png' , img,[cv2.IMWRITE_JPEG_QUALITY, 100])
            continue
        width=img.shape[0]
        hight=img.shape[1]
        #width, hight = img.size
        w = 1024  #切割成1000*1000
        id = 1
        i = 0
        while (i + w <= hight):
            j = 0
            while (j + w <= width):
                if len(img.shape) == 3:

                    new_img = img[i:i+w, j:j+w, :]
          
                    if new_img.shape != (w,w,4):
                        print('dim error')
                        pass
                    else:
                        try:
                            cv2.imwrite( args.path_out + '\\' + img_name + "__" + str(id) + '.png' , new_img,\
                                        [cv2.IMWRITE_JPEG_QUALITY, 100])
                        except:
                            print('error')

                id += 1
                j += w   #滑动步长
            i = i + w

            
            
        
if __name__ == '__main__':
    main()
