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
ImageFile.LOAD_TRUNCATED_IMAGES = True
from os import mkdir
from os.path import isdir
from skimage import io
import pandas as pd
import random
import shutil
random.seed(2)
import argparse




class check_x_size():
    def __init__(self, path, path_filter, threshold=667):
        '''
        check train_x size and delete small size data and corresponding label data
        '''
        self.path = path
        self.path_filter = path_filter
        self.threshold=threshold
 
        address_list=[]
        pattern=re.compile(r'.*.png')
        for home, dirs, files in os.walk(self.path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        self.address_list=address_list

    def check_size(self):
        res=[]
        count=0
        #size_list=[]

        for i in self.address_list:
            img_name=i.split('/')[-1].split('.')[0]
            data_size=os.path.getsize(i)
            data_size=data_size/float(1024)
            if data_size<self.threshold:
                res.append(img_name)
                #size_list.append(data_size)
                #count=count+1
        return res

    def delete_small_data(self, res):
        for i in res:
            os.remove(self.path+'/'+i+'.htmlmap.png')
            os.remove(self.path_filter+'/'+i+'.html_filter.png')


def main():
    # Get arguments
    parser = argparse.ArgumentParser(description='welcome to the fail image detection')
    parser.add_argument('-path_map', '--path_map', type=str,
        help='path of input map dir  ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel')
    parser.add_argument('-path_filter', '--path_filter', type=str,
        help='path of input filter dir  ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel')
#    parser.add_argument('-path_out', '--path_out', type=str,
#        help='data out abs address for picture ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')   
    parser.add_argument('-size', '--size', type=str, default=667,
            help='size of threshold for detection')
    args = parser.parse_args()



    res=check_x_size(args.path_map, args.path_filter).check_size()
    check_x_size(args.path_map, args.path_filter).delete_small_data(res)





if __name__ == '__main__':
    main()



