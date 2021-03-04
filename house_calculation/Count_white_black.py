#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pdb
from PIL import Image
import pandas as pd
import os
import re
from os import mkdir
from os.path import isdir
import csv
from io import StringIO
import argparse
from colorama import  init, Fore, Back, Style  
import cv2

init(autoreset=True)  
class Colored(object):  
    def red(self, s):  
        return Fore.RED + s + Fore.RESET  
    
    def yellow(self, s):  
        return Fore.YELLOW + s + Fore.RESET 
color = Colored() 


class house_ratio():
    def __init__(self,prefix, colour_mode=-1):
        ''' Make directory if not exist when initialize '''
        if not os.path.isdir(prefix):
            mkdir(prefix)
        self.prefix = prefix
        self.colour_mode = colour_mode
        # 1-> colour mode, 
        # 0-> gray mode, 
        #-1-> alpha mode

    def _record(self,id,  results, c):
        ''' 
        Save row to csv file 
        results: input, type: pd.Series
        id: saved name, type:str
        c: type for writerow
        '''
        f = open('{}/{}.csv'.format(self.prefix, id), 'a', encoding='gb18030')
        cw = csv.writer(f, lineterminator='\n')
        if c == 1:
            cw.writerow(results.iloc[row]+ '\t')
        else:
            cw.writerow(results)
        f.close()


    def ratio_count_small(self, df_map, df_filter):
        '''
        df_map:    1 1024*1024 map, absolute address, type:str
        df_filter: 1 1024*1024 filter, absolute address,  type:str
        '''
        name=df_map.split('.')[0].split('\\')[-1]
        map_=cv2.imread(df_map, self.colour_mode)
        filter_=cv2.imread(df_filter, self.colour_mode)
        filter_=filter_/255
        map_=map_/255
        available_pixel=np.sum(filter_[:,:,3] == 0)
        white_pixel= np.sum((filter_[:,:,3] == 0) & (map_[:,:,0]==1))
        ratio=round(white_pixel/available_pixel,5)
        df=pd.Series([str(name), ratio])  
        if len(df)!=0:
            self._record('out', df, c=0)
        else:
            print('No original_area in {}'.format(id))

    def ratio_count_large(self, df_map, df_filter):
        '''
        df_map:    1 3096*3096 map,    absolute address,  type:list, len=9
        df_filter: 1 3096*3096 filter, absolute address,  type:list, len=9
        '''
        available_pixel_res=[]
        white_res=[]
        name=df_map[-1].split('.')[0].split('\\')[-1]
        for j in range(len(df_map)):
            map_=cv2.imread(df_map[j], self.colour_mode)
            filter_=cv2.imread(df_filter[j], self.colour_mode)
            try:
                filter_=filter_/255
                map_=map_/255
            except:
                pdb.set_trace()
            available_pixel=np.sum(filter_[:,:,3] == 0)
            white_pixel= np.sum((filter_[:,:,3] == 0) & (map_[:,:,0] == 1))
            available_pixel_res.append(available_pixel)
            white_res.append(white_pixel)
            
            
        ratio=round(np.sum(white_res)/np.sum(available_pixel_res),5)
        df=pd.Series([str(name), ratio])  
        if len(df)!=0:
            self._record('out', df, c=0)
        else:
            print('No original_area in {}'.format(id))

def is_number(s):
    try:
        if s[-2]=='_':
            return True
    except ValueError:
        pass
    return False   
        
def main():
    # Get arguments
    parser = argparse.ArgumentParser(description='House Ratio Calculation')
    parser.add_argument('-path_map', '--path_map', type=str,
        help='dir of map file, ex: C:\\Users\\heca0002\\Desktop\\map_list')
    parser.add_argument('-path_filter', '--path_filter', type=str,
        help='dir of filter file, ex: C:\\Users\\heca0002\\Desktop\\filter_list')
    parser.add_argument('-p_out', '--prefix', type=str,
                help='data out abs address ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')    
    args = parser.parse_args()

    
    print(color.yellow('House Ratio Calculation Start')) 
    
    map_list=[]
    pattern=re.compile(r'.*.png')
    for home, dirs, files in os.walk(args.path_map):
        for filename in files:
            if pattern.findall(filename):
                map_list.append(os.path.join(home, filename))
    if map_list == []:
        raise NameError('please refill in the -p parameter')
    
    filter_list=[]
    pattern=re.compile(r'.*.png')
    for home, dirs, files in os.walk(args.path_filter):
        for filename in files:
            if pattern.findall(filename):
                filter_list.append(os.path.join(home, filename))    
    if filter_list == []:
        raise NameError('please refill in the -p_grid parameter')
       
    map_list.sort()
    filter_list.sort()
    
    if len(map_list)!=len(filter_list):
        print('List length Error')
        raise KeyboardInterrupt  
    house_ratio_=house_ratio(args.prefix)

    i=0   
    try:
        while i<len(map_list):
            if is_number(map_list[i].split('.')[0]) is True:
                original_area = house_ratio_.ratio_count_large(map_list[i:i+9], filter_list[i:i+9])
                i=i+9
            else:
                original_area = house_ratio_.ratio_count_small(map_list[i], filter_list[i])
                i=i+1
        print('Success')
    except:
        print('Fail')
        raise KeyboardInterrupt 
    

if __name__ == '__main__':
    main()



