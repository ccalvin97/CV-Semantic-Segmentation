#!/usr/bin/env python
# coding: utf-8
import os
import time
from selenium import webdriver
import geopandas as gpd
import pandas as pd 
import numpy as np
import pyproj
import folium
import fiona
from fiona.crs import from_epsg
import re
import shapely
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import matplotlib
from PIL import Image
import pdb
import math
from io import StringIO
import argparse
from multiprocessing import Pool
from functools import partial

'''
======================================
Version 10

Multipolygon bug solved
Multipreocessing Function ok
grid ID and block ID have different ID column name
Add color function for the picture frame
add filter fucntion
production ok
zoom function adjustment, large size
======================================
'''





class folium_parameter():
    def __init__(self, df_gpd, crs):
        self.df = df_gpd
        self.crs=crs

    
    def zoom_scale_area(self):
        '''
        input - df: pandas.core.series.Series, including geometry
        output- zoom_scale, int
        '''
        self.df=self.df.to_crs(epsg=self.crs)
        area_res=self.df['geometry'].area *0.000001
        if area_res[0] <= 1:
            return 16.7
        elif area_res[0] <= 2.05:
            return 16.5
        else:
            print('wrong size - {}'.format(df['new_id']))
 
    def zoom_scale_large_area(self):
        '''
        input - df: pandas.core.series.Series, including geometry
        output- zoom_scale, int
        '''
        self.df=self.df.to_crs(epsg=self.crs)
        area_res=self.df['geometry'].area *0.000001      
        if area_res[0] >= 2.05:
            return 16.5
        else:
            print('wrong size - {}'.format(df['new_id']))
        
class input_data_small():
    def __init__(self,p_out, p_in, colour, p_out_filter, crs):   
        self.pixel_width=1024
        self.pixel_height=1024
        self.threshold = 1
        self.delay = 5
        self.p_out = p_out
        self.p_in = p_in
        self.colour = colour
        self.p_out_filter = p_out_filter
        self.crs = crs

    def input_all_data_block(self, path):
        address_list=[]
        pattern=re.compile(r'.*dsv.shp')
        for home, dirs, files in os.walk(path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        df = gpd.GeoDataFrame(pd.concat([gpd.read_file(address_list[i]) for i in range(len(address_list))],
                                        ignore_index=True), crs='epsg:4326')
        df = df.rename(columns={'block_id': 'new_id'})
        df = df[['new_id','geometry']]
        df = df.drop_duplicates(subset='new_id')
        df = df[df.geometry != None]
        df = df[df['new_id'] != np.nan]
       # df=df[df['new_id'].isin(id_specific)]
       # df['geometry']=df['geometry'].apply(lambda x: x.buffer(0))
        return df
    
    def input_all_data_grid(self, path):
        address_list=[]
        pattern=re.compile(r'.*poires.shp')
        for home, dirs, files in os.walk(path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        df = gpd.GeoDataFrame(pd.concat([gpd.read_file(address_list[i]) for i in range(len(address_list))],
                                        ignore_index=True), crs='epsg:4326')
        df = df[['new_id','geometry']]
        df = df.drop_duplicates(subset='new_id')
        df = df[df.geometry != None]
        df = df[df['new_id'] != np.nan]
       # id_specific=['1401050302', '1401060041', '0043000024302', '0043000023400', '0043000021200', '0043000019600', '0043000019500', '0043000013001','1401050200']
       # df=df[df['new_id'].isin(id_specific)]
       # df['geometry']=df['geometry'].apply(lambda x: x.buffer(0))
        return df



    def cover_calculation(self, i, df):
        '''
        df: pandas series
        p_out: str, output dir
        colour: str, frame colour of the output
        '''
        df = df.iloc[i,:]
        df['geometry'] = df['geometry'].buffer(0)
        centroid_y=df['geometry'].centroid.y
        centroid_x=df['geometry'].centroid.x
        
        if df['geometry'].geom_type == 'Polygon':
            df_gpd = gpd.GeoDataFrame({'value1': [1],'geometry': df['geometry']}, crs=4326)
        elif df['geometry'].geom_type == 'MultiPolygon':
            num_polygon = len(list(df['geometry']))
            df_gpd = gpd.GeoDataFrame({'value1': [ i for i in range(num_polygon)],'geometry': df['geometry']}, crs=4326)
        else:
            print('unknown geo type{}'.format(df['new_id']))

        zoom_scale=folium_parameter(df_gpd, self.crs).zoom_scale_area()
        
        Map=folium.Map(location=[df['geometry'].centroid.y, df['geometry'].centroid.x],
            zoom_start=zoom_scale,
            control_scale=False, 
            zoom_control=False,
            tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
            attr='default',width=self.pixel_width, height=self.pixel_height )
        #style = { 'fillColor': self.colour, 'color':'blue','fillOpacity':1,"opacity":1}
        #folium.GeoJson(df_gpd['geometry'], style_function=lambda x: style).add_to(Map)
        #folium.GeoJson(df_gpd['geometry'],style_function=lambda x: style).add_to(Map) 
        # display(Map)
        name=str(df['new_id'])+'.html'
        Map.save(name)
        url='file://{path}/{mapfile}'.format(path=os.getcwd(),mapfile=name)
        save_fn=name+'map'+'.png'
        option = webdriver.ChromeOptions()
       # option.add_argument('--headless')
        option.headless = True
        option.add_argument('--disable-gpu')
        option.add_argument('disable-infobars')
        option.add_argument("--hide-scrollbars")
        option.add_argument('--no-sandbox')
        option.add_argument('window-size=0x0')
        driver = webdriver.Chrome('/home/GDDC9/web_crawler/chromedriver' , chrome_options=option)
        driver.get(url)
        # scroll_width = driver.execute_script('return document.body.parentNode.scrollWidth')
        # scroll_height = driver.execute_script('return document.body.parentNode.scrollHeight')
        driver.set_window_size(self.pixel_width, self.pixel_height)
        time.sleep(self.delay)
        driver.save_screenshot(self.p_out+'/'+save_fn)
        driver.quit()
        print('Sucess - new id {}'.format(df['new_id']))


    def filter_create(self, i, df):
        '''
        df: pandas series
        p_out: str, output dir
        colour: str, frame colour of the output
        '''
        df = df.iloc[i,:]
        df['geometry'] = df['geometry'].buffer(0)
        centroid_y=df['geometry'].centroid.y
        centroid_x=df['geometry'].centroid.x

        if df['geometry'].geom_type == 'Polygon':
            df_gpd = gpd.GeoDataFrame({'value1': [1],'geometry': df['geometry']}, crs=4326)
        elif df['geometry'].geom_type == 'MultiPolygon':
            num_polygon = len(list(df['geometry']))
            df_gpd = gpd.GeoDataFrame({'value1': [ i for i in range(num_polygon)],'geometry': df['geometry']}, crs=4326)
        else:
            print('unknown geo type{}'.format(df['new_id']))

        white_towel_1 = gpd.GeoDataFrame({'value1': [1],
            'geometry': [shapely.geometry.Polygon([(centroid_x-self.threshold , centroid_y-self.threshold),
            (centroid_x-self.threshold, centroid_y+self.threshold), (centroid_x+self.threshold, centroid_y+self.threshold),
            (centroid_x+self.threshold, centroid_y-self.threshold)])]},crs=4326)
        overlay_result = white_towel_1.difference(df_gpd['geometry'])
        
        zoom_scale=folium_parameter(df_gpd, self.crs).zoom_scale_area()

        Map=folium.Map(location=[df['geometry'].centroid.y, df['geometry'].centroid.x],
            zoom_start=zoom_scale,
            control_scale=False,
            zoom_control=False,
            tiles='Stamen Toner',
            attr='default',width=self.pixel_width, height=self.pixel_height )
        style = { 'fillColor': self.colour, 'color':'blue','fillOpacity':1,"opacity":1}
        folium.GeoJson(overlay_result,style_function=lambda x: style).add_to(Map)

        # display(Map)
        name=str(df['new_id'])+'.html'
        Map.save(name)
        url='file://{path}/{mapfile}'.format(path=os.getcwd(),mapfile=name)
        save_fn=name+'.jpeg'
        option = webdriver.ChromeOptions()
       # option.add_argument('--headless')
        option.headless = True
        option.add_argument('--disable-gpu')
        option.add_argument('disable-infobars')
        option.add_argument("--hide-scrollbars")
        option.add_argument('--no-sandbox')
        option.add_argument('window-size=0x0')
        driver = webdriver.Chrome('/home/GDDC9/web_crawler/chromedriver' , chrome_options=option)
        driver.get(url)
        # scroll_width = driver.execute_script('return document.body.parentNode.scrollWidth')
        # scroll_height = driver.execute_script('return document.body.parentNode.scrollHeight')
        driver.set_window_size(self.pixel_width, self.pixel_height)
        time.sleep(self.delay)
        driver.save_screenshot(self.p_out_filter+'/'+save_fn)
        driver.quit()

        # get filter
        img = Image.open(self.p_out_filter+'/'+save_fn)
        img = img.convert('RGBA')
        W, H = img.size
        blue_pixel = (0, 0, 255)
        for h in range(W):   ###循环图片的每个像素点
            for l in range(H):
                if img.getpixel((h,l))[:3] != blue_pixel:
                    img.putpixel((h,l),(0,0,0,0))
        img.save(self.p_out_filter+'/'+name+'_filter.png')

        try:
            os.remove(self.p_out_filter+'/'+save_fn)
            print('Sucess - new id {}'.format(df['new_id']))

        except:
            print('error no file')


class input_data_large():
    def __init__(self,p_out, p_in, colour, p_out_filter, crs):   
        self.pixel_width=3096
        self.pixel_height=3096
        self.threshold = 1
        self.delay = 5
        self.p_out = p_out
        self.p_in = p_in
        self.colour = colour
        self.p_out_filter = p_out_filter
        self.crs = crs

    def input_all_data_block(self, path):
        address_list=[]
        pattern=re.compile(r'.*dsv.shp')
        for home, dirs, files in os.walk(path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        df = gpd.GeoDataFrame(pd.concat([gpd.read_file(address_list[i]) for i in range(len(address_list))],
                                        ignore_index=True), crs='epsg:4326')

        df = df.rename(columns={'block_id': 'new_id'})
        df = df[['new_id','geometry']]
        df = df.drop_duplicates(subset='new_id')
        df = df[df.geometry != None]
        df = df[df['new_id'] != np.nan]
       # id_specific=['1401050302', '1401060041', '0043000024302', '0043000023400', '0043000021200', '0043000019600', '0043000019500', '0043000013001','1401050200']
       # df=df[df['new_id'].isin(id_specific)]
       # df['geometry']=df['geometry'].apply(lambda x: x.buffer(0))
        return df
    
    def input_all_data_grid(self, path):
        address_list=[]
        pattern=re.compile(r'.*poires.shp')
        for home, dirs, files in os.walk(path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        df = gpd.GeoDataFrame(pd.concat([gpd.read_file(address_list[i]) for i in range(len(address_list))],
                                        ignore_index=True), crs='epsg:4326')
        df = df[['new_id','geometry']]
        df = df.drop_duplicates(subset='new_id')
        df = df[df.geometry != None]
        df = df[df['new_id'] != np.nan]
       # id_specific=['1401050302', '1401060041', '0043000024302', '0043000023400', '0043000021200', '0043000019600', '0043000019500', '0043000013001','1401050200']
       # df=df[df['new_id'].isin(id_specific)]
       # df['geometry']=df['geometry'].apply(lambda x: x.buffer(0))
        return df



    def cover_calculation(self, i, df):
        '''
        df: pandas series
        p_out: str, output dir
        colour: str, frame colour of the output
        '''
        df = df.iloc[i,:]
        df['geometry'] = df['geometry'].buffer(0)
        centroid_y=df['geometry'].centroid.y
        centroid_x=df['geometry'].centroid.x
        
        if df['geometry'].geom_type == 'Polygon':
            df_gpd = gpd.GeoDataFrame({'value1': [1],'geometry': df['geometry']}, crs=4326)
        elif df['geometry'].geom_type == 'MultiPolygon':
            num_polygon = len(list(df['geometry']))
            df_gpd = gpd.GeoDataFrame({'value1': [ i for i in range(num_polygon)],'geometry': df['geometry']}, crs=4326)
        else:
            print('unknown geo type{}'.format(df['new_id']))
        '''
        white_towel_1 = gpd.GeoDataFrame({'value1': [1],
            'geometry': [shapely.geometry.Polygon([(centroid_x-self.threshold , centroid_y-self.threshold),
            (centroid_x-self.threshold, centroid_y+self.threshold), (centroid_x+self.threshold, centroid_y+self.threshold), 
            (centroid_x+self.threshold, centroid_y-self.threshold)])]},crs='epsg:4326')
        overlay_result = white_towel_1.difference(df_gpd['geometry'])
        '''

        zoom_scale=folium_parameter(df_gpd, self.crs).zoom_scale_large_area()
        
        Map=folium.Map(location=[df['geometry'].centroid.y, df['geometry'].centroid.x],
            zoom_start=zoom_scale,
            control_scale=False, 
            zoom_control=False,
            tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
            attr='default',width=self.pixel_width, height=self.pixel_height )
        #style = { 'fillColor': self.colour, 'color':'blue','fillOpacity':1,"opacity":1}
        #folium.GeoJson(df_gpd['geometry'], style_function=lambda x: style).add_to(Map)
        #folium.GeoJson(df_gpd['geometry'],style_function=lambda x: style).add_to(Map) 
        # display(Map)
        name=str(df['new_id'])+'.html'
        Map.save(name)
        url='file://{path}/{mapfile}'.format(path=os.getcwd(),mapfile=name)
        save_fn=name+'map'+'.png'
        option = webdriver.ChromeOptions()
       # option.add_argument('--headless')
        option.headless = True
        option.add_argument('--disable-gpu')
        option.add_argument('disable-infobars')
        option.add_argument("--hide-scrollbars")
        option.add_argument('--no-sandbox')
        option.add_argument('window-size=0x0')
        driver = webdriver.Chrome('/home/GDDC9/web_crawler/chromedriver' , chrome_options=option)
        driver.get(url)
        # scroll_width = driver.execute_script('return document.body.parentNode.scrollWidth')
        # scroll_height = driver.execute_script('return document.body.parentNode.scrollHeight')
        driver.set_window_size(self.pixel_width, self.pixel_height)
        time.sleep(self.delay)
        driver.save_screenshot(self.p_out+'/'+save_fn)
        driver.quit()
        print('Sucess - new id {}'.format(df['new_id']))


    def filter_create(self, i, df):
        '''
        df: pandas series
        p_out: str, output dir
        colour: str, frame colour of the output
        '''
        df = df.iloc[i,:]
        df['geometry'] = df['geometry'].buffer(0)
        centroid_y=df['geometry'].centroid.y
        centroid_x=df['geometry'].centroid.x

        if df['geometry'].geom_type == 'Polygon':
            df_gpd = gpd.GeoDataFrame({'value1': [1],'geometry': df['geometry']}, crs=4326)
        elif df['geometry'].geom_type == 'MultiPolygon':
            num_polygon = len(list(df['geometry']))
            df_gpd = gpd.GeoDataFrame({'value1': [ i for i in range(num_polygon)],'geometry': df['geometry']}, crs=4326)
        else:
            print('unknown geo type{}'.format(df['new_id']))

        white_towel_1 = gpd.GeoDataFrame({'value1': [1],
            'geometry': [shapely.geometry.Polygon([(centroid_x-self.threshold , centroid_y-self.threshold),
            (centroid_x-self.threshold, centroid_y+self.threshold), (centroid_x+self.threshold, centroid_y+self.threshold),
            (centroid_x+self.threshold, centroid_y-self.threshold)])]},crs=4326)
        overlay_result = white_towel_1.difference(df_gpd['geometry'])
        
        zoom_scale=folium_parameter(df_gpd, self.crs).zoom_scale_large_area()

        Map=folium.Map(location=[df['geometry'].centroid.y, df['geometry'].centroid.x],
            zoom_start=zoom_scale,
            control_scale=False,
            zoom_control=False,
            tiles='Stamen Toner',
            attr='default',width=self.pixel_width, height=self.pixel_height )
        style = { 'fillColor': self.colour, 'color':'blue','fillOpacity':1,"opacity":1}
        folium.GeoJson(overlay_result,style_function=lambda x: style).add_to(Map)

        # display(Map)
        name=str(df['new_id'])+'.html'
        Map.save(name)
        url='file://{path}/{mapfile}'.format(path=os.getcwd(),mapfile=name)
        save_fn=name+'.jpeg'
        option = webdriver.ChromeOptions()
       # option.add_argument('--headless')
        option.headless = True
        option.add_argument('--disable-gpu')
        option.add_argument('disable-infobars')
        option.add_argument("--hide-scrollbars")
        option.add_argument('--no-sandbox')
        option.add_argument('window-size=0x0')
        driver = webdriver.Chrome('/home/GDDC9/web_crawler/chromedriver' , chrome_options=option)
        driver.get(url)
        # scroll_width = driver.execute_script('return document.body.parentNode.scrollWidth')
        # scroll_height = driver.execute_script('return document.body.parentNode.scrollHeight')
        driver.set_window_size(self.pixel_width, self.pixel_height)
        time.sleep(self.delay)
        driver.save_screenshot(self.p_out_filter+'/'+save_fn)
        driver.quit()

        # get filter
        img = Image.open(self.p_out_filter+'/'+save_fn)
        img = img.convert('RGBA')
        W, H = img.size
        blue_pixel = (0, 0, 255)
        for h in range(W):   ###循环图片的每个像素点
            for l in range(H):
                if img.getpixel((h,l))[:3] != blue_pixel:
                    img.putpixel((h,l),(0,0,0,0))
        img.save(self.p_out_filter+'/'+name+'_filter.png')

        try:
            os.remove(self.p_out_filter+'/'+save_fn)
            print('Sucess - new id {}'.format(df['new_id']))

        except:
            print('error no file')         


def main():
    # Get arguments
    parser = argparse.ArgumentParser(description='welcome to the web crawling programme for croping the blocks and grids from open source map')
    parser.add_argument('-p_in', '--p_in', type=str,
        help='path of input data dir  ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel')
    parser.add_argument('-p_out', '--p_out', type=str,
        help='data out abs address for picture ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')   
    parser.add_argument('-p_out_filter', '--p_out_filter', type=str,
        help='data out abs address for filter files ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')
    parser.add_argument('-p_out_3096', '--p_out_3096', type=str,
        help='data out abs address for picture ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')
    parser.add_argument('-p_out_filter_3096', '--p_out_filter_3096', type=str,
        help='data out abs address for filter files ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')


    parser.add_argument('-crs', '--crs', type=int, default=0,
            help='crs is a 4 digits numbers ex: 4326')
    parser.add_argument('-f', '--f', type=str, default='crawl',
        help=' function crawl or filter or all')
    parser.add_argument('-colour', '--colour', type=str, default='blue',
        help=' colour for the frame of output, ex: red')

    args = parser.parse_args()
          

    df  = input_data_large(args.p_out_3096, args.p_in, args.colour, args.p_out_filter_3096, args.crs).input_all_data_block(args.p_in)
    df2 = input_data_large(args.p_out_3096, args.p_in, args.colour, args.p_out_filter_3096, args.crs).input_all_data_grid(args.p_in) 
    df  = pd.concat([df,df2],ignore_index=True,sort=False)
    df=df.to_crs(epsg=args.crs)
    df_large=df[df['geometry'].area *0.000001 >  2.05]
    df_large=df_large.to_crs(epsg=4326)
    del df2
    del df
    
    df  = input_data_small(args.p_out, args.p_in, args.colour, args.p_out_filter, args.crs).input_all_data_block(args.p_in)
    df2 = input_data_small(args.p_out, args.p_in, args.colour, args.p_out_filter, args.crs).input_all_data_grid(args.p_in) 
    df  = pd.concat([df,df2],ignore_index=True,sort=False) 
    df=df.to_crs(epsg=args.crs)    
    df_small=df[df['geometry'].area *0.000001 <=  2.05]
    df_small=df_small.to_crs(epsg=4326)
    del df2
    del df
    
    input_list_large=[i for i in range(df_large.shape[0])]
    input_list_small=[i for i in range(df_small.shape[0])]
    
    if args.f == 'crawl':
        partial_func = partial(input_data(args.p_out, args.p_in, args.colour, args.p_out_filter).cover_calculation, df=df)
        pool = Pool()
        pool.map(partial_func, input_list) 
        pool.close()
        pool.join()

    elif args.f == 'filter':
        partial_func = partial(input_data(args.p_out, args.p_in, args.colour, args.p_out_filter).filter_create, df=df)
        pool = Pool()
        pool.map(partial_func1, input_list)
        pool.close()
        pool.join()    
    
    elif args.f == 'small':
        df=df_small
        partial_func2 = partial(input_data_small(args.p_out, args.p_in, args.colour, args.p_out_filter, args.crs).cover_calculation,df=df)
        pool = Pool(1)
        pool.map(partial_func2, input_list_small)
        pool.close()
        pool.join()
        
        
    elif args.f == 'all':
        print('Start all function')
        print('Start Large')
        df=df_large
        partial_func = partial(input_data_large(args.p_out_3096, args.p_in, args.colour, args.p_out_filter_3096, args.crs).cover_calculation, df=df)
        pool = Pool()
        pool.map(partial_func, input_list_large)
        pool.close()
        pool.join()

        partial_func1 = partial(input_data_large(args.p_out_3096, args.p_in, args.colour, args.p_out_filter_3096, args.crs).filter_create, df=df)
        pool = Pool()
        pool.map(partial_func1, input_list_large)
        pool.close()
        pool.join()
        print('Finish Large')
        
        
        print('Start Small')
        df=df_small
        partial_func2 = partial(input_data_small(args.p_out, args.p_in, args.colour, args.p_out_filter, args.crs).cover_calculation,df=df)
        pool = Pool()
        pool.map(partial_func2, input_list_small)
        pool.close()
        pool.join()

        partial_func2 = partial(input_data_small(args.p_out, args.p_in, args.colour, args.p_out_filter, args.crs).filter_create,df=df)
        pool = Pool()
        pool.map(partial_func2, input_list_small)
        pool.close()
        pool.join()
        print('Finish Small')
    else:
        print('error of args.f')






if __name__ == '__main__':
    main()
    
    
    
    
