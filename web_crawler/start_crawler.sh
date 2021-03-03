#!/bin/bash

## Start crawler solution ##


p_in="/home/GDDC9/web_crawler/shanxi_0043_1130_gcj"
p_out_map="/home/GDDC9/web_crawler/out_shanxi_0043_1024_map"
p_out_filter="/home/GDDC9/web_crawler/out_shanxi_0043_1024_filter"
p_out_map_3096="/home/GDDC9/web_crawler/out_shanxi_0043_3096_map"
p_out_filter_3096="/home/GDDC9/web_crawler/out_shanxi_0043_3096_filter"


crs="2382"
path_out_map="/home/GDDC9/web_crawler/out_shanxi_0043_1024_final_map"
path_out_filter="/home/GDDC9/web_crawler/out_shanxi_0043_1024_final_filter"





python -W ignore print_map_png_v10.py -p_in $p_in  -p_out $p_out_map -f all -p_out_filter $p_out_filter -crs $crs -p_out_3096 $p_out_map_3096 -p_out_filter_3096 $p_out_filter_3096


#python -W ignore web_crawler_image_crop.py -path  /home/GDDC9/web_crawler/out_shanxi_0043_1024 -path_out /home/GDDC9/web_crawler/out_shanxi_0043_1024_final
#python -W ignore web_crawler_image_crop.py -path  /home/GDDC9/web_crawler/out_shanxi_0043_1024 -path_out /home/GDDC9/web_crawler/out_shanxi_0043_1024_final


: << !
    parser = argparse.ArgumentParser(description='welcome to the web crawling programme for croping the blocks and grids from open source map')
    parser.add_argument('-p_in', '--p_in', type=str,
        help='path of input data dir  ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel')
    parser.add_argument('-p_out', '--p_out', type=str,
        help='data out abs address for picture ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')   
    parser.add_argument('-colour', '--colour', type=str, default='blue',
        help=' colour for the frame of output, ex: red')
    parser.add_argument('-f', '--f', type=str, default='crawl',
        help=' function crawl or filter or all')
    parser.add_argument('-p_out_filter', '--p_out_filter', type=str,
        help='data out abs address for filter files ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')
    parser.add_argument('-crs', '--crs', type=int, default=0,
            help='crs is a 4 digits numbers ex: 4326')
    args = parser.parse_args()



    parser = argparse.ArgumentParser(description='welcome to the web crawling programme for splitting image to 1024*1024')
    parser.add_argument('-path', '--path', type=str,
        help='path of input data dir  ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel')
    parser.add_argument('-path_out', '--path_out', type=str,
        help='data out abs address for picture ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')
!
