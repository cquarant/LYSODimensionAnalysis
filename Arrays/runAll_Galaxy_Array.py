## Python stuff
from array import array
import os
import math as mt
import numpy as np
import re
import argparse
from collections import defaultdict
import pandas as pd
import sys
import csv
import glob
import shutil
import os.path




#--- Options

parser = argparse.ArgumentParser(usage="python3 runAll_Galaxy_Array.py --inputdir  ArrayData")


parser.add_argument('--inputdir',dest='inputdir',required=True)
args = parser.parse_args()

for filename in os.listdir(args.inputdir):

    print('+++++++++++++++++++++++++++')
    print('        NEXT  FILE         ')
    print('+++++++++++++++++++++++++++')


    print (filename)

    filename_trunc = filename.split("_")[0]
    print (filename_trunc)

    array = filename_trunc
    print (array)

    command = ( "python3 Galaxy_raw_Data_Array_MTDDB.py   --array {} --data {}".format( array , args.inputdir+"/"+filename) ) 
 
    print (command)
    os.system(command)
    



print('')
print('+++++++++++++++++++++++++++')
print('      MOVING FILES         ')
print('+++++++++++++++++++++++++++')


#+++++++++++++++++++++++++++++++++++++++++++
#   Moving files in different directories
#+++++++++++++++++++++++++++++++++++++++++++

path = os.chdir("ArrayData")
path = os.getcwd() 


print('=====> Raw Data directory: ',os.path.basename(path))


res_path = os.path.join(os.getcwd(),"../ArrayData_results")
processed_path = os.path.join(os.getcwd(),"../ArrayData_processed")
output_path = os.path.join(os.getcwd(),"../ArrayData_output")
plot_path = os.path.join(os.getcwd(),"../ArrayData_plot")


print('=====> Results directory: ',os.path.basename(res_path))


if not os.path.isdir(res_path):
    os.makedirs(res_path)
    print('=====> Created ArrayData_results directory: ', os.path.basename(res_path))
else:
    print('=====>',os.path.basename(res_path) ,'directory already exists!')


if not os.path.isdir(processed_path):
    os.makedirs(processed_path)
    print('=====> Created ArrayData_processed directory: ', os.path.basename(processed_path))
else:
    print('=====>',os.path.basename(processed_path) ,'directory already exists!')


if not os.path.isdir(output_path):
    os.makedirs(output_path)
    print('=====> Created ArrayData_output directory: ', os.path.basename(output_path))
else:
    print('=====>',os.path.basename(output_path) ,'directory already exists!')
    
if not os.path.isdir(plot_path):
    os.makedirs(plot_path)
    print('=====> Created ArrayData_output directory: ', os.path.basename(plot_path))
else:
    print('=====>',os.path.basename(plot_path) ,'directory already exists!')


files = os.listdir(path)


for file in files:

    filename = os.path.join(path, file)
    filedest = os.path.join(res_path, file)
    filedest1 = os.path.join(processed_path, file)
    filedest2 = os.path.join(output_path, file)
    filedest3 = os.path.join(plot_path, file)


    if (filename.endswith('.csv') or filename.endswith('.json')):
       

        print('+++++++++++++++++++++++++++++++++++++++++++++')       
        print('-----> Processing file: ',os.path.basename(filename))



        if os.path.exists(filedest):
            os.remove(filedest)
            print('-----> Removing file: ', os.path.basename(filedest), 'from',os.path.basename(res_path),'as it exists!')


            
        else:
            print('-----> Can not delete the file',os.path.basename(filedest), "as it doesn't exist!")


        shutil.move(filename, filedest)
        print('-----> File', file ,'moved successfully to', os.path.basename(res_path) ,'directory')

        

    elif (filename.endswith('.TXT')):
       

        print('+++++++++++++++++++++++++++++++++++++++++++++')       
        print('-----> Processing file: ',os.path.basename(filename))



        if os.path.exists(filedest1):
            os.remove(filedest1)
            print('-----> Removing file: ', os.path.basename(filedest1), 'from',os.path.basename(processed_path),'as it exists!')

            
        else:
            print('-----> Can not delete the file',os.path.basename(filedest1), "as it doesn't exist!")


        shutil.move(filename, filedest1)
        print('-----> File', file ,'moved successfully to', os.path.basename(processed_path) ,'directory')



    elif (filename.endswith('.txt') or filename.endswith('.pdf')):
       

        print('+++++++++++++++++++++++++++++++++++++++++++++')       
        print('-----> Processing file: ',os.path.basename(filename))



        if os.path.exists(filedest2):
            os.remove(filedest2)
            print('-----> Removing file: ', os.path.basename(filedest2), 'from',os.path.basename(output_path),'as it exists!')


            
        else:
            print('-----> Can not delete the file',os.path.basename(filedest2), "as it doesn't exist!")


        shutil.move(filename, filedest2)
        print('-----> File', file ,'moved successfully to', os.path.basename(output_path) ,'directory')


    elif (filename.endswith('.png')):
       

        print('+++++++++++++++++++++++++++++++++++++++++++++')       
        print('-----> Processing file: ',os.path.basename(filename))



        if os.path.exists(filedest3):
            os.remove(filedest3)
            print('-----> Removing file: ', os.path.basename(filedest3), 'from',os.path.basename(output_path),'as it exists!')


            
        else:
            print('-----> Can not delete the file',os.path.basename(filedest3), "as it doesn't exist!")


        shutil.move(filename, filedest3)
        print('-----> File', file ,'moved successfully to', os.path.basename(plot_path) ,'directory')








    




