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

#parser = argparse.ArgumentParser(usage="python3 runAll_Galaxy_Bar_new_clean.py --inputdir  BarData")
parser = argparse.ArgumentParser(usage="python3 runAll_Galaxy_Bar.py --inputdir  BarData")

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

    #command = ( "python3 Galaxy_raw_Data_Bar_new_clean.py   --bar {} --data {}".format(array , args.inputdir+"/"+filename) ) 
    command = ( "python3 Galaxy_raw_Data_Bar_MTDDB.py   --bar {} --data {}".format( array , args.inputdir+"/"+filename) ) 

    print (command)
    os.system(command)






print('')
print('+++++++++++++++++++++++++++')
print('      MOVING FILES         ')
print('+++++++++++++++++++++++++++')


    #moving = ( "python3 MoveFiles.py" ) 

    #print (moving)
    #os.system(moving)
    

    

#+++++++++++++++++++++++++++++++++++++++++++
#   Moving files in different directories
#+++++++++++++++++++++++++++++++++++++++++++

path = os.chdir("BarData")
path = os.getcwd() 

#print('path = Starting directory: ',path)

print('=====> Raw Data directory: ',os.path.basename(path))


#res_path = os.chdir("../BarData_results")
#res_path = os.getcwd() 

res_path = os.path.join(os.getcwd(),"../BarData_results")
processed_path = os.path.join(os.getcwd(),"../BarData_processed")
output_path = os.path.join(os.getcwd(),"../BarData_output")


#print('res_path = Final directory: ',res_path)
print('=====> Results directory: ',os.path.basename(res_path))



if not os.path.isdir(res_path):
    os.makedirs(res_path)
    #print("-----> BarData_results folder created: ", res_path)
    print('=====> Created BarData_results directory: ', os.path.basename(res_path))
else:
    print('=====>',os.path.basename(res_path) ,'directory already exists!')


if not os.path.isdir(processed_path):
    os.makedirs(processed_path)
    #print("-----> BarData_results folder created: ", res_path)
    print('=====> Created BarData_processed directory: ', os.path.basename(processed_path))
else:
    print('=====>',os.path.basename(processed_path) ,'directory already exists!')


if not os.path.isdir(output_path):
    os.makedirs(output_path)
    #print("-----> BarData_results folder created: ", res_path)
    print('=====> Created BarData_output directory: ', os.path.basename(output_path))
else:
    print('=====>',os.path.basename(output_path) ,'directory already exists!')
    



files = os.listdir(path)
#print('List of files: ',files)


for file in files:

    filename = os.path.join(path, file)
    filedest = os.path.join(res_path, file)
    filedest1 = os.path.join(processed_path, file)
    filedest2 = os.path.join(output_path, file)





    if (filename.endswith('.csv') or filename.endswith('.json')):
       
        #print('-----> Processing file: ',filename)

        print('+++++++++++++++++++++++++++++++++++++++++++++')       
        print('-----> Processing file: ',os.path.basename(filename))



        if os.path.exists(filedest):
            os.remove(filedest)
            #print('-----> Removing file: ', filedest, 'as it exists')
            print('-----> Removing file: ', os.path.basename(filedest), 'from',os.path.basename(res_path),'as it exists!')


            
        else:
            print('-----> Can not delete the file',os.path.basename(filedest), "as it doesn't exist!")


        shutil.move(filename, filedest)
        #print("File is moved successfully to: ", res_path)
        #print('-----> File', file ,'moved successfully to', res_path ,'folder')
        print('-----> File', file ,'moved successfully to', os.path.basename(res_path) ,'directory')

        

    elif (filename.endswith('.TXT')):
       
        #print('-----> Processing file: ',filename)

        print('+++++++++++++++++++++++++++++++++++++++++++++')       
        print('-----> Processing file: ',os.path.basename(filename))



        if os.path.exists(filedest1):
            os.remove(filedest1)
            #print('-----> Removing file: ', filedest, 'as it exists')
            print('-----> Removing file: ', os.path.basename(filedest1), 'from',os.path.basename(processed_path),'as it exists!')


            
        else:
            print('-----> Can not delete the file',os.path.basename(filedest1), "as it doesn't exist!")


        shutil.move(filename, filedest1)
        #print("File is moved successfully to: ", res_path)
        #print('-----> File', file ,'moved successfully to', res_path ,'folder')
        print('-----> File', file ,'moved successfully to', os.path.basename(processed_path) ,'directory')



    elif (filename.endswith('.txt')):
       
        #print('-----> Processing file: ',filename)

        print('+++++++++++++++++++++++++++++++++++++++++++++')       
        print('-----> Processing file: ',os.path.basename(filename))



        if os.path.exists(filedest2):
            os.remove(filedest2)
            #print('-----> Removing file: ', filedest, 'as it exists')
            print('-----> Removing file: ', os.path.basename(filedest2), 'from',os.path.basename(output_path),'as it exists!')


            
        else:
            print('-----> Can not delete the file',os.path.basename(filedest2), "as it doesn't exist!")


        shutil.move(filename, filedest2)
        #print("File is moved successfully to: ", res_path)
        #print('-----> File', file ,'moved successfully to', res_path ,'folder')
        print('-----> File', file ,'moved successfully to', os.path.basename(output_path) ,'directory')






    



