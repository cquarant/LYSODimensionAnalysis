#!/usr/bin/env python
# coding: utf-8



import os
import sys
import optparse
import datetime
import subprocess
from glob import glob
from collections import defaultdict
from collections import OrderedDict
from array import array
import time
import re
import numpy as np
import copy
import shutil
from os import path



'''
# folder path
dir_path = '/Users/merlino/Desktop/LYSO_Roma/misure3D/Galaxy3D/ArraySizeAna/ForRoberta_Script/GalaxyAnalysis/BarData'
# list to store files
res = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)
print(res)
'''




  
runNumberFileName='/data/cmsdaq/DimensionBench/Bars/RunNumberGalaxyBar.txt'
print(runNumberFileName)




dir = 'BarData/'
ext = ('.TXT')


for file in os.listdir(dir):
    if file.endswith(ext):

        old_filepath = os.path.join(dir, file)
        currentRun = 0
        outputFileName = runNumberFileName
    
        file_runs = open(outputFileName, 'a+')
    
        lastRun = subprocess.check_output(['tail', '-1', outputFileName])
        lastRun = lastRun.rstrip(b'\n')
    
        if not lastRun:
            currentRun = 1
        else:
            currentRun = int(lastRun) + 1
       
        file_runs.write(str(currentRun)+'\n')
        newlabel = 'Run'+str(currentRun).zfill(6)+str('_')
            
        new_name = newlabel + file

        new_filepath = os.path.join(dir, new_name)

        os.rename(old_filepath, new_filepath)
    #print(file)
   
        file_runs.close()
    
#print(file)



