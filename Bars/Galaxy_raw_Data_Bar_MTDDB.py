import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from array import array
import sys, os, argparse
import math as mt
import numpy as np
import re
from collections import defaultdict
import pandas as pd
import csv
import json
import datetime
import itertools
import glob
import timeit
from datetime import datetime
from prettytable import PrettyTable
import shutil


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     HOW TO RUN OVER ONE SINGLE FILE
#  python3 Galaxy_raw_Data_Bar_MTDDB.py --data BarData/1115* --bar 1115
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

start = timeit.default_timer()

#+++++++++++++++++++++++++++++++++++++++++++
#  Parsing - Date - Producers definifition
#+++++++++++++++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()

parser.add_argument('--data',dest='data', help='data file *.TXT', type=str,required=True,default=False)
parser.add_argument('--bar',dest='bar',required=False,default=False)

args = parser.parse_args()


#+++++++++++++++++++++++++++++++++++++
# filename: 415_2022-05-04-12-52.txt
# Date and time as 2022-05-04-12-52
#+++++++++++++++++++++++++++++++++++++

data = args.data
data = data[:-2]
info = data.split('_')

run = info[0]
args.xtalbar = barcode = info[1]

date = info[2]

tag = info[3]

date = date.replace('.T','')
data = data.replace('.T','')
tag = tag.replace('.T','')


print('Filename :', data)
print('Barcode :', barcode)
print('Tag :', tag)
print('RunNumber :', run)
#print('Barcode :', barcode.split('/')[1]) # this when we run it using runAll script otherwise we have to replace with print('Barcode :', barcode)
#print('Barcode :', barcode)
print('Date in filename :', date)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# convert datetime string into date,month,day and
# hours:minutes:and seconds format using strptime
#++++++++++++++++++++++++++++++++++++++++++++++++++

time = datetime.strptime(date, '%Y-%m-%d-%H-%M') #in use time format
timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  #new time format
print('Time after conversion :',timestamp)
print('Info LYSO Single Xtal Bar: {}'.format(str(args)))     


#+++++++++++++++++++++++++++++++++++++++++++
#   JUST FOR REFERENCE: NOMINAL DIMENSIONS
#+++++++++++++++++++++++++++++++++++++++++++

#----------------------------------------------------------------------------
#  Array type  |     w        |       t     |         L
#----------------------------------------------------------------------------
#        1     | 51.50+-0.10  | 4.05+-0.10  | 55(56.30)+-0.020
#        2     | 51.50+-0.10  | 3.30+-0.10  | 55(56.30)+-0.020
#        3     | 51.50+-0.10  | 2.70+-0.10  | 55(56.30)+-0.020
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# Bar type  |      w      |       t           |         L
#----------------------------------------------------------------------------
#     1     | 3.12+-0.10  | 3.75+-0.10        | 55(56.30)+-0.020
#     2     | 3.12+-0.10  | 3.00 (3.30)+-0.10 | 55(56.30)+-0.020  MS3 w and w/o ESR
#     3     | 3.12+-0.10  | 2.40 +-0.10       | 55(56.30)+-0.020  MS3 w/o ESR
#----------------------------------------------------------------------------

#+++++++++++++++++++++++++++++++++++++++++++
#    Tolerances and ranges definition
#+++++++++++++++++++++++++++++++++++++++++++


lenght_m = 55.0 #mm
e_low_lenght = 0.02 #mm
e_high_lenght = 0.02 #mm

lenght_min = 54.9 #for plot only
lenght_max = 55.1 #for plot only
lenght_bin = 40 #for plot only

#width_m = 51.50 #mm #Array
width_m = 3.12 #mm #Bar

e_low_width = 0.10 #mm
e_high_width = 0.10 #mm
width_min = 2.72 #for plot only #Bar
width_max = 3.42 #for plot only #Bar
width_bin = 80 #for plot only

thickness_m = 0. #mm
e_low_thickness = 0. #mm
e_high_thickness = 0. #mm
thickness_min = 0. #for plot only
thickness_max = 0. #for plot only
thickness_bin = 0. #for plot only

## Bars w/o ESR

if(type==1):
    #thickness_m = 4.05 #mm #Array
    thickness_m = 3.75 #mm #Bar
    e_low_thickness = 0.10 #mm
    e_high_thickness = 0.10 #mm
    thickness_min = 3.35 #for plot only #Bar
    thickness_max = 4.15 #for plot only #Bar    
    thickness_bin = 80 #for plot only

if(type==2):
    #thickness_m = 3.30 #mm #Array
    thickness_m = 3.00 #mm #Bar    
    e_low_thickness = 0.10 #mm
    e_high_thickness = 0.10 #mm
    #thickness_min = 2.90 #for plot only  #Array
    #thickness_max = 3.70 #for plot only  #Array
    thickness_min = 2.60 #for plot only #Bar
    thickness_max = 3.40 #for plot only #Bar   
    thickness_bin = 80 #for plot only

if(type==3):
    #thickness_m = 2.70 #mm #Array
    thickness_m = 2.40 #mm #Bar    
    e_low_thickness = 0.10 #mm
    e_high_thickness = 0.10 #mm
    #thickness_min = 2.30 #for plot only #Array
    #thickness_max = 3.10 #for plot only #Array
    thickness_min = 2.00 #for plot only #Bar 
    thickness_max = 2.80 #for plot only #Bar    
    thickness_bin = 80 #for plot only


#+++++++++++++++++++++++++
#     Reading data
#+++++++++++++++++++++++++


df_LS = pd.DataFrame(columns=['X', 'Y', 'Z'])
counter_LS = 0

df_LN = pd.DataFrame(columns=['X', 'Y', 'Z'])
counter_LN = 0

df_FS = pd.DataFrame(columns=['X', 'Y', 'Z'])
counter_FS = 0

df_LO = pd.DataFrame(columns=['X', 'Y', 'Z'])
counter_LO = 0

df_LE = pd.DataFrame(columns=['X', 'Y', 'Z'])
counter_LE = 0




for line in open(args.data,errors='ignore'):    
    line = line.rstrip()
    #line.strip()
    splitline = line.split()
    n_elements = len(splitline)
    if(n_elements<5):
        continue    
    
    n = x = y = z = 0
    n = splitline[0]
    x = splitline[1]
    y = splitline[2]
    z = splitline[3]

    side = splitline[0]

    if( ('_LS' in side) or ('_LN' in side) or ('_FS' in side) or ('_LO' in side) or ('_LE' in side) ):
        n = splitline[1]
        x = float(splitline[2])
        y = float(splitline[3])
        z = float(splitline[4])

    if('_LS' in side):
        counter_LS = 1
    if('_LN' in side):
        counter_LN = 1
    if('_FS' in side):
        counter_FS = 1
    if('_LO' in side):
        counter_LO = 1
    if('_LE' in side):
        counter_LE = 1
        
#++++++++++++++++++++++++++++++++++++++++++++++
#   HERE IS WHERE WE SELECT GALAXY POINTS
#++++++++++++++++++++++++++++++++++++++++++++++

    if(counter_LS>0 and counter_LS<19):
        values_to_add = {'X': x, 'Y': y, 'Z': z}
        row_to_add = pd.Series(values_to_add, name=n)
        df_LS = df_LS.append(row_to_add)
        counter_LS = counter_LS + 1

    if(counter_LN>0 and counter_LN<19):
        values_to_add = {'X': x, 'Y': y, 'Z': z}
        row_to_add = pd.Series(values_to_add, name=n)
        df_LN = df_LN.append(row_to_add)
        counter_LN = counter_LN + 1

    if(counter_FS>0 and counter_FS<19):         
        values_to_add = {'X': x, 'Y': y, 'Z': z}
        row_to_add = pd.Series(values_to_add, name=n)
        df_FS = df_FS.append(row_to_add)
        counter_FS = counter_FS + 1

    if(counter_LO>0 and counter_LO<5):
        values_to_add = {'X': x, 'Y': y, 'Z': z}
        row_to_add = pd.Series(values_to_add, name=n)
        df_LO = df_LO.append(row_to_add)
        counter_LO = counter_LO + 1

    if(counter_LE>0 and counter_LE<5):
        values_to_add = {'X': x, 'Y': y, 'Z': z}
        row_to_add = pd.Series(values_to_add, name=n)
        df_LE = df_LE.append(row_to_add)
        counter_LE = counter_LE + 1




#++++++++++++++++++++++++++++++++++++++++++++++++++
#      FINAL DATASETS TO ANALYZE
#++++++++++++++++++++++++++++++++++++++++++++++++++

df_LS = df_LS.astype({'X': float, 'Y': float, 'Z': float})
df_LN = df_LN.astype({'X': float, 'Y': float, 'Z': float})
df_FS = df_FS.astype({'X': float, 'Y': float, 'Z': float})
df_LO = df_LO.astype({'X': float, 'Y': float, 'Z': float})
df_LE = df_LE.astype({'X': float, 'Y': float, 'Z': float})

#++++++++++++++++++++++++++++++++++++++++++
# lenght: Single Bars in the array Style
#++++++++++++++++++++++++++++++++++++++++++


l_lenght = []
l1_lenght = []
l2_lenght = []


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   POINTS ASSOCIATION FOR SINGLE BAR lenght MEASUREMENT
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
print('++++++++++++++++++++++++++')
print('   lenght1 SUMMARY TABLE  ')
print('++++++++++++++++++++++++++')   
'''    
for point in range(1,5,2):
#for point in range(2,5,1):

    x_LE = float(df_LE.loc[str(point)]['X'])
    x_LO = float(df_LO.loc[str(point)]['X']) 
    lenght1 = x_LE - x_LO
    #print ('point, y_LS, y_LN, lenght :',point, y_LS, y_LN, lenght)
    l1_lenght.append(lenght1)

'''
    p = PrettyTable(['Point LE','Point LO','x_LE', 'x_LO','lenght1','Z LE','Z LO','Y LE','Y LO','X LE','X LO'])
    p.add_row([str(point), str(point),x_LE,x_LO,lenght1,df_LE.loc[str(point)]['Z'],df_LO.loc[str(point)]['Z'],df_LE.loc[str(point)]['Y'],df_LO.loc[str(point)]['Y'],df_LE.loc[str(point)]['X'],df_LO.loc[str(point)]['X']])

    print(p)
'''

'''
print('++++++++++++++++++++++++++')
print('   lenght2 SUMMARY TABLE  ')
print('++++++++++++++++++++++++++')   
'''

for point in range(2,5,2):
    x_LE = float(df_LE.loc[str(point)]['X'])
    x_LO = float(df_LO.loc[str(point)]['X']) 
    lenght2 = x_LE - x_LO
    #print ('point, y_LS, y_LN, lenght :',point, y_LS, y_LN, lenght)
    l2_lenght.append(lenght2)
    
  
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Combine lenght in two different ranges
#++++++++++++++++++++++++++++++++++++++++++++++++++

    l_lenght = l1_lenght + l2_lenght   
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
    p = PrettyTable(['Point LE','Point LO','x_LE', 'x_LO','lenght2','Z LE','Z LO','Y LE','Y LO','X LE','X LO'])
    p.add_row([str(point), str(point),x_LE,x_LO,lenght2,df_LE.loc[str(point)]['Z'],df_LO.loc[str(point)]['Z'],df_LE.loc[str(point)]['Y'],df_LO.loc[str(point)]['Y'],df_LE.loc[str(point)]['X'],df_LO.loc[str(point)]['X']])
    print(p)
'''
np_lenght_all = np.asarray(l_lenght)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#       SINGLE BARS MEAN 
#++++++++++++++++++++++++++++++++++++++++++++++++++

np_lenght = np.mean(np_lenght_all.reshape(-1, 2), axis=1) #NEW
#np_lenght = np.mean(np_lenght_all.reshape(-1, 1), axis=1) #OLD
np_lenght = np_lenght.round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#       SINGLE BARS STD
#++++++++++++++++++++++++++++++++++++++++++++++++++

np_lenght_mean = np_lenght.mean().round(3)
np_lenght_std = np_lenght.std().round(3)

#++++++++++++++++++++++++++++++++++++++++++
#     Bar lenght (Mean +/- std)
#++++++++++++++++++++++++++++++++++++++++++

lenght_mean = np_lenght.mean().round(3)
lenght_std = np_lenght.std().round(3)

#--------------------------------------------
# ADDITIONAL OUTPUT: OUT OF RANGE/TOLERANCE
#--------------------------------------------

n_lenght_outOfPlotRange = np.count_nonzero( (np_lenght < lenght_min) | (np_lenght > lenght_max) )
n_lenght_outOfTolerance = np.count_nonzero( (np_lenght < lenght_m-e_low_lenght) | (np_lenght > lenght_m+e_high_lenght) )

'''
print('')
print ('All lenght',np_lenght_all)
#print (np_lenght_all.reshape(-1, 2))
print ('Single bar Mean lenght :       ',np_lenght)
print('Bars Mean lenght :              ',lenght_mean)
print ('Single bar Mean lenght Std :   ',np_lenght_std)
print('Bars std lenght             :   ',lenght_std)
print('')


p = PrettyTable(['Bar','Bar lenght out Plot range','Bar lenght out of Tolerance'])
p.add_row([barcode.split('/')[1], str(n_lenght_outOfPlotRange),str(n_lenght_outOfTolerance)])  # this when we run it using runAll script
print(p)
print('')
'''

# ================
# ===  lenght  ===
# ================

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  lenght: Y -> MaxVar - Mean - Spread - Max array size
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++++++++++++++
#  Max. X variation on ovest/east side (lenght)
#++++++++++++++++++++++++++++++++++++++++++++++++++

max_x_LO = np.amax(df_LO['X'].to_numpy())
min_x_LO = np.amin(df_LO['X'].to_numpy())
delta_x_LO = (max_x_LO - min_x_LO).round(3)

max_x_LE = np.amax(df_LE['X'].to_numpy())
min_x_LE = np.amin(df_LE['X'].to_numpy())
delta_x_LE = (max_x_LE - min_x_LE).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Mean on X on 'ovest'/east side (lenght)
#++++++++++++++++++++++++++++++++++++++++++++++++++

mean_x_LO = (df_LO['X'].to_numpy()).mean().round(3)
mean_x_LE = (df_LE['X'].to_numpy()).mean().round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Std. dev. on X on 'ovest'/east side (lenght)
#++++++++++++++++++++++++++++++++++++++++++++++++++

std_x_LO = (df_LO['X'].to_numpy()).std().round(3)
std_x_LE = (df_LE['X'].to_numpy()).std().round(3)
std_deltax = round(mt.sqrt(std_x_LO**2+std_x_LE**2),3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Max. bar size along X (lenght)
#++++++++++++++++++++++++++++++++++++++++++++++++++

array_lenght_max = (max_x_LE - min_x_LO).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Average array size along X (lenght)
#++++++++++++++++++++++++++++++++++++++++++++++++++

array_lenght_mean = (mean_x_LE - mean_x_LO).round(3)
array_lenght_mean_std = round(mt.sqrt( (std_x_LO/mt.sqrt((df_LO['X'].to_numpy()).size))**2
                               + (std_x_LE/mt.sqrt((df_LE['X'].to_numpy()).size))**2  ) , 3)


#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Simulating Mitutoyo (lenght)
#++++++++++++++++++++++++++++++++++++++++++++++++++

np_mitutoyo_lenght_LO = max_x_LE - df_LO['X'].to_numpy()
np_mitutoyo_lenght_LE = df_LE['X'].to_numpy() - min_x_LO
np_mitutoyo_lenght = np_mitutoyo_lenght_LO
np_mitutoyo_lenght = np.concatenate([np_mitutoyo_lenght,np_mitutoyo_lenght_LE])

mitutoyo_array_lenght_mean = (np_mitutoyo_lenght.mean()).round(3)
mitutoyo_array_lenght_std = (np_mitutoyo_lenght.std()).round(3)


# =============
# === Width ===
# =============

np_width = df_FS['Z'].to_numpy()
np_width = np_width.round(3)

width_mean = np_width.mean().round(3)
width_std = np_width.std().round(3)


#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Max. Z variation on front side (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

max_z_FS = np.amax(df_FS['Z'].to_numpy())
min_z_FS = np.amin(df_FS['Z'].to_numpy())
delta_z_FS = (max_z_FS - min_z_FS).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Mean on Z on front side (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

mean_z_FS = (df_FS['Z'].to_numpy()).mean().round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Std. dev. on Z on front side (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

std_z_FS = (df_FS['Z'].to_numpy()).std().round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Max. bar size along Z (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

array_width_max = (max_z_FS - 0).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Average bar size along Z (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

array_width_mean = (mean_z_FS - 0).round(3)
array_width_mean_std = round( std_z_FS / mt.sqrt( (df_FS['Z'].to_numpy()).size ), 3 )


#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Simulating Mitutoyo (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

np_mitutoyo_width = df_FS['Z'].to_numpy()

mitutoyo_array_width_mean = (np_mitutoyo_width.mean()).round(3)
mitutoyo_array_width_std = (np_mitutoyo_width.std()).round(3)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    Saving width_var array in case of swap check
#++++++++++++++++++++++++++++++++++++++++++++++++++++++



width_var = [max_z_FS,   #0
             '',         #1
             min_z_FS,   #2
             '',         #3
             delta_z_FS, #4
             '',         #5
             mean_z_FS,  #6
             '',         #7
             std_z_FS,   #8
             '',         #9
             '',         #10
             array_width_max,  #11
             array_width_mean,  #12
             array_width_mean_std, #13
             mitutoyo_array_width_mean, #14
             mitutoyo_array_width_std]   #15





# =================
# === Thickness ===
# =================


#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Max. Y variation on north/sud side (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++

max_y_LS = np.amax(df_LS['Y'].to_numpy())
min_y_LS = np.amin(df_LS['Y'].to_numpy())
delta_y_LS = (max_y_LS - min_y_LS).round(3)

max_y_LN = np.amax(df_LN['Y'].to_numpy())
min_y_LN = np.amin(df_LN['Y'].to_numpy())
delta_y_LN = (max_y_LN - min_y_LN).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#   Mean on Y on north/sud side (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++

mean_y_LS = (df_LS['Y'].to_numpy()).mean().round(3)
mean_y_LN = (df_LN['Y'].to_numpy()).mean().round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Spread on Y on north/sud side (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++

std_y_LS = (df_LS['Y'].to_numpy()).std().round(3)
std_y_LN = (df_LN['Y'].to_numpy()).std().round(3)
std_deltay = round(mt.sqrt(std_y_LS**2+std_y_LN**2),3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Max. bar size along Y (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++

array_thickness_max = (max_y_LN - min_y_LS).round(3)


#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Average bar size along Y (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++

array_thickness_mean = (mean_y_LN - mean_y_LS).round(3)
array_thickness_mean_std = round(mt.sqrt( (std_y_LS/ mt.sqrt((df_LS['Y'].to_numpy()).size) )**2
                               + (std_y_LN/mt.sqrt((df_LN['Y'].to_numpy()).size) )**2  ) , 3)


#+++++++++++++++++++++++++++++++++++++++++++
# ******** GEOMETRY DEFINITION ******** 
#+++++++++++++++++++++++++++++++++++++++++++


geo=['geo1','geo2','geo3']

if (array_thickness_mean >= 3.3):
    geo=geo[0]
elif (array_thickness_mean < 2.6):
    geo=geo[2]
else:
    geo=geo[1]

    print('Geometry :',str(geo))
    print('')

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Simulating Mitutoyo (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++

np_mitutoyo_thickness_LS = max_y_LN - df_LS['Y'].to_numpy()
np_mitutoyo_thickness_LN = df_LN['Y'].to_numpy() - min_y_LS
np_mitutoyo_thickness = np_mitutoyo_thickness_LS
np_mitutoyo_thickness = np.concatenate([np_mitutoyo_thickness,np_mitutoyo_thickness_LN])

mitutoyo_array_thickness_mean = (np_mitutoyo_thickness.mean()).round(3)
mitutoyo_array_thickness_std = (np_mitutoyo_thickness.std()).round(3)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    Saving thickness_var array in case of swap check
#++++++++++++++++++++++++++++++++++++++++++++++++++++++


thickness_var = [max_y_LS, #0
                 max_y_LN, #1
                 min_y_LS, #2
                 min_y_LN, #3
                 delta_y_LS, #4
                 delta_y_LN, #5
                 mean_y_LS, #6
                 mean_y_LN, #7
                 std_y_LS, #8
                 std_y_LN, #9
                 std_deltay, #10
                 array_thickness_max, #11
                 array_thickness_mean, #12
                 array_thickness_mean_std, #13
                 mitutoyo_array_thickness_mean, #14
                 mitutoyo_array_thickness_std] #15

#+++++++++++++++++++++++++++++++
#   Check thickness & width
#+++++++++++++++++++++++++++++++
 

if (thickness_var[12] > 2.5 and thickness_var[12] < 3.3 ): 

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Did you fit the single crystal bar in the right position?')    
    '''
    print('++++++++++ Before swap ++++++++++++++++++')
    print('')
    print('THICKNESS :', thickness_var)
    print('WIDTH :', width_var)   
    print('thickness before =', thickness_var[12])
    print('width before =', width_var[12])
    '''
    if (array_thickness_mean > array_width_mean): #this is the real check


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Here is where we force to swap thickness with width
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
        
        thickness_var, width_var = width_var, thickness_var #swapping variables array

        print('NO the single crystal bar position in the frame was NOT CORRECT!') 
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
         
        '''
        print('++++++++++ After swap ++++++++++++++++++')
        print('')
        print('THICKNESS AFTER :', thickness_var)
        print('WIDTH AFTER:', width_var)
        print('thickness after =', thickness_var[12])
        print('width after =', width_var[12])
        print('++++++++++++++++++++++++++++++++++++++++')
        '''
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This follows the LYSO Array GALAXY logic: for consistency
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
        json_results_swap = {
            'runName': run.split('/')[-1]+'_BAR'+barcode+str('_')+tag,
            'id': barcode,
            #'producer': prod,
            #'geometry': geo,
            'time': date,
            #'L_bar_mu':  lenght_mean,
            #'L_bar_std':  lenght_std,
            'L_bar_mu': '',
            'L_bar_std': '',
            'L_maxVar_LS': '',
            'L_maxVar_LN': '',
            'L_std_LS': '',
            'L_std_LN': '',
            'L_std_tot': '',
            'L_max': '',
            'L_mean': array_lenght_mean,
            'L_mean_std': array_lenght_mean_std,
            'L_mean_mitu': '',
            'L_std_mitu': '',
    	    'W_maxVar_LO': '',
            'W_maxVar_LE': '',
            'W_std_LO': '',
            'W_std_LE': '',
            'W_std_tot': '',
            'W_max': '',
            'W_mean': width_var[12],
            'W_mean_std': width_var[13],
            #'W_mean': array_width_mean,
            #'W_mean_std': array_width_mean_std,
            'W_mean_mitu': '',
            'W_std_mitu': '',
            'T_maxVar_FS': '',
            'T_std_FS': '',
            'T_max': '',
            'T_mean': thickness_var[12],
            'T_mean_std': thickness_var[13],
            #'T_mean': array_thickness_mean,
            #'T_mean_std': array_thickness_mean_std,
            'T_mean_mitu': '',
            'T_std_mitu': '',
            'bar': '',
            'bar_lenght': '',
            'bar_lenght_std': '',
            'type': 'xtal'       
        }


        l_results_names_swap = ['runName','id','time','L_bar_mu','L_bar_std',
                           'L_maxVar_LS','L_maxVar_LN','L_std_LS','L_std_LN','L_std_tot','L_max','L_mean','L_mean_std',
                           'L_mean_mitu','L_std_mitu',
                           'W_maxVar_LO','W_maxVar_LE','W_std_LO','W_std_LE','W_std_tot','W_max','W_mean','W_mean_std',
                           'W_mean_mitu','W_std_mitu',
                           'T_maxVar_FS','T_std_FS','T_max','T_mean','T_mean_std','T_mean_mitu','T_std_mitu',
                           'bar','bar_length','bar_length_std','type']


        l_results_swap = [[run.split('/')[-1]+'_BAR'+barcode+str('_')+tag,barcode,date,'','',
                         '','','','','','',array_lenght_mean,array_lenght_mean_std,'','',
                         '','','','','','',width_var[12],width_var[13],'','',
                         '','','',thickness_var[12],thickness_var[13],'','','' ,'','','xtal']] 

       #+++++++++++++++++++
       #   SAVE csv FILE
       #+++++++++++++++++++
          
        #with open(str(args.xtalbar)+'_swap'+'.csv', 'w') as file:
        with open(run+str('_BAR')+barcode+str('_')+tag+str('_swap')+'.csv','w') as file:    
            writer = csv.writer(file, delimiter=',')
            # row by row         
            writer.writerow(l_results_names_swap)
            writer.writerows(l_results_swap)
       
        os.system('cp '+run+str('_BAR')+barcode+str('_')+tag+str('_swap')+'.csv /home/cmsdaq/MTDDB/uploader/files_to_upload/galaxy-bars/')
            
	#+++++++++++++++++++
	#  SAVE json FILE
	#+++++++++++++++++++

        #with open(str(args.xtalbar)+'_swap'+'.json', 'w') as json_file:
        with open(run+str('_BAR')+barcode+str('_')+tag+str('_swap')+'.json', 'w') as json_file:
            json.dump(json_results_swap, json_file, indent=4) 
            

	#++++++++++++++++++++++++
	#   SUMMARY TABLE
	#++++++++++++++++++++++++


            p = PrettyTable(['Bar',' Date&Time','Geometry'])
            p.add_row([barcode+str('_')+tag, str(timestamp),str(geo)])
            print(p)
            data = p.get_string()

            
            print('+++++++++++++')
            print('   lenght    ')
            print('+++++++++++++')

            p1 = PrettyTable(['Bar','Mean Bar Lenght along X (LE - LO)'])
            p1.add_row([barcode+str('_')+tag,str(array_lenght_mean)+ ' +/- '+ str(array_lenght_mean_std)+ ' mm'])

            print(p1)
            mean_lenght = p1.get_string()

            
            print('+++++++++++++++++')
            print('  Thickness swap ')
            print('+++++++++++++++++')

            p1 = PrettyTable(['Bar','Mean bar thickness along Z (FS - 0)'])
            p1.add_row([barcode+str('_')+tag,str(thickness_var[12])+ ' +/- '+ str(thickness_var[13])+ ' mm'])

            print(p1)
            mean_thickness_swap = p1.get_string()
            

            print('+++++++++++++++++')
            print('    Width swap   ')
            print('+++++++++++++++++')

            p1 = PrettyTable(['Bar','Mean bar width along Y (LN - LS)'])
            p1.add_row([barcode+str('_')+tag,str(width_var[12])+ ' +/- '+ str(width_var[13])+ ' mm'])

            print(p1)
            mean_width_swap = p1.get_string()



            #+++++++++++++++++++++++++++++++++
	    #   SAVING TABLE IN OUTPUT FILE
	    #+++++++++++++++++++++++++++++++++


            #with open(str(barcode)+str('_')+tag+'_output.txt','w') as f:
            with open(run+str('_BAR')+barcode+str('_')+tag+'_output.txt','w') as f:

            
            
                f.write(data) 
                f.write('\n') 
                f.write('++++++++++++++++++++++++++++\n')
                f.write('         lenght             \n')
                f.write('++++++++++++++++++++++++++++\n')
                f.write(mean_lenght)
                f.write('\n') 
                f.write('++++++++++++++++++++++++++++\n')
                f.write('         Width              \n')
                f.write('++++++++++++++++++++++++++++\n')
                f.write(mean_width_swap)
                f.write('\n')     
                f.write('++++++++++++++++++++++++++++\n')
                f.write('        Thickness           \n')
                f.write('++++++++++++++++++++++++++++\n')
                f.write(mean_thickness_swap)
                f.write('\n') 

           
             
            print('System Exiting ..........')
            print('Program End') 

            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Type 2: thickness > width ! --> Swapped Thickness and Width    ')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')              
            
            exit()

    else:
        #print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('YES the single crystal bar position in the frame was CORRECT!')
        print('Program Continue')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

     

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
#  This part is in case you measured bar in the right position
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 


#++++++++++++++++++++++++
#   SUMMARY TABLE
#++++++++++++++++++++++++

from prettytable import PrettyTable

p = PrettyTable(['Bar',' Date&Time','Geometry'])
p.add_row([barcode+str('_')+tag, str(timestamp),str(geo)])
print(p)
data = p.get_string()
print ('')


print('+++++++++++++')
print('   lenght    ')
print('+++++++++++++')


p1 = PrettyTable(['Bar','Mean bar lenght along X (LE - LO)'])
p1.add_row([barcode+str('_')+tag,str(array_lenght_mean)+ ' +/- '+ str(array_lenght_mean_std)+ ' mm'])

print(p1)
mean_lenght = p1.get_string()


print ('')


print('+++++++++++++')
print('  Thickness  ')
print('+++++++++++++')


p1 = PrettyTable(['Bar','Mean bar thickness along Y (LN - LS)'])
p1.add_row([barcode+str('_')+tag,str(array_thickness_mean)+ ' +/- '+ str(array_thickness_mean_std)+ ' mm'])

print(p1)
mean_thickness = p1.get_string()


print ('')



print('+++++++++++++')
print('    Width    ')
print('+++++++++++++')


p1 = PrettyTable(['Bar','Mean bar width along Z (FS - 0)'])
p1.add_row([barcode+str('_')+tag,str(array_width_mean)+ ' +/- '+ str(array_width_mean_std)+ ' mm'])

print(p1)
mean_width = p1.get_string()

print ('')

#+++++++++++++++++++++++++++++++++
#   SAVING TABLE IN OUTPUT FILE
#+++++++++++++++++++++++++++++++++


#with open(str(barcode)+str('_')+tag+ '_output'+'_output.txt','w') as f:
with open(run+str('_BAR')+barcode+str('_')+tag+'_output.txt','w') as f:


    f.write(data) 
    f.write('\n') 
    f.write('++++++++++++++++++++++++++++\n')
    f.write('         lenght             \n')
    f.write('++++++++++++++++++++++++++++\n')
    f.write(mean_lenght)
    f.write('\n') 
    f.write('++++++++++++++++++++++++++++\n')
    f.write('         Width              \n')
    f.write('++++++++++++++++++++++++++++\n')
    f.write(mean_width)
    f.write('\n')     
    f.write('++++++++++++++++++++++++++++\n')
    f.write('        Thickness           \n')
    f.write('++++++++++++++++++++++++++++\n')
    f.write(mean_thickness)
    f.write('\n') 




#++++++++++++++++++++++++++++++++++++++++++++++++++
#      WRITING OUTPUT IN CSV FILE
#++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This follows the LYSO Array GALAXY logic: for consistency
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

json_bar = {
    'runName': run.split('/')[-1]+'_BAR'+barcode+str('_')+tag,
    'id': barcode,
    #'producer': prod,
    #'geometry': geo,
    'time': date,
    'L_bar_mu': '',
    'L_bar_std': '',
    'L_maxVar_LS': '',
    'L_maxVar_LN': '',
    'L_std_LS': '',
    'L_std_LN': '',
    'L_std_tot': '',
    'L_max': '',
    'L_mean': array_lenght_mean,
    'L_mean_std': array_lenght_mean_std,
    'L_mean_mitu': '',
    'L_std_mitu': '',
    'W_maxVar_LO': '',
    'W_maxVar_LE': '',
    'W_std_LO': '',
    'W_std_LE': '',
    'W_std_tot': '',
    'W_max': '',
    'W_mean': array_width_mean,
    'W_mean_std': array_width_mean_std,
    'W_mean_mitu': '',
    'W_std_mitu': '',
    'T_maxVar_FS': '',
    'T_std_FS': '',
    'T_max': '',
    'T_mean': array_thickness_mean,
    'T_mean_std': array_thickness_mean_std,
    'T_mean_mitu': '',
    'T_std_mitu': '',
    'bar': '',
    'bar_lenght': '',
    'bar_lenght_std': '',
    'type': 'xtal'       
}


l_results_names = ['runName','id','time','L_bar_mu','L_bar_std',
                   'L_maxVar_LS','L_maxVar_LN','L_std_LS','L_std_LN','L_std_tot','L_max','L_mean','L_mean_std',
                   'L_mean_mitu','L_std_mitu',
                   'W_maxVar_LO','W_maxVar_LE','W_std_LO','W_std_LE','W_std_tot','W_max','W_mean','W_mean_std',
                   'W_mean_mitu','W_std_mitu',
                   'T_maxVar_FS','T_std_FS','T_max','T_mean','T_mean_std','T_mean_mitu','T_std_mitu',
                    'bar','bar_length','bar_length_std','type']


l_results = [[run.split('/')[-1]+'_BAR'+barcode+str('_')+tag,barcode,date,'','',
                         '','','','','','',array_lenght_mean,array_lenght_mean_std,'','',
                         '','','','','','',array_width_mean,array_width_mean_std,'','',
                         '','','',array_thickness_mean,array_thickness_mean_std,'','','' ,'','','xtal']]   


             
#+++++++++++++++++++
#  SAVE .csv FILE
#+++++++++++++++++++

#with open(str(args.xtalbar)+str('_')+tag+'.csv', 'w') as file:   
with open(run+str('_BAR')+barcode+str('_')+tag+'.csv','w') as file: 
    writer = csv.writer(file, delimiter=',')
    # row by row         
    writer.writerow(l_results_names)
    writer.writerows(l_results)

os.system('cp '+run+str('_BAR')+barcode+str('_')+tag+'.csv /home/cmsdaq/MTDDB/uploader/files_to_upload/galaxy-bars/')       
#+++++++++++++++++++
#SAVE json FILE
#+++++++++++++++++++

#with open(str(args.xtalbar)+str('_')+tag+'.json', 'w') as json_file:
with open(run+str('_BAR')+barcode+str('_')+tag+'.json','w') as json_file:
    json.dump(json_bar, json_file, indent=4) 

    
    print('+++++++++++++++++++++++++++++++++++++++++++++')
    print('Type 2: thickness < width ! -> No swap needed')
    print('+++++++++++++++++++++++++++++++++++++++++++++')



