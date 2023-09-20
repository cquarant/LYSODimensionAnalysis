import sys
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as plticker
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
from datetime import datetime
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import argparse
import glob
import timeit
from scipy.ndimage.interpolation import shift


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     HOW TO RUN OVER ONE SINGLE FILE
#  python3 Galaxy_raw_Data_Array_MTDDB.py --data ArrayData/812* --array 812
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

start = timeit.default_timer()

#+++++++++++++++++++++++++++++++++++++++++++
#  Parsing - Date - Producers definifition
#+++++++++++++++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()

parser.add_argument('--data',dest='data', help='data file *.TXT', type=str,required=True,default=False)
parser.add_argument('--array',dest='array',required=False,default=False)

args = parser.parse_args()


#+++++++++++++++++++++++++++++++++++++
# filename: 415_2022-05-04-12-52.txt
# Date and time as 2022-05-04-12-52
# Run001_416_2023-01-19-12-40_OLD.TXT 
#+++++++++++++++++++++++++++++++++++++
data = args.data
data = data[:-2]


info = data.split('_')


run = info[0]
args.array = barcode = info[1]

date = info[2]

tag = info[3]

date = date.replace('.T','')
data = data.replace('.T','')
tag = tag.replace('.T','')

print('Filename :', data)
print('Barcode :', barcode)
print('Tag :', tag)
print('RunNumber :', run)


#print('Barcode :', barcode.split('/')[2]) # this when we run it using runAll script otherwise we have to replace with print('Barcode :', barcode)
print('Date in filename :', date)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# convert datetime string into date,month,day and
# hours:minutes:and seconds format using strptime
#++++++++++++++++++++++++++++++++++++++++++++++++++

time = datetime.strptime(date, '%Y-%m-%d-%H-%M') #in use time format
timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  #new time format
print('Time after conversion :',timestamp)
print('Info LYSO Array: {}'.format(str(args)))


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

#length_m = 56.30 #mm  #old MS3/OPT
#length_min = 56.2 #for plot only #old MS3/OPT
#length_max = 56.4 #for plot only #old MS3/OPT


length_m = 55.0 #mm
e_low_length = 0.02 #mm
e_high_length = 0.02 #mm

length_min = 54.9 #for plot only
length_max = 55.1 #for plot only
length_bin = 40 #for plot only

width_m = 51.50 #mm
e_low_width = 0.10 #mm
e_high_width = 0.10 #mm
width_min = 51.10 #for plot only
width_max = 51.90 #for plot only
width_bin = 80 #for plot only

thickness_m = 0. #mm
e_low_thickness = 0. #mm
e_high_thickness = 0. #mm
thickness_min = 0. #for plot only
thickness_max = 0. #for plot only
thickness_bin = 0. #for plot only

if(type==1):
    thickness_m = 4.05 #mm
    e_low_thickness = 0.10 #mm
    e_high_thickness = 0.10 #mm
    thickness_min = 3.65 #for plot only
    thickness_max = 4.45 #for plot only
    thickness_bin = 80 #for plot only

if(type==2):
    thickness_m = 3.30 #mm
    e_low_thickness = 0.10 #mm
    e_high_thickness = 0.10 #mm
    thickness_min = 2.90 #for plot only
    thickness_max = 3.70 #for plot only
    thickness_bin = 80 #for plot only

if(type==3):
    thickness_m = 2.70 #mm
    e_low_thickness = 0.10 #mm
    e_high_thickness = 0.10 #mm
    thickness_min = 2.30 #for plot only
    thickness_max = 3.10 #for plot only
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

    if(counter_LS>0 and counter_LS<40):      
        values_to_add = {'X': x, 'Y': y, 'Z': z}
        row_to_add = pd.Series(values_to_add, name=n)
        df_LS = df_LS.append(row_to_add)
        counter_LS = counter_LS + 1

    if(counter_LN>0 and counter_LN<40):
        values_to_add = {'X': x, 'Y': y, 'Z': z}
        row_to_add = pd.Series(values_to_add, name=n)
        df_LN = df_LN.append(row_to_add)
        counter_LN = counter_LN + 1

    if(counter_FS>0 and counter_FS<104): 
        values_to_add = {'X': x, 'Y': y, 'Z': z}
        row_to_add = pd.Series(values_to_add, name=n)
        df_FS = df_FS.append(row_to_add)
        counter_FS = counter_FS + 1

    if(counter_LO>0 and counter_LO<12):
        values_to_add = {'X': x, 'Y': y, 'Z': z}
        row_to_add = pd.Series(values_to_add, name=n)
        df_LO = df_LO.append(row_to_add)
        counter_LO = counter_LO + 1

    if(counter_LE>0 and counter_LE<12):
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


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   REORDERED DATASETS TO BE CONSISTENT WITH THE OLD CODE
#        LS -> ascending X      LN -> descending X
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('LS inverted')
df_LS = df_LS.reindex(['37','38','39','34','35','36','31','32','33','28','29','30','25','26','27','22','23','24',
                       '19','20','21','16','17','18','13','14','15','10','11','12','7','8','9','4','5','6',
                       '1','2','3'])
df_LS.index = pd.RangeIndex(1,1 + len(df_LS))

print(df_LS)

df_LN = df_LN.reindex(['37','38','39','34','35','36','31','32','33','28','29','30','25','26','27','22','23','24',
                       '19','20','21','16','17','18','13','14','15','10','11','12','7','8','9','4','5','6',
                       '1','2','3'])
df_LN.index = pd.RangeIndex(1,1 + len(df_LN))
print('LN inverted')
print(df_LN)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  CHECK REORDERED DATASETS: uncomment this if you want the print out 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#print ('LATO SUD')
#print(df_LS)
#print ('LATO NORD')
#print(df_LN)


#+++++++++++++++++++++++++++++++
#   VERSION WITH WRAPPING
#+++++++++++++++++++++++++++++++

l_length = []  #l1_length+l2_length
l1_length = [] #length 1st row of measurements
l2_length = [] #length 2nd row of measurements
wrap_length =[] #length wrapping included

#+++++++++++++++++++++++++++++++++++++++++++++++++++++
# South and North definition for plots LS-LN length
#+++++++++++++++++++++++++++++++++++++++++++++++++++++
y_sud1 = [] 
y_nord1 = []
y_sud2 = []
y_nord2 = []
y_sud = []
y_nord = []

south_side = []
north_side = []

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   POINTS ASSOCIATION FOR WRAPPING EXCESS MEASUREMENT
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
print('++++++++++++++++++++++++++')
print('  WRAPPING SUMMARY TABLE  ')
print('++++++++++++++++++++++++++')
'''
for point in range(3,31,3):
    
    y_wrap_LS = float(df_LS.loc[point]['Y'])
    y_wrap_LN = float(df_LN.loc[33-point]['Y']) 
    
    wrapping =  y_wrap_LN - y_wrap_LS
        
    wrap_length.append(wrapping)
        
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  WRAPPING SUMMARY TABLE: comment this if you don't want the print out of points association 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
    p = PrettyTable(['Point LS','Point LN','y_wrap_LS', 'y_wrap_LN','length w/wrapping','Z LS','Z LN','X LS','X LN'])
    p.add_row([point, (33-point),y_wrap_LS,y_wrap_LN,wrapping,df_LS.loc[(point)]['Z'],df_LN.loc[(33-point)]['Z'],df_LS.loc[(point)]['X'],df_LN.loc[(33-point)]['X']])
    print(p)
'''
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   POINTS ASSOCIATION FOR SINGLE BAR length MEASUREMENT
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
print('++++++++++++++++++++++++++')
print('   length1 SUMMARY TABLE  ')
print('++++++++++++++++++++++++++')   
'''    
for point in range(1,31,3):
    y_LS = float(df_LS.loc[(point)]['Y'])
    y_LN = float(df_LN.loc[(29-point)]['Y']) 

    length1 = y_LN - y_LS
    print('point, 29-point, y1_LS, y1_LN, l1_length')
    print(point, 29-point, y_LS, y_LN,length1)

    l1_length.append(length1)    
    
#++++++++++++++++++++++++++++++++++++++++++++++++++    
#    This is just for length histo/plot LS vs LN
#++++++++++++++++++++++++++++++++++++++++++++++++++

    sud1 = y_LS
    nord1 = y_LN   
    y_sud1.append(sud1)
    y_nord1.append(nord1)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  length1 SUMMARY TABLE: comment this if you don't want the print out of points association 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''       
    p = PrettyTable(['Point LS','Point LN','y_LS', 'y_LN','length1','Z LS','Z LN','X LS','X LN'])
    p.add_row([(point), (29-point),y_LS,y_LN,length1,df_LS.loc[(point)]['Z'],df_LN.loc[(29-point)]['Z'],df_LS.loc[(point)]['X'],df_LN.loc[(29-point)]['X']])
    print(p)
'''
'''
print('++++++++++++++++++++++++++')
print('   length2 SUMMARY TABLE  ')
print('++++++++++++++++++++++++++')   
'''
for point1 in range(2,31,3):
    y_LS = float(df_LS.loc[(point1)]['Y'])
    y_LN = float(df_LN.loc[(31-point1)]['Y'])  
    length2 = y_LN - y_LS
    l2_length.append(length2)

    print('point1, 31-point1, y2_LS, y2_LN, l2_length')
    print(point1, 31-point1, y_LS, y_LN,length2)
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
#    This is just for length histo/plot LS vs LN
#++++++++++++++++++++++++++++++++++++++++++++++++++
    sud2 = y_LS
    nord2 = y_LN   
    y_sud2.append(sud2)
    y_nord2.append(nord2)    
    
#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Combine length in two different ranges
#++++++++++++++++++++++++++++++++++++++++++++++++++

    l_length = l1_length + l2_length
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  length2 SUMMARY TABLE: comment this if you don't want the print out of points association 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
    p = PrettyTable(['Point LS','Point LN','y_LS', 'y_LN','length2','Z LS','Z LN','X LS','X LN'])
    p.add_row([(point1), (31-point1),y_LS,y_LN,length2,df_LS.loc[(point1)]['Z'],df_LN.loc[(31-point1)]['Z'],df_LS.loc[(point1)]['X'],df_LN.loc[(31-point1)]['X']])
    print(p)
'''
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Useful printout to check step by step if all is working fine: uncomment this if you want the print out 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#print('++++++++++++++++++++++++++++++++++++')
#print('length 1st set of points:',l1_length)
#print('++++++++++++++++++++++++++++++++++++')
#print('length 2nd set of points:',l2_length)
#print('++++++++++++++++++++++++++++++++++++')


#++++++++++++++++++++++++++++++++++++
#   This is for LS-LN Plots
#++++++++++++++++++++++++++++++++++++
south_side1 = np.array(y_sud1)
south_side2 = np.array(y_sud2)
north_side1 = np.array(y_nord1)
north_side2 = np.array(y_nord2)
south_side = []
north_side = []

south_side = np.concatenate((south_side1, south_side2))
north_side = np.concatenate((north_side1, north_side2))


#++++++++++++++++++++++++++++++
# Create some mock data
#++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         REMINDER: points used to compute the single bars length
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   south_side1 = np.array(y_sud1)  # 1st set of points on LS
#   south_side2 = np.array(y_sud2)  # 2nd set of points on LS
#   north_side1 = np.array(y_nord1) # 1st set of points on LN
#   north_side2 = np.array(y_nord2) # 2nd set of points on LN
#   south_side = []
#   north_side = []
#   south_side = np.concatenate((south_side1, south_side2)) #combine the 2 set of measurements on LS
#   north_side = np.concatenate((north_side1, north_side2)) #combine the 2 set of measurements on LN
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

sud = south_side
nord = north_side

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Useful printout to check step by step if all is working fine: uncomment this if you want the print out 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 
#+++++++++++++++++++++++++++++++++++
# length: Single Bars in the array
#+++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#              REMINDER: 
#      l_length = []  #l1_length+l2_length
#     l1_length = [] #length 1st row of measurements
#     l2_length = [] #length 2nd row of measurements
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

np_length_all = np.asarray(l_length)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#       SINGLE BARS MEAN 
#++++++++++++++++++++++++++++++++++++++++++++++++++
np_length = np.mean(np_length_all.reshape(2, 10), axis=0) #.reshape(2, 10) --> 10 columns and 2 rows
np_length = np_length.round(3)

print('+++++++++++++++++++++')
print('l_length:')
print(l_length)
print('+++++++++++++++++++++')
print('np_length_all:')
print(np_length_all)
print('+++++++++++++++++++++')
print('np_length_reshaped:')
print(np_length_all.reshape(-1, 2))
print('+++++++++++++++++++++')
print('np_length:')
print(np_length)
print('+++++++++++++++++++++')

#++++++++++++++++++++++++++++++++++++++++++++++++++
#       SINGLE BARS STD
#++++++++++++++++++++++++++++++++++++++++++++++++++
np_length_std = np.std(np_length_all.reshape(-1, 2), axis=1) #.reshape(-1, 2) --> 2 columns and n rows
np_length_std = np_length_std.round(3)
#print(np_length_std)

np_wrap_length_all = np.asarray(wrap_length)
np_wrap_length = np.mean(np_wrap_length_all.reshape(-1, 2), axis=1)
np_wrap_length = np_wrap_length.round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#   This is just for length histo/plot LS vs LN
#++++++++++++++++++++++++++++++++++++++++++++++++++
np_lato_sud = np.asarray(y_sud)
np_lato_nord = np.asarray(y_nord)
np_yLS = np_lato_sud.round(3)
np_yLN = np_lato_nord.round(3)

np_lato_sud = np_lato_sud.reshape(-1, 1)
np_lato_nord = np_lato_nord.reshape(-1, 1)

#++++++++++++++++++++++++++++++++++++++++++
#      Bar length (Mean +/- std)
#++++++++++++++++++++++++++++++++++++++++++
length_mean = np_length.mean().round(3)
length_std = np_length.std().round(3)

#++++++++++++++++++++++++++++++++++++++++++
#          With Wrap length
#++++++++++++++++++++++++++++++++++++++++++
wrap_length_mean = np_wrap_length.mean().round(3)
wrap_length_std = np_wrap_length.std().round(3)

#+++++++++++++++++++++++++++++++++++++++++
#         Only wrap length
#+++++++++++++++++++++++++++++++++++++++++
wrap = (wrap_length_mean - length_mean).round(3)

print('------------------------------------') 
print('  Wrapping Excess:',str(wrap) +' mm',)
print('------------------------------------') 
print('')


#-------------------------------------------------------------------------------------
# ADDITIONAL OUTPUT: OUT OF RANGE/TOLERANCE: uncomment this if you want the print out 
#-------------------------------------------------------------------------------------

n_length_outOfPlotRange = np.count_nonzero( (np_length < length_min) | (np_length > length_max) )
n_length_outOfTolerance = np.count_nonzero( (np_length < length_m-e_low_length) | (np_length > length_m+e_high_length))


#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print ('Bars out plot range:',n_length_outOfPlotRange)
#print ('Bars out tolerance:', n_length_outOfTolerance)
#print ('Number of points:',np_length_all.size)
#print ('Number of points after mean:',np_length.size)
#print ('All length:',np_length_all)
##print (np_length_all.reshape(-1, 2))
#print ('Single bar Mean length :       ',np_length)
#print('Bars Mean length :              ',length_mean)
#print ('Single bar Mean length Std :   ',np_length_std)
#print('Bars std length             :   ',length_std)
#print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

p = PrettyTable(['Array','Bar length out Plot range','Bar length out of Tolerance'])
p.add_row([barcode+str('_')+tag, str(n_length_outOfPlotRange),str(n_length_outOfTolerance)])
print(p)
print('')


#+++++++++++++++++++++++++++++++++++++++++++++++++++++
#  PLOT 2: BarOutRange - single bar out of tolerance
#+++++++++++++++++++++++++++++++++++++++++++++++++++++

fig, ax = plt.subplots()
ax.plot(np_length,linestyle = 'dashed', marker = 'o', label='Bar length')

# Define bbox style
box_style=dict(boxstyle='round', facecolor='green', alpha=0.3)
box_style1=dict(boxstyle='round', facecolor='yellow', alpha=0.5)


plt.text(0.3, 55.06, '# Bars out plot range = ' + str(n_length_outOfPlotRange),fontsize = 10,bbox=box_style)
plt.text(0.3, 55.08, '# Bars out of tolerance = ' + str(n_length_outOfTolerance), fontsize = 10,bbox=box_style1)


plt.xlim([-3.5,12.5]) #to see all bars


plt.ylim(length_min,length_max)
plt.grid()
plt.legend()

plt.ylabel('length [mm]')
plt.xlabel('# Bars')

bars = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']

ax.set_xticks(np.arange(-3,len(np_length)+3,1)) 
ax.set_xticklabels(bars)

color = 'tab:red'

#-----> For Production
x1, y1 = [-3.5, 12.5], [54.67, 54.67]
x2, y2 = [-3.5, 12.5], [54.65, 54.65]
x3, y3 = [-3.5, 12.5], [54.73, 54.73]

#------> For MS3 and OPT
#x1, y1 = [-3.5, 12.5], [55., 55.]
#x2, y2 = [-3.5, 12.5], [54.98, 54.98]
#x3, y3 = [-3.5, 12.5], [55.02, 55.02]



plt.plot(x1, y1,color=color,linestyle='dashed')
plt.plot(x2, y2,color='k',linestyle='dashed')
plt.plot(x3, y3,color='k',linestyle='dashed')

plt.title('CMS MTD' + str(args.array) + ' - Bar length')
fig.savefig(run+str('_ARRAY')+barcode+str('_')+tag+'_BarOutRange.pdf',bbox_inches='tight')
#fig.savefig(str(args.array)+str('_')+tag+'_BarOutRange.pdf',bbox_inches='tight')

# Set axes limit
plt.ylim(54.9,55.1)
plt.ylim(length_min,length_max)
#plt.grid()
plt.legend()
#plt.show() #uncomment this if you want display plots while running code


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  length: Y -> MaxVar - Mean - Spread - Max array size
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  - with 39 points used to compute wrapping excess -
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Max. Y variation on north/sud side (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++


max_y_LS = np.amax(df_LS['Y'].to_numpy()) #considering wrapping
min_y_LS = np.amin(df_LS['Y'].to_numpy()) #considering wrapping

delta_y_LS = (max_y_LS - min_y_LS).round(3)

max_y_LN = np.amax(df_LN['Y'].to_numpy()) #considering wrapping
min_y_LN = np.amin(df_LN['Y'].to_numpy()) #considering wrapping

delta_y_LN = (max_y_LN - min_y_LN).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#      Mean on Y on north/sud side (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++
mean_y_LS = (df_LS['Y'].to_numpy()).mean().round(3)
mean_y_LN = (df_LN['Y'].to_numpy()).mean().round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Spread on Y on north/sud side (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++
std_y_LS = (df_LS['Y'].to_numpy()).std().round(3)
std_y_LN = (df_LN['Y'].to_numpy()).std().round(3)
std_deltay = round(mt.sqrt(std_y_LS**2+std_y_LN**2),3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Max. array size along Y (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++
array_length_max = (max_y_LN - min_y_LS).round(3)

'''
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('------ with wrapping and all points (39 points on each side) ------')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
p = PrettyTable(['Array','MaxVar Y LS','MaxVar Y LN','Mean Y LS','Mean Y LN','Spread Y LS/LN', 'Max array length size (Y)' ])
p.add_row([barcode.split('/')[1], str(delta_y_LS)+' mm', str(delta_y_LN)+' mm', str(mean_y_LS)+' mm',str(mean_y_LN)+' mm',str(std_deltay)+' mm',str(array_length_max)+' mm'])
print(p)
print('')
print('MAX_LN:',max_y_LN)
print('MIN_LN:',min_y_LN)
print('MAXVAR_LN:',delta_y_LN)
print('')
'''

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  length: Y -> MaxVar - Mean - Spread - Max array size
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  - w/o points used to compute wrapping excess (26 points) -
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    Max. Y variation on north/sud side (length) excluding points used to compute wrapping excess
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create some mock data for all point collected along LS and LN - 26 points on each side -

sud = df_LS['Y'].to_numpy() #considering wrapping
print('sud with wrapping: ', sud, len(sud))
sud = np.delete(sud,(2,5,8,11,14,17,20,23,26,29,32,35,38),axis=0) #not considering wrapping
print('sud w/o wrapping: ', sud)
nord = df_LN['Y'].to_numpy() #considering wrapping
nord = np.delete(nord,(2,5,8,11,14,17,20,23,26,29,32,35,38),axis=0) #not considering wrapping

#-------------------------------------------------------------------------------------
# ADDITIONAL OUTPUT: uncomment this if you want the print out 
#-------------------------------------------------------------------------------------

#print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print('Lato Sud considering w/o points for wrapping')
#print(sud)
#print('Lato Nord considering w/o points for wrapping')
#print(nord)
#print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print('')


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Test: plot for all 26 points collected on LS and LN - w/o points for wrapping
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#------------------------------------------
#    TEST MAX VAR excluding wrapping
#------------------------------------------
max_y_LS_nowr = np.amax(sud)
min_y_LS_nowr = np.amin(sud)
np.sort(sud)
print('sud:', np.sort(sud))
delta_y_LS_nowr = (max_y_LS_nowr - min_y_LS_nowr).round(3)

max_y_LN_nowr = np.amax(nord)
min_y_LN_nowr = np.amin(nord)

delta_y_LN_nowr= (max_y_LN_nowr - min_y_LN_nowr).round(3)

print('HERE ARE THE LMAXVAR RESULTS')
print('LN: ', delta_y_LN_nowr)
print('LS: ', delta_y_LS_nowr)
#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Mean on Y on north/sud side (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++
mean_y_LS_nowr = (sud.mean().round(3))
mean_y_LN_nowr = (nord.mean().round(3))
#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Spread on Y on north/sud side (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++
std_y_LS_nowr = (sud.std().round(3))
std_y_LN_nowr = (nord.std().round(3))
std_deltay_nowr = round(mt.sqrt(std_y_LS_nowr**2+std_y_LN_nowr**2),3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Max. array size along Y (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++
array_length_max_nowr = (max_y_LN_nowr - min_y_LS_nowr).round(3)

'''
print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
print('------ w/o wrapping (26 points on each side) ------')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++')

p = PrettyTable(['Array','MaxVar Y LS','MaxVar Y LN','Mean Y LS','Mean Y LN','Spread Y LS/LN', 'Max array length size (Y)' ])
p.add_row([barcode.split('/')[1], str(delta_y_LS_nowr)+' mm', str(delta_y_LN_nowr)+' mm', str(mean_y_LS_nowr)+' mm',str(mean_y_LN_nowr)+' mm',str(std_deltay_nowr)+' mm',str(array_length_max_nowr)+' mm'])
print(p)
print('')
print('MAX_LN_nowr:',max_y_LN_nowr)
print('MIN_LN_nowr:',min_y_LN_nowr)
print('MAXVAR_LN_nowr:',delta_y_LN_nowr)
print('')
'''

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create some mock data for all point collected along LS and LN - 39 points on each side -
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

sud = df_LS['Y'].to_numpy() #considering wrapping
           
nord = df_LN['Y'].to_numpy() #considering wrapping

#------------------------------------------
#    TEST MAX VAR including wrapping
#------------------------------------------
max_y_LS = np.amax(sud)
min_y_LS = np.amin(sud)

delta_y_LS = (max_y_LS - min_y_LS).round(3)

max_y_LN = np.amax(nord)
min_y_LN = np.amin(nord)
delta_y_LN = (max_y_LN - min_y_LN).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Mean on Y on north/sud side (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++
mean_y_LS = (sud.mean().round(3))
mean_y_LN = (nord.mean().round(3))
#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Spread on Y on north/sud side (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++
std_y_LS = (sud.std().round(3))
std_y_LN = (nord.std().round(3))
std_deltay = round(mt.sqrt(std_y_LS**2+std_y_LN**2),3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#    Max. array size along Y (length)
#++++++++++++++++++++++++++++++++++++++++++++++++++
array_length_max = (max_y_LN - min_y_LS).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Stuff for new plot upon Paolo's request  - FOR VENDORS -
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   Removing points used to compute wrapping measurements
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

df1_LS = df_LS.drop([3,6,9,12,15,18,21,23,27,30,33,36,39])
df1_LN = df_LN.drop([3,6,9,12,15,18,21,23,27,30,33,36,39])

#-------------------------------------------------------------------------------------
# ADDITIONAL OUTPUT: uncomment this if you want the print out 
#-------------------------------------------------------------------------------------

#print('++++++++++++++++++++++++++++++')
#print('Initial dataset for LS',df_LS)
#print('-----------------------------')
#print('Initial dataset for LN',df_LN)
#print('-----------------------------')
#print('South Side Points',df1_LS)
#print('-----------------------------')
#print('North Side Points',df1_LN)
#print('-----------------------------')


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#      New dataset after removing points used to compute wrapping measurements
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

south_side = []
north_side = []

LS = df1_LS['Y']
LN = df1_LN['Y']

south_side.append(LS)
north_side.append(LN)

sud = np.array(LS)
print('Original sud', sud, len(sud))
nord = np.array(north_side)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Compute Mean Value for the two points/measurements on each side (LS-LN) 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("sud plot 3 before average: ", sud)
n=2
sud = np.average(sud.reshape(-1, n), axis=1)
nord = np.average(nord.reshape(-1, n), axis=1)
print('Sud after averages: ',sud)
sud_min = np.amin(sud)
sud_max = np.amax(sud)
nord_min = np.amin(nord)
nord_max= np.amax(nord)

nord = nord[::-1]

sud_misalign = sud - sud.mean()
nord_misalign = nord - nord.mean()

# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # PLOT 3 LN-LS_MaxVar_13_measured_points:  New plot upon Paolo's request  - FOR VENDORS - (mean of 26 points w/o wrapping)
# #                 VERSION WITH DIFFERENT Y AXES RANGE FOR LN AND LS
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# fig, ax1 = plt.subplots()

# #++++++++++++++++++++++++
# #    South Side
# #++++++++++++++++++++++++
# sud_misalign = np.append(sud_misalign, [np.nan,np.nan,np.nan])
# sud_x_index = np.arange(0,16,1)

# color = 'tab:blue'

# ax1.set_xlabel('# Bar')
# ax1.set_ylabel('South Side [mm]', color=color)
# ax1.plot(sud_x_index, sud_misalign, color=color,linestyle='dashed', marker='o')
# loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
# ax1.xaxis.set_major_locator(loc)
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.grid()

# # points = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']
# # ax1.set_xticklabels(points)
# #++++++++++++++++++++++++
# #    North Side
# #++++++++++++++++++++++++

# nord_x_index = np.arange(3,16,1)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# # ax2.invert_xaxis() # to invert x axis on LNax2.set_xticks(np.arange(-3,len(sud),1)) #to include first and last 3 bars
# # # ax2.set_xticks(np.arange(-3,len(sud),1)) #to include first and last 3 bars

# color = 'tab:red'
# ax2.set_ylabel('North Side [mm]', color=color)  # we already handled the x-label with ax1
# ax2.plot(nord_x_index,nord_misalign, '-r',color=color,linestyle='dashed', marker='o',label='North Side')

# # Adding legend
# fig.legend(loc='upper left', bbox_to_anchor=(0.03,0.15), bbox_transform=ax1.transAxes)
# ax2.tick_params(axis='y', labelcolor=color)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.suptitle('CMS MTD' + str(args.array) + ' - North-South Side Misalignment', y=1.02,x=0.5)

# #Adding labels
# #plt.subplots_adjust(hspace=1.5)
# plt.savefig(str(args.array)+str('_')+tag+'_LN-LS_MaxVar_13_measured_points.pdf',bbox_inches='tight')
# #plt.show() #uncomment this if you want display plots while running code

# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # PLOT 4 LN-LS_MaxVar_13_measured_points:  Same as plot 3 but only 10 central bars
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# fig, ax1 = plt.subplots()

# #++++++++++++++++++++++++
# #    South Side
# #++++++++++++++++++++++++
# sud_misalign = np.delete(sud_misalign, [0,1,2,13,14,15], None)
# sud_x_index = np.arange(3,13,1)

# color = 'tab:blue'
# ax1.set_xlabel('# Bar')
# ax1.set_ylabel('South Side mis-alignment [mm]', color=color)
# ax1.plot(sud_x_index, sud_misalign, color=color,linestyle='dashed', marker='o')
# loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
# ax1.xaxis.set_major_locator(loc)
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.grid()

# #++++++++++++++++++++++++
# #    North Side
# #++++++++++++++++++++++++
# nord_misalign = np.delete(nord_misalign, [10,11,12], None)
# nord_x_index = np.arange(3,13,1)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:red'
# ax2.set_ylabel('North Side mis-alignment [mm]', color=color)  # we already handled the x-label with ax1
# ax2.plot(nord_x_index,nord_misalign, '-r',color=color,linestyle='dashed', marker='o',label='North Side')

# # Adding legend
# fig.legend(loc='upper left', bbox_to_anchor=(0.03,0.15), bbox_transform=ax1.transAxes)
# ax2.tick_params(axis='y', labelcolor=color)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.suptitle('CMS MTD' + str(args.array) + ' - North-South Side Misalignment', y=1.02,x=0.5)

# #Adding labels
# #plt.subplots_adjust(hspace=1.5)
# plt.savefig(str(args.array)+str('_')+tag+'_LN-LS_MaxVar_centra_bars.png',bbox_inches='tight')
# #plt.show() #uncomment this if you want display plots while running code

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PLOT 3 LN-LS_MaxVar_13_measured_points:  New plot upon Paolo's request  - FOR VENDORS - (mean of 26 points w/o wrapping)
#                               VERSION WITH FIXED Y AXIS RANGE
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

fig, ax1 = plt.subplots()

#++++++++++++++++++++++++
#    South Side
#++++++++++++++++++++++++
sud_misalign = np.append([np.nan,np.nan,np.nan], sud_misalign)
sud_x_index = np.arange(0,16,1)

color = 'tab:red'
ax1.plot(sud_x_index, sud_misalign, color=color,linestyle='dashed', marker='o',label='Barcode Side') 

#++++++++++++++++++++++++
#    North Side
#++++++++++++++++++++++++

nord_x_index = np.arange(0,13,1)
color = 'tab:blue'
ax1.plot(nord_x_index,nord_misalign, '-r',color=color,linestyle='dashed', marker='o',label='Opposite to Barcode Side')

# Labels and appearance
ax1.grid()

ax1.set_xlabel('# Bar')
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax1.xaxis.set_major_locator(loc)

ax1.set_ylabel('Mis-alignment [mm]')
plt.ylim([-0.09,0.09]) # fix y range

ax1.text(0.1,-0.038,'MTD tolerance (0.060 mm)',color='r')
plt.axhline(y = 0.03, color = 'r', linestyle = '--') # MTD acceptance
plt.axhline(y =-0.03, color = 'r', linestyle = '--') # MTD acceptance

# Adding legend
fig.legend(loc='upper left', bbox_to_anchor=(0.03,0.15), bbox_transform=ax1.transAxes)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.suptitle('CMS MTD' + str(args.array) + ' - North-South Side Misalignment', y=1.02,x=0.5)

# Plot show/saving
plt.savefig(run+"_"+date+"_"+str(args.array)+str('_')+tag+'_LN-LS_MaxVar_13_measured_points.png',bbox_inches='tight')
#plt.show() #uncomment this if you want display plots while running code

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PLOT 4 LN-LS_MaxVar_central:  Same as plot 3 but only 10 central bars
#                  VERSION WITH FIXED Y AXIS RANGE
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

fig, ax1 = plt.subplots()

#++++++++++++++++++++++++
#    South Side
#++++++++++++++++++++++++
sud_misalign = np.delete(sud_misalign, [0,1,2,13,14,15], None)
sud_x_index = np.arange(3,13,1)

color = 'tab:blue'
ax1.plot(sud_x_index, sud_misalign, color=color,linestyle='dashed', marker='o',label='Barcode Side')

#++++++++++++++++++++++++
#    North Side
#++++++++++++++++++++++++

nord_misalign = np.delete(nord_misalign, [10,11,12], None)
nord_x_index = np.arange(3,13,1)

color = 'tab:red'
ax1.plot(nord_x_index,nord_misalign, '-r',color=color,linestyle='dashed', marker='o',label='Opposite to Barcode Side')

# Labels and appearance
ax1.grid()

ax1.set_xlabel('# Bar')
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax1.xaxis.set_major_locator(loc)

ax1.set_ylabel('Mis-alignment [mm]')
plt.ylim([-0.09,0.09]) # fix y range

ax1.text(3.1,-0.038,'MTD tolerance (0.060 mm)',color='r')
plt.axhline(y = 0.03, color = 'r', linestyle = '--') # MTD acceptance
plt.axhline(y =-0.03, color = 'r', linestyle = '--') # MTD acceptance

# Adding legend
fig.legend(loc='upper left', bbox_to_anchor=(0.03,0.15), bbox_transform=ax1.transAxes)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.suptitle('CMS MTD' + str(args.array) + ' - North-South Side Misalignment', y=1.02,x=0.5)

# Plot show/saving
plt.savefig(run+"_"+date+"_"+str(args.array)+str('_')+tag+'_LN-LS_MaxVar_central.png',bbox_inches='tight')
#plt.show() #uncomment this if you want display plots while running code


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('    New mean dataset after removing points used to compute wrapping measurements: USE ONLY to match points in Paolo plot     ')
print('  Mean computed over 26 points --> plotted 13 points for each side corresponding to 13 single bars we can measure per array  ')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

# p2 = PrettyTable(['Array','Min LS','Max LS', 'MaxVar LS','Min LN','Max LN', 'MaxVar LN','MaxVar mean','Max length'])
# p2.add_row([barcode.split('/')[1],str(sud_min.round(3))+' mm',str(sud_max.round(3))+' mm',str((sud_max-sud_min).round(3))+' mm',str(nord_min.round(3))+' mm',str(nord_max.round(3))+' mm',str((nord_max-nord_min).round(3))+' mm',str((((sud_max-sud_min)+(nord_max-nord_min))/2).round(3))+' mm',str(array_length_max_nowr.round(3))+' mm'])
# print(p2)

#++++++++++++++++++++++++++++++++++++++++
#  length: average array size along Y
#++++++++++++++++++++++++++++++++++++++++


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     Average array size along Y (length) with wrapping points
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
array_length_mean = (mean_y_LN - mean_y_LS).round(3)
array_length_mean_std = round(mt.sqrt( (std_y_LS/ mt.sqrt((df_LS['Y'].to_numpy()).size) )**2
                               + (std_y_LN/mt.sqrt((df_LN['Y'].to_numpy()).size) )**2  ) , 3)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     Average array size along Y (length) w/o wrapping points
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
array_length_mean_nowr = (mean_y_LN_nowr - mean_y_LS_nowr).round(3)
array_length_mean_std_nowr = round(mt.sqrt( (std_y_LS_nowr/ mt.sqrt(sud.size) )**2
                               + (std_y_LN_nowr/mt.sqrt(nord.size) )**2  ) , 3)

#++++++++++++++++++++++++++++++++++++++++
#     length: Mitutoyo simulation
#++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Simulating Mitutoyo (length) with wrapping
#++++++++++++++++++++++++++++++++++++++++++++++++++
np_mitutoyo_length_LS = max_y_LN - df_LS['Y'].to_numpy()
np_mitutoyo_length_LN = df_LN['Y'].to_numpy() - min_y_LS
np_mitutoyo_length = np_mitutoyo_length_LS
np_mitutoyo_length = np.concatenate([np_mitutoyo_length,np_mitutoyo_length_LN])

mitutoyo_array_length_mean = (np_mitutoyo_length.mean()).round(3)
mitutoyo_array_length_std = (np_mitutoyo_length.std()).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Simulating Mitutoyo (length) w/o wrapping
#++++++++++++++++++++++++++++++++++++++++++++++++++

np_mitutoyo_length_LS_nowr = max_y_LN - sud
np_mitutoyo_length_LN_nowr = nord - min_y_LS
np_mitutoyo_length_nowr = np_mitutoyo_length_LS_nowr
np_mitutoyo_length_nowr = np.concatenate([np_mitutoyo_length_nowr,np_mitutoyo_length_LN_nowr])

mitutoyo_array_length_nowr_mean = (np_mitutoyo_length_nowr.mean()).round(3)
mitutoyo_array_length_nowr_std = (np_mitutoyo_length_nowr.std()).round(3)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  WIDTH: X -> MaxVar - Mean - Spread - Max array size - average array size
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =============
# === Width ===
# =============

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Max. X variation on 'ovest'/east side (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

max_x_LO = np.amax(df_LO['X'].to_numpy())
min_x_LO = np.amin(df_LO['X'].to_numpy())
delta_x_LO = (max_x_LO - min_x_LO).round(3)

max_x_LE = np.amax(df_LE['X'].to_numpy())
min_x_LE = np.amin(df_LE['X'].to_numpy())
delta_x_LE = (max_x_LE - min_x_LE).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Mean on X on 'ovest'/east side (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

mean_x_LO = (df_LO['X'].to_numpy()).mean().round(3)
mean_x_LE = (df_LE['X'].to_numpy()).mean().round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Std. dev. on X on 'ovest'/east side (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

std_x_LO = (df_LO['X'].to_numpy()).std().round(3)
std_x_LE = (df_LE['X'].to_numpy()).std().round(3)
std_deltax = round(mt.sqrt(std_x_LO**2+std_x_LE**2),3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Max. array size along X (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

array_width_max = (max_x_LE - min_x_LO).round(3)


#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Average array size along X (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

array_width_mean = (mean_x_LE - mean_x_LO).round(3)
array_width_mean_std = round(mt.sqrt( (std_x_LO/mt.sqrt((df_LO['X'].to_numpy()).size))**2
                               + (std_x_LE/mt.sqrt((df_LE['X'].to_numpy()).size))**2  ) , 3)

#++++++++++++++++++++++++++++++++++
#  WIDTH: Mitutoyo simulation
#++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Simulating Mitutoyo (width)
#++++++++++++++++++++++++++++++++++++++++++++++++++

np_mitutoyo_width_LO = max_x_LE - df_LO['X'].to_numpy()
np_mitutoyo_width_LE = df_LE['X'].to_numpy() - min_x_LO
np_mitutoyo_width = np_mitutoyo_width_LO
np_mitutoyo_width = np.concatenate([np_mitutoyo_width,np_mitutoyo_width_LE])

mitutoyo_array_width_mean = (np_mitutoyo_width.mean()).round(3)
mitutoyo_array_width_std = (np_mitutoyo_width.std()).round(3)

  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  THICKNESS: Z -> MaxVar - Mean - Spread - Max array size - average array size
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# =================
# === Thickness ===
# =================

np_thickness = df_FS['Z'].to_numpy()
np_thickness = np_thickness.round(3)

###################################
### TEMPORARY FIX FOR FIRST PRE-PROD BATCH (subtract 75um from label thickness)
print ("before fix: ", np_thickness.mean().round(3),np_thickness.std().round(3))
#np_thickness = np_thickness - 0.075
print ("after fix: ", np_thickness.mean().round(3),np_thickness.std().round(3))
###################################

thickness_mean = np_thickness.mean().round(3)
thickness_std = np_thickness.std().round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Max. Z variation on front side (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++
max_z_FS = np.amax(np_thickness)
min_z_FS = np.amin(np_thickness)
delta_z_FS = (max_z_FS - min_z_FS).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Mean on Z on front side (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++
mean_z_FS = (np_thickness).mean().round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Std. dev. on Z on front side (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++
std_z_FS = (np_thickness).std().round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Max. array size along Z (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++
array_thickness_max = (max_z_FS - 0).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Average array size along Z (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++
array_thickness_mean = (mean_z_FS - 0).round(3)
array_thickness_mean_std = round( std_z_FS / mt.sqrt( (np_thickness).size ), 3 )

#+++++++++++++++++++++++++++++++++++++++++++
# ******** GEOMETRY DEFINITION ******** 
#+++++++++++++++++++++++++++++++++++++++++++

geo=['geo1','geo2','geo3']

if (array_thickness_mean > 3.95): 
    geo=geo[0]
elif (array_thickness_mean < 2.9): 
    geo=geo[2]
else:
    geo=geo[1]
    print('')
    print('++++++++++++++++++++++++++')
    print('Array Geometry :',str(geo) )
    print('++++++++++++++++++++++++++')    
    print('')

#++++++++++++++++++++++++++++++++++++++++++++++++++
#  THICKNESS: Mitutoyo simulation
#++++++++++++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++
#     Simulating Mitutoyo (thickness)
#++++++++++++++++++++++++++++++++++++++++++++++++++
np_mitutoyo_thickness = np_thickness

mitutoyo_array_thickness_mean = (np_mitutoyo_thickness.mean()).round(3)
mitutoyo_array_thickness_std = (np_mitutoyo_thickness.std()).round(3)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# === WRITING OUTPUT IN CSV FILE: Output .csv & .json files
#++++++++++++++++++++++++++++++++++++++++++++++++++

json_array = [{
    'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
    'id': barcode,
    #'producer': str(prod),
    #'geometry': str(geo),
    'time': date,
    'L_bar_mu':  length_mean,
    'L_bar_std':  length_std,
    'L_maxVar_LS': delta_y_LS_nowr,
    'L_maxVar_LN': delta_y_LN_nowr,
    'L_std_LS': std_y_LS_nowr,
    'L_std_LN': std_y_LN_nowr,
    'L_std_tot': std_deltay_nowr,
    'L_max': array_length_max_nowr,
    'L_mean': array_length_mean_nowr,
    'L_mean_std': array_length_mean_std_nowr,
    'L_mean_mitu': '',
    'L_std_mitu': '',
    'W_maxVar_LO':delta_x_LO,
    'W_maxVar_LE':delta_x_LE,
    'W_std_LO': std_x_LO,
    'W_std_LE': std_x_LE,
    'W_std_tot': std_deltax,
    'W_max': array_width_max,
    'W_mean': array_width_mean,
    'W_mean_std': array_width_mean_std,
    'W_mean_mitu': '',
    'W_std_mitu': '',
    'T_maxVar_FS': delta_z_FS,
    'T_std_FS': std_z_FS,
    'T_max': array_thickness_max,
    'T_mean': array_thickness_mean,
    'T_mean_std': array_thickness_mean_std,
    'T_mean_mitu': '',
    'T_std_mitu': '',
    'bar': '',
    'bar_length': '',
    'bar_length_std': '',
    'type': 'array'                  
}, 
     {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        #'producer': str(prod),
        #'geometry': str(geo),
        'time': date,
         
        'bar' : '0',
        'bar_length':'',
        'bar_length_std':'',
        'type': 'array'
      },
    {  
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),
 
        'bar' : '1',
        'bar_length':'',
        'bar_length_std': '',
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '2',
        'bar_length':'',
        'bar_length_std': '',
        'type': 'array'
      },
    {  
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        
 
        'bar' : '3',
        'bar_length':np_length[0],
        'bar_length_std': np_length_std[0],
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '4',
        'bar_length':np_length[1],
        'bar_length_std': np_length_std[1],
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '5',
        'bar_length':np_length[2],
        'bar_length_std': np_length_std[2],
        'type': 'array'
      },
    {  
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        
 
        'bar' : '6',
        'bar_length':np_length[3],
        'bar_length_std': np_length_std[3],
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '7',
        'bar_length':np_length[4],
        'bar_length_std': np_length_std[4],
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '8',
        'bar_length':np_length[5],
        'bar_length_std': np_length_std[5],
        'type': 'array'
      },   
 {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),     
     
        'bar' : '9',
        'bar_length':np_length[6],
        'bar_length_std': np_length_std[6],
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '10',
        'bar_length':np_length[7],
        'bar_length_std': np_length_std[7],
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '11',
        'bar_length':np_length[8],
        'bar_length_std': np_length_std[8],
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '12',
        'bar_length':np_length[9],
        'bar_length_std': np_length_std[9],
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '13',
        'bar_length':'',
        'bar_length_std': '',
        'type': 'array'
      },
    {   
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),        

        'bar' : '14',
        'bar_length':'',
        'bar_length_std': '',
        'type': 'array'
      },
     {
        'runName': run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,
        'id': barcode,
        'time': date,
        #'producer': str(prod),
        #'geometry': str(geo),         

        'bar' : '15',
        'bar_length':'',
        'bar_length_std': '',
        'type': 'array'
      }
]

l_results_names = ['runName','id','time',
                       'L_bar_mu','L_bar_std',
                       'L_maxVar_LS','L_maxVar_LN','L_std_LS','L_std_LN','L_std_tot','L_max','L_mean','L_mean_std',
                       'L_mean_mitu','L_std_mitu',
                       'W_maxVar_LO','W_maxVar_LE','W_std_LO','W_std_LE','W_std_tot','W_max','W_mean','W_mean_std',
                       'W_mean_mitu','W_std_mitu',
                       'T_maxVar_FS','T_std_FS','T_max','T_mean','T_mean_std','T_mean_mitu','T_std_mitu','bar',
                       'bar_length','bar_length_std','type']


l_results = [[run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,          
                 length_mean,length_std,
                 delta_y_LS_nowr,delta_y_LN_nowr,std_y_LS_nowr,std_y_LN_nowr,std_deltay_nowr,array_length_max_nowr,array_length_mean_nowr,array_length_mean_std_nowr, #w/o wrapping
                 #delta_y_LS,delta_y_LN,std_y_LS,std_y_LN,std_deltay,array_length_max,array_length_mean,array_length_mean_std, #with wrapping
                 '','',
                 delta_x_LO,delta_x_LE,std_x_LO,std_x_LE,std_deltax,array_width_max,array_width_mean,array_width_mean_std,
                 '','',
                 delta_z_FS,std_z_FS,array_thickness_max,array_thickness_mean,array_thickness_mean_std,
                 '','',
                 '','' ,'',''],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',0,'','','array'],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',1,'','','array'],    
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',2,'','','array'],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',3,np_length[0],np_length_std[0],'array'],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',4,np_length[1],np_length_std[1],'array'],    
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',5,np_length[2],np_length_std[2],'array'],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',6,np_length[3],np_length_std[3],'array'],    
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',7,np_length[4],np_length_std[4],'array'],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',8,np_length[5],np_length_std[5],'array'],    
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',9,np_length[6],np_length_std[6],'array'],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',10,np_length[7],np_length_std[7],'array'],  
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',11,np_length[8],np_length_std[8],'array'],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',12,np_length[9],np_length_std[9],'array'],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',13,'','','array'],
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',14,'','','array'],    
             [run.split('/')[-1]+'_ARRAY'+barcode+str('_')+tag,barcode,date,'','','','','','','','','','','','','','','','','','','','','','','','','','','','','',15,'','','array']]
             


#+++++++++++++++++++
#  SAVE .csv FILE
#+++++++++++++++++++

#with open(str(args.array)+str('_')+tag+'.csv', 'w') as file:
with open(run+str('_ARRAY')+barcode+str('_')+tag+'.csv','w') as file:
    writer = csv.writer(file, delimiter=',')
    # row by row         
    writer.writerow(l_results_names)
    writer.writerows(l_results)

os.system('cp '+run+str('_ARRAY')+barcode+str('_')+tag+'.csv /home/cmsdaq/MTDDB/uploader/files_to_upload/galaxy-arrays/')


#+++++++++++++++++++
#SAVE json FILE
#+++++++++++++++++++

#with open(str(args.array)+str('_')+tag+'.json', 'w') as json_file:
with open(run+str('_ARRAY')+barcode+str('_')+tag+'.json','w') as json_file:
    json.dump(json_array, json_file, indent=4)

#++++++++++++++++++++++++
#   SUMMARY TABLE
#++++++++++++++++++++++++
    
from prettytable import PrettyTable



p = PrettyTable(['Array',' Date&Time'])
p.add_row([barcode+str('_')+tag, str(timestamp)])
#print(p)
data = p.get_string()
print(data)
print ('')
print('++++++++++++++++++++++++++++')
print('     Single Bars length     ')
print('++++++++++++++++++++++++++++')

p3 = PrettyTable(['Bar3','Bar4','Bar5','Bar6','Bar7','Bar8','Bar9','Bar10','Bar11','Bar12'])
p3.add_row([str(np_length[0]),str(np_length[1]),str(np_length[2]),str(np_length[3]),str(np_length[4]),str(np_length[5]),str(np_length[6]),str(np_length[7]),str(np_length[8]),str(np_length[9])])     
p3.add_row([str(np_length_std[0])+ ' mm',str(np_length_std[1])+ ' mm',str(np_length_std[2])+ ' mm',str(np_length_std[3])+ ' mm',str(np_length_std[4])+ ' mm',str(np_length_std[5])+ ' mm',str(np_length_std[6])+ ' mm',str(np_length_std[7])+ ' mm',str(np_length_std[8])+ ' mm',str(np_length_std[9])+ ' mm'])                  
print(p3)
single_bar = p3.get_string()

print ('')
print('+++++++++++++')
print('   length    ')
print('+++++++++++++')

p = PrettyTable(['Array',' Bar length Mean','length Mean w/wrap.', 'Wrapping'])
p.add_row([barcode+str('_')+tag, str(length_mean)+ ' +/- '+ str(length_std)+' mm',str(wrap_length_mean)+ ' +/- '+ str(wrap_length_std)+' mm',str(wrap)+' mm'])
print(p)
length = p.get_string()

print ('')
print('++++++++++++++++++++++++++++++++')
print('    length including wrapping   ')
print('++++++++++++++++++++++++++++++++')

p = PrettyTable(['Array','Max. Y var. (LS)','Max. Y var. (LN)','Std. dev. Y (LS)','Std. dev. Y (LN)', 'Std. dev. DeltaY (LN-LS)'])
p.add_row([barcode+str('_')+tag,str(delta_y_LS)+' mm',str(delta_y_LN)+' mm',str(std_y_LS)+' mm',str(std_y_LN)+' mm', str(std_deltay)+' mm'])
print(p)
max_var_length = p.get_string()
#print ('')
p1 = PrettyTable(['Array','Max. array size along Y (LN - LS)','Mean array size along Y (LN - LS)','Mitutoyo Simulation '])
p1.add_row([barcode+str('_')+tag,str(array_length_max)+' mm',str(array_length_mean)+ ' +/- '+ str(array_length_mean_std)+ ' mm',str(mitutoyo_array_length_mean)
           + ' +/- '+ str(mitutoyo_array_length_std)+ ' mm'])
print(p1)
max_length = p1.get_string()
print ('')
print('++++++++++++++++++++++++++++++++')
print('      length w/o wrapping       ')
print('++++++++++++++++++++++++++++++++')

p = PrettyTable(['Array','Max. Y var. (LS)','Max. Y var. (LN)','Std. dev. Y (LS)','Std. dev. Y (LN)', 'Std. dev. DeltaY (LN-LS)'])
p.add_row([barcode+str('_')+tag,str(delta_y_LS_nowr)+' mm',str(delta_y_LN_nowr)+' mm',str(std_y_LS_nowr)+' mm',str(std_y_LN_nowr)+' mm', str(std_deltay_nowr)+' mm'])
print(p)
max_var_length_nowr = p.get_string()
#print ('')
p1 = PrettyTable(['Array','Max. array size along Y (LN - LS)','Mean array size along Y (LN - LS)','Mitutoyo Simulation '])
p1.add_row([barcode+str('_')+tag,str(array_length_max_nowr)+' mm',str(array_length_mean_nowr)+ ' +/- '+ str(array_length_mean_std_nowr)+ ' mm',str(mitutoyo_array_length_nowr_mean)
           + ' +/- '+ str(mitutoyo_array_length_nowr_std)+ ' mm'])
print(p1)
max_length_nowr = p1.get_string()
print ('')

print('+++++++++++++')
print('    Width    ')
print('+++++++++++++')

p = PrettyTable(['Array','Max. X var. (LO)','Max. X var. (LE)','Std. dev. X (LO)','Std. dev. X (LE)', 'Std. dev. DeltaX (LE-LO)'])
p.add_row([barcode+str('_')+tag,str(delta_x_LO)+' mm',str(delta_x_LE)+' mm',str(std_x_LO)+' mm',str(std_x_LE)+' mm', str(std_deltax)+' mm'])
print(p)
max_var_width = p.get_string()
#print ('')

p1 = PrettyTable(['Array','Max. array size along X (LE - LO)','Mean array size along X (LE - LO)','Mitutoyo Simulation '])
p1.add_row([barcode+str('_')+tag,str(array_width_max)+' mm',str(array_width_mean)+ ' +/- '+ str(array_width_mean_std)+ ' mm',str(mitutoyo_array_width_mean)
           + ' +/- '+ str(mitutoyo_array_width_std)+ ' mm'])
print(p1)
max_width = p1.get_string()
print ('')

print('+++++++++++++')
print('  Thickness  ')
print('+++++++++++++')

p2 = PrettyTable(['Array','Max. Z var. (FS)','Std. dev. Z (FS)'])
p2.add_row([barcode+str('_')+tag,str(delta_z_FS)+' mm',str(std_z_FS)+' mm'])
print(p2)
max_var_thickness = p2.get_string()
#print ('')
p1 = PrettyTable(['Array','Max. array size along Z (FS - 0)','Mean array size along Z (FS - 0)','Mitutoyo Simulation '])
p1.add_row([barcode+str('_')+tag,str(array_thickness_max)+' mm',str(array_thickness_mean)+ ' +/- '+ str(array_thickness_mean_std)+ ' mm',str(mitutoyo_array_thickness_mean)
           + ' +/- '+ str(mitutoyo_array_thickness_std)+ ' mm'])
print(p1)
max_thickness = p1.get_string()
print ('')


#+++++++++++++++++++++++++++++++++
#        FOR TENDER
#+++++++++++++++++++++++++++++++++

print('---Length')
Array_L_mean = array_length_mean_nowr
print('Array_L_mean:',str(array_length_mean_nowr)+ ' +/- '+ str(array_length_mean_std_nowr)+ ' mm')
#Array_L_MaxVar = (((sud_max-sud_min)+(nord_max-nord_min))/2).round(3) #old version with mean value
Array_L_MaxVar = max(delta_y_LS_nowr,delta_y_LN_nowr).round(3) #new version with max(maxvarLS,maxvarLN)
print('Array_L_MaxVar:',str(Array_L_MaxVar)+ ' mm')
Array_L_spread = ((std_y_LS_nowr+std_y_LN_nowr)/2).round(3)
print('Array_L_spread:',str(Array_L_spread)+' mm')
Array_L_MaxSize = array_length_max_nowr
print('Array_L_MaxSize:',str(array_length_max_nowr)+' mm')
Bar_L_mean = length_mean
print('Bar_L_mean:', str(length_mean)+ ' +/- '+ str(length_std)+' mm')
print('---Width')
Array_W_mean = array_width_mean
print('Array_W_mean:',str(array_width_mean)+ ' +/- '+ str(array_width_mean_std)+ ' mm')
#Array_W_MaxVar = ((delta_x_LO+delta_x_LE)/2).round(3)
Array_W_MaxVar = max(delta_x_LO,delta_x_LE).round(3)
print('Array_W_MaxVar:',str(Array_W_MaxVar)+ ' mm')
Array_W_spread = ((std_x_LO+std_x_LE)/2).round(3)
print('Array_W_spread:',str(Array_W_spread)+' mm')
Array_W_MaxSize = array_width_max
print('Array_W_MaxSize:',str(array_width_max)+' mm')
print('---Thickness')
Array_T_mean = array_thickness_mean
print('Array_T_mean:',str(array_thickness_mean)+ ' +/- '+ str(array_thickness_mean_std)+ ' mm')
Array_T_MaxVar = str(delta_z_FS)
print('Array_T_MaxVar:',str(Array_T_MaxVar)+ ' mm')
Array_T_spread = (std_z_FS).round(3)
print('Array_T_spread:',str(Array_T_spread)+' mm')
Array_T_MaxSize = array_thickness_max
print('Array_T_MaxSize:',str(array_thickness_max)+' mm')

#+++++++++++++++++++++++++++++++++
#     Specifications for Tender
#+++++++++++++++++++++++++++++++++

#----------------------------------------------------------------------------------------------
#                            MS3 - OPT - OPT2
#----------------------------------------------------------------------------------------------
# Array type |     w        |       t     |         L
#----------------------------------------------------------------------------------------------
#     1      | 51.50+-0.10  | 4.05+-0.10  | 55(56.30)+-0.020
#     2      | 51.50+-0.10  | 3.30+-0.10  | 55(56.30)+-0.020
#     3      | 51.50+-0.10  | 2.70+-0.10  | 55(56.30)+-0.020
#----------------------------------------------------------------------------------------------
# Bar type  |      w      |       t           |         L
#----------------------------------------------------------------------------------------------
#     1     | 3.12+-0.10  | 3.75+-0.10        | 55(56.30)+-0.020
#     2     | 3.12+-0.10  | 3.00 (3.30)+-0.10 | 55(56.30)+-0.020  MS3 w and w/o ESR
#     3     | 3.12+-0.10  | 2.40 +-0.10       | 55(56.30)+-0.020  MS3 w/o ESR
#----------------------------------------------------------------------------------------------

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                 TENDER
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Type |       Width            |        Thickness        |          Lenght
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 #1   |   51.48 (+0.10/-0.10)  |    4.11 (+0.10/-0.10)   |      54.70 (+0.05/-0.05)
 #2   |   51.48 (+0.10/-0.10)  |    3.36 (+0.10/-0.10)   |      54.70 (+0.05/-0.05)
 #3   |   51.48 (+0.10/-0.10)  |    2.76 (+0.10/-0.10)   |      54.70 (+0.05/-0.05)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Single Bars within Array: thickness of the reflector between crystals between 60 and 100 μm.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 #1   |    3.12 (+0.03/-0.03)  |    3.75 (+0.03/-0.03)   |      54.70 (+0.03/-0.03)
 #2   |    3.12 (+0.03/-0.03)  |    3.00 (+0.03/-0.03)   |      54.70 (+0.03/-0.03)
 #3   |    3.12 (+0.03/-0.03)  |    2.40 (+0.03/-0.03)   |      54.70 (+0.03/-0.03)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''
#------Array: After Final Tender------
L_mean_min = 54.65 #Final
L_mean_max = 54.75 #Final
L_Max_min = 54.65 #Final
L_Max_max = 54.75 #Final
Bar_L_mean_min = 54.67 #Final
Bar_L_mean_max = 54.73 #Final

L_MaxVar = 60 #Final
W_mean_min =  51.38 #Final
W_mean_max =  51.58 #Final
W_Max_min =  51.38 #Final
W_Max_max =  51.58 #Final
T1_mean_min = 4.01 #Final
T1_mean_max = 4.21 #Final
T1_Max_min = 4.01 #Final
T1_Max_max = 4.21 #Final
T2_mean_min = 3.26 #Final
T2_mean_max = 3.46 #Final
T2_Max_min = 3.26 #Final
T2_Max_max = 3.46 #Final
T3_mean_min = 2.66 #Final
T3_mean_max = 2.86 #Final
T3_Max_min = 2.66 #Final
T3_Max_max = 2.86 #Final
'''

#+++++++++++++++++++++++++++++++++
#L_mean_min = 56.28 #MS3/OPT
#L_mean_max = 56.32 #MS3/OPT
#L_Max_min = 56.25 #MS3/OPT
#L_Max_max = 56.35 #MS3/OPT
#Bar_L_mean_min = 56.28  #MS3/OPT
#Bar_L_mean_max = 56.32  #MS3/OPT

L_mean_min = 54.98 #OPT2
L_mean_max = 55.02 #OPT2
L_Max_min = 54.95 #OPT2
L_Max_max = 55.05 #OPT2
Bar_L_mean_min = 54.98 #OPT2
Bar_L_mean_max = 55.02 #OPT2

L_MaxVar = 50
W_mean_min =  51.4
W_mean_max =  51.6
W_Max_min =  51.4
W_Max_max =  51.6
T1_mean_min = 3.95
T1_mean_max = 4.15
T1_Max_min = 3.95
T1_Max_max = 4.15
T2_mean_min = 3.2
T2_mean_max = 3.4
T2_Max_min = 3.2
T2_Max_max = 3.4
T3_mean_min = 2.6
T3_mean_max = 2.8
T3_Max_min = 2.6
T3_Max_max = 2.8

True == 1
False == 1

#---Length
Array_L_mean_pass = 1
Array_L_MaxVar_pass = 1
Array_L_spread_pass = 1
Array_L_MaxSize_pass = 1
Bar_L_mean_pass = 1
#---Width
Array_W_mean_pass = 1
Array_W_MaxVar_pass = 1
Array_W_spread_pass = 1
Array_W_MaxSize_pass = 1
#---Thickness
Array_T_mean_pass = 1
Array_T_MaxVar_pass = 1
Array_T_spread_pass = 1
Array_T_MaxSize_pass = 1
# Array_T1_mean_pass = 1
# Array_T1_MaxVar_pass = 1
# Array_T1_spread_pass = 1
# Array_T1_MaxSize_pass = 1
# Array_T2_mean_pass = 1
# Array_T2_MaxVar_pass = 1
# Array_T2_spread_pass = 1
# Array_T2_MaxSize_pass = 1
# Array_T3_mean_pass = 1
# Array_T3_MaxVar_pass = 1
# Array_T3_spread_pass = 1


print('-----> Length')
if (Array_L_mean > L_mean_max or Array_L_mean < L_mean_min ):
    Array_L_mean_pass = 0
    print('Array_L_mean: failed')
else:
    print('Array_L_mean: passed') 

if (Array_L_MaxVar*1000 > L_MaxVar ):
    Array_L_MaxVar_pass = 0
    print('Array_L_MaxVar: failed')
else:
    print('Array_L_MaxVar: passed')   
    
if (Array_L_MaxSize > L_Max_max or Array_L_MaxSize < L_Max_min ):
    Array_L_MaxSize_pass = 0
    print('Array_L_MaxSize: failed')
else:
    print('Array_L_MaxSize: passed')  
    
if (Bar_L_mean > Bar_L_mean_max or Bar_L_mean < Bar_L_mean_min ):
    Bar_L_mean_pass = 0
    print('Bar_L_mean: failed')
else:
    print('Bar_L_mean: passed')  
    
print('-----> Width')    
if (Array_W_mean > W_mean_max or Array_W_mean < W_mean_min ):
    Array_W_mean_pass = 0
    print('Array_W_mean: failed')
else:
    print('Array_W_mean: passed')    
      
if (Array_W_MaxSize > W_Max_max or Array_W_MaxSize < W_Max_min ):
    Array_W_MaxSize_pass = 0
    print('Array_W_MaxSize: failed')
else:
    print('Array_W_MaxSize: passed')  
    
print('-----> Thickness') 
#print(array_thickness_mean)

if (array_thickness_mean < 4.15 and array_thickness_mean > 3.95):

    print('Array Geometry :',str(geo))
    if (Array_T_mean > T1_mean_max or Array_T_mean < T1_mean_min ):
        Array_T_mean_pass = 0
        print('Array_T_mean: failed')
    else:
        print('Array_T_mean: passed')    
      
    if (Array_T_MaxSize > T1_Max_max or Array_T_MaxSize < T1_Max_min ):
        Array_T_MaxSize_pass = 0
        print('Array_T_MaxSize: failed')
    else:
        print('Array_T_MaxSize: passed') 
                                
elif (array_thickness_mean < 3.4 and array_thickness_mean > 3.2):
    print('Array Geometry :',str(geo))
    if (Array_T_mean > T2_mean_max or Array_T_mean < T2_mean_min ):
        Array_T_mean_pass = 0
        print('Array_T_mean: failed')
    else:
        print('Array_T_mean: passed')    
      
    if (Array_T_MaxSize > T2_Max_max or Array_T_MaxSize < T2_Max_min ):
        Array_T_MaxSize_pass = 0
        print('Array_T_MaxSize: failed')
    else:
        print('Array_T_MaxSize: passed')        
                       
#if (array_thickness_mean < 2.8 and array_thickness_mean > 2.6):
else:
    print('Array Geometry :',str(geo))
    if (Array_T_mean > T3_mean_max or Array_T_mean < T3_mean_min ):
        Array_T_mean_pass = 0
        print('Array_T_mean: failed')
    else:
        print('Array_T_mean: passed')    
      
    if (Array_T_MaxSize > T3_Max_max or Array_T_MaxSize < T3_Max_min ):
        Array_T_MaxSize_pass = 0
        print('Array_T_MaxSize: failed')
    else:
        print('Array_T_MaxSize: passed')

print('')
print('**************************')
print(' INFORMATIONS FOR TENDER  ')
print('**************************')

ptender = PrettyTable(['Array','Array_L_mean','Array_L_MaxVar','Array_L_spread','Array_L_MaxSize','Bar_L_mean'])
ptender.add_row([barcode+str('_')+tag,str(array_length_mean_nowr)+ ' mm',str(Array_L_MaxVar*1000)+ ' \u03BCm',
                 str(Array_L_spread*1000)+ ' \u03BCm',str(array_length_max_nowr)+' mm', str(length_mean)+' mm'])
print(ptender)
tender_L = ptender.get_string()


ptender = PrettyTable(['Array','Array_W_mean','Array_W_MaxVar','Array_W_spread','Array_W_MaxSize'])
ptender.add_row([barcode+str('_')+tag,str(array_width_mean)+ ' mm',str(Array_W_MaxVar*1000)+ ' \u03BCm',
                 str(Array_W_spread*1000)+ ' \u03BCm',str(array_width_max)+' mm'])
print(ptender)
tender_W = ptender.get_string()

ptender = PrettyTable(['Array','Array_T_mean','Array_T_MaxVar','Array_T_spread','Array_T_MaxSize'])
ptender.add_row([barcode+str('_')+tag,str(array_thickness_mean)+ ' mm',str(delta_z_FS*1000)+' \u03BCm',
                 str(Array_T_spread*1000)+' \u03BCm',str(array_thickness_max)+' mm'])
print(ptender)
tender_T = ptender.get_string()


print('************************************************************')
print(' INFORMATIONS FOR TENDER: Variables passing specifications  ')
print('************************************************************')

pfinal = PrettyTable(['Array','Bar_L_mean','Array_L_mean','Array_L_MaxVar','Array_L_spread','Array_L_MaxSize'])
pfinal.add_row([barcode+str('_')+tag,Bar_L_mean_pass,Array_L_mean_pass,Array_L_MaxVar_pass,Array_L_spread_pass,
                 Array_L_MaxSize_pass])
print(pfinal)
tender_L_pass = pfinal.get_string()


pfinal = PrettyTable(['Array','Array_W_mean','Array_W_MaxSize','Array_T_mean','Array_T_MaxSize'])
pfinal.add_row([barcode+str('_')+tag,Array_W_mean_pass,Array_W_MaxSize_pass,Array_T_mean_pass,Array_T_MaxSize_pass])
print(pfinal)
tender_W_T_pass = pfinal.get_string()

print('')
print('Passed = 1')
print('Failed = 0')


#+++++++++++++++++++++++++++++++++
#   SAVING TABLE IN OUTPUT FILE
#+++++++++++++++++++++++++++++++++

#with open(str(args.array)+str('_')+tag+'_output.txt','w') as f:
with open(run+str('_ARRAY')+barcode+str('_')+tag+'_output.txt','w') as f:


    f.write(data) 
    f.write('\n\r') 
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write('     Single Bars length     \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(single_bar)   
    f.write('\n\r') 
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write('         length             \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(length)
    f.write('\n\r') 
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(' length  including wrapping \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(max_var_length)
    f.write('\n\r') 
    f.write(max_length)
    f.write('\n\r') 
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write('   length  w/o wrapping     \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(max_var_length_nowr)
    f.write('\n\r') 
    f.write(max_length_nowr)
    f.write('\n\r') 
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write('         Width              \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(max_var_width)
    f.write('\n\r') 
    f.write(max_width)
    f.write('\n\r')     
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write('        Thickness           \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(max_var_thickness)
    f.write('\n\r') 
    f.write(max_thickness)
    f.write('\n\r') 
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write('       For Tender Length    \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(tender_L)
    f.write('\n\r') 
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write('       For Tender Width     \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(tender_W)
    f.write('\n\r') 
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write('   For Tender Thickness     \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(tender_T)
    f.write('\n\r') 
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write('   For Tender Length Pass   \n\r')
    f.write('++++++++++++++++++++++++++++\n\r')
    f.write(tender_L_pass)
    f.write('\n\r') 
    f.write('+++++++++++++++++++++++++++++++++\n\r')
    f.write(' For Tender Width/Thickness Pass \n\r')
    f.write('+++++++++++++++++++++++++++++++++\n\r')
    f.write(tender_W_T_pass)
    f.write('\n\r') 



    stop = timeit.default_timer()

    print('++++++++++++++++++++++++++++++')
    print('Execution Time: ', stop - start)
    print('++++++++++++++++++++++++++++++')
  






