import sys, os, argparse
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import re

import ROOT as rt

pd.set_option('display.max_rows', None)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     HOW TO RUN OVER ONE SINGLE FILE
#  python3 Galaxy_raw_Data_Array_MTDDB.py --data ArrayData/812* --array 812
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++++++++++++++++
#  Parsing - Date - Producers definifition
#+++++++++++++++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()

parser.add_argument('--data',dest='data', help='data file *.TXT', type=str,required=True,default=False)
parser.add_argument('--LYdata',dest='LYdata', help='data file *.root', type=str,required=True,default=False)
parser.add_argument('--array',dest='array',required=False,default=False)

args = parser.parse_args()


#+++++++++++++++++++++++++++++++++++++
# FORMATS
# filename: 415_2022-05-04-12-52.txt
# Date and time as 2022-05-04-12-52
# Run001_416_2023-01-19-12-40_OLD.TXT 
#+++++++++++++++++++++++++++++++++++++
data = args.data
data = data[:-2]

LYdata = args.LYdata

info = data.split('_')


run = info[0]
args.array = barcode = info[1]

date = info[2]

tag = info[3]

date = date.replace('.T','')
data = data.replace('.T','')
tag = tag.replace('.T','')

print('Galaxy Filename :', data)
print('LY Filename :', LYdata)
print('Barcode :', barcode)
print('Tag :', tag)
print('RunNumber :', run)

#+++++++++++++++++++++++++++++++++++++++++++
#   JUST FOR REFERENCE: NOMINAL DIMENSIONS
#+++++++++++++++++++++++++++++++++++++++++++

#----------------------------------------------------------------------------
#  Array type  |     w        |       t     |         L
#----------------------------------------------------------------------------
#        1     | 51.50+-0.10  | 4.05+-0.10  | 54.70+-0.030
#        2     | 51.50+-0.10  | 3.30+-0.10  | 54.70+-0.030
#        3     | 51.50+-0.10  | 2.70+-0.10  | 54.70+-0.030
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

length_m = 54.7 #mm
e_low_length = 0.03 #mm
e_high_length = 0.03 #mm

#+++++++++++++++++++++++++++++++
#     Reading data from Galaxy
#+++++++++++++++++++++++++++++++

df = {}
df['LS'] = pd.DataFrame(columns=['X', 'Y', 'Z']) # Dataframe for South Side (Lato Sud)  points
df['LN'] = pd.DataFrame(columns=['X', 'Y', 'Z']) # Dataframe for South Side (Lato Nord) points

Npoints = 40
for line in open(args.data,errors='ignore'):    
    line = line.rstrip()
    splitline = line.split()
    n_elements = len(splitline)

    if(line.startswith(('DIM','AS','M','DATI','PART'))):
        continue    
    if line=='':
        continue

    # Detect if line is first row of LN or LS side
    # otherwise add data to current side
    if re.match(r'[0-9][0-9]_[A-Z][A-Z]', splitline[0]):
        current_side = re.sub(r'[0-9][0-9]_', '', splitline[0])
        n = splitline[1]
        x = float(splitline[2])
        y = float(splitline[3])
        z = float(splitline[4])
    else:
        n = splitline[0]
        x = float(splitline[1])
        y = float(splitline[2])
        z = float(splitline[3])

    if current_side not in df.keys():
        continue

    values_to_add = {'X': x, 'Y': y, 'Z': z}
    row_to_add = pd.Series(values_to_add, name=n)
    df[current_side] = df[current_side].append(row_to_add)

print(df['LS'])
print(df['LN'])



#+++++++++++++++++++++++++++++++++
#     Reading LYdata from Milan
#+++++++++++++++++++++++++++++++++
#
# LY_LN = Light Yield Lato Nord (North Side)
# LY_LS = Light Yield Lato Sud  (South Side)
#

LYgraph = {}
for tag in ['LY_LN', 'LY_LS']:
    file = rt.TFile(LYdata)

    if tag == 'LY_LN':
        LYgraph[tag] = file.Get("g_charge_511_vs_chR")
    elif tag == 'LY_LS':
        LYgraph[tag] = file.Get('g_charge_511_vs_chL')

    df[tag] = pd.DataFrame(columns=['BARID','LY','LY_Err'])

    for i in range(LYgraph[tag].GetN()):

        bar_id = LYgraph[tag].GetPointX(i)
        bar_ly = LYgraph[tag].GetPointY(i) # LY from SiPM on one side
        ly_err = LYgraph[tag].GetErrorY(i)

        values_to_add = { 'BARID': bar_id, 'LY':bar_ly, 'LY_Err':ly_err }
        row_to_add = pd.Series(values_to_add, name=str(bar_id))
        df[tag] = df[tag].append(row_to_add)
    print(tag)
    print(df[tag])

#+++++++++++++++++++++++++++++++++++++++++++++#
#        Filter/Ordering of Dataframes        #
#+++++++++++++++++++++++++++++++++++++++++++++#

for tag in ['LS','LN']:
    
    # Order by ascending X, Z
    df[tag] = df[tag].sort_values(['X','Z'], ascending=[True,True])
    df[tag].index = pd.RangeIndex(1,1 + len(df[tag]))

    # Remove points measured on wrapping (Z~4.05mm)
    df[tag] = df[tag][ df[tag]['Z'] < 3.0 ]
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # print('tag ',tag)
    # print(df[tag])
    #           Average over two consecutive row           #
    #   (mean Y on a XTAL face, X and Z are meaningless)   #
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++#

    df[tag+'_avg'] = pd.concat([(df[tag].iloc[::2, :2] + df[tag].iloc[1::2, :2].values) / 2, df[tag].iloc[::2, 2]], axis=1)
    df[tag+'_avg']['Y_misalign'] = df[tag+'_avg']['Y'] - df[tag+'_avg']['Y'].mean()

    # add bar_id column
    if 'LS' in tag:
        id_array = range(3,16,1)
        df[tag+'_avg']['BARID'] = id_array
    if 'LN' in tag:
        id_array = range(0,13,1)
        df[tag+'_avg']['BARID'] = id_array
    print(tag+'_avg')
    print(df[tag+'_avg'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PLOT 3 LN-LS_MaxVar_13_measured_points:  New plot upon Paolo's request  - FOR VENDORS - (mean of 26 points w/o wrapping)
#                               VERSION WITH FIXED Y AXIS RANGE
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

fig, ax1 = plt.subplots()

#++++++++++++++++++++++++
#    South Side
#++++++++++++++++++++++++
south_misalign = df['LS_avg']['Y_misalign'].to_numpy()
south_misalign = np.append([np.nan,np.nan,np.nan], south_misalign)
south_x_index = np.arange(0,16,1)

color = 'tab:red'
ax1.plot(south_x_index, south_misalign, color=color,linestyle='dashed', marker='o',label='Barcode Side') 

#++++++++++++++++++++++++
#    North Side
#++++++++++++++++++++++++

north_misalign = df['LN_avg']['Y_misalign'].to_numpy()
north_misalign = np.append(north_misalign, [np.nan,np.nan,np.nan])
north_x_index = np.arange(0,16,1)
color = 'tab:blue'
ax1.plot(north_x_index,north_misalign, '-r',color=color,linestyle='dashed', marker='o',label='Opposite to Barcode Side')

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
plt.savefig(str(args.array)+str('_')+tag+'_LN-LS_MaxVar_13_measured_points_new.png',bbox_inches='tight')
#plt.show() #uncomment this if you want display plots while running code


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PLOT 4 LN-LS_MaxVar_central:  Same as plot 3 but only 10 central bars
#                  VERSION WITH FIXED Y AXIS RANGE
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

fig, ax1 = plt.subplots()

#++++++++++++++++++++++++
#    South Side
#++++++++++++++++++++++++
south_misalign = np.delete(south_misalign, [0,1,2,13,14,15], None)
south_misalign = south_misalign - south_misalign.mean()
south_x_index  = np.arange(3,13,1)

color = 'tab:red'
ax1.plot(south_x_index, south_misalign, color=color,linestyle='dashed', marker='o',label='Barcode Side')

#++++++++++++++++++++++++
#    North Side
#++++++++++++++++++++++++

north_misalign = np.delete(north_misalign, [0,1,2,13,14,15], None)
north_misalign = north_misalign - north_misalign.mean()
north_x_index = np.arange(3,13,1)

color = 'tab:blue'
ax1.plot(north_x_index,north_misalign, '-r',color=color,linestyle='dashed', marker='o',label='Opposite to Barcode Side')

# Labels and appearance
ax1.grid()

ax1.set_xlabel('# Bar')
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax1.xaxis.set_major_locator(loc)

ax1.set_ylabel('Mis-alignment (central bars only) [mm]')
plt.ylim([-0.09,0.09]) # fix y range

ax1.text(3.1,-0.038,'MTD tolerance (0.060 mm)',color='r')
plt.axhline(y = 0.03, color = 'r', linestyle = '--') # MTD acceptance
plt.axhline(y =-0.03, color = 'r', linestyle = '--') # MTD acceptance

# Adding legend
fig.legend(loc='upper left', bbox_to_anchor=(0.03,0.15), bbox_transform=ax1.transAxes)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.suptitle('CMS MTD' + str(args.array) + ' - North-South Side Misalignment', y=1.02,x=0.5)

# Plot show/saving
plt.savefig(str(args.array)+str('_')+tag+'_LN-LS_MaxVar_central_new.png',bbox_inches='tight')
#plt.show() #uncomment this if you want display plots while running code

##########################
# Join misalign and LY

fig, ax1 = plt.subplots()

for tag in ['LS','LN']:
    df['LY_vs_Planarity_'+tag] = pd.merge(df[tag+'_avg'],df['LY_'+tag],how='inner',left_on='BARID',right_on='BARID')
    xval = df['LY_vs_Planarity_'+tag]['Y_misalign']
    yval = df['LY_vs_Planarity_'+tag]['LY']
    # print('planarity df ',tag,'\n', df[tag+'_avg'])
    # print('LY df ',tag,'\n', df['LY_'+tag])
    print(f'planarity vd LY df {tag}\n',df['LY_vs_Planarity_'+tag],'\n\n')    

    if tag == 'LS':
        color_ = 'tab:red'
        label_ = 'Barcode Side'
    else:
        color_ = 'tab:blue'
        label_ = 'Opposite to Barcode Side'

    ax1.plot(xval, yval, color=color_,linestyle='', marker='o',label=label_)
    
    # Fit searching for correlation
    par, cov = np.polyfit(xval, yval, 1, cov=True)        
    print(np.polyfit(xval, yval, 1,cov=True)    )
    print(math.sqrt(np.diag(cov)[0]))
    xseq = np.linspace(min(xval), max(xval), num=100)

    from scipy.stats import pearsonr
    corr, _ = pearsonr(xval, yval)
        
    plt.plot(xseq, par[1] + par[0] * xseq, color=color_, lw=2.5, linestyle='--');

    # Use plt.legend to automatically place text in plot
    text0 = tag+':'
    text1 = "slope: {:.3e}".format(par[0])
    text1p5 = "slope err: {:.3e}".format( math.sqrt(np.diag(cov)[0]) )
    text2 = f"offset: {round(par[1],3)}"
    text2p5 = "offset err: {:.3e}".format( math.sqrt(np.diag(cov)[1]) )
    text3 = f"Pears. r: {round(corr,3)}"
    text = text0 + '\n' + text1 + '\n' + text1p5 + '\n' +text2+'\n'+ text2p5 + '\n' + text3


# Labels and appearance
ax1.grid()

ax1.set_xlabel('Misalignment [mm]', labelpad=10, size=14)
ax1.set_ylabel('511 keV charge peak [pV * s]', labelpad=10, size=14)
# plt.ylim([-0.09,0.09]) # fix y range

# ax1.text(3.1,-0.038,'MTD tolerance (0.060 mm)',color='r')
# plt.axhline(y = 0.03, color = 'r', linestyle = '--') # MTD acceptance
# plt.axhline(y =-0.03, color = 'r', linestyle = '--') # MTD acceptance

# Adding legend
fig.legend(loc='upper left', bbox_to_anchor=(0.03,0.15), bbox_transform=ax1.transAxes)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.tick_params(axis='both', which='major', labelsize=14)
plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
plt.suptitle('Array ' + str(args.array) + ' - Bar response vs Misalignment ', y=1.05,x=0.5, fontsize=15)


# Plot show/saving
plt.savefig(str(args.array)+'_LY_vs_Misalign.png',bbox_inches='tight')
#plt.show() #uncomment this if you want display plots while running code



fig, axs = plt.subplots(2, figsize=(7,6))

i_subplot = 0
for tag in ['LS','LN']:
    
    df['LY_vs_Planarity_'+tag] = pd.merge(df[tag+'_avg'],df['LY_'+tag],how='inner',left_on='BARID',right_on='BARID')
    xval = df['LY_vs_Planarity_'+tag]['Y_misalign']
    yval = df['LY_vs_Planarity_'+tag]['LY']
    # print('planarity df ',tag,'\n', df[tag+'_avg'])
    # print('LY df ',tag,'\n', df['LY_'+tag])
    # print('planarity vd LY df \n',df['LY_vs_Planarity_'+tag],'\n\n')    

    if tag == 'LS':
        color_ = 'tab:red'
        label_ = 'Barcode Side'
    else:
        color_ = 'tab:blue'
        label_ = 'Opposite to Barcode Side'

    axs[i_subplot].plot(xval, yval, color=color_,linestyle='', marker='o',label=label_)
    
    par, cov = np.polyfit(xval, yval, 1, cov=True)        
    print(np.polyfit(xval, yval, 1,cov=True)    )
    print(math.sqrt(np.diag(cov)[0]))
    xseq = np.linspace(min(xval), max(xval), num=100)

    from scipy.stats import pearsonr
    corr, _ = pearsonr(xval, yval)
        
    axs[i_subplot].plot(xseq, par[1] + par[0] * xseq, color=color_, lw=2.5, linestyle='--');

    # Use plt.legend to automatically place text in plot
    text1 = "slope: {:.3e}".format(par[0])
    text1p5 = "slope err: {:.3e}".format( math.sqrt(np.diag(cov)[0]) )
    text2 = f"offset: {round(par[1],3)}"
    text2p5 = "offset err: {:.3e}".format( math.sqrt(np.diag(cov)[1]) )
    text3 = f"Pears. r: {round(corr,3)}"
    text = text1 + '\n' + text1p5 + '\n' +text2+'\n'+ text2p5 + '\n' + text3

    axs[i_subplot].text(axs[i_subplot].get_xlim()[1]*1.1, (axs[i_subplot].get_ylim()[1] + axs[i_subplot].get_ylim()[0])*0.5, text)

    # Labels and appearance
    axs[i_subplot].grid()

    # Adding legend
    axs[i_subplot].legend(loc='upper left', bbox_to_anchor=(0.03,0.2), bbox_transform=axs[i_subplot].transAxes)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    axs[i_subplot].tick_params(axis='both', which='major', labelsize=12)
    axs[i_subplot].ticklabel_format(axis='y', style='sci', scilimits=(4,4))

    i_subplot += 1


plt.suptitle('Array ' + str(args.array) + ' - Bar response vs Misalignment ', y=1.0,x=0.5, fontsize=15)
plt.tick_params(top=False, bottom=False, left=False, right=False)
plt.xlabel('Misalignment [mm]', labelpad=10, size=14, ha='center')
plt.ylabel('511 keV charge peak [pV * s]', labelpad=12, size=14, ha='left')

fig.align_labels()
fig.tight_layout(pad=0.6)

# Plot show/saving
plt.savefig(str(args.array)+'_LY_vs_Misalign_split.png',bbox_inches='tight')
#plt.show() #uncomment this if you want display plots while running code
