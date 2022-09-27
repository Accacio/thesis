#!/usr/bin/env python3

import sys
import os
import getopt
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.font_manager as font_manager
from matplotlib import rc
from matplotlib import rcParams
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import tikzplotlib

import scipy.io as sio
import numpy as np
import numpy.matlib
from datetime import datetime, timedelta
import pandas as pd

serif_font = {'family': 'serif', 'serif': ['Computer Modern']};
arial_font = {'family': 'sans-serif'};
rc('font', **serif_font)
rc('text', usetex=True)
inputFile = "../../data/example_introduction/example.mat"
outputFolder= "../../img/example_introduction/"
mat = sio.loadmat(inputFile)
Jhist=mat['Jhist']
k=mat['k'][0][0]
taus=mat['taus'][0]
thetahist=mat['thetahist']
lambdahist=mat['lambdahist']
fig, ax = plt.subplots(figsize=(7,3),facecolor=(.0, .0, .0, .0))

one_day=timedelta(days=7)
days_minus_length=pd.to_datetime('2022-12-24')-(np.size(taus)-2)*one_day
dates=np.repeat(days_minus_length,np.size(taus))+np.arange(0,np.size(taus),1)*one_day

plt.plot(dates, np.sum(Jhist,axis=0))
plt.plot(dates, Jhist[0],'-')
plt.plot(dates, Jhist[1],'--')
plt.plot(dates, Jhist[2],'-.')
plt.plot(dates, Jhist[3],':')

import matplotlib.dates as dates
days=dates.DayLocator(interval=7)
day_format = dates.DateFormatter('%d/%m')

ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(day_format)
plt.ylim([0, 20])

ax.set_title('Global and local loss functions $J^\star$ and  $J_i^\star$', fontsize=16,usetex=True)

start = mdates.date2num(pd.to_datetime('2022-12-23'))
end = mdates.date2num(pd.to_datetime('2022-12-30'))
width = end - start

plt.xlim([mdates.date2num(pd.to_datetime('2022-10-17')), mdates.date2num(pd.to_datetime('2022-12-25'))])

rect = Rectangle((start, 0), width, 20, color='red',hatch='/',lw=0,fill=False)
ax.add_patch(rect)
plt.legend(('$J^\star$', '$J^\star_1$', '$J^\star_2$','$J^\star_3$','$J^\star_4$'),ncol=2,fontsize=15)
plt.savefig(outputFolder + "/example_J" +  ".pdf",facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "/example_J" +  ".png",facecolor=fig.get_facecolor())
