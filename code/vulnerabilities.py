#!/usr/bin/env python3
# import scipy.io
# mat = scipy.io.loadmat('file.mat')
# nuitka --recurse-on --python-version=3.6 daoct;
import sys
import os
import getopt
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.font_manager as font_manager
# import matplotlib
from matplotlib import rc
from matplotlib import rcParams
import tikzplotlib

import scipy.io as sio
import numpy as np
import numpy.matlib

import warnings

# shamelessly copied from https://stackoverflow.com/a/17131750/9781176 and modified
def smallmatrix(a):
    """Returns a LaTeX smallmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{smallmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{smallmatrix}']
    return '\n'.join(rv)

# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.sans-serif'] = 'cm'
serif_font = {'family': 'serif', 'serif': ['Computer Modern']};
arial_font = {'family': 'sans-serif'};
rc('font', **serif_font)
rc('text', usetex=True)
warnings.filterwarnings("ignore")
# config variables

example = sio.loadmat("../data/example_dmpc.mat")
lastp=np.transpose(example['lastp'])[0]
theta=example['theta']
lambdaHist=example['lambdaHist']
xt=example['xt'][0]
Wt=example['Wt']
simK=example['simK'][0][0]

# exit()

color_map = ['#ff7f0e','#FF9C45',  '#1f77b4','#7CBCE9','#1ca02c','#65E474']

k=5
fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,lastp[k]),np.transpose(theta[0,0:lastp[k],k,0]),color_map[0])
axs.plot(np.arange(0,lastp[k]),np.transpose(theta[1,0:lastp[k],k,0]),color_map[1])

axs.plot(np.arange(0,lastp[k]),np.transpose(theta[0,0:lastp[k],k,1]),color_map[2])
axs.plot(np.arange(0,lastp[k]),np.transpose(theta[1,0:lastp[k],k,1]),color_map[3])

axs.plot(np.arange(0,lastp[k]),np.transpose(theta[0,0:lastp[k],k,2]),color_map[4])
axs.plot(np.arange(0,lastp[k]),np.transpose(theta[1,0:lastp[k],k,2]),color_map[5])

plt.xlabel('Negotiation step (p)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\theta_{1_1}$', '$\\theta_{1_2}$', '$\\theta_{2_1}$', '$\\theta_{2_2}$','$\\theta_{3_1}$', '$\\theta_{3_2}$'),ncol=3,fontsize=16)
plt.savefig("../img/example_theta" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())

fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,0]),color_map[0])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,0]),color_map[1])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,1]),color_map[2])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,1]),color_map[3])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,2]),color_map[4])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,2]),color_map[5])

axs.plot(np.arange(0,lastp[k]),np.transpose(np.mean(lambdaHist[:,0:lastp[k],k,:],axis=2)),'--k')

plt.xlabel('Negotiation step (p)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\lambda_{1_1}$', '$\\lambda_{1_2}$', '$\\lambda_{2_1}$', '$\\lambda_{2_2}$','$\\lambda_{3_1}$', '$\\lambda_{3_2}$'),ncol=3,fontsize=16)
plt.savefig("../img/example_lambda" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())

fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,simK),np.repeat(Wt[0],simK),lineStyle='--',color=color_map[0])
axs.plot(np.arange(0,simK),np.transpose(xt[0:simK,0]),color_map[1])

axs.plot(np.arange(0,simK),np.repeat(Wt[1],simK),lineStyle='--',color=color_map[2])
axs.plot(np.arange(0,simK),np.transpose(xt[0:simK,1]),color_map[3])

axs.plot(np.arange(0,simK),np.repeat(Wt[2],simK),lineStyle='--',color=color_map[4])
axs.plot(np.arange(0,simK),np.transpose(xt[0:simK,2]),color_map[5])

plt.xlabel('Time ($k$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(('$w_1[k]$', '$x_1[k]$','$w_2[k]$', '$x_2[k]$','$w_3[k]$', '$x_3[k]$'),loc='right',fontsize=16)
plt.savefig("../img/example_state" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())


# plt.show()

exit()
