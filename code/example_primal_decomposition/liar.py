#!/usr/bin/env python3

import sys
import os
import getopt
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
import matplotlib.font_manager as font_manager
# import matplotlib
from matplotlib import rc
from matplotlib import rcParams
import tikzplotlib

import scipy.io as sio
import numpy as np
import numpy.matlib

serif_font = {'family': 'serif', 'serif': ['Computer Modern']};
arial_font = {'family': 'sans-serif'};
rc('font', **serif_font)
rc('text', usetex=True)

color_map = ['#ff7f0e','#FF9C45',  '#1f77b4','#7CBCE9','#1ca02c','#65E474']
outputFolder= "../../img/example_primal_decomposition/"

liar = sio.loadmat("../../data/example_primal_decomposition/example_dmpc_liar.mat")
lastp=np.transpose(liar['lastp'])[0]
theta=liar['theta']
lambdaHist=liar['lambdaHist']
xt=liar['xt'][0]
Wt=liar['Wt']
simK=liar['simK'][0][0]

k=0
fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,0]),color_map[0])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,0]),color_map[1])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,1]),color_map[2])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,1]),color_map[3])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,2]),color_map[4])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,2]),color_map[5])

axs.plot(np.arange(0,lastp[k]),np.transpose(np.mean(lambdaHist[:,0:lastp[k],k,:],axis=2)),'--k')

plt.xlabel('Negotiation step ($p$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\lambda_{1_1}$', '$\\lambda_{1_2}$', '$\\lambda_{2_1}$', '$\\lambda_{2_2}$','$\\lambda_{3_1}$', '$\\lambda_{3_2}$'),ncol=3,fontsize=16)
plt.savefig(outputFolder + "example_liar_lambda_k_" + str(k)  +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_liar_lambda_k_" + str(k)  +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

k=4
fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,0]),color_map[0])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,0]),color_map[1])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,1]),color_map[2])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,1]),color_map[3])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,2]),color_map[4])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,2]),color_map[5])

axs.plot(np.arange(0,lastp[k]),np.transpose(np.mean(lambdaHist[:,0:lastp[k],k,:],axis=2)),'--k')

plt.xlabel('Negotiation step ($p$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\lambda_{1_1}$', '$\\lambda_{1_2}$', '$\\lambda_{2_1}$', '$\\lambda_{2_2}$','$\\lambda_{3_1}$', '$\\lambda_{3_2}$'),ncol=3,fontsize=16)
plt.savefig(outputFolder + "example_liar_lambda_k_" + str(k)  +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_liar_lambda_k_" + str(k)  +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

k=9
fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,0]),color_map[0])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,0]),color_map[1])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,1]),color_map[2])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,1]),color_map[3])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,2]),color_map[4])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,2]),color_map[5])

axs.plot(np.arange(0,lastp[k]),np.transpose(np.mean(lambdaHist[:,0:lastp[k],k,:],axis=2)),'--k')

plt.xlabel('Negotiation step ($p$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\lambda_{1_1}$', '$\\lambda_{1_2}$', '$\\lambda_{2_1}$', '$\\lambda_{2_2}$','$\\lambda_{3_1}$', '$\\lambda_{3_2}$'),ncol=3,fontsize=16)
plt.savefig(outputFolder + "example_liar_lambda_k_" + str(k)  +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_liar_lambda_k_" + str(k)  +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

# k=19
# fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
# axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,0]),color_map[0])
# axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,0]),color_map[1])
# axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,1]),color_map[2])
# axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,1]),color_map[3])
# axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[0,0:lastp[k],k,2]),color_map[4])
# axs.plot(np.arange(0,lastp[k]),np.transpose(lambdaHist[1,0:lastp[k],k,2]),color_map[5])

# axs.plot(np.arange(0,lastp[k]),np.transpose(np.mean(lambdaHist[:,0:lastp[k],k,:],axis=2)),'--k')

# plt.xlabel('Negotiation step ($p$)',usetex=True,fontsize=16)
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.legend((  '$\\lambda_{1_1}$', '$\\lambda_{1_2}$', '$\\lambda_{2_1}$', '$\\lambda_{2_2}$','$\\lambda_{3_1}$', '$\\lambda_{3_2}$'),ncol=3,fontsize=16)
# plt.savefig(outputFolder + "example_liar_lambda_k_" + str(k)  +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
# plt.savefig(outputFolder + "example_liar_lambda_k_" + str(k)  +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())



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
plt.savefig(outputFolder + "example_liar_state" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_liar_state" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())
