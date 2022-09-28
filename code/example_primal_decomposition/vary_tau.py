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

var_tau = sio.loadmat("../../data/example_primal_decomposition/example_dmpc_vary_tau.mat")
taus=var_tau['tau'][0]
var_tau_Ji=2*var_tau['J']+var_tau['cHist']
Ji_accum=np.sum(var_tau_Ji,axis=0)

fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
plt.plot(taus,np.sum(Ji_accum,axis=0),'b')
plt.plot(taus,Ji_accum[0,:],color_map[0])
plt.plot(taus,Ji_accum[1,:],color_map[2])
plt.plot(taus,Ji_accum[2,:],color_map[4])

rect = Rectangle((25, 0), 20, 1000, color='red',hatch='/',lw=0,fill=False)
axs.add_patch(rect)
j_legend=('$J^{\\mathrm{acc}}$', '$J^{\\mathrm{acc}}_{1}$', '$J^{\\mathrm{acc}}_{2}$','$J^{\\mathrm{acc}}_{3}$')

plt.legend(j_legend,ncol=2,fontsize=16)
plt.xlabel('Non-cooperative coefficient ($\\tau_{3}$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlim([0,40])
plt.ylim([0,900])
fig.tight_layout()
plt.savefig(outputFolder + "example_vary_tau_J" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_vary_tau_J" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())


fig, axs = plt.subplots(2, 1,figsize=(4, 3),facecolor=(.0, .0, .0, .0))
axs[0].plot(taus,np.sum(Ji_accum,axis=0),'b')
plt.plot(taus,Ji_accum[0,:],color_map[0])
plt.plot(taus,Ji_accum[1,:],color_map[2])
plt.plot(taus,Ji_accum[2,:],color_map[4])

rect = Rectangle((25, 0), 20, 1000, color='red',hatch='/',lw=0,fill=False)
# plt.legend(j_legend,ncol=2,fontsize=16)
plt.xlabel('Non-cooperative coefficient ($\\tau_{3}$)',usetex=True,fontsize=16)
# plt.xticks(fontsize = 20)
axs[0].set_yticks(np.arange(300,400,20))
axs[1].set_yticks(np.arange(100,160,25))
axs[0].tick_params(axis='both', which='major', labelsize=20)
axs[1].tick_params(axis='both', which='major', labelsize=20)

axs[0].set_xlim([0,2])
axs[1].set_xlim([0,2])
axs[0].set_ylim([320,360])
axs[1].set_ylim([100,150])
fig.tight_layout()

plt.savefig(outputFolder + "example_vary_tau_J_detail" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_vary_tau_J_detail" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

lambda_hist=var_tau['lambdaHist']
lastp=var_tau['lastp'][0]

k=5

tau_index=10
fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,0,tau_index]),color_map[0])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,0,tau_index]),color_map[1])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,1,tau_index]),color_map[2])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,1,tau_index]),color_map[3])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,2,tau_index]),color_map[4])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,2,tau_index]),color_map[5])
plt.xlim([0,60])

plt.xlabel('Negotiation step ($p$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\lambda_{1_1}$', '$\\lambda_{1_2}$', '$\\lambda_{2_1}$', '$\\lambda_{2_2}$','$\\lambda_{3_1}$', '$\\lambda_{3_2}$'),ncol=3,fontsize=16)

plt.savefig(outputFolder + "example_vary_tau_lambda_tau_" + str(taus[tau_index])  +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_vary_tau_lambda_tau_" + str(taus[tau_index])  +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())


tau_index=13
fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,0,tau_index]),color_map[0])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,0,tau_index]),color_map[1])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,1,tau_index]),color_map[2])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,1,tau_index]),color_map[3])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,2,tau_index]),color_map[4])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,2,tau_index]),color_map[5])
plt.xlim([0,60])
plt.xlabel('Negotiation step ($p$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\lambda_{1_1}$', '$\\lambda_{1_2}$', '$\\lambda_{2_1}$', '$\\lambda_{2_2}$','$\\lambda_{3_1}$', '$\\lambda_{3_2}$'),ncol=3,fontsize=16)

plt.savefig(outputFolder + "example_vary_tau_lambda_tau_" + str(taus[tau_index])  +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_vary_tau_lambda_tau_" + str(taus[tau_index])  +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

tau_index=18
fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,0,tau_index]),color_map[0])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,0,tau_index]),color_map[1])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,1,tau_index]),color_map[2])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,1,tau_index]),color_map[3])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,2,tau_index]),color_map[4])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,2,tau_index]),color_map[5])
plt.xlim([0,60])
plt.xlabel('Negotiation step ($p$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\lambda_{1_1}$', '$\\lambda_{1_2}$', '$\\lambda_{2_1}$', '$\\lambda_{2_2}$','$\\lambda_{3_1}$', '$\\lambda_{3_2}$'),ncol=3,fontsize=16)

plt.savefig(outputFolder + "example_vary_tau_lambda_tau_" + str(taus[tau_index])  +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_vary_tau_lambda_tau_" + str(taus[tau_index])  +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())


tau_index=22
fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,0,tau_index]),color_map[0])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,0,tau_index]),color_map[1])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,1,tau_index]),color_map[2])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,1,tau_index]),color_map[3])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[0,0:lastp[k],k,2,tau_index]),color_map[4])
axs.plot(np.arange(0,lastp[k]),np.transpose(lambda_hist[1,0:lastp[k],k,2,tau_index]),color_map[5])
plt.xlim([0,60])
plt.xlabel('Negotiation step ($p$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\lambda_{1_1}$', '$\\lambda_{1_2}$', '$\\lambda_{2_1}$', '$\\lambda_{2_2}$','$\\lambda_{3_1}$', '$\\lambda_{3_2}$'),loc='lower right',ncol=3,fontsize=16)

plt.savefig(outputFolder + "example_vary_tau_lambda_tau_" + str(taus[tau_index])  +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_vary_tau_lambda_tau_" + str(taus[tau_index])  +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

# plt.show()
