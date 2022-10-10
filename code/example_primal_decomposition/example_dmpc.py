#!/usr/bin/env python3
# import scipy.io
# mat = scipy.io.loadmat('file.mat')
# nuitka --recurse-on --python-version=3.6 daoct;
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

example = sio.loadmat("../../data/example_primal_decomposition/example_dmpc.mat")
outputFolder= "../../img/example_primal_decomposition/"

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

plt.xlabel('Negotiation step ($p$)',usetex=True,fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend((  '$\\theta_{1_1}$', '$\\theta_{1_2}$', '$\\theta_{2_1}$', '$\\theta_{2_2}$','$\\theta_{3_1}$', '$\\theta_{3_2}$'),ncol=3,fontsize=16)
plt.savefig(outputFolder + "example_theta" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_theta" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

plt.xlabel('Pas de la négociation ($p$)',usetex=True,fontsize=16)
plt.savefig(outputFolder + "example_theta_fr" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_theta_fr" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())


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
plt.savefig(outputFolder + "example_lambda" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_lambda" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

plt.xlabel('Pas de la négociation ($p$)',usetex=True,fontsize=16)
plt.savefig(outputFolder + "example_lambda_fr" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_lambda_fr" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

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
plt.savefig(outputFolder + "example_state" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_state" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())
exit()


# example_J=2*example['J']+example['cHist']
# example_sumJ=np.sum(example_J,axis=0)

# example_I=example_sumJ[0]
# example_II=example_sumJ[1]
# example_III=example_sumJ[2]
# example_global=np.sum(example_sumJ)

# selfish_I=selfish_sumJ[0]
# selfish_II=selfish_sumJ[1]
# selfish_III=selfish_sumJ[2]
# selfish_IV=selfish_sumJ[3]
# selfish_global=np.sum(selfish_sumJ)


# corrected_I=corrected_sumJ[0]
# corrected_II=corrected_sumJ[1]
# corrected_III=corrected_sumJ[2]
# corrected_IV=corrected_sumJ[3]
# corrected_global=np.sum(corrected_sumJ)
# with open('../article/reference.tex','w') as f:
#     print(nominal_Wt[0][0],file=f)
# with open('../article/sampling.tex','w') as f:
#     print(nominal['Te'][0][0],file=f)
# with open('../article/simulation_time.tex','w') as f:
#     print(nominal['Te'][0][0]*nominal['simK'][0][0],file=f)
# with open('../article/prediction_horizon.tex','w') as f:
#     print(nominal['Np'][0][0],file=f)


# with open('../article/table_costs_all_rooms_error.tex', 'w') as f:
#     print("I & $", round(nominal_I,1),"$ ($",100*round((nominal_I-nominal_I)/nominal_I,1), "$)& $",round(selfish_I,1),"$ ($", 100*round((selfish_I-nominal_I)/nominal_I,1),"$)& $",round(corrected_I,1),"$ ($",100*round((corrected_I-nominal_I)/nominal_I,1) ,"$)\\\\",file=f)
#     print("II & $", round(nominal_II,1),"$ ($",100*round((nominal_II-nominal_II)/nominal_II,1), "$)& $",round(selfish_II,1),"$ ($", 100*round((selfish_II-nominal_II)/nominal_II,1),"$)& $",round(corrected_II,1),"$ ($",100*round((corrected_II-nominal_II)/nominal_II,1) ,"$)\\\\",file=f)
#     print("III & $", round(nominal_III,1),"$ ($",100*round((nominal_III-nominal_III)/nominal_III,1), "$)& $",round(selfish_III,1),"$ ($", 100*round((selfish_III-nominal_III)/nominal_III,1),"$)& $",round(corrected_III,1),"$ ($",100*round((corrected_III-nominal_III)/nominal_III,1) ,"$)\\\\",file=f)
#     print("IV & $", round(nominal_IV,1),"$ ($",100*round((nominal_IV-nominal_IV)/nominal_IV,1), "$)& $",round(selfish_IV,1),"$ ($", 100*round((selfish_IV-nominal_IV)/nominal_IV,1),"$)& $",round(corrected_IV,1),"$ ($",100*round((corrected_IV-nominal_IV)/nominal_IV,1) ,"$)\\\\",file=f)
#     print("Global & $", round(nominal_global,1),"$ ($",100*round((nominal_global-nominal_global)/nominal_global,1), "$)& $",round(selfish_global,1),"$ ($", 100*round((selfish_global-nominal_global)/nominal_global,1),"$)& $",round(corrected_global,1),"$ ($",100*round((corrected_global-nominal_global)/nominal_global,1) ,"$)",file=f)


# with open('../article/table_costs_only_global.tex', 'w') as f:
#     print("Global & $", round(nominal_global,1),"$ ($",100*round((nominal_global-nominal_global)/nominal_global,1), "$)& $",round(selfish_global,1),"$ ($", 100*round((selfish_global-nominal_global)/nominal_global,1),"$)& $",round(corrected_global,1),"$ ($",100*round((corrected_global-nominal_global)/nominal_global,1) ,"$)",file=f)



# print("===")
# print("I &", round(nominal_I,2),"(",nominal_I/nominal_I, ")& ",round(selfish_I,2),"(", round((selfish_I-nominal_I)/nominal_I,2),")&",round(corrected_I,2)," (",round(corrected_I/nominal_I,2) ,")\\\\")
# print("II &", round(nominal_II,2),"(",nominal_II/nominal_II, ")& ",round(selfish_II,2),"(", round((selfish_II-nominal_II)/nominal_II,2),")&",round(corrected_II,2)," (",round(corrected_II/nominal_II,2) ,")\\\\")
# print("III &", round(nominal_III,2),"(",nominal_III/nominal_III, ")& ",round(selfish_III,2),"(", round((selfish_III-nominal_III)/nominal_III,2),")&",round(corrected_III,2)," (",round(corrected_III/nominal_III,2) ,")\\\\")
# print("IV &", round(nominal_IV,2),"(",nominal_IV/nominal_IV, ")& ",round(selfish_IV,2),"(", round((selfish_IV-nominal_IV)/nominal_IV,2),")&",round(corrected_IV,2)," (",round(corrected_IV/nominal_IV,2) ,")\\\\")
# print("Global &", round(nominal_global,2),"(",nominal_global/nominal_global, ")& ",round(selfish_global,2),"(", round((selfish_global-nominal_global)/nominal_global,2),")&",round(corrected_global,2)," (",round(corrected_global/nominal_global,2) ,")\\\\")
# print("===")
# print("I &", round(nominal_I,2),"(",nominal_I/nominal_I, ")& ",round(selfish_I,2),"(", round(selfish_I/nominal_I,2),")&",(corrected_I)," (",round(corrected_I/nominal_I,2) ,")\\\\")
# print("II &", round(nominal_II,2),"(",nominal_II/nominal_II, ")& ",round(selfish_II,2),"(", round(selfish_II/nominal_II,2),")&",(corrected_II)," (",round(corrected_II/nominal_II,2) ,")\\\\")
# print("III &", round(nominal_III,2),"(",nominal_III/nominal_III, ")& ",round(selfish_III,2),"(", round(selfish_III/nominal_III,2),")&",(corrected_III)," (",round(corrected_III/nominal_III,2) ,")\\\\")
# print("IV &", round(nominal_IV,2),"(",nominal_IV/nominal_IV, ")& ",round(selfish_IV,2),"(", round(selfish_IV/nominal_IV,2),")&",(corrected_IV)," (",round(corrected_IV/nominal_IV,2) ,")\\\\")
# print("Global &", round(nominal_global,2),"(",nominal_global/nominal_global, ")& ",round(selfish_global,2),"(", round(selfish_global/nominal_global,2),")&",(corrected_global)," (",round(corrected_global/nominal_global,2) ,")\\\\")

# fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))

# nominal_bar=(round(nominal_global,2),round(nominal_IV,2),round(nominal_III,2),round(nominal_II,2),round(nominal_I,2))
# selfish_bar=(round(selfish_global,2),round(selfish_IV,2),round(selfish_III,2),round(selfish_II,2),round(selfish_I,2))
# corrected_bar=(round(corrected_global,2),round(corrected_IV,2),round(corrected_III,2),round(corrected_II,2),round(corrected_I,2))
# labels=[ 'Global','IV', 'III', 'II', 'I']

# barHeight=0.25
# br_nominal = np.arange(5)
# br_selfish = [x - barHeight for x in br_nominal]
# br_corrected = [x - barHeight for x in br_selfish]

# plt.barh(br_nominal, nominal_bar, height=barHeight,label='Nominal') # nominal
# plt.barh(br_selfish, selfish_bar, height=barHeight,label='Selfish') # nominal
# plt.barh(br_corrected, corrected_bar, height=barHeight,label='+ Correction') # nominal
# plt.yticks([r - barHeight for r in range(5)],
#            labels,fontsize=20)
# plt.xticks(fontsize = 20)

# plt.ylabel('Agents', fontweight ='bold', fontsize = 25)
# plt.xlabel('Objective function', fontweight ='bold', fontsize = 25)
# plt.legend(fontsize=20)

# plt.savefig("../img/barplot_results_poster" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
# plt.savefig("../img/barplot_results_poster" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

# sys.exit()
# print("Nominal")
# print(np.sum((Wt[0:simK,0]-xt[0:simK,0])*(Wt[0:simK,0]-xt[0:simK,0])+uHist[0,-1,:,0]*uHist[0,-1,:,0])) # J1
# print(np.sum((Wt[0:simK,1]-xt[0:simK,1])*(Wt[0:simK,1]-xt[0:simK,1])+uHist[0,-1,:,1]*uHist[0,-1,:,1])) # J2
# print(np.sum((Wt[0:simK,2]-xt[0:simK,2])*(Wt[0:simK,2]-xt[0:simK,2])+uHist[0,-1,:,2]*uHist[0,-1,:,2])) # J3
# print(np.sum((Wt[0:simK,3]-xt[0:simK,3])*(Wt[0:simK,3]-xt[0:simK,3])+uHist[0,-1,:,3]*uHist[0,-1,:,3])) # J4

# print(np.sum((Wt[0:simK,0]-xt[0:simK,0])*(Wt[0:simK,0]-xt[0:simK,0])+uHist[0,-1,:,0]*uHist[0,-1,:,0])
#       +np.sum((Wt[0:simK,1]-xt[0:simK,1])*(Wt[0:simK,1]-xt[0:simK,1])+uHist[0,-1,:,1]*uHist[0,-1,:,1]) # J2
#       +np.sum((Wt[0:simK,2]-xt[0:simK,2])*(Wt[0:simK,2]-xt[0:simK,2])+uHist[0,-1,:,2]*uHist[0,-1,:,2]) # J3
#       +np.sum((Wt[0:simK,3]-xt[0:simK,3])*(Wt[0:simK,3]-xt[0:simK,3])+uHist[0,-1,:,3]*uHist[0,-1,:,3])) # J4

# print("Selfish")
# print(np.sum((selfWt[0:simK,0]-selfxt[0:simK,0])*(selfWt[0:simK,0]-selfxt[0:simK,0])+selfuHist[0,-1,:,0]*selfuHist[0,-1,:,0])) # J1
# print(np.sum((selfWt[0:simK,1]-selfxt[0:simK,1])*(selfWt[0:simK,1]-selfxt[0:simK,1])+selfuHist[0,-1,:,1]*selfuHist[0,-1,:,1])) # J2
# print(np.sum((selfWt[0:simK,2]-selfxt[0:simK,2])*(selfWt[0:simK,2]-selfxt[0:simK,2])+selfuHist[0,-1,:,2]*selfuHist[0,-1,:,2])) # J3
# print(np.sum((selfWt[0:simK,3]-selfxt[0:simK,3])*(selfWt[0:simK,3]-selfxt[0:simK,3])+selfuHist[0,-1,:,3]*selfuHist[0,-1,:,3])) # J4

# print(np.sum((selfWt[0:simK,0]-selfxt[0:simK,0])*(selfWt[0:simK,0]-selfxt[0:simK,0])+selfuHist[0,-1,:,0]*selfuHist[0,-1,:,0]) # J1
#       +np.sum((selfWt[0:simK,1]-selfxt[0:simK,1])*(selfWt[0:simK,1]-selfxt[0:simK,1])+selfuHist[0,-1,:,1]*selfuHist[0,-1,:,1]) # J2
#       +np.sum((selfWt[0:simK,2]-selfxt[0:simK,2])*(selfWt[0:simK,2]-selfxt[0:simK,2])+selfuHist[0,-1,:,2]*selfuHist[0,-1,:,2]) # J3
#       +np.sum((selfWt[0:simK,3]-selfxt[0:simK,3])*(selfWt[0:simK,3]-selfxt[0:simK,3])+selfuHist[0,-1,:,3]*selfuHist[0,-1,:,3])) # J4

# print("Corrected")
# print(np.sum((correctWt[0:simK,0]-correctxt[0:simK,0])*(correctWt[0:simK,0]-correctxt[0:simK,0])+correctuHist[0,-1,:,0]*correctuHist[0,-1,:,0])) # J1
# print(np.sum((correctWt[0:simK,1]-correctxt[0:simK,1])*(correctWt[0:simK,1]-correctxt[0:simK,1])+correctuHist[0,-1,:,1]*correctuHist[0,-1,:,1])) # J2
# print(np.sum((correctWt[0:simK,2]-correctxt[0:simK,2])*(correctWt[0:simK,2]-correctxt[0:simK,2])+correctuHist[0,-1,:,2]*correctuHist[0,-1,:,2])) # J3
# print(np.sum((correctWt[0:simK,3]-correctxt[0:simK,3])*(correctWt[0:simK,3]-correctxt[0:simK,3])+correctuHist[0,-1,:,3]*correctuHist[0,-1,:,3])) # J4

# print(np.sum((correctWt[0:simK,0]-correctxt[0:simK,0])*(correctWt[0:simK,0]-correctxt[0:simK,0])+correctuHist[0,-1,:,0]*correctuHist[0,-1,:,0]) # J1
#       +np.sum((correctWt[0:simK,1]-correctxt[0:simK,1])*(correctWt[0:simK,1]-correctxt[0:simK,1])+correctuHist[0,-1,:,1]*correctuHist[0,-1,:,1]) # J2
#       +np.sum((correctWt[0:simK,2]-correctxt[0:simK,2])*(correctWt[0:simK,2]-correctxt[0:simK,2])+correctuHist[0,-1,:,2]*correctuHist[0,-1,:,2]) # J3
#       +np.sum((correctWt[0:simK,3]-correctxt[0:simK,3])*(correctWt[0:simK,3]-correctxt[0:simK,3])+correctuHist[0,-1,:,3]*correctuHist[0,-1,:,3])) # J4

# plt.savefig("../img/airtemp_roomI" + "/control" +  ".pdf",bbox_inches='tight',transparent=True)
# plt.savefig("../img/airtemp_roomI" + "/control" +  ".png",bbox_inches='tight',transparent=True)

# plt.savefig("../img/airtemp_roomI" + "/control_poster" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
# plt.savefig("../img/airtemp_roomI" + "/control_poster" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

# NOTE(accacio): Costs
# fig, axs = plt.subplots(1, 1)
# axs.plot(np.arange(0,simK),example_J,'-',drawstyle='steps-post') # error line
# axs.plot(np.arange(0,simK),np.sum(example_J,axis=1),'-',drawstyle='steps-post') # error line
# axs.set_xticks(np.arange(0,simK+1,2))
# axs.set_xlim([1, simK])
# axs.set_title('',fontsize=16)
# axs.legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=16)
# print("oi")
# plt.show()


# axs[1].plot(np.arange(1,simK+1),(selfish_J),'-',drawstyle='steps-post') # error line
# axs[1].plot(np.arange(1,simK+1),np.sum(selfish_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[1].legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=16)
# axs[2].plot(np.arange(1,simK+1),(corrected_J),'-',drawstyle='steps-post') # error line
# axs[2].plot(np.arange(1,simK+1),np.sum(corrected_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[2].legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=16)
# axs[3].plot(np.arange(1,simK+1),np.sum(nominal_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[3].plot(np.arange(1,simK+1),np.sum(selfish_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[3].plot(np.arange(1,simK+1),np.sum(corrected_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[3].legend(( 'N', 'S','C'),loc='bottom right',ncol=3,fontsize=16)

# plt.show()
# sys.exit()

# plt.savefig("../img/airtemp_roomI" + "/control" +  ".pdf",bbox_inches='tight')
# plt.savefig("../img/airtemp_roomI" + "/control" +  ".png",bbox_inches='tight')

# plot(reshape(subsystems.uHist(1,end,:,i),[simK 1]),linS{i},'Color',colors{i})

# axs[0].plot(np.arange(1,simK+1), Wt[0,0:0+simK],drawstyle='steps')
# axs[0].plot(np.arange(0,ktotal), W[0,0:0+ktotal],'r',drawstyle='steps')
# axs[0].plot(np.arange(2,ktotal+1), W[0,0:0+ktotal-1],'--r',drawstyle='steps')
# # axs[0].set_xlabel('Iteration (k)',usetex=True)
# axs[0].set_title('Temperature of room 1',fontsize=16)
# axs[0].set_ylim([18 ,22])
# axs[0].set_yticks(np.arange(18,23,1))
# axs[0].set_xticks(np.arange(0,ktotal+1,1))
# axs[0].set_xlabel('Time',usetex=True,fontsize=16)
# axs[1].plot(np.arange(1,ktotal+1), Y[1,0:0+ktotal],drawstyle='steps')
# axs[1].plot(np.arange(2,ktotal+1), W[1,0:0+ktotal-1],'--r',drawstyle='steps')
# axs[1].set_ylim([18 ,22])
# axs[1].set_yticks(np.arange(18,23,1))
# axs[1].set_xticks(np.arange(0,ktotal+1,1))
# # axs[1].set_xlabel('Iteration (k)',usetex=True)
# axs[1].set_title('Temperature of room 2',fontsize=16)
# # axs[2].plot(np.arange(0,ktotal+0), J[0,-1,:])
# # axs[2].set_title('Global cost $J^{\star}$')
# axs[1].set_xlabel('Time',usetex=True,fontsize=16)

# #
# #
# color="red"
# fig, axs = plt.subplots(2,1)
# axs[0].plot(np.arange(0,ktotal), Y[0,0:0+ktotal])
# # axs[0].set_xlabel('Iteration (k)',usetex=True)
# axs2 = axs[0].twinx()
# axs2.plot(np.arange(1,ktotal+1), W[0,0:0+ktotal],'--',drawstyle='steps',color=color)
# axs2.tick_params(axis='y', labelcolor=color)
# axs[0].set_title('Temperature of room 1',fontsize=16)

# axs2.set_ylim([19 ,22])
# axs[0].set_xlim([1 ,20])
# axs[0].set_ylim([16 ,18])
# axs[0].set_xticks(np.arange(1,20,1))
# axs[1].plot(np.arange(0,ktotal), Y[1,0:0+ktotal])
# # axs[1].set_xlabel('Iteration (k)',usetex=True)
# axs2 = axs[1].twinx()
# axs2.plot(np.arange(1,ktotal+1), W[1,0:0+ktotal],'--r',drawstyle='steps')
# axs2.tick_params(axis='y', labelcolor=color)
# axs[1].set_title('Temperature of room 2',fontsize=16)

# axs2.set_ylim([19 ,22])
# axs[1].set_ylim([17 ,18])
# axs[1].set_xlim([1 ,20])
# axs[1].set_xticks(np.arange(1,20,1))
# # axs[2].plot(np.arange(0,ktotal), J[0,-1,:])
# # axs[2].set_xlim([1 ,20])
# # axs[2].set_xticks(np.arange(1,20,1))
# # axs[2].set_ylim([60 ,110])
# # axs[2].set_title('Global cost $J^{\star}$',fontsize=16)
# axs[1].set_xlabel('Time',usetex=True,fontsize=16)
# fig.tight_layout()
# plt.savefig(outputFolder + "/" + os.path.basename(inputFile) + "__TempAndJtogether" +  ".pdf",bbox_inches='tight')
# plt.savefig(outputFolder + "/" + os.path.basename(inputFile) + "__TempAndJtogether" +  ".png",bbox_inches='tight')

# # EigenValues
# eigAest=mat['eigAestHist']

# plt.figure()
# plt.plot(np.arange(1,ktotal+1), eigAest[1:-1,0:0+ktotal].T,'*')

# # plt.ylim(1, 3)
# # plt.xlim(0, 20)

# plt.title('Estimated eigenvalues of $\\mathcal{A}$',usetex=True,fontsize=16)
# plt.xlabel('Time (k)',usetex=True,fontsize=16)
# # plt.legend(loc='right')

# plt.xticks(np.arange(1,21,1))



# plt.savefig(outputFolder + "/" + os.path.basename(inputFile) + "__eigAest" +  ".pdf",bbox_inches='tight')
# plt.savefig(outputFolder + "/" + os.path.basename(inputFile) + "__eigAest" +  ".png",bbox_inches='tight')

# # plt.show()
