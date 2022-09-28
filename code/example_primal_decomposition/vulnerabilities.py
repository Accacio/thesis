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


# plt.show()
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

k=19
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

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlim([0,40])
plt.ylim([0,900])
plt.legend(('$J^\star$', '$J^\star_1$', '$J^\star_2$','$J^\star_3$'),ncol=2,fontsize=16)
plt.savefig(outputFolder + "example_vary_tau_J" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_vary_tau_J" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

plt.xlim([0,2])
plt.ylim([50,400])
plt.legend(('$J^\star$', '$J^\star_1$', '$J^\star_2$','$J^\star_3$'),ncol=2,fontsize=16)
plt.savefig(outputFolder + "example_vary_tau_J_detail" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "example_vary_tau_J_detail" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())
sys.exit()


# fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
# plt.plot(taus,Ji_accum[0,:],color_map[0])
# plt.plot(taus,Ji_accum[1,:],color_map[2])
# plt.plot(taus,Ji_accum[2,:],color_map[4])
# plt.plot(taus,np.sum(Ji_accum,axis=0),'b')
# plt.xlim(0,20)
# plt.ylim(0,500)

# ktotal=mat['ktotal'][0][0].astype(int)
nominal_J=2*nominal['J']+nominal['cHist']
nominal_sumJ=np.sum(nominal_J,axis=0)

selfish_J=2*selfish['J']+selfish['cHist']
selfish_sumJ=np.sum(selfish_J,axis=0)

corrected_J=2*corrected['J']+corrected['cHist']
corrected_sumJ=np.sum(corrected_J,axis=0)

nominal_I=nominal_sumJ[0]
nominal_II=nominal_sumJ[1]
nominal_III=nominal_sumJ[2]
nominal_IV=nominal_sumJ[3]
nominal_global=np.sum(nominal_sumJ)

selfish_I=selfish_sumJ[0]
selfish_II=selfish_sumJ[1]
selfish_III=selfish_sumJ[2]
selfish_IV=selfish_sumJ[3]
selfish_global=np.sum(selfish_sumJ)


corrected_I=corrected_sumJ[0]
corrected_II=corrected_sumJ[1]
corrected_III=corrected_sumJ[2]
corrected_IV=corrected_sumJ[3]
corrected_global=np.sum(corrected_sumJ)
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



print("===")
print("I &", round(nominal_I,2),"(",nominal_I/nominal_I, ")& ",round(selfish_I,2),"(", round((selfish_I-nominal_I)/nominal_I,2),")&",round(corrected_I,2)," (",round(corrected_I/nominal_I,2) ,")\\\\")
print("II &", round(nominal_II,2),"(",nominal_II/nominal_II, ")& ",round(selfish_II,2),"(", round((selfish_II-nominal_II)/nominal_II,2),")&",round(corrected_II,2)," (",round(corrected_II/nominal_II,2) ,")\\\\")
print("III &", round(nominal_III,2),"(",nominal_III/nominal_III, ")& ",round(selfish_III,2),"(", round((selfish_III-nominal_III)/nominal_III,2),")&",round(corrected_III,2)," (",round(corrected_III/nominal_III,2) ,")\\\\")
print("IV &", round(nominal_IV,2),"(",nominal_IV/nominal_IV, ")& ",round(selfish_IV,2),"(", round((selfish_IV-nominal_IV)/nominal_IV,2),")&",round(corrected_IV,2)," (",round(corrected_IV/nominal_IV,2) ,")\\\\")
print("Global &", round(nominal_global,2),"(",nominal_global/nominal_global, ")& ",round(selfish_global,2),"(", round((selfish_global-nominal_global)/nominal_global,2),")&",round(corrected_global,2)," (",round(corrected_global/nominal_global,2) ,")\\\\")
print("===")
print("I &", round(nominal_I,2),"(",nominal_I/nominal_I, ")& ",round(selfish_I,2),"(", round(selfish_I/nominal_I,2),")&",(corrected_I)," (",round(corrected_I/nominal_I,2) ,")\\\\")
print("II &", round(nominal_II,2),"(",nominal_II/nominal_II, ")& ",round(selfish_II,2),"(", round(selfish_II/nominal_II,2),")&",(corrected_II)," (",round(corrected_II/nominal_II,2) ,")\\\\")
print("III &", round(nominal_III,2),"(",nominal_III/nominal_III, ")& ",round(selfish_III,2),"(", round(selfish_III/nominal_III,2),")&",(corrected_III)," (",round(corrected_III/nominal_III,2) ,")\\\\")
print("IV &", round(nominal_IV,2),"(",nominal_IV/nominal_IV, ")& ",round(selfish_IV,2),"(", round(selfish_IV/nominal_IV,2),")&",(corrected_IV)," (",round(corrected_IV/nominal_IV,2) ,")\\\\")
print("Global &", round(nominal_global,2),"(",nominal_global/nominal_global, ")& ",round(selfish_global,2),"(", round(selfish_global/nominal_global,2),")&",(corrected_global)," (",round(corrected_global/nominal_global,2) ,")\\\\")

fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))

nominal_bar=(round(nominal_global,2),round(nominal_IV,2),round(nominal_III,2),round(nominal_II,2),round(nominal_I,2))
selfish_bar=(round(selfish_global,2),round(selfish_IV,2),round(selfish_III,2),round(selfish_II,2),round(selfish_I,2))
corrected_bar=(round(corrected_global,2),round(corrected_IV,2),round(corrected_III,2),round(corrected_II,2),round(corrected_I,2))
labels=[ 'Global','IV', 'III', 'II', 'I']

barHeight=0.25
br_nominal = np.arange(5)
br_selfish = [x - barHeight for x in br_nominal]
br_corrected = [x - barHeight for x in br_selfish]

plt.barh(br_nominal, nominal_bar, height=barHeight,label='Nominal') # nominal
plt.barh(br_selfish, selfish_bar, height=barHeight,label='Selfish') # nominal
plt.barh(br_corrected, corrected_bar, height=barHeight,label='+ Correction') # nominal
plt.yticks([r - barHeight for r in range(5)],
           labels,fontsize=20)
plt.xticks(fontsize = 20)

plt.ylabel('Agents', fontweight ='bold', fontsize = 25)
plt.xlabel('Objective function', fontweight ='bold', fontsize = 25)
plt.legend(fontsize=20)

# plt.savefig("../img/barplot_results_poster" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
# plt.savefig("../img/barplot_results_poster" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

print((corrected_global-nominal_global)/nominal_global)
print((selfish_global-nominal_global)/nominal_global)

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

# NOTE(accacio): Detection
fig, axs = plt.subplots(2, 1,facecolor=(.0, .0, .0, .0))

axs[0].plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0],simK+1,1),'-',drawstyle='steps-post')
axs[0].plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,0],'-',color='magenta',drawstyle='steps-post')
axs[0].plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,0],'-',drawstyle='steps-post')
axs[0].scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,0],s=15,color='black')

axs[0].legend(( '$w_{\mathrm{I}}(k)$','$y_{\mathrm{I}}^N(k)$','$y_{\mathrm{I}}^S(k)$','$y_{\mathrm{I}}^C(k)$'),loc='bottom center',ncol=4,fontsize=13)

axs[0].set_xticks(np.arange(0,simK+1,2))
axs[0].set_xlim([1, simK])
axs[0].set_ylim([15, 27])
axs[0].set_title('Air temperature in room I ($^oC$)',fontsize=16)

axs[1].plot(np.arange(1,simK+1),1e-4*np.ones([simK,1]),'-',drawstyle='steps-post',label='$\epsilon_p$') # error line
axs[1].scatter(np.arange(1,simK+1),nominal_err[0:simK,0],color='magenta',s=10,label='$E_{\mathrm{I}}^N(k)$')
axs[1].plot(np.arange(1,simK+1),selfish_err[0:simK,0],'-',color='darkorange',drawstyle='steps-post',label='$E_{\mathrm{I}}^S(k)$')
axs[1].scatter(np.arange(1,simK+1),corrected_err[0:simK,0],color='black',s=10,label='$E_{\mathrm{I}}^C(k)$')

handles,labels=axs[1].get_legend_handles_labels();
handles=[handles[0],handles[2],handles[1],handles[3]];
labels=[labels[0],labels[2],labels[1],labels[3]];
axs[1].legend(handles,labels,loc='center',ncol=4,fontsize=13)

axs[1].set_title("Norm of error $\| \widehat{\\tilde{P}^{1}_I}[k]-\\bar{P}^{1}_{I}\|_{F}$",fontsize=16)

axs[1].set_xticks(np.arange(0,simK+1,2))
axs[1].set_xlim([1, simK])
axs[1].set_xlabel('Time (k)',usetex=True,fontsize=16)

fig.tight_layout()
rc('font', **serif_font)
# plt.savefig("../img/airtemp_roomI" + "/__ErrorWX_command_normErrH" +  ".pdf",bbox_inches='tight',transparent=True)
# plt.savefig("../img/airtemp_roomI" + "/__ErrorWX_command_normErrH" +  ".png",bbox_inches='tight',transparent=True)

# plt.rcParams.update({'font.size': 50})
# fig.tight_layout()
# rc('font', **arial_font)


fig, axs = plt.subplots(2, 1,facecolor=(.0, .0, .0, .0),figsize=(6,5))

axs[0].plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0],simK+1,1),'-',drawstyle='steps-post')
axs[0].plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,0],'-',color='magenta',drawstyle='steps-post')
axs[0].plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,0],'-',drawstyle='steps-post')
axs[0].scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,0],s=15,color='black')
axs[0].set_title('Air temperature in house I ($^oC$)',fontsize=20)
axs[0].set_ylim([15, 27])


axs[1].plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[1],simK+1,1),'-',drawstyle='steps-post')
axs[1].plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,1],'-',color='magenta',drawstyle='steps-post')
axs[1].plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,1],'-',drawstyle='steps-post')
axs[1].scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,1],s=15,color='black')
axs[1].set_title('Air temperature in house II ($^oC$)',fontsize=20)
axs[1].set_ylim([15, 27])

axs[0].set_xlim([1, simK])
axs[1].set_xlim([1, simK])

axs[1].set_xlabel('Time (k)',usetex=True,fontsize=16)

# axs[0].set_xticks(np.arange(0,simK+1,4))
axs[0].set_xticks([])
axs[1].set_xticks(np.arange(0,simK+1,4))
axs[0].tick_params(axis='both', which='major', labelsize=20)
axs[1].tick_params(axis='both', which='major', labelsize=20)

axs[1].legend(( 'Reference i','Nominal','Agent I is Selfish','+ Correction'),loc='bottom', bbox_to_anchor=(.835, -0.5),ncol=2,fontsize=14)
# axs[1].legend(handles,labels,loc='center',ncol=4,fontsize=15)

fig.tight_layout()
# plt.savefig("../img/airtemp_roomI" + "/__ErrorWX_command_normErrH_poster" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
# plt.savefig("../img/airtemp_roomI" + "/__ErrorWX_command_normErrH_poster" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

# NOTE(accacio): control
fig, axs = plt.subplots(3, 1,facecolor=(.0, .0, .0, .0))

axs[0].plot(np.arange(1,simK+1),nominal_u,'-',drawstyle='steps-post') # error line
axs[0].set_xticks(np.arange(0,simK+1,2))
axs[0].set_xlim([1, simK])
axs[0].set_title('Applied control $u_i$ ($kW$) ',fontsize=16)
axs[0].text(6, 3, "Nominal", ha="center", va="center",  size=16,
    bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, fc="w", ec="gray", lw=2))

axs[0].legend(( 'I', 'II','III','IV'),loc='upper right',ncol=4,fontsize=13)
axs[1].plot(np.arange(1,simK+1),selfish_u,'-',drawstyle='steps-post') # error line
axs[1].set_xticks(np.arange(0,simK+1,2))
axs[1].set_xlim([1, simK])
axs[1].set_title('',fontsize=16)
axs[1].text(6, 3, "Selfish", ha="center", va="center",  size=16,
    bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, fc="w", ec="gray", lw=2))

axs[1].legend(( 'I', 'II','III','IV'),loc='upper right',ncol=4,fontsize=13)
axs[2].plot(np.arange(1,simK+1),corrected_u,'-',drawstyle='steps-post') # error line
axs[2].set_xticks(np.arange(0,simK+1,2))
axs[2].set_xlim([1, simK])
axs[2].set_title('',fontsize=16)
axs[2].text(8, 3, "+ Correction", ha="center", va="center",  size=16,
    bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, fc="w",ec="gray",lw=2))
axs[2].legend(( 'I', 'II','III','IV'),loc='upper right',ncol=4,fontsize=13)
axs[2].set_xlabel('Time (k)',usetex=True,fontsize=16)

fig.tight_layout()
rc('font', **serif_font)
# plt.savefig("../img/airtemp_roomI" + "/control" +  ".pdf",bbox_inches='tight',transparent=True)
# plt.savefig("../img/airtemp_roomI" + "/control" +  ".png",bbox_inches='tight',transparent=True)

# rc('font', **arial_font)
axs[0].set_xticks([])
axs[1].set_xticks([])
axs[2].set_xticks(np.arange(0,simK+1,4))
axs[0].tick_params(axis='both', which='major', labelsize=20)
axs[1].tick_params(axis='both', which='major', labelsize=20)
axs[2].tick_params(axis='both', which='major', labelsize=20)

axs[0].legend([])
axs[1].legend([])
axs[2].legend(( 'I', 'II','III','IV'),loc='bottom', bbox_to_anchor=(.835, -0.7),ncol=4,fontsize=14)
axs[0].set_title('Applied control $u_i$ ($kW$) ',fontsize=20)
fig.tight_layout()



# plt.savefig("../img/airtemp_roomI" + "/control_poster" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
# plt.savefig("../img/airtemp_roomI" + "/control_poster" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

# NOTE(accacio): Costs
# fig, axs = plt.subplots(4, 1)
# axs[0].plot(np.arange(1,simK+1),nominal_J,'-',drawstyle='steps-post') # error line
# axs[0].plot(np.arange(1,simK+1),np.sum(nominal_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[0].set_xticks(np.arange(0,simK+1,2))
# axs[0].set_xlim([1, simK])
# axs[0].set_title('',fontsize=16)
# axs[0].legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=13)
# axs[1].plot(np.arange(1,simK+1),(selfish_J),'-',drawstyle='steps-post') # error line
# axs[1].plot(np.arange(1,simK+1),np.sum(selfish_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[1].legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=13)
# axs[2].plot(np.arange(1,simK+1),(corrected_J),'-',drawstyle='steps-post') # error line
# axs[2].plot(np.arange(1,simK+1),np.sum(corrected_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[2].legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=13)
# axs[3].plot(np.arange(1,simK+1),np.sum(nominal_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[3].plot(np.arange(1,simK+1),np.sum(selfish_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[3].plot(np.arange(1,simK+1),np.sum(corrected_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[3].legend(( 'N', 'S','C'),loc='bottom right',ncol=3,fontsize=13)

# plt.show()
sys.exit()

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
