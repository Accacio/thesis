#!/usr/bin/env python3
# import scipy.io
# mat = scipy.io.loadmat('file.mat')
# nuitka --recurse-on --python-version=3.6 daoct;

from matplotlib import rc
from matplotlib import rcParams
from matplotlib.patches import Circle, Wedge, Polygon
import getopt
import math
# import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import os
import scipy.io as sio
import sys
import tikzplotlib
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
def latex_matrix(a):
    """Returns a LaTeX smallmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{matrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{matrix}']
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
outputFolder = "../../img/resilient_eq/"

nominal = sio.loadmat("../../data/resilient_eq/dmpc4rooms_chSetpoint__0_selfish__0_secure__0_.mat")
selfish = sio.loadmat("../../data/resilient_eq/dmpc4rooms_chSetpoint__0_selfish__1_secure__0_.mat")
corrected = sio.loadmat("../../data/resilient_eq/dmpc4rooms_chSetpoint__0_selfish__1_secure__1_.mat")
# nominal = sio.loadmat("../../../../docsThese/data/matlab/dmpc4rooms_chSetpoint__0_cheating__0_secure__0_.mat")
# selfish = sio.loadmat("../../../../docsThese/data/matlab/dmpc4rooms_chSetpoint__0_cheating__1_secure__0_.mat")
# corrected = sio.loadmat("../../../../docsThese/data/matlab/dmpc4rooms_chSetpoint__0_cheating__1_secure__1_.mat")

# exit()

# print(nominal['umin'])
# print(nominal['umax'])

# print(nominal['a'])
# print(nominal['b'])
# print(nominal['theta'])

simK=nominal['simK'][0][0]
subsystems=nominal['subsystems']

coordinator=nominal['coordinator']
nominal_Wt=subsystems['Wt'][0][0][0]
nominal_xt=subsystems['xt'][0][0]
err=coordinator['err'][0][0]
uHist=subsystems['uHist'][0][0]
lambda_hist=nominal['lambdaHist']

# nominal_err=nominal['norm_err']
# nominal_u=nominal['uHist'][0]


# exit()
# selfish_Wt=selfish['Wt']
# selfish_xt=selfish['xt']
# selfish_err=selfish['norm_err']
# selfish_u=selfish['uHist'][0]


# corrected_Wt=corrected['Wt']
# corrected_xt=corrected['xt']
# corrected_err=corrected['norm_err']
# corrected_u=corrected['uHist'][0]
dataFolder = "../../data/resilient_eq/"
with open(dataFolder + 'initial_states.tex', 'w') as f:
    print("${\\vec{x}_{\\text{I}}[0]=[" , latex_matrix(nominal_xt[:,0,0]), "]\T}$,", file=f)
    print("${\\vec{x}_{\\text{II}}[0]=[" , latex_matrix(nominal_xt[:,0,1]), "]\T}$,", file=f)
    print("${\\vec{x}_{\\text{III}}[0]=[" , latex_matrix(nominal_xt[:,0,2]), "]\T}$,", file=f)
    print("and",file=f)
    print("${\\vec{x}_{\\text{IV}}[0]=[", latex_matrix(nominal_xt[:,0,3]), "]\T}$", file=f,end="")

with open(dataFolder + 'references.tex', 'w') as f:
    print("${w_{\\text{I}}[0]=" , nominal_Wt[0][0], "}$,", file=f)
    print("${w_{\\text{II}}[0]=" , nominal_Wt[1][0], "}$,", file=f)
    print("${w_{\\text{III}}[0]=" , nominal_Wt[2][0], "}$,", file=f)
    print("and",file=f)
    print("${w_{\\text{IV}}[0]=", nominal_Wt[3][0], "}$", file=f,end="")

with open('../../data/resilient_eq/triche_matrix_I.tex', 'w') as f:
    print(smallmatrix((nominal['tau'][:,:,0])),file=f)

Cs   = nominal['Cs'  ][0]
Cres = nominal['Cres'][0]
Rf   = nominal['Rf'  ][0]
Ri   = nominal['Ri'  ][0]
Ro   = nominal['Ro'][0]
# print(Cs)
# print(Cres)
# print(Rf)
# print(Ri)
# print(Ro)

with open('../../data/resilient_eq/thermic_params.tex', 'w') as f:
    print("$C^{\\text{walls}}$ & $", Cs[0]  , "$ & $",Cs[1]  ,"$ & $",Cs[2]  ,"$ & $",Cs[3]  ,"$ & $","10^{4}\mathrm{J/K}$\\\\",file=f)
    print("$C^{\\text{air}}$   & $", Cres[0], "$ & $",Cres[1],"$ & $",Cres[2],"$ & $",Cres[3],"$ & $","10^{4}\mathrm{J/K}$\\\\",file=f)
    print("$R^{\\text{oa/ia}}$ & $", Rf[0]  , "$ & $",Rf[1]  ,"$ & $",Rf[2]  ,"$ & $",Rf[3]  ,"$ & $","10^{-3}\mathrm{K/W}$\\\\",file=f)
    print("$R^{\\text{iw/ia}}$ & $", Ri[0]  , "$ & $",Ri[1]  ,"$ & $",Ri[2]  ,"$ & $",Ri[3]  ,"$ & $","10^{-4}\mathrm{K/W}$\\\\",file=f)
    print("$R^{\\text{ow/oa}}$ & $", Ro[0]  , "$ & $",Ro[1]  ,"$ & $",Ro[2]  ,"$ & $",Ro[3]  ,"$ & $","10^{-4}\mathrm{K/W}$",file=f)

# with open('/dev/stdout', 'w') as f:
#     print("$C^{\\text{walls}}$ & $", Cs[0]  , "$ & $",Cs[1]  ,"$ & $",Cs[2]  ,"$ & $",Cs[3]  ,"$ & $","10^{4}\mathrm{J/K}$\\\\",file=f)
#     print("$C^{\\text{air}}$   & $", Cres[0], "$ & $",Cres[1],"$ & $",Cres[2],"$ & $",Cres[3],"$ & $","10^{4}\mathrm{J/K}$\\\\",file=f)
#     print("$R^{\\text{oa/ia}}$ & $", Rf[0]  , "$ & $",Rf[1]  ,"$ & $",Rf[2]  ,"$ & $",Rf[3]  ,"$ & $","10^{-3}\mathrm{K/W}$\\\\",file=f)
#     print("$R^{\\text{iw/ia}}$ & $", Ri[0]  , "$ & $",Ri[1]  ,"$ & $",Ri[2]  ,"$ & $",Ri[3]  ,"$ & $","10^{-4}\mathrm{K/W}$\\\\",file=f)
#     print("$R^{\\text{ow/oa}}$ & $", Ro[0]  , "$ & $",Ro[1]  ,"$ & $",Ro[2]  ,"$ & $",Ro[3]  ,"$ & $","10^{-4}\mathrm{K/W}$",file=f)



# with open('../../data/resilient_eq/triche_matrix_IV.tex', 'w') as f:
#     print(smallmatrix(nominal['tau'][:,:,0]),file=f)

nominal_J=2*subsystems['J'][0][0][0]+np.transpose(subsystems['const'][0][0])
nominal_sumJ=np.sum(nominal_J,axis=0)

subsystems=selfish['subsystems']
coordinator=selfish['coordinator']
selfWt=subsystems['Wt'][0][0][0]
selfish_xt=subsystems['xt'][0][0]
selferr=coordinator['err'][0][0]
selfuHist=subsystems['uHist'][0][0]
selfish_J=2*subsystems['J'][0][0][0]+np.transpose(subsystems['const'][0][0])
selfish_sumJ=np.sum(selfish_J,axis=0)

subsystems=corrected['subsystems']
coordinator=corrected['coordinator']
corrected_Wt=subsystems['Wt'][0][0][0]
corrected_xt=subsystems['xt'][0][0]
correcterr=coordinator['err'][0][0]
correctuHist=subsystems['uHist'][0][0]

corrected_J=2*subsystems['J'][0][0][0]+np.transpose(subsystems['const'][0][0])
corrected_sumJ=np.sum(corrected_J,axis=0)


nominal_I=nominal_sumJ[0]
nominal_II=nominal_sumJ[1]
nominal_III=nominal_sumJ[2]
nominal_IV=nominal_sumJ[3]
nominal_global=np.sum(nominal_sumJ)


# # ktotal=mat['ktotal'][0][0].astype(int)
# nominal_J=2*nominal['J']+nominal['cHist']
# nominal_sumJ=np.sum(nominal_J,axis=0)

# selfish_J=2*selfish['J']+selfish['cHist']
# selfish_sumJ=np.sum(selfish_J,axis=0)

# corrected_J=2*corrected['J']+corrected['cHist']
# corrected_sumJ=np.sum(corrected_J,axis=0)

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

# with open('../../data/resilient_eq/reference.tex','w') as f:
#     print(nominal_Wt[0][0],file=f)
# with open('../../data/resilient_eq/sampling.tex','w') as f:
#     print(nominal['Te'][0][0],file=f)
# with open('../../data/resilient_eq/simulation_time.tex','w') as f:
#     print(nominal['Te'][0][0]*nominal['simK'][0][0],file=f)
# with open('../../data/resilient_eq/prediction_horizon.tex','w') as f:
#     print(nominal['Np'][0][0],file=f)
# with open('../../data/resilient_eq/selfish_time_I.tex','w') as f:
#     print(nominal['selfish_time'][0][0],file=f)
# with open('../../data/resilient_eq/selfish_time_IV.tex','w') as f:
#     print(nominal['selfish_time'][3][0],file=f)
# with open('../../data/resilient_eq/hours.tex','w') as f:
#     print(nominal['Te'][0][0]*nominal['simK'][0][0],file=f)


with open('../../data/resilient_eq/table_costs_all_houses_error.tex', 'w') as f:
    print("I & $", round(nominal_I,1),"$ & $",round(selfish_I,1),"$ ($", round(100*(selfish_I-nominal_I)/nominal_I,1),"$)& $",round(corrected_I,1),"$ ($",100*round((corrected_I-nominal_I)/nominal_I,1) ,"$)\\\\",file=f)
    print("II & $", round(nominal_II,1),"$ & $",round(selfish_II,1),"$ ($", round(100*(selfish_II-nominal_II)/nominal_II,1),"$)& $",round(corrected_II,1),"$ ($",round(100*(corrected_II-nominal_II)/nominal_II,1) ,"$)\\\\",file=f)
    print("III & $", round(nominal_III,1),"$ & $",round(selfish_III,1),"$ ($", round(100*(selfish_III-nominal_III)/nominal_III,1),"$)& $",round(corrected_III,1),"$ ($",round(100*(corrected_III-nominal_III)/nominal_III,1) ,"$)\\\\",file=f)
    print("IV & $", round(nominal_IV,1),"$ & $",round(selfish_IV,1),"$ ($", round(100*(selfish_IV-nominal_IV)/nominal_IV,1),"$)& $",round(corrected_IV,1),"$ ($",round(100*(corrected_IV-nominal_IV)/nominal_IV,1) ,"$)\\\\",file=f)
    print("Global & $", round(nominal_global,1),"$ & $",round(selfish_global,1),"$ ($", round(100*(selfish_global-nominal_global)/nominal_global,1),"$)& $",round(corrected_global,1),"$ ($",round(100*(corrected_global-nominal_global)/nominal_global,1) ,"$)",file=f)

# with open('/dev/stdout', 'w') as f:
#     print("I & $", round(nominal_I,1),"$ & $",round(selfish_I,1),"$ ($", round(100*(selfish_I-nominal_I)/nominal_I,1),"$)& $",round(corrected_I,1),"$ ($",100*round((corrected_I-nominal_I)/nominal_I,1) ,"$)\\\\",file=f)
#     print("II & $", round(nominal_II,1),"$ & $",round(selfish_II,1),"$ ($", round(100*(selfish_II-nominal_II)/nominal_II,1),"$)& $",round(corrected_II,1),"$ ($",round(100*(corrected_II-nominal_II)/nominal_II,1) ,"$)\\\\",file=f)
#     print("III & $", round(nominal_III,1),"$ & $",round(selfish_III,1),"$ ($", round(100*(selfish_III-nominal_III)/nominal_III,1),"$)& $",round(corrected_III,1),"$ ($",round(100*(corrected_III-nominal_III)/nominal_III,1) ,"$)\\\\",file=f)
#     print("IV & $", round(nominal_IV,1),"$ & $",round(selfish_IV,1),"$ ($", round(100*(selfish_IV-nominal_IV)/nominal_IV,1),"$)& $",round(corrected_IV,1),"$ ($",round(100*(corrected_IV-nominal_IV)/nominal_IV,1) ,"$)\\\\",file=f)
#     print("Global & $", round(nominal_global,1),"$ & $",round(selfish_global,1),"$ ($", round(100*(selfish_global-nominal_global)/nominal_global,1),"$)& $",round(corrected_global,1),"$ ($",round(100*(corrected_global-nominal_global)/nominal_global,1) ,"$)",file=f)

# with open('../../data/resilient_eq/table_costs_only_global.tex', 'w') as f:
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

# plt.savefig(outputFolder + "barplot_results_poster" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
# plt.savefig(outputFolder + "barplot_results_poster" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

# print((corrected_global-nominal_global)/nominal_global)
# print((selfish_global-nominal_global)/nominal_global)

# # sys.exit()
# # print("Nominal")
# # print(np.sum((Wt[0:simK,0]-xt[0:simK,0])*(Wt[0:simK,0]-xt[0:simK,0])+uHist[0,-1,:,0]*uHist[0,-1,:,0])) # J1
# # print(np.sum((Wt[0:simK,1]-xt[0:simK,1])*(Wt[0:simK,1]-xt[0:simK,1])+uHist[0,-1,:,1]*uHist[0,-1,:,1])) # J2
# # print(np.sum((Wt[0:simK,2]-xt[0:simK,2])*(Wt[0:simK,2]-xt[0:simK,2])+uHist[0,-1,:,2]*uHist[0,-1,:,2])) # J3
# # print(np.sum((Wt[0:simK,3]-xt[0:simK,3])*(Wt[0:simK,3]-xt[0:simK,3])+uHist[0,-1,:,3]*uHist[0,-1,:,3])) # J4

# # print(np.sum((Wt[0:simK,0]-xt[0:simK,0])*(Wt[0:simK,0]-xt[0:simK,0])+uHist[0,-1,:,0]*uHist[0,-1,:,0])
# #       +np.sum((Wt[0:simK,1]-xt[0:simK,1])*(Wt[0:simK,1]-xt[0:simK,1])+uHist[0,-1,:,1]*uHist[0,-1,:,1]) # J2
# #       +np.sum((Wt[0:simK,2]-xt[0:simK,2])*(Wt[0:simK,2]-xt[0:simK,2])+uHist[0,-1,:,2]*uHist[0,-1,:,2]) # J3
# #       +np.sum((Wt[0:simK,3]-xt[0:simK,3])*(Wt[0:simK,3]-xt[0:simK,3])+uHist[0,-1,:,3]*uHist[0,-1,:,3])) # J4

# # print("Selfish")
# # print(np.sum((selfWt[0:simK,0]-selfxt[0:simK,0])*(selfWt[0:simK,0]-selfxt[0:simK,0])+selfuHist[0,-1,:,0]*selfuHist[0,-1,:,0])) # J1
# # print(np.sum((selfWt[0:simK,1]-selfxt[0:simK,1])*(selfWt[0:simK,1]-selfxt[0:simK,1])+selfuHist[0,-1,:,1]*selfuHist[0,-1,:,1])) # J2
# # print(np.sum((selfWt[0:simK,2]-selfxt[0:simK,2])*(selfWt[0:simK,2]-selfxt[0:simK,2])+selfuHist[0,-1,:,2]*selfuHist[0,-1,:,2])) # J3
# # print(np.sum((selfWt[0:simK,3]-selfxt[0:simK,3])*(selfWt[0:simK,3]-selfxt[0:simK,3])+selfuHist[0,-1,:,3]*selfuHist[0,-1,:,3])) # J4

# # print(np.sum((selfWt[0:simK,0]-selfxt[0:simK,0])*(selfWt[0:simK,0]-selfxt[0:simK,0])+selfuHist[0,-1,:,0]*selfuHist[0,-1,:,0]) # J1
# #       +np.sum((selfWt[0:simK,1]-selfxt[0:simK,1])*(selfWt[0:simK,1]-selfxt[0:simK,1])+selfuHist[0,-1,:,1]*selfuHist[0,-1,:,1]) # J2
# #       +np.sum((selfWt[0:simK,2]-selfxt[0:simK,2])*(selfWt[0:simK,2]-selfxt[0:simK,2])+selfuHist[0,-1,:,2]*selfuHist[0,-1,:,2]) # J3
# #       +np.sum((selfWt[0:simK,3]-selfxt[0:simK,3])*(selfWt[0:simK,3]-selfxt[0:simK,3])+selfuHist[0,-1,:,3]*selfuHist[0,-1,:,3])) # J4

# # print("Corrected")
# # print(np.sum((correctWt[0:simK,0]-corrected_xt[0:simK,0])*(correctWt[0:simK,0]-correctxt[0:simK,0])+correctuHist[0,-1,:,0]*correctuHist[0,-1,:,0])) # J1
# # print(np.sum((correctWt[0:simK,1]-corrected_xt[0:simK,1])*(correctWt[0:simK,1]-correctxt[0:simK,1])+correctuHist[0,-1,:,1]*correctuHist[0,-1,:,1])) # J2
# # print(np.sum((correctWt[0:simK,2]-corrected_xt[0:simK,2])*(correctWt[0:simK,2]-correctxt[0:simK,2])+correctuHist[0,-1,:,2]*correctuHist[0,-1,:,2])) # J3
# # print(np.sum((correctWt[0:simK,3]-corrected_xt[0:simK,3])*(correctWt[0:simK,3]-correctxt[0:simK,3])+correctuHist[0,-1,:,3]*correctuHist[0,-1,:,3])) # J4

# # print(np.sum((correctWt[0:simK,0]-corrected_xt[0:simK,0])*(correctWt[0:simK,0]-correctxt[0:simK,0])+correctuHist[0,-1,:,0]*correctuHist[0,-1,:,0]) # J1
# #       +np.sum((correctWt[0:simK,1]-corrected_xt[0:simK,1])*(correctWt[0:simK,1]-correctxt[0:simK,1])+correctuHist[0,-1,:,1]*correctuHist[0,-1,:,1]) # J2
# #       +np.sum((correctWt[0:simK,2]-corrected_xt[0:simK,2])*(correctWt[0:simK,2]-correctxt[0:simK,2])+correctuHist[0,-1,:,2]*correctuHist[0,-1,:,2]) # J3
# #       +np.sum((correctWt[0:simK,3]-corrected_xt[0:simK,3])*(correctWt[0:simK,3]-correctxt[0:simK,3])+correctuHist[0,-1,:,3]*correctuHist[0,-1,:,3])) # J4

# # NOTE(accacio): Detection
# fig, axs = plt.subplots(2, 1,facecolor=(.0, .0, .0, .0))

# axs[0].plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0],simK+1,1),'-',drawstyle='steps-post')
# axs[0].plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,0],'-',color='magenta',drawstyle='steps-post')
# axs[0].plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,0],'-',drawstyle='steps-post')
# axs[0].scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,0],s=15,color='black')

# axs[0].legend(( '$w_{\mathrm{I}}(k)$','$y_{\mathrm{I}}^N(k)$','$y_{\mathrm{I}}^S(k)$','$y_{\mathrm{I}}^C(k)$'),loc='bottom center',ncol=4,fontsize=13)

# axs[0].set_xticks(np.arange(0,simK+1,2))
# axs[0].set_xlim([1, simK])
# axs[0].set_ylim([15, 27])
# axs[0].set_title('Air temperature in house I ($^oC$)',fontsize=16)
# axs[0].tick_params(axis='both', which='major', labelsize=20)

# axs[1].plot(np.arange(1,simK+1),1e-4*np.ones([simK,1]),'-',drawstyle='steps-post',label='$\epsilon_p$') # error line
# axs[1].scatter(np.arange(1,simK+1),nominal_err[0:simK,0],color='magenta',s=10,label='$E_{\mathrm{I}}^N(k)$')
# axs[1].plot(np.arange(1,simK+1),selfish_err[0:simK,0],'-',color='darkorange',drawstyle='steps-post',label='$E_{\mathrm{I}}^S(k)$')
# axs[1].scatter(np.arange(1,simK+1),corrected_err[0:simK,0],color='black',s=10,label='$E_{\mathrm{I}}^C(k)$')

# handles,labels=axs[1].get_legend_handles_labels();
# handles=[handles[0],handles[2],handles[1],handles[3]];
# labels=[labels[0],labels[2],labels[1],labels[3]];
# axs[1].legend(handles,labels,loc='center',ncol=4,fontsize=13)

# axs[1].set_title("Norm of error $\| \\widehat{\\widetilde{P_{I}}}^{(0)}[k]-\\bar{P}^{(0)}_{I}\|_{F}$",fontsize=16)
# # \widehat{\widetilde{\Plin[#1]}}^{\left(#2\right)}[k]

# axs[1].set_xticks(np.arange(0,simK+1,2))
# axs[1].set_xlim([1, simK])
# axs[1].set_xlabel('Time (k)',usetex=True,fontsize=16)
# axs[1].tick_params(axis='both', which='major', labelsize=20)

# fig.tight_layout()
# rc('font', **serif_font)

# plt.savefig(outputFolder + "/__ErrorWX_command_normErrH" +  ".pdf",bbox_inches='tight',transparent=True)
# plt.savefig(outputFolder + "/__ErrorWX_command_normErrH" +  ".png",bbox_inches='tight',transparent=True)

# # plt.rcParams.update({'font.size': 50})
# # fig.tight_layout()
# # rc('font', **arial_font)


# fig, axs = plt.subplots(2, 1,facecolor=(.0, .0, .0, .0),figsize=(6,5))

# axs[0].plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0],simK+1,1),'-',drawstyle='steps-post')
# axs[0].plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,0],'-',color='magenta',drawstyle='steps-post')
# axs[0].plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,0],'-',drawstyle='steps-post')
# axs[0].scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,0],s=15,color='black')
# axs[0].set_title('Air temperature in house I ($^oC$)',fontsize=20)
# axs[0].set_ylim([15, 27])


# axs[1].plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[1],simK+1,1),'-',drawstyle='steps-post')
# axs[1].plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,1],'-',color='magenta',drawstyle='steps-post')
# axs[1].plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,1],'-',drawstyle='steps-post')
# axs[1].scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,1],s=15,color='black')
# axs[1].set_title('Air temperature in house II ($^oC$)',fontsize=20)
# axs[1].set_ylim([15, 27])

# axs[0].set_xlim([1, simK])
# axs[1].set_xlim([1, simK])

# axs[1].set_xlabel('Time (k)',usetex=True,fontsize=16)

# # axs[0].set_xticks(np.arange(0,simK+1,4))
# axs[0].set_xticks([])
# axs[1].set_xticks(np.arange(0,simK+1,4))
# axs[0].tick_params(axis='both', which='major', labelsize=20)
# axs[1].tick_params(axis='both', which='major', labelsize=20)

# axs[1].legend(( 'Reference i','Nominal','Agent I is Selfish','+ Correction'),loc='bottom', bbox_to_anchor=(.835, -0.5),ncol=2,fontsize=14)
# # axs[1].legend(handles,labels,loc='center',ncol=4,fontsize=15)

# fig.tight_layout()

# plt.savefig(outputFolder + "ErrorWX_command_normErrH_poster" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
# plt.savefig(outputFolder + "ErrorWX_command_normErrH_poster" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

# # NOTE(accacio): control
# fig, axs = plt.subplots(3, 1,facecolor=(.0, .0, .0, .0))

# axs[0].plot(np.arange(1,simK+1),nominal_u,'-',drawstyle='steps-post') # error line
# axs[0].legend(( '$u_{I}$', '$u_{II}$','$u_{III}$','$u_{IV}$'),loc='upper right',ncol=4,fontsize=14)
# axs[0].set_xticks(np.arange(0,simK+1,2))
# axs[0].set_xlim([1, simK])
# axs[0].set_title('Applied control $u_i$ ($kW$) ',fontsize=16)
# axs[0].text(2, 2, "N", ha="center", va="center",  size=16,
#     bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, fc="w", ec="gray", lw=2))
# axs[0].tick_params(axis='both', which='major', labelsize=20)

# axs[1].plot(np.arange(1,simK+1),selfish_u,'-',drawstyle='steps-post') # error line
# # axs[1].legend(( '$u_{I}$', '$u_{II}$','$u_{III}$','$u_{IV}$'),loc='upper right',ncol=4,fontsize=14)
# axs[1].set_xticks(np.arange(0,simK+1,2))
# axs[1].set_xlim([1, simK])
# axs[1].set_title('',fontsize=16)
# axs[1].text(2, 2, "S", ha="center", va="center",  size=16,
#     bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, fc="w", ec="gray", lw=2))
# axs[1].tick_params(axis='both', which='major', labelsize=20)

# axs[2].plot(np.arange(1,simK+1),corrected_u,'-',drawstyle='steps-post') # error line
# # axs[2].legend(( '$u_{I}$', '$u_{II}$','$u_{III}$','$u_{IV}$'),loc='upper right',ncol=4,fontsize=14)
# axs[2].set_xticks(np.arange(0,simK+1,2))
# axs[2].set_xlim([1, simK])
# axs[2].set_title('',fontsize=16)
# axs[2].text(2, 2, "C", ha="center", va="center",  size=16,
#     bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, fc="w",ec="gray",lw=2))
# axs[2].set_xlabel('Time (k)',usetex=True,fontsize=16)
# axs[2].tick_params(axis='both', which='major', labelsize=20)

# fig.tight_layout()
# rc('font', **serif_font)

# plt.savefig(outputFolder + "control" +  ".pdf",bbox_inches='tight',transparent=True)
# plt.savefig(outputFolder + "control" +  ".png",bbox_inches='tight',transparent=True)

# # rc('font', **arial_font)
# axs[0].set_xticks([])
# axs[1].set_xticks([])
# axs[2].set_xticks(np.arange(0,simK+1,4))
# axs[0].tick_params(axis='both', which='major', labelsize=20)
# axs[1].tick_params(axis='both', which='major', labelsize=20)
# axs[2].tick_params(axis='both', which='major', labelsize=20)

# axs[0].legend([])
# axs[1].legend([])
# axs[2].legend(( 'I', 'II','III','IV'),loc='bottom', bbox_to_anchor=(.835, -0.7),ncol=4,fontsize=14)
# axs[0].set_title('Applied control $u_i$ ($kW$) ',fontsize=20)
# fig.tight_layout()
# plt.savefig(outputFolder + "control_poster" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
# plt.savefig(outputFolder + "control_poster" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())

# NOTE(accacio): Costs
# fig, axs = plt.subplots(4, 1)
# axs[0].plot(np.arange(1,simK+1),nominal_J,'-',drawstyle='steps-post') # error line
# axs[0].plot(np.arange(1,simK+1),np.sum(nominal_J,axis=1),'-',drawstyle='steps-post') # error line
# axs[0].set_xticks(np.arange(0,simK+1,2))
# axs[0].set_xlim([1, simK])
# axs[0].set_title('',fontsize=16)
# axs[0].legend(( 'I', 'II','III','IV','Global'),loc='bottom right',ncol=5,fontsize=16)
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






k=0
# print(np.shape(lambda_hist))
# print(corrected_xt)
tau_index=10

fig, axs = plt.subplots(1, 1,figsize=(7, 3),facecolor=(.0, .0, .0, .0))
for i in np.arange(0,4,1):
    axs.plot(np.arange(0,100),np.transpose(lambda_hist[0,0:100,k,i]))
    axs.plot(np.arange(0,100),np.transpose(lambda_hist[1,0:100,k,i]))
    axs.plot(np.arange(0,100),np.transpose(lambda_hist[2,0:100,k,i]))
    axs.plot(np.arange(0,100),np.transpose(lambda_hist[3,0:100,k,i]))

# plt.show()
# axs.plot(np.arange(0,simK),np.transpose(lambda_hist[1,0:simK,k,0,tau_index]),color_map[1])
# axs.plot(np.arange(0,simK),np.transpose(lambda_hist[0,0:simK,k,1,tau_index]),color_map[2])
# axs.plot(np.arange(0,simK),np.transpose(lambda_hist[1,0:simK,k,1,tau_index]),color_map[3])
# axs.plot(np.arange(0,simK),np.transpose(lambda_hist[0,0:simK,k,2,tau_index]),color_map[4])
# axs.plot(np.arange(0,simK),np.transpose(lambda_hist[1,0:simK,k,2,tau_index]),color_map[5])

# ktotal=mat['ktotal'][0][0].astype(int)

# W=mat['W']
# Y=mat['Y']
fig, axs = plt.subplots(2, 1,facecolor=(.0, .0, .0, .0))

axs[0].plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0][0],simK+1,1),'-',drawstyle='steps-post')
# axs[0].plot(np.arange(0,simK+1),nominal_Wt[0:simK+1,0],'-',drawstyle='steps-post')
axs[0].plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,0],'-',color='magenta',drawstyle='steps-post')
axs[0].plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,0],'-',drawstyle='steps-post')
axs[0].scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,0],s=15,color='black')
# axs[0].set_ylim([0, 7])


# axs[0].legend(( '$Sub_1$', '$Sub_2$','$Sub_3$','$Sub_4$'),loc='upper center',ncol=4,fontsize=13)
# axs[0].legend(( '$Sub_1$', '$Sub_2$','$Sub_3$','$Sub_4$'),loc='upper center',ncol=4,fontsize=13)
# axs[0].legend(( '$I$', '$II$','$III$','$IV$'),loc='upper center',ncol=4,fontsize=13)
# axs[0].legend(( 'I', 'II','III','IV'),loc='upper center',ncol=4,fontsize=13)
axs[0].legend(( '$w_{\mathrm{I}}[k]$','$y_{\mathrm{I}}^N[k]$','$y_{\mathrm{I}}^S[k]$','$y_{\mathrm{I}}^C[k]$'),loc='upper center',ncol=4,fontsize=13)

# axs[0].legend(( '$I$', '$II$','$III$','$IV$'),loc='upper right',ncol=2,fontsize=13)
#
axs[0].set_xticks(np.arange(0,simK+1,1))
axs[0].set_xlim([1, simK])
axs[0].set_title('Air temperature in house I ($^oC$)',fontsize=16)
# axs[0].set_xlabel('Time (k)',usetex=True,fontsize=16)

# axs[2].plot(np.arange(1,simK+1),np.sum(uHist[0,-1,:,:],axis=1),'-r',drawstyle='steps-post')
# # axs[1].plot(np.arange(1,simK+1),4*np.ones([simK,1]),'ob',drawstyle='steps-post',fillstyle='none')

# axs[2].plot(np.arange(1,simK+1),uHist[0,-1,:,0],'-',drawstyle='steps-post')
# axs[2].plot(np.arange(1,simK+1),uHist[0,-1,:,1],'--',drawstyle='steps-post')
# axs[2].plot(np.arange(1,simK+1),uHist[0,-1,:,2],'-.',drawstyle='steps-post')
# axs[2].plot(np.arange(1,simK+1),uHist[0,-1,:,3],':',drawstyle='steps-post')

# axs[2].set_xlim([0, simK])
# axs[2].set_title('Command $u_i(k)$',usetex=True,fontsize=16)
# # axs[1].legend(( '$u_{max}$', '$\\Sigma u_{i}$', '$\\lambda_1$', '$\\lambda_2$','$\\lambda_3$','$\\lambda_4$'),loc='upper center',ncol=3,fontsize=13)
# axs[2].legend(( '$\\Sigma u_{i}$', '$u_1$', '$u_2$','$u_3$','$u_4$'),loc='upper center',ncol=5,fontsize=13)
# axs[2].set_xlabel('Time (k)',usetex=True,fontsize=16)

axs[1].plot(np.arange(1,simK+1),1e-4*np.ones([simK,1]),'-',drawstyle='steps-post',label='$\epsilon_p$') # error line
axs[1].scatter(np.arange(1,simK+1),np.linalg.norm(err[:,-1,:,0],axis=0),color='magenta',s=10,label='$E_{\mathrm{I}}^N[k]$')
axs[1].plot(np.arange(1,simK+1),np.linalg.norm(selferr[:,-1,:,0],axis=0),'-',color='darkorange',drawstyle='steps-post',label='$E_{\mathrm{I}}^S[k]$')
axs[1].scatter(np.arange(1,simK+1),np.linalg.norm(correcterr[:,-1,:,0],axis=0),color='black',s=10,label='$E_{\mathrm{I}}^C[k]$')

# axs[1].scatter(np.arange(1,simK+1),corrected_err[0:simK,0],color='black',s=10,label='$E_{\mathrm{I}}^C[k]$')

handles,labels=axs[1].get_legend_handles_labels();
handles=[handles[0],handles[2],handles[1],handles[3]];
labels=[labels[0],labels[2],labels[1],labels[3]];
axs[1].legend(handles,labels,loc='upper center',ncol=4,fontsize=13)

axs[1].set_title("Norm of error $\| \hat{P_i}-P_{0_i}\|$",fontsize=16)
axs[1].set_ylim([-1, 17])

axs[1].set_xticks(np.arange(0,simK+1,1))
axs[1].set_xlim([1, simK])

axs[1].set_xlabel('Time (k)',usetex=True,fontsize=16)
axs[0].tick_params(axis='both', which='major', labelsize=20)
axs[1].tick_params(axis='both', which='major', labelsize=20)

fig.tight_layout()
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
plt.savefig(outputFolder + "/" +  "ErrorWX_command_normErrH" +  ".pdf",bbox_inches='tight')
plt.savefig(outputFolder + "/" +  "ErrorWX_command_normErrH" +  ".png",bbox_inches='tight')

axs[0].set_title('Température de l\'air dans maison I ($^oC$)',fontsize=16)
axs[1].set_title("Norme de l'erreur $\| \hat{P_i}-P_{0_i}\|$",fontsize=16)
axs[1].set_xlabel('Temps (k)',usetex=True,fontsize=16)

plt.savefig(outputFolder + "/" +  "ErrorWX_command_normErrH" +  "_fr.pdf",bbox_inches='tight')
plt.savefig(outputFolder + "/" +  "ErrorWX_command_normErrH" +  "_fr.png",bbox_inches='tight')


fig, axs = plt.subplots(4, 1,facecolor=(.0, .0, .0, .0),figsize=(8,8))
ax0=plt.subplot(4,1,1,aspect=1/1.1)
ax1=plt.subplot(4,1,2,aspect=1/1.1)
ax2=plt.subplot(4,1,3,aspect=1/1.1)
ax3=plt.subplot(4,1,4,aspect=1/1.1)

ax0.plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0][0],simK+1,1),'-',drawstyle='steps-post')
ax0.plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,0],'-',color='magenta',drawstyle='steps-post')
ax0.plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,0],'-',drawstyle='steps-post')
ax0.scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,0],s=15,color='black')

ax1.plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0][1],simK+1,1),'-',drawstyle='steps-post')
ax1.plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,1],'-',color='magenta',drawstyle='steps-post')
ax1.plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,1],'-',drawstyle='steps-post')
ax1.scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,1],s=15,color='black')

ax2.plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0][2],simK+1,1),'-',drawstyle='steps-post')
ax2.plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,2],'-',color='magenta',drawstyle='steps-post')
ax2.plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,2],'-',drawstyle='steps-post')
ax2.scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,2],s=15,color='black')

ax3.plot(np.arange(0,simK+1),np.matlib.repmat(nominal_Wt[0][2],simK+1,1),'-',drawstyle='steps-post')
ax3.plot(np.arange(0,simK+1),nominal_xt[0,0:simK+1,2],'-',color='magenta',drawstyle='steps-post')
ax3.plot(np.arange(0,simK+1),selfish_xt[0,0:simK+1,2],'-',drawstyle='steps-post')
ax3.scatter(np.arange(0,simK+1),corrected_xt[0,0:simK+1,2],s=15,color='black')

ax0.set_title('Air temperature in house I ($^oC$)',fontsize=20)
ax1.set_title('Air temperature in house II ($^oC$)',fontsize=20)
ax2.set_title('Air temperature in house III ($^oC$)',fontsize=20)
ax3.set_title('Air temperature in house IV ($^oC$)',fontsize=20)

ax0.set_ylim([17.5, 21])
ax1.set_ylim([17.5, 21])
ax2.set_ylim([17.5, 21])
ax3.set_ylim([17.5, 21])

ax0.set_xlim([1, simK])
ax1.set_xlim([1, simK])
ax2.set_xlim([1, simK])
ax3.set_xlim([1, simK])

ax0.set_xticks([])
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xlabel('Time (k)',usetex=True,fontsize=16)
ax0.tick_params(axis='both', which='major', labelsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax3.tick_params(axis='both', which='major', labelsize=20)

ax1.legend(( '$w_{i}[k]$','$y_{i}^{N}[k]$','$y_{i}^{S}[k]$','$y_{i}^{C}[k]$'),loc='bottom', bbox_to_anchor=(1.275, 0.5),ncol=1,fontsize=16)

# axs[1].legend(handles,labels,loc='center',ncol=4,fontsize=15)

# fig.tight_layout()
plt.savefig(outputFolder + "ErrorWX_command_normErrH_all_houses" +  ".pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "ErrorWX_command_normErrH_all_houses" +  ".png",bbox_inches='tight',facecolor=fig.get_facecolor())


ax0.set_title('Température de l\'air dans maison I ($^oC$)',fontsize=20)
ax1.set_title('Température de l\'air dans maison II ($^oC$)',fontsize=20)
ax2.set_title('Température de l\'air dans maison III ($^oC$)',fontsize=20)
ax3.set_title('Température de l\'air dans maison IV ($^oC$)',fontsize=20)
ax3.set_xlabel('Temps (k)',usetex=True,fontsize=16)

plt.savefig(outputFolder + "ErrorWX_command_normErrH_all_houses" +  "_fr.pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "ErrorWX_command_normErrH_all_houses" +  "_fr.png",bbox_inches='tight',facecolor=fig.get_facecolor())











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
