#!/usr/bin/env python3

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib import rc

serif_font = {'family': 'serif', 'serif': ['Computer Modern']};
arial_font = {'family': 'sans-serif'};
rc('font', **serif_font)
rc('text', usetex=True)

outputFolder= "../../img/resilient_ineq/"

color_map=plt.get_cmap('Greys')

fig, axs = plt.subplots(facecolor=(.0, .0, .0, .0))


x1 = np.linspace(2,5,100);
x2 = np.linspace(-5,10,100);
[X1 ,X2 ] = np.meshgrid(x1,x2);
Z=1000/(np.sqrt(2*np.pi))*np.exp(-(X2-2*X1+4)**2/2)
CS = axs.contour(X1, X2, Z,cmap=color_map,levels=1000)

x1 = np.linspace(0,2,100);
x2 = np.linspace(-5,10,100);
[X1 ,X2 ] = np.meshgrid(x1,x2);
Z=1000/(np.sqrt(2*np.pi))*np.exp(-(X2-0*X1+0)**2/2)
CS = axs.contour(X1, X2, Z,cmap=color_map,levels=1000)
# axs.plot_surface(X1,X2,Z,cmap=color_map)

plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
# plt.ylim(-4,4)
# plt.xlim(0,4)

# axs.view_init(90,-90)
# plt.show()
fig.tight_layout()
plt.savefig(outputFolder + "pwa.pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "pwa.png",bbox_inches='tight',facecolor=fig.get_facecolor())
