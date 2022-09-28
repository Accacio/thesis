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

outputFolder= "../../img/resilient_eq/"
region_color="#9CDCF9";
color_map=plt.get_cmap('inferno')

fig, axs = plt.subplots(facecolor=(.0, .0, .0, .0))
delta = 0.01
x = np.arange(0, 3.0, delta)
y = np.arange(0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = (X-1.5)**2 + (Y-2)**2
Z = (Z1)
CS = axs.contour(X, Y, Z,extent=[-0,1,0,4],cmap=color_map,levels=10)
axs.plot(x,1.5-.5*x)
nx=1
ny=1.5-.5*nx
axs.plot(nx,ny,'bo',markersize=14)

rect = Polygon(((0, 0),(0,1.5),(3.,0)),color=region_color)
axs.add_patch(rect)

plt.xlim(min(x),max(x))
plt.ylim(min(y),max(y))
axs.clabel(CS, inline=1, fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)


plt.savefig(outputFolder + "original-minimum.pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "original-minimum.png",bbox_inches='tight',facecolor=fig.get_facecolor())

fig, axs = plt.subplots(facecolor=(.0, .0, .0, .0))
delta = 0.01
X, Y = np.meshgrid(x, y)
Z = (X-1.5)**2 + (Y-2.)**2
Z1 = 4*(X-1.5)**2 + (Y-2.)**2
# CS = ax.contour(X, Y, Z,extent=[-0,1,0,4],cmap=color_map,levels=10)
axs.plot(x,1.5-.5*x)
CS1 = axs.contour(X, Y, Z1,extent=[-0,1,0,4],cmap=color_map)
nx=1.5
ny=1.5-.5*nx
axs.plot(nx,ny,'go',markersize=14)
nx=1
ny=1.5-.5*nx


rect = Polygon(((0, 0),(0,1.5),(3.,0)),color=region_color)
axs.add_patch(rect)

axs.plot(nx,ny,'bo',markersize=14)
plt.xlim(min(x),max(x))
plt.ylim(min(y),max(y))
# ax.clabel(CS, inline=1, fontsize=10)
axs.clabel(CS1, inline=1, fontsize=16)
# ax.set_title('New minimum - Selfish Behavior.',fontsize=16)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)


plt.savefig(outputFolder + "new-minimum-selfish.pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "new-minimum-selfish.png",bbox_inches='tight',facecolor=fig.get_facecolor())

fig, axs = plt.subplots(facecolor=(.0, .0, .0, .0))

delta = 0.01
X, Y = np.meshgrid(x, y)
Z = (X-1.5)**2 + (Y-2.35)**2
Z1 = (Y-2.35)**2
# CS = ax.contour(X, Y, Z,extent=[-0,1,0,4],cmap=color_map,levels=10)
axs.plot(x,1.5-.5*x)
CS1 = axs.contour(X, Y, Z1,extent=[-0,1,0,4],cmap=color_map)
nx=0
ny=1.5-.5*nx
axs.plot(nx,ny,'o',markersize=14,color="#ff7f0e")
nx=1
ny=1.5-.5*nx
axs.plot(nx,ny,'bo',markersize=10)

rect = Polygon(((0, 0),(0,1.5),(3.,0)),color=region_color)
axs.add_patch(rect)

plt.xlim(min(x),max(x))
plt.ylim(min(y),max(y))
# ax.clabel(CS, inline=1, fontsize=10)
axs.clabel(CS1, inline=1, fontsize=16)
# ax.set_title('Ignore Selfish')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)


plt.savefig(outputFolder + "ignoreX.pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "ignoreX.png",bbox_inches='tight',facecolor=fig.get_facecolor())

fig, axs = plt.subplots(facecolor=(.0, .0, .0, .0))
delta = 0.01
X, Y = np.meshgrid(x, y)
Z = (X-1.5)**2 + (Y-2.35)**2
CS = axs.contour(X, Y, Z,extent=[-0,1,0,4],cmap=color_map,levels=10)
axs.plot(x,1.5-.5*x)
nx=1
ny=1.5-.5*nx
axs.plot(nx,ny,'bo',markersize=10)
circle=plt.Circle((nx,ny),0.1,color='r')
axs.add_artist(circle)


rect = Polygon(((0, 0),(0,1.5),(3.,0)),color=region_color)
axs.add_patch(rect)

plt.xlim(min(x),max(x))
plt.ylim(min(y),max(y))
axs.clabel(CS, inline=1, fontsize=16)
# ax.set_title('Correct Selfish')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.savefig(outputFolder + "correctX.pdf",bbox_inches='tight',facecolor=fig.get_facecolor())
plt.savefig(outputFolder + "correctX.png",bbox_inches='tight',facecolor=fig.get_facecolor())
