# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:40:16 2020

@author: Zhi Li
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def plot3d (X, y, figname):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    color_type = ['orange','blue','red']
    for i in range(0,3):
        ax.scatter(X[:,0][y==i], X[:,1][y==i], X[:,2][y==i],\
                    c = color_type[i], s=5, alpha=0.5)   

    # set axis, limit.
    #ax.set_axis_off()
    ax.view_init(elev=7, azim=30)
    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    ax.set_zlim(-4,4)
    plt.savefig('Graphs/'+figname+'.svg', format = "svg", dpi = 1000, transparent=True)