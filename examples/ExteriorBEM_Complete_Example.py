# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:22:08 2020

@author: gutoa
"""

import bemder as bd
import bempp.api
import numpy as np
import matplotlib.pyplot as plt

AP = bd.controlsair.AirProperties() #Initialized class for air properties
AC = bd.controlsair.AlgControls(AP.c0,freq_vec= [100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500]) #Defines the Algorithm Controls

grid = bempp.api.import_grid('Mshs/eric_06_06_0025.msh') #Loading Mesh

S = bd.sources.Source("spherical") #Setting Sources
S.coord = np.array(([0,0,1.525],[0,0,-1.525])) #Custom source coordinates
S.q = np.array([1,1]) #Custom source amplitudes

R = bd.receivers.Receiver()
R.coord = np.array(([0,0,0.01+0.025],[0,0,0.0227+0.025])) #Setting Receivers

BC = bd.BC(AC,AP) #Setting Boundary Conditions
BC.delany(domain_index=1,RF=10900,d=0.025) #Attributes Delnay-Bazley porous impedance for domain_index=1

P1 = bd.ExteriorBEM(grid,AC,AP,S,R,BC.mu)

#%%

bSP1 = P1.impedance_bemsolve()
P1.bem_save('Mshs/Eric Porous/06_06_0025_delany/P1_sol')

#%%
ps,pt = P1.point_evaluate(bSP1,R)

#%%

plt.semilogx(AC.freq,np.abs(ps))

#%% To load saved data

# P1,bSP1 = bd.bem_load('Mshs/Eric Porous/06_06_0025_delany/P1_sol')

# ps,pt = P1.point_evaluate(bSP1,R)
