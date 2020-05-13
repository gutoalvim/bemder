# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:22:08 2020

@author: gutoa
"""

import bemder as bd
import bempp.api

AP = bd.controlsair.AirProperties() #Initialized class for air properties, allowing to set c0 and rho0 or calculate
                                # it from temperature and humidity.

AC = bd.controlsair.AlgControls(AP.c0) #Defines the Algorithm Controls

AC.third_octave_fvec(100,2500,7) #Using AlgControls method to generete a frequency vector of 7 frequencies per 1/3 octave band

#Loading Mesh
p1 = bempp.api.import_grid('Mshs/eric_06_06_0025.msh')
S = bd.sources.Source("spherical",coord=[0,0,1.525])

# S.arc_sources(100,10,[10,90],axis='y')
# S.plot()


R = bd.receivers.Receiver(coord=[0,0,0])
# R.arc_receivers(50,37,[0,180],axis = "y")
R.coord

BC = bd.BC(AC,AP)
BC.delany(1,10900,0.025)