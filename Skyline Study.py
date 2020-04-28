# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:27:35 2020

@author: gutoa
"""

import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
import numpy as np
from bemder.bem_api_new import ExteriorBEM
import bemder.plot as bplt
import bemder.helpers as hh
from bemder import sources 
from bemder import receivers
from bemder import controlsair as ctrl
from bemder.bem_api_new import bem_load
from matplotlib import style
style.use("seaborn-talk")
#% Defining Air Properties and Algorithm Controls

AP = ctrl.AirProperties() #Initialized class for air properties, allowing to set c0 and rho0 or calculate
                                # it from temperature and humidity.


AP.c0 = 343
AP.rho0 = 1.21


AC = ctrl.AlgControls(AP.c0, 200,2500,500) #Defines frequencies of analysis

AC.freq = [500,1000,1500,2000,2500]
#% Load mesh

sky = 'Skyline_QRD.msh'
sky_ref = 'Skyline_REF.msh'


sky = bempp.api.import_grid('Mshs/Diffusers/Skyline/'+sky)
sky_ref = bempp.api.import_grid('Mshs/Diffusers/Skyline/'+sky_ref)

#Defining Sources and Receivers

S = sources.Source("spherical",coord=[200,0,0])
# S.set_ssph_sources(radius = 2.0, ns = 100, random = False, plot=False)
# S.arc_sources(200,37,[-90,90],axis = "z",random=False)
# S.plot()


R = receivers.Receiver()
R.spherical_receivers(radius = 20, ns = 100, axis='y',random = False, plot=False)
R.coord

#%%
SKY = ExteriorBEM(sky,AC,AP,S,R)
SKY_REF = ExteriorBEM(sky_ref,AC,AP,S,R)

#%%
bs = SKY.impedance_bemsolve()
SKY.bem_save("Mshs/Diffusers/Skyline/Skyline")
bsr = SKY_REF.impedance_bemsolve()
SKY_REF.bem_save("Mshs/Diffusers/QRD_Study/Skyline_ref")

#%%