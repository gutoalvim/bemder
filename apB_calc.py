# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:18:28 2020

@author: gutoa
"""
import os
os.chdir('../bempp-cl')
import matplotlib.pyplot as plt
import pandas as pd
import bempp.api
os.chdir('../bemder')

import numpy as np
from bemder import ExteriorBEM
import bemder.plot as bplt
import bemder.helpers as hh
from bemder import sources 
from bemder import receivers
from bemder import controlsair as ctrl
from bemder.bem_api_new import bem_load
 
#% Defining Air Properties and Algorithm Controls

AP = ctrl.AirProperties() #Initialized class for air properties, allowing to set c0 and rho0 or calculate
                                # it from temperature and humidity.


AP.c0 = 343
AP.rho0 = 1.21


AC = ctrl.AlgControls(AP.c0)#, freq_vec= [100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500]) #Defines frequencies of analysis

AC.third_octave_fvec(100,800,7)
# AC.freq = [500,1000]#[#500,1000,1500,2000,2500]
#% Load mesh

p12 = bempp.api.import_grid('Mshs/200mm/p1_gmsh.msh')
p12_ref = bempp.api.import_grid('Mshs/200mm/p1_ref_gmsh.msh')
p12m = bempp.api.import_grid('Mshs/2m/p1_virtual_gmsh_800.msh')
p12m_ref = bempp.api.import_grid('Mshs/2m/p1_virtual_ref_800_gmsh.msh')
p14m = bempp.api.import_grid('Mshs/4m/p1_virtual_gmsh_800.msh')
p14m_ref = bempp.api.import_grid('Mshs/4m/p1_virtual_ref_gmsh_800.msh')

S = sources.Source("spherical")
S.coord = np.array(([100,0.04,0],[100,-0.04,0]))
S.q = np.array(([1],[1]))

Sr = sources.Source("spherical")
Sr.coord = np.array(([-100,0.04,0],[-100,-0.04,0]))
Sr.q = np.array(([1],[1]))

R = receivers.Receiver()
R.arc_receivers(50,37,[-90,90],axis = "y")

Rr = receivers.Receiver()
Rr.arc_receivers(50,37,[90,270],axis = "y")


P12 = ExteriorBEM(p12,AC,AP,S,R)
P12_inv = ExteriorBEM(p12,AC,AP,Sr,Rr)
P12_REF = ExteriorBEM(p12_ref,AC,AP,S,R)

P12m = ExteriorBEM(p12m,AC,AP,S,R)
P12m_inv = ExteriorBEM(p12m,AC,AP,Sr,Rr)
P12m_REF = ExteriorBEM(p12m_ref,AC,AP,S,R)

P14m = ExteriorBEM(p14m,AC,AP,S,R)
P14m_inv = ExteriorBEM(p14m,AC,AP,Sr,Rr)
P14m_REF = ExteriorBEM(p14m_ref,AC,AP,S,R)

bp12m = P12m.impedance_bemsolve(individual_sources=False)
P12m.bem_save('Solutions/2m/P1_gmsh_0')
bp12mr = P12m_REF.impedance_bemsolve(individual_sources=False)
P12m_REF.bem_save('Solutions/2m/P1_ref_gmsh_0')
bp12mi = P12m_inv.impedance_bemsolve(individual_sources=False)
P12m_inv.bem_save('Solutions/2m/P1_inv_gmsh_0')

bp14m = P14m.impedance_bemsolve(individual_sources=False)
P14m.bem_save('Solutions/4m/P1_gmsh_0')
bp14mr = P14m_REF.impedance_bemsolve(individual_sources=False)
P14m_REF.bem_save('Solutions/4m/P1_ref_gmsh_0')
bp14mi = P14m_inv.impedance_bemsolve(individual_sources=False)
P14m_inv.bem_save('Solutions/4m/P1_inv_gmsh_0')