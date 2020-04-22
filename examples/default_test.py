# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:06:02 2020

@author: gutoa
"""
from bemder import bem_api_new as BEM
s1 = BEM.ExteriorBEM()
bp1 = s1.hard_bemsolve()

s1.bem_save("test")
#%%
bp = s1.bem_load("test")


p1 = s1.point_evaluate(boundD=bp1)
