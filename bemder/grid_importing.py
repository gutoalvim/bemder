# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:51:47 2020

@author: gutoa
"""
# import numpy as np
try:  
    import gmsh
except :
    import gmsh_api.gmsh as gmsh
import bempp.api
import sys
import os
def import_grid(path_to_msh,show_mesh=False):
    """
    This function imports a .msh file and orders the domain_index from 0 to len(domain_index).

    Parameters
    ----------
    path : String
        Path to .msh file.

    Returns
    -------
    Bempp Grid.

    """
    
    gmsh.initialize(sys.argv)
    gmsh.open(path_to_msh) # Open msh
    phgr = gmsh.model.getPhysicalGroups(2)
    odph = []
    for i in range(len(phgr)):
        odph.append(phgr[i][1]) 
    phgr_ordered = [i for i in range(0, len(phgr))]
    phgr_ent = []
    for i in range(len(phgr)):
        phgr_ent.append(gmsh.model.getEntitiesForPhysicalGroup(phgr[i][0],phgr[i][1]))
    gmsh.model.removePhysicalGroups()
    for i in range(len(phgr)):
        gmsh.model.addPhysicalGroup(2, phgr_ent[i],phgr_ordered[i])
        
        
    path_name = os.path.dirname(path_to_msh)
    gmsh.write(path_name+'/current_mesh.msh')        
    if show_mesh == True:
        try:
            gmsh.fltk.run()
        except:
            bempp.api.PLOT_BACKEND = "jupyter"
            bempp.api.import_grid(path_name+'/current_mesh.msh').plot()
            

    gmsh.finalize()
    return bempp.api.import_grid(path_name+'/current_mesh.msh')