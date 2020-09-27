# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:51:47 2020

@author: gutoa
"""
# import numpy as np
import gmsh
import bempp.api
import sys
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
        
    if show_mesh == True:
        gmsh.fltk.run()
    gmsh.write('C:\\Users\\gutoa\\Documents\\UFSM\\TCC\\Bemder Projects\\Double_Mesh_Symmetry\\Mshs\\current_mesh.msh')
    gmsh.finalize()
    return bempp.api.import_grid('C:\\Users\\gutoa\\Documents\\UFSM\\TCC\\Bemder Projects\\Double_Mesh_Symmetry\\Mshs\\current_mesh.msh')