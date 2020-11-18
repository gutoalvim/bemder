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
def import_grid(path_to_msh,show_mesh=False,gmsh_filepath=None,reorder_domain_index=True):
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
    if reorder_domain_index:
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
            
        # gmsh.fltk.run()   
        path_name = os.path.dirname(path_to_msh)
        gmsh.write(path_name+'/current_mesh.msh')   
        gmsh.finalize()     
        if show_mesh == True:
            try:
                bempp.api.PLOT_BACKEND = "jupyter_notebook"
                bempp.api.import_grid(path_name+'/current_mesh.msh').plot()
            except:
                bempp.api.GMSH_PATH = gmsh_filepath
                bempp.api.PLOT_BACKEND = "gmsh"
                bempp.api.import_grid(path_name+'/current_mesh.msh').plot()
    
    
    
        msh = bempp.api.import_grid(path_name+'/current_mesh.msh')
        os.remove(path_name+'/current_mesh.msh')
    else:
        msh = bempp.api.import_grid(path_to_msh)
    return [path_to_msh,msh]

def import_geo(path_to_geo, max_freq, num_freq,show_mesh=False):
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
    gmsh.open(path_to_geo) # Open msh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 343/max_freq/num_freq)
    gmsh.model.mesh.generate(2)  
    # gmsh.fltk.run()
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
        
    

    path_name = os.path.dirname(path_to_geo)
    # gmsh.fltk.run()
    gmsh.write(path_name+'/current_mesh.msh')        
    # if show_mesh == True:
    #     try:
    #         bempp.api.PLOT_BACKEND = "jupyter_notebook"
    #         bempp.api.import_grid(path_name+'/current_mesh.msh').plot()
    #     except:
    #         # gmsh.fltk.run()
    #         # bempp.api.GMSH_PATH = gmsh_filepath
    #         # bempp.api.PLOT_BACKEND = "gmsh"
    #         # bempp.api.import_grid(path_name+'/current_mesh.msh').plot()



    
    gmsh.finalize() 
    return [path_to_geo,bempp.api.import_grid(path_name+'/current_mesh.msh')]