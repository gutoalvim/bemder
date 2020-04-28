# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 22:20:20 2020

@author: gutoa
"""
import numpy as np
import plotly
import matplotlib.pyplot as plt
from matplotlib import style
from bemder import receivers
style.use("seaborn-talk")

def plot_problem(obj,S=None,R=None,grid_pts=None, pT=None, mode="element", transformation=None):
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    from bempp.api import GridFunction
    from bempp.api.grid.grid import Grid
    import numpy as np
    
    if transformation is None:
        transformation = np.abs
        
    p2dB = lambda x: 20*np.log10(np.abs(x)/2e-5) 
    if transformation is None:
        transformation = np.abs
    if transformation == "dB":
        transformation = p2dB

    # plotly.offline.init_notebook_mode()

    if isinstance(obj, Grid):
        vertices = obj.vertices
        elements = obj.elements
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(255, 222, 173)"],
        )

        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        
        if R != None:
            fig.add_trace(go.Scatter3d(x = R.coord[:,0], y = R.coord[:,1], z = R.coord[:,2],mode='markers',name="Receivers"))
            
        if S != None:    
            if S.wavetype == "spherical":
                fig.add_trace(go.Scatter3d(x = S.coord[:,0], y = S.coord[:,1], z = S.coord[:,2],mode='markers',name="Sources"))
        
        # fig.show()
        plotly.offline.iplot(fig)

    elif isinstance(obj, GridFunction):


        grid = obj.space.grid
        vertices = grid.vertices
        elements = grid.elements

        local_coordinates = np.array([[1.0 / 3], [1.0 / 3]])
        values = np.zeros(grid.entity_count(0), dtype="float64")
        for element in grid.entity_iterator(0):
            index = element.index
            local_values = np.real(
                transformation(obj.evaluate(index, local_coordinates))
            )
            values[index] = local_values.flatten()

        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            color_func=values,
            colormap = "Jet",
            simplices=elements.T,
            show_colorbar = True,
 
        )
     
        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        if R != None:
            fig.add_trace(go.Scatter3d(x = R.coord[:,0], y = R.coord[:,1], z = R.coord[:,2],name="Receivers"))
            
        if S != None:    
            if S.wavetype == "spherical":
                fig.add_trace(go.Scatter3d(x = S.coord[:,0], y = S.coord[:,1], z = S.coord[:,2],name="Sources"))
        
            
        plotly.offline.iplot(fig)
        
def polar_plot(theta,p,normalize=True,transformation= None,ylim=[-40,0],title = None):
    import numpy as np
    p2dB = lambda x: 20*np.log10(np.abs(x)/2e-5) 
    if transformation is None:
        transformation = np.abs
    if transformation == "dB":
        transformation = p2dB
        P = transformation(p)
    

    if normalize == True:
        for i in range(P.shape[0]):
            P[i,:] = P[i,:]- np.ones_like(P[i,:])*np.amax(P[i,:])
    for i in range(p.shape[0]):
        plt.polar(theta, (P[i,:].reshape(len(theta),1)))
        plt.ylim(ylim)
        if title != None:
            plt.title(title)
        plt.show()
        
def polar_3d(R,ps):
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    
    Rr = receivers.Receiver()
    Rr.spherical_receivers(radius=1.0,ns=200, axis='x',random = False, plot=False)
    P = 1+np.real((20*np.log10(ps/2e-5) - max(20*np.log10(ps/2e-5)))/max(20*np.log10(ps/2e-5)))

    cx = Rr.coord[:,0]*0.5*P
    cy = Rr.coord[:,1]*0.5*P
    cz = Rr.coord[:,2]*0.5*P

    fig = go.Figure(data=[go.Mesh3d(x = cx,y=cy,z=cz,alphahull=-1)])
    fig.show()