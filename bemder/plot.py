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

def plot_problem(obj,S=None,R=None,grid_pts=None, pT=None, opacity = 0.75, mode="element", transformation=None):
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    from bempp.api import GridFunction
    from bempp.api.grid.grid import Grid
    import numpy as np
    obj = obj[1]
    if transformation is None:
        transformation = np.abs
        
    p2dB = lambda x: 20*np.log10(np.abs(x)/2e-5) 
    if transformation is None:
        transformation = np.abs
    if transformation == "dB":
        transformation = p2dB

    def configure_plotly_browser_state():
        import IPython
        display(IPython.core.display.HTML('''
                <script src="/static/components/requirejs/require.js"></script>
                <script>
                  requirejs.config({
                    paths: {
                      base: '/static/base',
                      plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
                    },
                  });
                </script>
                '''))
            
    plotly.offline.init_notebook_mode()
    
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
        fig['data'][0].update(opacity=opacity)
        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        
        if R != None:
            fig.add_trace(go.Scatter3d(x = R.coord[:,0], y = R.coord[:,1], z = R.coord[:,2],marker=dict(size=8, color='rgb(0, 0, 128)', symbol='circle'),name="Receivers"))
            
        if S != None:    
            if S.wavetype == "spherical":
                fig.add_trace(go.Scatter3d(x = S.coord[:,0], y = S.coord[:,1], z = S.coord[:,2],marker=dict(size=8, color='rgb(128, 0, 0)', symbol='square'),name="Sources"))

        fig.add_trace(go.Mesh3d(x=[-6,6,-6,6], y=[-6,6,-6,6], z=0 * np.zeros_like([-6,6,-6,6]), color='red', opacity=0.5, showscale=False))

        configure_plotly_browser_state() 
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
        fig['data'][0].update(opacity=opacity)
        
        fig['layout']['scene'].update(go.layout.Scene(aspectmode='data'))
        if R != None:
            fig.add_trace(go.Scatter3d(x = R.coord[:,0], y = R.coord[:,1], z = R.coord[:,2],name="Receivers"))
            
        if S != None:    
            if S.wavetype == "spherical":
                fig.add_trace(go.Scatter3d(x = S.coord[:,0], y = S.coord[:,1], z = S.coord[:,2],name="Sources"))
        
        configure_plotly_browser_state()    
        plotly.iplot(fig)
        
def polar_plot(theta,p,normalize=True,transformation= None,s_number=0,ylim=[-40,0],title = None,n_average=0):
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
        
def polar_plot_3(P,R,AC,fi,n_average=7,ylim=[-30,0],title=None,hold=False,linestyle='-',linewidht=3,stackFig = None,normalize=True,color='tab:blue'):
    import numpy as np
    pDiffuser = P
    ppd = {}
    ddp = {}
    f_range = AC.freq
    a=0
    for i in range(int(len(f_range)/n_average)):
        dp = np.zeros_like(pDiffuser[0])
        rp = np.zeros_like(pDiffuser[0])# np.zeros((len(S.coord),pDiffuser[0][0].size),dtype=complex)
        iic=0
        ii=0
        for ii in range(n_average):
            iic = ii + (i*a)
            pD = np.abs(pDiffuser[iic,:])
            dp += pD #np.array([pD[c] for c in pD.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            # print(iic)
        # print("STOP")
        ddp[i] = dp/(n_average)
        
        ppd[i] = ddp[i]
        a=n_average

    if normalize == True:
        normalz = np.amax(20*np.log10(ppd[fi]/2e-5))
    else:
        normalz= 0

    if stackFig == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_thetamin(np.amin(np.rad2deg(R.theta)))
        ax.set_thetamax(np.amax(np.rad2deg(R.theta)))

    else:
        ax = stackFig

    ax.plot(R.theta,  20*np.log10(ppd[fi]/2e-5)-normalz,ls=linestyle,lw=linewidht,color=color)
    # plt.ylim(ylim)
    if title != None:
        plt.title(title)
    if hold == False:    
        plt.show()
        
    return ax,20*np.log10(ppd[fi]/2e-5)-normalz


def polar_plot_2(P,R,AC,fi,n_average=7,s_number=0,ylim=[-30,0],title=None,hold=False,scatter=False,linestyle='-',linewidht=3,stackFig = None,normalize=True):
    import numpy as np
    ir = s_number
    pDiffuser = P
    ppd = {}
    ddp = {}
    f_range = AC.freq


    a=0
    for i in range(int(len(f_range)/n_average)):
        dp = np.zeros_like(pDiffuser[0])
        rp = np.zeros_like(pDiffuser[0])# np.zeros((len(S.coord),pDiffuser[0][0].size),dtype=complex)
        iic=0
        ii=0
        for ii in range(n_average):
            iic = ii + (i*a)
            pD = np.abs(pDiffuser.get(iic))
            dp += pD #np.array([pD[c] for c in pD.keys()]).reshape(len(S.coord),(pDiffuser[0][0].size))
            # print(iic)
        # print("STOP")
        ddp[i] = dp/(n_average)
        
        ppd[i] = ddp[i]
        a=n_average
        
    if normalize == True:
        normalz = np.amax(20*np.log10(ppd[fi][ir,:]/2e-5))
    else:
        normalz= 0
    if scatter == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        c = ax.scatter(R.theta, 20*np.log10(ppd[fi][ir,:]/2e-5)-normalz)

    else:
        if stackFig == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
            ax.set_thetamin(np.amin(np.rad2deg(R.theta)))
            ax.set_thetamax(np.amax(np.rad2deg(R.theta)))

        else:
            ax = stackFig

        ax.plot(R.theta, 20*np.log10(ppd[fi][ir,:]/2e-5)-normalz,ls=linestyle,lw=linewidht)
    # plt.ylim(ylim)
    if title != None:
        plt.title(title)
    if hold == False:    
        plt.show()
        
    return ax
        
def polar_3d(R,ps):
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    
    Rr = receivers.Receiver()
    Rr.spherical_receivers(radius=1.0,ns=200, axis='x',random = False, plot=False)
    P = 1+np.real((20*np.log10(ps/2e-5) - np.amax(20*np.log10(ps/2e-5)))/np.amax(20*np.log10(ps/2e-5)))

    cx = Rr.coord[:,0]*0.5*P
    cy = Rr.coord[:,1]*0.5*P
    cz = Rr.coord[:,2]*0.5*P

    fig = go.Figure(data=[go.Mesh3d(x = cx,y=cy,z=cz,alphahull=-1)])
    fig.show()