'''
Created on 2015-12-27

@author: chelsea

Module containing common plotting routines
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

TEXT_SIZE = 20

def plot2DImage(img,extent,colourmap = plt.get_cmap('Greys'), aspect='auto',
                label_x = None, label_y = None, label_cb='', plotTitle = None,
                saveFile = None, plot=True, imkwargs = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if imkwargs is not None:
        cax = ax.imshow(img, cmap = colourmap, extent=extent, interpolation = 'none', **imkwargs)
    else:
        cax = ax.imshow(img, cmap=colourmap, extent=extent, interpolation = 'none')
    ax.set_aspect(aspect)
    cbar = fig.colorbar(cax)
    cbar.set_label(label_cb, rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    cbar.ax.tick_params(labelsize=TEXT_SIZE)
    plt.xticks(size=TEXT_SIZE)
    plt.yticks(size = TEXT_SIZE)
    plt.xlabel(label_x,size=TEXT_SIZE,labelpad=2)
    plt.ylabel(label_y,size=TEXT_SIZE,labelpad=1)
    plt.xticks(ticks = [-1.5, 0, 1.5])
    plt.yticks(ticks = [-1.5, 0, 1.5])
    #plt.title(plotTitle)
    if saveFile is not None:
        plt.savefig(saveFile + ".png",dpi=300)
        plt.savefig(saveFile + ".eps",format="eps",dpi=300)
    if plot:
        plt.show()
    else:
        plt.close()

def subplot2DImage(img,extent,colourmap = plt.get_cmap('Greys'), aspect='auto',
                label_x = None, label_y = None, label_cb='', plotTitle = None,
                saveFile = None, plot=True, imkwargs = None):
    ax = plt.gca()
    fig = plt.gcf()
    if imkwargs is not None:
        cax = ax.imshow(img, cmap = colourmap, extent=extent, interpolation = 'none', **imkwargs)
    else:
        cax = ax.imshow(img, cmap=colourmap, extent=extent, interpolation = 'none')
        
    ax.set_aspect(aspect)
    cbar = fig.colorbar(cax, ax = ax)
    cbar.set_label(label_cb, rotation=270,labelpad=TEXT_SIZE,size=TEXT_SIZE)
    cbar.ax.tick_params(labelsize = TEXT_SIZE)
    plt.xticks(size = TEXT_SIZE)
    plt.yticks(size = TEXT_SIZE)
    plt.xlabel(label_x,size=TEXT_SIZE,labelpad=2)
    plt.ylabel(label_y,size=TEXT_SIZE,labelpad=1)
    plt.xticks(ticks = [-1.5, 0, 1.5])
    plt.yticks(ticks = [-1.5, 0, 1.5])
    #plt.title(plotTitle)
    if saveFile is not None:
        plt.savefig(saveFile + ".png",dpi=300)
        plt.savefig(saveFile + ".eps",format="eps",dpi=300)
    if plot:
        plt.show()
    else:
        plt.close()
    
def plot1DSpectrum(x,y,fmt,color,xerr=None,yerr=None,xlog=False,
                   ylog = False, label_x = None,
                   label_y = None, plotTitle = None, legend = None,
                   saveFile = None):
    '''
    :param x: array of arrays if plotting multiple data sets (x axis)
    :param y: array of arrays if plotting multiple data sets (y axis)
    :param fmt:
    :param color:
    :param xerr: len(x) == len(xerr)
    :param yerr: len(x) == len(xerr)
    :param xlog: semilog plot in x
    :param ylog: semilog plot in y
    :param label_x: string label of x axis
    :param label_y: string label of y axis
    :param plotTitle: string label of title
    :param legend: len(x), array of strings representing data set labels in legend
    :param saveFile: string label of save file to save figure to
    :return: void
    '''
    fig = plt.figure()
    for i in range(0,len(x)):
        ax = fig.add_subplot(111)
        if fmt[i] == '-':
            thick=1
        else:
            thick=3
        if yerr is None:
            ax.errorbar(x[i],y[i],fmt=fmt[i],color=color[i],lw=thick)
        else:
            ax.errorbar(x[i],y[i],yerr=yerr[i],fmt=fmt[i],color=color[i],lw=thick)
    
    plt.grid(True)
    if ylog:
        plt.yscale('log')
    if xlog:
        plt.xscale('log')
    if legend is not None:
        plt.legend(legend,loc=0)
    plt.xlabel(label_x,size=TEXT_SIZE,labelpad = 1)
    plt.ylabel(label_y,size=TEXT_SIZE,labelpad = -3)
    plt.ylim([0,None])
    plt.xticks(size=TEXT_SIZE)
    plt.yticks(size=TEXT_SIZE)
    #plt.title(plotTitle)

    x=np.array(x)
    #XLOW = np.amin(np.reshape(x,np.size(x)))*0.97
    #print x
    '''
    XHIGH = np.max(x)*1.03
    XLOW = np.min(x) * 0.97
    plt.xlim([XLOW,XHIGH])
    '''

    if saveFile is not None:
        plt.savefig(saveFile + ".png", dpi=300)
        plt.savefig(saveFile + ".eps", format="eps", dpi=300)
    plt.show()

