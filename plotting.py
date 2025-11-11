# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def contours1(X,Y,phi,phel,cden,ix,iy,facarr,fac,maxcol,phel_level):
    phel_level = phel_level*maxcol
    "Plots the contours of the entire field, for current density, its components and temperature"
    plt.figure()
    plt.contourf(X[1:-1,1: -1],Y[1:-1,1:-1],phel[1:-1,1:-1],vmax=maxcol,vmin=0,levels=phel_level) # ,vmax=maxcol,vmin=0,levels=phel_level
    plt.colorbar()
    plt.contour(X[1:-1,1: -1], Y[1:-1,1: -1], phi[1:-1,1: -1], levels=[0], colors='black',linestyles='dotted')  
    plt.xlabel('z (m)')
    plt.ylabel('x (m)')
    plt.title(r'Electric potential contour (V)')
    plt.savefig(r'bump_plots/scalar/phel_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
    plt.figure()
    plt.contourf(X[1:-1,1:-1],Y[1:-1,1:-1],cden[1:-1,1:-1],levels=60)
    plt.colorbar()
    plt.xlabel('z (m)')
    plt.ylabel('x (m)')
    plt.title(r'Current density magnitude $(\; A/m^2 \; )$')
    plt.savefig(r'bump_plots/scalar/cden_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
    plt.figure()
    plt.contourf(X[1:-1,1:-1],Y[1:-1,1:-1],ix[1:-1,1:-1],levels=60,cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('z (m)')
    plt.ylabel('x (m)')
    plt.title(r'Current density x component $(\; A/m^2 \; )$')
    plt.savefig(r'bump_plots/scalar/cdenx_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
    plt.figure()
    plt.contourf(X[1:-1,1:-1],Y[1:-1,1:-1],iy[1:-1,1:-1],levels=60,cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('z (m)')
    plt.ylabel('x (m)')
    plt.title(r'Current density z component $(\; A/m^2 \; )$')
    plt.savefig(r'bump_plots/scalar/cdenz_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
    
def contours2(X,Y,gx,gy,phel,phi,n,dx,facarr,fac,ulim,llim,maxcol2,mincol2): #,maxcol2,mincol2,phel_level
    "Plots the zoomed in contour of electric potential overlaid with gradient of potential."
    zoom_levels = np.linspace(mincol2, maxcol2, 60)
    xpl = X[llim:ulim,llim:ulim]
    ypl = Y[llim:ulim,llim:ulim]
    fld = phel[llim:ulim,llim:ulim]
    fldx= gx[llim:ulim,llim:ulim]
    fldy= gy[llim:ulim,llim:ulim]
    fldp= phi[llim:ulim,llim:ulim]
    g = 6
    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.contourf(xpl, ypl, fld, levels=zoom_levels, cmap='PuOr') #BrBG
    plt.colorbar()
    plt.contour(xpl, ypl, fldp,levels=[0],colors='white',linestyles='dotted',linewidths=1) #BrBG
    plt.quiver(xpl[::g,::g],ypl[::g,::g], fldx[::g,::g], fldy[::g,::g],scale=2.5e4, color='black', width=0.002)
    plt.title(r'Electric potential contour and gradients (V)')
    plt.savefig(r'bump_plots/vector/cdenvec_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=400)

