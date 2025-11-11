# -*- coding: utf-8 -*-
import numpy as np

def AWEvals(ival):
    Fc    = 96485 #Faradays constant, C/mol 
    Rc    = 8.314 #Gas constant     , J/mol.K 
    T0    = 353   #Temperature (80C), K
    # cin   = 6700  #KOH concentration, mol/m^3
    # Dkoh  = 3.2*10**-9 #Mass diff c., m^2/s
    i0c   = 1     #Exchange c, den  , A/m^2
    alc   = 0.5   #Charge trans. coeff.
    
    # kel   = 2*Dkoh*cin*Fc**2 / (Rc*T0)
    bvarg= np.arcsinh(ival/2/i0c)    
    eta  = bvarg*Rc*T0/(alc*Fc)
    
    ibv  = 2*i0c*np.sinh(alc*Fc*eta/Rc/T0)
    return eta,ibv

def Tanal(n,al,L,Q,bm,bp,dv,C):
    # L   = 2
    # al  = 0.51
    Qc  = np.max(abs(Q))    
    dx  = L/(n-2)
    xarr= np.linspace(-0-dx/2,L+dx/2,n)
    Tan = np.zeros_like(xarr)
    mask1 = (xarr >= 0) & (xarr < al)
    mask2 = (xarr >= al)& (xarr <= L)
    
    Tan[mask1] = (C-Qc*al+Qc*L)*xarr[mask1]/bm + dv
    Tan[mask2] = -Qc*(xarr[mask2]**2-al**2)/(2*bp) + Qc*L*(xarr[mask2]-al)/bp + (C -Qc*al + Qc*L)*al/bm + dv
    # Tan[0]     = 2*dv - Tan[1]
    # Tan[-1]    = Tan[-2]
    return Tan,xarr
